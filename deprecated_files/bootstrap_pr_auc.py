import argparse
import gc
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score

from src.model import LSTMTransFormer
from src.utils import PTMDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap PR-AUC confidence intervals for the PTM ensemble."
    )

    parser.add_argument("--test-data", required=True)
    parser.add_argument("--model-dir", default="./new_model")
    parser.add_argument("--model-pattern", default="MIND_fold_{fold}")
    parser.add_argument("--pr-auc-json", default="./new_model/MIND_PRAU.json")
    parser.add_argument("--output-dir", default="./bootstrap_pr_auc")

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of valid bootstrap realizations per PTM.",
    )
    parser.add_argument(
        "--unit",
        choices=("protein", "site"),
        default="protein",
        help="Bootstrap proteins (cluster bootstrap) or individual evaluated sites.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help=(
            "Number of bootstrap units drawn per realization. "
            "0 uses the full observed number of units for each PTM."
        ),
    )
    parser.add_argument("--seed", type=int, default=252)

    # Model/data settings matching the supplied training code.
    parser.add_argument("--seq-len", type=int, default=514)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--fill-cont", type=int, default=5)

    parser.add_argument(
        "--graph",
        action="store_true",
        help="Use graph inputs if the trained fold models require them.",
    )
    parser.add_argument("--adj-dir", default="")

    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Prediction-cache pickle. If omitted, "
            "<output-dir>/ensemble_predictions.pkl is used."
        ),
    )
    parser.add_argument(
        "--recompute-predictions",
        action="store_true",
        help="Ignore an existing prediction cache and rerun all fold models.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create the bootstrap-distribution PDF.",
    )

    return parser.parse_args()


def make_generator_config(args):
    """Configuration object required by PTMDataGenerator."""
    return SimpleNamespace(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        fill_cont=args.fill_cont,
        graph=args.graph,
        adj_dir=args.adj_dir,
        ensemble=False,
        random_ensemble=False,
        data_path=".",
    )


def load_validation_pr_aucs(path):
    with open(path, "r") as f:
        scores = json.load(f)

    if not scores:
        raise ValueError(f"No fold PR-AUC values found in {path}")

    fold_ids = sorted(int(k) for k in scores.keys())
    expected = list(range(len(fold_ids)))
    if fold_ids != expected:
        raise ValueError(f"Expected consecutive fold IDs {expected}, found {fold_ids}.")

    unique_labels = sorted(scores[str(fold_ids[0])].keys())

    for fold in fold_ids:
        missing = set(unique_labels) - set(scores[str(fold)].keys())
        if missing:
            raise ValueError(
                f"Fold {fold} is missing PR-AUC values for: {sorted(missing)}"
            )

    return scores, fold_ids, unique_labels


def ensemble_weights(pr_aucs, fold_ids, unique_labels):
    """Reproduce ensemble_get_weights() from train_model.py."""
    weights = {}

    for ptm in unique_labels:
        w = np.asarray(
            [float(pr_aucs[str(fold)][ptm]) for fold in fold_ids],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(w)):
            raise ValueError(f"Non-finite validation PR-AUC weight for {ptm}: {w}")

        total = w.sum()
        if total <= 0:
            raise ValueError(f"Validation PR-AUC weights sum to <= 0 for {ptm}: {w}")

        weights[ptm] = w / total

    return weights


def build_test_generator(args, unique_labels):
    cfg = make_generator_config(args)

    test_aug = PTMDataGenerator(
        args.test_data,
        cfg,
        shuffle=False,
        ind=None,
        eval=True,
        binary=False,
        class_weights=None,
    )

    # The output-column ordering must match the trained models exactly.
    test_labels = list(test_aug.unique_labels)
    if test_labels != list(unique_labels):
        raise ValueError(
            "PTM labels in the curated test generator do not exactly match "
            "the labels stored in MIND_PRAU.json.\n"
            f"Test labels : {test_labels}\n"
            f"Model labels: {unique_labels}"
        )

    return test_aug


def batch_record_ids(aug, batch_number):
    """Recover record IDs used internally by PTMDataGenerator.__getitem__()."""
    lo = batch_number * aug.batch_size
    hi = (batch_number + 1) * aug.batch_size
    positions = aug.index[lo:hi]
    return [aug.list_id[k] for k in positions]


def collect_truth_groups(test_aug, unique_labels, seq_len):
    """
    Collect the exact evaluation targets selected by PTMDataGenerator and retain
    the protein identity associated with every evaluated residue/PTM example.
    """
    n_labels = len(unique_labels)
    y_true = {ptm: [] for ptm in unique_labels}
    groups = {ptm: [] for ptm in unique_labels}

    for batch_number in range(len(test_aug)):
        record_ids = batch_record_ids(test_aug, batch_number)
        batch_uids = [test_aug.uid[i] for i in record_ids]
        protein_ids = np.asarray(
            [uid.split("~", 1)[0] for uid in batch_uids], dtype=object
        )

        _, test_y, test_sw = test_aug[batch_number]
        y = np.asarray(test_y).reshape(-1, seq_len, n_labels)
        sw = np.asarray(test_sw).reshape(-1, seq_len, n_labels)

        if y.shape[0] != len(protein_ids):
            raise RuntimeError(
                f"Batch {batch_number}: generator returned {y.shape[0]} records "
                f"but {len(protein_ids)} UIDs were recovered."
            )

        group_grid = np.repeat(protein_ids[:, None], seq_len, axis=1)

        for j, ptm in enumerate(unique_labels):
            mask = sw[:, :, j] == 1
            if np.any(mask):
                y_true[ptm].append(y[:, :, j][mask].astype(np.int8))
                groups[ptm].append(group_grid[mask])

    for ptm in unique_labels:
        if not y_true[ptm]:
            raise RuntimeError(f"No evaluation examples were selected for {ptm}.")
        y_true[ptm] = np.concatenate(y_true[ptm])
        groups[ptm] = np.concatenate(groups[ptm]).astype(str)

    return y_true, groups


def predict_fold(model, test_aug, unique_labels, seq_len, batch_size):
    """Generate one fold's predictions in the same masked order as y_true."""
    n_labels = len(unique_labels)
    fold_pred = {ptm: [] for ptm in unique_labels}

    for batch_number in range(len(test_aug)):
        test_x, _, test_sw = test_aug[batch_number]

        pred = model.predict(test_x, batch_size=batch_size, verbose=0)
        pred = np.asarray(pred).reshape(-1, seq_len, n_labels)
        sw = np.asarray(test_sw).reshape(-1, seq_len, n_labels)

        for j, ptm in enumerate(unique_labels):
            mask = sw[:, :, j] == 1
            if np.any(mask):
                fold_pred[ptm].append(pred[:, :, j][mask].astype(np.float64))

    for ptm in unique_labels:
        if not fold_pred[ptm]:
            raise RuntimeError(f"Fold prediction produced no values for {ptm}.")
        fold_pred[ptm] = np.concatenate(fold_pred[ptm])

    return fold_pred


def generate_ensemble_predictions(args, test_aug, pr_aucs, fold_ids, unique_labels):
    """Load one fold at a time and accumulate the weighted ensemble prediction."""
    weights = ensemble_weights(pr_aucs, fold_ids, unique_labels)
    y_true, groups = collect_truth_groups(test_aug, unique_labels, args.seq_len)

    ensemble_pred = {
        ptm: np.zeros(y_true[ptm].shape[0], dtype=np.float64)
        for ptm in unique_labels
    }

    print("\nGenerating fold predictions:")
    for fold_position, fold in enumerate(fold_ids):
        model_path = Path(args.model_dir) / args.model_pattern.format(fold=fold)
        if not model_path.exists():
            raise FileNotFoundError(f"Fold model not found: {model_path}")

        print(f"  [{fold_position + 1:2d}/{len(fold_ids):2d}] {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)

        fold_pred = predict_fold(
            model,
            test_aug,
            unique_labels,
            args.seq_len,
            args.batch_size,
        )

        for ptm in unique_labels:
            if fold_pred[ptm].shape != y_true[ptm].shape:
                raise RuntimeError(
                    f"{ptm}: fold {fold} prediction length "
                    f"{fold_pred[ptm].size} does not match target length "
                    f"{y_true[ptm].size}."
                )

            ensemble_pred[ptm] += weights[ptm][fold_position] * fold_pred[ptm]

        del fold_pred
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    return {
        "unique_labels": unique_labels,
        "fold_ids": fold_ids,
        "y_true": y_true,
        "y_pred": ensemble_pred,
        "groups": groups,
    }


def load_or_generate_predictions(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        Path(args.cache)
        if args.cache is not None
        else output_dir / "ensemble_predictions.pkl"
    )

    if cache_path.exists() and not args.recompute_predictions:
        print(f"Loading cached ensemble predictions: {cache_path}")
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload, cache_path

    pr_aucs, fold_ids, unique_labels = load_validation_pr_aucs(args.pr_auc_json)
    test_aug = build_test_generator(args, unique_labels)

    payload = generate_ensemble_predictions(
        args, test_aug, pr_aucs, fold_ids, unique_labels
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved ensemble prediction cache: {cache_path}")
    return payload, cache_path


def bootstrap_site_level(y_true, y_pred, n_bootstrap, sample_size, rng):
    n = len(y_true)
    draw_size = n if sample_size <= 0 else sample_size

    scores = []
    attempts = 0
    max_attempts = max(100, n_bootstrap * 20)

    while len(scores) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=draw_size)
        yt = y_true[idx]

        if np.unique(yt).size < 2:
            continue

        scores.append(average_precision_score(yt, y_pred[idx]))

    if len(scores) < n_bootstrap:
        raise RuntimeError(
            f"Obtained only {len(scores)} valid bootstrap realizations out of "
            f"{n_bootstrap} requested after {attempts} attempts."
        )

    return np.asarray(scores, dtype=np.float64), draw_size, attempts


def bootstrap_protein_level(y_true, y_pred, groups, n_bootstrap, sample_size, rng):
    unique_groups = np.unique(groups)
    draw_size = len(unique_groups) if sample_size <= 0 else sample_size

    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }

    scores = []
    attempts = 0
    max_attempts = max(100, n_bootstrap * 20)

    while len(scores) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        sampled_groups = rng.choice(unique_groups, size=draw_size, replace=True)

        # If a protein is drawn more than once, its whole cluster is repeated.
        idx = np.concatenate([group_indices[g] for g in sampled_groups])
        yt = y_true[idx]

        if np.unique(yt).size < 2:
            continue

        scores.append(average_precision_score(yt, y_pred[idx]))

    if len(scores) < n_bootstrap:
        raise RuntimeError(
            f"Obtained only {len(scores)} valid bootstrap realizations out of "
            f"{n_bootstrap} requested after {attempts} attempts."
        )

    return np.asarray(scores, dtype=np.float64), draw_size, attempts


def bootstrap_all(payload, args):
    unique_labels = payload["unique_labels"]
    y_true_all = payload["y_true"]
    y_pred_all = payload["y_pred"]
    groups_all = payload["groups"]

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    distribution_rows = []

    print("\nBootstrap PR-AUC:")
    for i, ptm in enumerate(unique_labels, start=1):
        y_true = np.asarray(y_true_all[ptm], dtype=np.int8)
        y_pred = np.asarray(y_pred_all[ptm], dtype=np.float64)
        groups = np.asarray(groups_all[ptm]).astype(str)

        if not (len(y_true) == len(y_pred) == len(groups)):
            raise RuntimeError(
                f"{ptm}: y_true, y_pred, and groups have different lengths."
            )

        if np.unique(y_true).size < 2:
            raise RuntimeError(
                f"{ptm}: curated evaluation set does not contain both classes."
            )

        original = average_precision_score(y_true, y_pred)

        if args.unit == "protein":
            scores, effective_size, attempts = bootstrap_protein_level(
                y_true,
                y_pred,
                groups,
                args.n_bootstrap,
                args.sample_size,
                rng,
            )
        else:
            scores, effective_size, attempts = bootstrap_site_level(
                y_true,
                y_pred,
                args.n_bootstrap,
                args.sample_size,
                rng,
            )

        ci_low, ci_high = np.percentile(scores, [2.5, 97.5])

        summary_rows.append(
            {
                "ptm": ptm,
                "n_evaluated_sites": int(len(y_true)),
                "n_positive": int(np.sum(y_true == 1)),
                "n_negative": int(np.sum(y_true == 0)),
                "n_proteins": int(np.unique(groups).size),
                "bootstrap_unit": args.unit,
                "bootstrap_sample_size": int(effective_size),
                "n_bootstrap": int(len(scores)),
                "bootstrap_attempts": int(attempts),
                "original_pr_auc": float(original),
                "bootstrap_mean": float(np.mean(scores)),
                "bootstrap_median": float(np.median(scores)),
                "bootstrap_std": float(np.std(scores, ddof=1)),
                "ci95_lower": float(ci_low),
                "ci95_upper": float(ci_high),
            }
        )

        distribution_rows.extend(
            {
                "ptm": ptm,
                "bootstrap_index": b,
                "pr_auc": float(score),
            }
            for b, score in enumerate(scores)
        )

        print(
            f"  [{i:2d}/{len(unique_labels):2d}] "
            f"{ptm:<30s} "
            f"PR-AUC={original:.4f}  "
            f"95% CI=({ci_low:.4f}, {ci_high:.4f})"
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(distribution_rows)


def write_plot(summary, distribution, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        for _, row in summary.iterrows():
            ptm = row["ptm"]
            scores = distribution.loc[
                distribution["ptm"] == ptm, "pr_auc"
            ].to_numpy()

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(scores, bins=40, alpha=0.8)
            ax.axvline(
                row["original_pr_auc"],
                linestyle="-",
                linewidth=1.5,
                label=f"Observed = {row['original_pr_auc']:.4f}",
            )
            ax.axvline(
                row["ci95_lower"],
                linestyle="--",
                linewidth=1.5,
                label=(
                    f"95% CI = {row['ci95_lower']:.4f}–"
                    f"{row['ci95_upper']:.4f}"
                ),
            )
            ax.axvline(row["ci95_upper"], linestyle="--", linewidth=1.5)

            ax.set_xlabel("PR-AUC")
            ax.set_ylabel("Bootstrap count")
            ax.set_title(
                f"{ptm}\n"
                f"{row['n_bootstrap']} bootstrap realizations; "
                f"unit={row['bootstrap_unit']}; "
                f"N={row['bootstrap_sample_size']}"
            )
            ax.legend(frameon=False)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    args = parse_args()

    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be >= 1")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be >= 0")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, cache_path = load_or_generate_predictions(args)
    summary, distribution = bootstrap_all(payload, args)

    summary_path = output_dir / "bootstrap_pr_auc_summary.csv"
    distribution_path = output_dir / "bootstrap_pr_auc_distribution.csv"

    summary.to_csv(summary_path, index=False)
    distribution.to_csv(distribution_path, index=False)

    print(f"\nSaved summary      : {summary_path}")
    print(f"Saved distribution : {distribution_path}")
    print(f"Prediction cache   : {cache_path}")

    if not args.no_plot:
        plot_path = output_dir / "bootstrap_pr_auc_distributions.pdf"
        write_plot(summary, distribution, plot_path)
        print(f"Saved distributions: {plot_path}")

    print("\nSummary:")
    cols = [
        "ptm",
        "original_pr_auc",
        "bootstrap_mean",
        "bootstrap_median",
        "ci95_lower",
        "ci95_upper",
        "n_evaluated_sites",
        "n_proteins",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
