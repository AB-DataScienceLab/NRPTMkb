#!/usr/bin/env python3
"""Plot three-model PTM performance directly from the curated NR dataset.

Each of the seven PTM folders contains ONLY three PNG figures:
    <PTM_type>/confusion_matrix.png  (three models, threshold 0.5 by default)
    <PTM_type>/roc_curve.png         (three models, full negatives)
    <PTM_type>/pr_curve.png          (three models, full negatives)

Run from the repository root:
    python plot_ptm_performance.py \\
        --sequences data/curated/NR_Sequences_New.csv \\
        --curated data/curated/NR_InputData_New.csv \\
        --prediction NRPTMpred=results_new/correct_predictions.csv \\
        --prediction MIND=results_original/correct_predictions_newlabels.csv \\
        --prediction MusiteDeep=results_musitedeep/correct_predictions.csv \\
        --output-dir ptm_performance_plots

"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless plotting on Linux/remote servers.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PTM_AA = {
    "Acetylation_K": "K",
    "Methylation_K": "K",
    "Phosphorylation_S": "S",
    "Phosphorylation_T": "T",
    "Phosphorylation_Y": "Y",
    "Sumoylation_K": "K",
    "Ubiquitination_K": "K",
}
KEY = ["uid", "site", "PTM_type"]
MODEL_ORDER = ("NRPTMpred", "MIND", "MusiteDeep")
MODEL_LABELS = {"NRPTMpred": "NRPTMpred", "MIND": "MIND-S", "MusiteDeep": "MusiteDeep"}
PLOT_COLORS = {"NRPTMpred": "#2166ac", "MIND": "#d95f02", "MusiteDeep": "#1b9e77"}


def arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequences", default="data/curated/NR_Sequences_New.csv")
    parser.add_argument("--curated", default="data/curated/NR_InputData_New.csv")
    parser.add_argument("--prediction", action="append", required=True, metavar="MODEL=CSV",
                        help="Pass once each for NRPTMpred, MIND and MusiteDeep")
    parser.add_argument("--output-dir", default="ptm_performance_plots")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for confusion matrices (default: 0.5)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--strict-positive-coverage", action="store_true",
                        help="Fail if any annotated positive lacks a score from any model")
    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be in [0, 1]")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def canonical_model(name):
    names = {
        "NRPTMpred": "NRPTMpred", "NRPTMKB": "NRPTMpred",
        "MIND": "MIND", "MIND-S": "MIND", "MusiteDeep": "MusiteDeep",
    }
    try:
        return names[name.strip()]
    except KeyError as err:
        raise ValueError(f"Unknown model {name!r}. Expected {MODEL_ORDER}.") from err


def prediction_paths(specs):
    found = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --prediction argument: {spec!r}; expected MODEL=CSV")
        name = canonical_model(name)
        if name in found:
            raise ValueError(f"Duplicate prediction path for {name}")
        found[name] = Path(path.strip())
    if set(found) != set(MODEL_ORDER):
        raise ValueError(f"Supply exactly these three models: {MODEL_ORDER}; got {tuple(found)}")
    return found


def read_csv(path, required):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return frame


def load_sequences(path):
    table = read_csv(path, ("UniprotID", "Sequence"))[["UniprotID", "Sequence"]].copy()
    if table.isna().any().any():
        raise ValueError(f"{path}: sequence or accession is missing")
    table["UniprotID"] = table["UniprotID"].astype(str).str.strip()
    table["Sequence"] = table["Sequence"].astype(str).str.strip().str.upper()
    if (table == "").any().any() or table["UniprotID"].duplicated().any():
        raise ValueError(f"{path}: empty fields or duplicated UniProt IDs")
    sequences = dict(zip(table["UniprotID"], table["Sequence"]))
    if not sequences:
        raise ValueError(f"{path}: no sequences")
    print(f"Curated sequences: {len(sequences)}")
    return sequences


def load_positive_keys(path, sequences):
    table = read_csv(path, ("UniprotID", "Site", "PTM_Type"))
    positives = set()
    ignored = 0
    for uid, annotation, ptm_base in table[["UniprotID", "Site", "PTM_Type"]].itertuples(
            index=False, name=None):
        uid = str(uid).strip()
        match = re.fullmatch(r"([A-Za-z])(\d+)", str(annotation).strip())
        if match is None:
            raise ValueError(f"{path}: invalid annotated site {annotation!r} for {uid}")
        aa, pos = match.group(1).upper(), int(match.group(2))
        ptm = f"{str(ptm_base).strip()}_{aa}"
        if ptm not in PTM_AA:
            ignored += 1
            continue
        sequence = sequences.get(uid)
        if sequence is None:
            raise ValueError(f"{path}: accession {uid!r} absent from curated sequences")
        if pos < 1 or pos > len(sequence) or sequence[pos - 1] != aa:
            raise ValueError(f"{path}: annotated {uid} {aa}{pos} does not match sequence")
        key = (uid, pos, ptm)
        if key in positives:
            raise ValueError(f"{path}: duplicate curated positive {key}")
        positives.add(key)
    if not positives:
        raise ValueError(f"{path}: no supported annotated PTM positives found")
    print(f"Curated positives for seven PTM classes: {len(positives)} (ignored {ignored} others)")
    return positives


def eligible_sites(sequences, positives):
    amino_to_ptm = {}
    for ptm, aa in PTM_AA.items():
        amino_to_ptm.setdefault(aa, []).append(ptm)
    data = (
        (uid, pos, ptm, int((uid, pos, ptm) in positives))
        for uid, sequence in sequences.items()
        for pos, aa in enumerate(sequence, start=1)
        for ptm in amino_to_ptm.get(aa, ())
    )
    sites = pd.DataFrame(data, columns=KEY + ["label"])
    if sites.empty or int(sites["label"].sum()) != len(positives):
        raise ValueError("Curated positives do not match generated residue candidates")
    if sites.duplicated(KEY).any():
        raise AssertionError("Duplicate generated candidate PTM-residue key")
    print(f"Eligible PTM-residue examples before score matching: {len(sites):,}")
    return sites


def load_prediction(path, model, sequences):
    pred = read_csv(path, KEY + ["pred_score"])[KEY + ["pred_score"]].copy()
    original_count = len(pred)
    pred["uid"] = pred["uid"].astype(str).str.strip()
    pred["PTM_type"] = pred["PTM_type"].astype(str).str.strip()
    pred = pred.loc[pred["uid"].isin(sequences) & pred["PTM_type"].isin(PTM_AA)].copy()
    pos = pd.to_numeric(pred["site"], errors="raise")
    if pos.isna().any() or (pos < 1).any() or (pos % 1).ne(0).any():
        raise ValueError(f"{model}: site must be an integer using one-based indexing")
    pred["site"] = pos.astype(np.int64)
    if pred.duplicated(KEY).any():
        examples = pred.loc[pred.duplicated(KEY, keep=False), KEY].head(5)
        raise ValueError(f"{model}: duplicate predicted sites in {path}:\n{examples}")
    residues = np.asarray([
        sequences[uid][site - 1] if site <= len(sequences[uid]) else "?"
        for uid, site in zip(pred["uid"], pred["site"])
    ])
    if (residues != pred["PTM_type"].map(PTM_AA).to_numpy()).any():
        raise ValueError(f"{model}: prediction positions do not agree with curated sequences")
    pred["pred_score"] = pd.to_numeric(pred["pred_score"], errors="raise")
    scores = pred["pred_score"].to_numpy(dtype=float)
    available = ~np.isnan(scores)
    if (~np.isfinite(scores[available])).any() or ((scores[available] < 0) |
                                                  (scores[available] > 1)).any():
        raise ValueError(f"{model}: scores must be probabilities in [0,1]")
    print(f"  {MODEL_LABELS[model]}: {int(available.sum()):,} valid scores; "
          f"{original_count - len(pred):,} unrelated predictions ignored")
    return pred.rename(columns={"pred_score": f"score_{model}"})


def build_matched_dataset(args):
    sequences = load_sequences(args.sequences)
    positives = load_positive_keys(args.curated, sequences)
    sites = eligible_sites(sequences, positives)
    paths = prediction_paths(args.prediction)
    columns = {}
    for model in MODEL_ORDER:
        column = f"score_{model}"
        sites = sites.merge(load_prediction(paths[model], model, sequences), on=KEY,
                            how="left", validate="one_to_one", sort=False)
        columns[model] = column
    matched = sites[list(columns.values())].notna().all(axis=1)
    missed = sites.loc[(sites["label"] == 1) & ~matched]
    if len(missed):
        print(f"WARNING: {len(missed)} curated positives are missing scores from "
              "at least one model and cannot enter a paired comparison.")
        for model, col in columns.items():
            missing = int(missed[col].isna().sum())
            print(f"  {MODEL_LABELS[model]}: {missing} missing curated-positive scores")
        if args.strict_positive_coverage:
            raise ValueError("Some curated positives lack scores; --strict-positive-coverage set")
    evaluated = sites.loc[matched].copy()
    if evaluated.empty:
        raise ValueError("No common scored positions across all three models")
    print(f"Matched evaluation examples: {len(evaluated):,} "
          f"(N+={int(evaluated['label'].sum()):,}, "
          f"N-={int((evaluated['label'] == 0).sum()):,})")
    return evaluated, columns


def plot_confusion_matrices(y, probabilities, ptm, destination, threshold, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.75), constrained_layout=True)
    for ax, model in zip(axes, MODEL_ORDER):
        decisions = (probabilities[model] >= threshold).astype(np.int8)
        matrix = confusion_matrix(y, decisions, labels=[0, 1])
        row_totals = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(matrix, row_totals, where=(row_totals > 0),
                              out=np.zeros_like(matrix, dtype=float))
        ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        for row in range(2):
            for col in range(2):
                pct = fractions[row, col]
                ax.text(col, row, f"{matrix[row, col]:,}\n({pct:.1%})",
                        ha="center", va="center", fontsize=11,
                        color="white" if pct >= 0.55 else "black")
        acc = accuracy_score(y, decisions)
        precision = precision_score(y, decisions, zero_division=0)
        recall = recall_score(y, decisions, zero_division=0)
        ax.set_title(f"{MODEL_LABELS[model]}\nAccuracy {acc:.3f}  |  Precision {precision:.3f}  |  Recall {recall:.3f}",
                     fontsize=10)
        ax.set_xticks([0, 1], ["Negative", "Positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
    fig.suptitle(f"{ptm.replace('_', ' ')} — confusion matrices at threshold {threshold:g}", fontsize=13)
    fig.supxlabel("Cells show count (percentage within each true-label row)", fontsize=9)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y, probabilities, ptm, destination, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model in MODEL_ORDER:
        scores = probabilities[model]
        fpr, tpr, _ = roc_curve(y, scores)
        auroc = roc_auc_score(y, scores)
        ax.plot(fpr, tpr, lw=2, color=PLOT_COLORS[model],
                label=f"{MODEL_LABELS[model]} (ROC-AUC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1.2, label="Random classifier")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate (1 − specificity)",
           ylabel="True positive rate (sensitivity)",
           title=f"{ptm.replace('_', ' ')} — ROC curve")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y, probabilities, ptm, destination, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model in MODEL_ORDER:
        scores = probabilities[model]
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.step(recall, precision, where="post", lw=1.9, color=PLOT_COLORS[model],
                label=f"{MODEL_LABELS[model]} (AP = {ap:.3f})")
    prevalence = float(np.mean(y))
    ax.axhline(prevalence, linestyle="--", lw=1.2, color="0.5",
               label=f"Random baseline = {prevalence:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title=f"{ptm.replace('_', ' ')} — precision–recall curve")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = arguments()
    frame, score_columns = build_matched_dataset(args)
    output = Path(args.output_dir)
    count = 0
    for ptm in PTM_AA:
        sub = frame.loc[frame["PTM_type"] == ptm]
        y = sub["label"].to_numpy(dtype=np.int8)
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        if n_positive == 0 or n_negative == 0:
            raise ValueError(f"Cannot plot {ptm}: N+={n_positive}, N-={n_negative}; "
                             "ROC and PR metrics require both classes")
        probabilities = {
            model: sub[col].to_numpy(dtype=float)
            for model, col in score_columns.items()
        }
        directory = output / ptm
        directory.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrices(y, probabilities, ptm, directory / "confusion_matrix.png",
                                args.threshold, args.dpi)
        plot_roc(y, probabilities, ptm, directory / "roc_curve.png", args.dpi)
        plot_pr(y, probabilities, ptm, directory / "pr_curve.png", args.dpi)
        count += 1
        print(f"{ptm:22s} N+={n_positive:4d} N-={n_negative:5d}  ->  {directory}")
        for model in MODEL_ORDER:
            scores = probabilities[model]
            print(f"    {MODEL_LABELS[model]:12s} ROC-AUC={roc_auc_score(y, scores):.6f} "
                  f"AP={average_precision_score(y, scores):.6f}")
    print(f"\nSaved {3 * count} PNG figures in {count} PTM subfolders beneath {output.resolve()}")
    

if __name__ == "__main__":
    main()
