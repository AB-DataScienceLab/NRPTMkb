#!/usr/bin/env python3
"""Held-out, matched-site evaluation of NR-PTMpred, MIND-S, and MusiteDeep.

The source of ground-truth labels is *test_data_new.json*, not the pooled
benchmark's curated labels. Positive annotations in train_data_new.json are
excluded. All other amino-acid-eligible sites on TEST sequences are PRESUMED
negatives, not experimentally verified negatives.

For a fair three-model comparison, each PTM class is evaluated on the exact
intersection of sites carrying finite prediction scores from all three models.
This may differ from a single-model evaluation with wider score coverage.

Outputs include:
  test_metrics.csv                  Per-class, per-model full-negative metrics
  test_metrics_readable.csv         Key metrics with parent-protein bootstrap CIs
  test_comparison_full.csv          Matched full-negative comparison
  test_comparison_balanced.csv      Matched balanced-negative median comparison
  test_model_comparison.xlsx        'Full negatives' / 'Balanced negatives' sheets
  paired_pr_auc_differences.csv     Matched, parent-bootstrap AP differences
  bootstrap_pr_auc_distribution.csv  Bootstrap AP values for all three models
  test_sites.csv                    Site-level scores, flags, and labels
  omitted_sites.csv                 Exclusion reasons and missing models
  test_summary_by_ptm.csv           Per-class coverage and annotation counts
  test_split_summary.json           Provenance and train/test independence audit

Example (run from the repository root):
    python evaluate_test_data.py \\
        --train sequence_split/train_data_new.json \\
        --test sequence_split/test_data_new.json \\
        --prediction NRPTMpred=results_new/correct_predictions.csv \\
        --prediction MIND=results_original/correct_predictions_newlabels.csv \\
        --prediction MusiteDeep=results_musitedeep/correct_predictions.csv \\
        --output-dir test_only_comparison \\
        --require-parent-disjoint

The full negatives use all matched scored positive and presumed-negative sites.
Balanced negatives hold matched positives fixed and sample N+ presumed negatives
WITH replacement, identically across models, for 1,000 iterations (default).
Accuracy/precision/recall use threshold 0.5 by default; PR-AUC is average
precision and ROC-AUC uses continuous probabilities, without a fixed threshold.
Bootstrap confidence intervals are percentile intervals from parent UniProt
accession resampling; related isoforms enter the resample together.

NOTE: no train/test split can establish independence from the *pretraining*
data of MIND-S. Annotation-level splits can share receptor sequences and may
have sampled withheld positives as training negatives. These issues are audited
rather than concealed.

Dependencies: numpy, pandas, scikit-learn; openpyxl for the Excel comparison.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

KEY = ["uid", "site", "PTM_type"]
PTM_AA = {
    "Acetylation_K": "K", "Methylation_K": "K", "Phosphorylation_S": "S",
    "Phosphorylation_T": "T", "Phosphorylation_Y": "Y", "Sumoylation_K": "K",
    "Ubiquitination_K": "K",
}
MODELS = ("NRPTMpred", "MIND", "MusiteDeep")
DISPLAY = {"NRPTMpred": "NR-PTMpred", "MIND": "MIND-S", "MusiteDeep": "MusiteDeep"}
COMPARISON_METRICS = ("Accuracy", "Precision", "Recall", "PR-AUC", "ROC-AUC")
BOOTSTRAP_METRICS = ("Accuracy", "Balanced_accuracy", "Precision", "Recall", "Specificity",
                     "F1", "MCC", "ROC_AUC", "PR_AUC_AP", "Brier", "ECE")


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="train_data_new.json")
    p.add_argument("--test", default="test_data_new.json")
    p.add_argument("--prediction", action="append", required=True, metavar="MODEL=CSV",
                   help="Supply NRPTMpred, MIND, MusiteDeep (repeat 3 times)")
    p.add_argument("--output-dir", default="test_only_comparison")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Parent UniProt cluster-bootstrap resamples for full-negative metrics")
    p.add_argument("--balanced-repeats", type=int, default=1000,
                   help="Balanced-negative draws with replacement for test-set comparison")
    p.add_argument("--seed", type=int, default=252)
    p.add_argument("--classes", nargs="+", default=list(PTM_AA))
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--min-score-coverage", type=float, default=0.7,
                   help="Required overall fraction of eligible test sites scored by ALL models")
    p.add_argument("--require-parent-disjoint", action="store_true",
                   help="Abort if test and train share parent accessions (sequence-level evaluation)")
    p.add_argument("--no-excel", action="store_true", help="Write CSVs only")
    a = p.parse_args()
    if not 0 <= a.threshold <= 1 or a.bootstrap < 0 or a.balanced_repeats < 1:
        p.error("threshold must be in [0,1], bootstrap >= 0 and balanced-repeats >= 1")
    if not 0 <= a.min_score_coverage <= 1 or a.ece_bins < 2:
        p.error("min-score-coverage must be in [0,1] and ece-bins >= 2")
    if len(a.classes) != len(set(a.classes)) or set(a.classes) - set(PTM_AA):
        p.error("--classes must be unique names in " + str(tuple(PTM_AA)))
    return a


def parse_prediction_paths(specs):
    aliases = {"NRPTMKB": "NRPTMpred", "NR-PTMpred": "NRPTMpred", "MIND-S": "MIND"}
    paths = {}
    for spec in specs:
        name, sep, filename = spec.partition("=")
        name = aliases.get(name.strip(), name.strip())
        if not sep or not filename.strip() or name not in MODELS or name in paths:
            raise ValueError(f"Invalid/duplicate --prediction {spec!r}; supply each model once")
        paths[name] = Path(filename.strip())
    if set(paths) != set(MODELS):
        raise ValueError("Exactly three models required: NRPTMpred, MIND, MusiteDeep")
    return paths


def read_dataset(path):
    with Path(path).open(encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected mapping of uid to {{'seq': str, 'label': list}}")
    for uid, rec in d.items():
        if not isinstance(rec, dict) or not isinstance(rec.get("seq"), str) or not isinstance(rec.get("label"), list):
            raise ValueError(f"{path}: invalid record for {uid}")
        if not rec["seq"]:
            raise ValueError(f"{path}: empty sequence for {uid}")
    return d


def positives(dataset, supported):
    """JSON positions are zero-based; convert to one-based prediction positions."""
    pos = set()
    for uid, record in dataset.items():
        seq = record["seq"]
        for annotation in record["label"]:
            index = int(annotation["site"])
            ptm = str(annotation["ptm_type"])
            if ptm not in supported:
                continue
            if index < 0 or index >= len(seq) or seq[index] != PTM_AA[ptm]:
                raise ValueError(f"PTM/sequence mismatch {uid}:{index+1} ({ptm})")
            key = (uid, index + 1, ptm)
            if key in pos:
                raise ValueError(f"Duplicate positive annotation {key}")
            pos.add(key)
    return pos


def get_scores(model, path, supported, test):
    df = pd.read_csv(path, dtype={"uid": str, "PTM_type": str}, low_memory=False)
    if "pred_score" not in df and "score" in df:
        df = df.rename(columns={"score": "pred_score"})
    needed = KEY + ["pred_score"]
    if not set(needed).issubset(df):
        raise ValueError(f"{model} {path}: expected columns {needed}, including FULL continuous scores")
    df = df[needed].copy()
    df["uid"] = df["uid"].str.strip()
    df["PTM_type"] = df["PTM_type"].str.strip()
    df = df[df["PTM_type"].isin(supported)].copy()
    position = pd.to_numeric(df["site"], errors="raise")
    if position.isna().any() or not np.isfinite(position).all() or not (position >= 1).all() or not (position == position.astype(int)).all():
        raise ValueError(f"{model}: sites must be positive one-based integer positions")
    df["site"] = position.astype(int)
    df["pred_score"] = pd.to_numeric(df["pred_score"], errors="raise")
    has = df["pred_score"].notna()
    if not np.isfinite(df.loc[has, "pred_score"]).all() or not df.loc[has, "pred_score"].between(0, 1).all():
        raise ValueError(f"{model}: prediction scores must be finite probabilities in [0,1]")
    if df.duplicated(KEY).any():
        raise ValueError(f"{model}: duplicated uid/site/PTM prediction keys")
    # Fail on inconsistent prediction positions/residues for test sequences.
    # Scores for unrelated proteins may be present and are harmlessly ignored.
    for uid, grp in df.loc[df["uid"].isin(test)].groupby("uid"):
        seq = test[uid]["seq"]
        indices = grp["site"].to_numpy(dtype=int)
        if np.any(indices > len(seq)):
            raise ValueError(f"{model}: out-of-range test prediction site for {uid}")
        aa = grp["PTM_type"].map(PTM_AA).to_numpy(dtype=str)
        actual = np.asarray([seq[i - 1] for i in indices])
        if np.any(actual != aa):
            bad_i = np.flatnonzero(actual != aa)[0]
            raise ValueError(f"{model}: residue mismatch: {uid}:{indices[bad_i]} "
                             f"for {grp['PTM_type'].iloc[bad_i]}")
    return df.rename(columns={"pred_score": f"score_{model}"})


def ece_score(y, scores, bins):
    ix = np.minimum((scores * bins).astype(int), bins - 1)
    return float(sum(np.mean(ix == b) * abs(np.mean(y[ix == b]) - np.mean(scores[ix == b]))
                     for b in range(bins) if (ix == b).any()))


def compute_metrics(y, scores, threshold, bins):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Accuracy": float(accuracy_score(y, predicted)),
        "Precision": float(precision_score(y, predicted, zero_division=0)),
        "Recall": float(recall_score(y, predicted, zero_division=0)),
        "Specificity": float(tn / nneg) if nneg else np.nan,
        "F1": float(f1_score(y, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predicted)),
        "Balanced_accuracy": float((tp / npos + tn / nneg) / 2) if npos and nneg else np.nan,
        "ROC_AUC": float(roc_auc_score(y, scores)) if npos and nneg else np.nan,
        "PR_AUC_AP": float(average_precision_score(y, scores)) if npos and nneg else np.nan,
        "Brier": float(brier_score_loss(y, scores)),
        "ECE": float(ece_score(y, scores, bins)),
    }


def rank_metrics(y, probability):
    """Compute exact average precision and trapezoidal ROC-AUC, grouping ties."""
    npos = int(np.sum(y)); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan, np.nan
    order = np.argsort(-probability, kind="mergesort")
    sorted_prob = probability[order]
    cumulative_tp = np.cumsum(y[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(np.diff(sorted_prob) != 0), len(y) - 1]
    tps = cumulative_tp[ends]
    fps = ends + 1 - tps
    gains = np.diff(np.r_[0, tps])
    ap = float(np.dot(gains / npos, tps / (ends + 1)))
    roc = float(np.dot(np.diff(np.r_[0, fps]) / nneg,
                       (np.r_[0, tps[:-1]] + tps) / (2 * npos)))
    return ap, roc


def five_metrics(y, scores, threshold):
    predicted = scores >= threshold
    target = y.astype(bool)
    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    tn = len(y) - tp - fp - fn
    ap, roc = rank_metrics(y, scores)
    return np.asarray(((tp + tn) / len(y),
                       tp / (tp + fp) if tp + fp else 0.0,
                       tp / (tp + fn) if tp + fn else 0.0,
                       ap, roc), dtype=float)


def bootstrap_confidence_intervals(frame, threshold, bins, n_draws, seed):
    """Resample parent UniProt groups; same draw/seed across compared models.

    Repeated groups are represented by weights, preserving all isoforms/sites.
    Returns percentile CIs and resample AP values in matching draw order.
    """
    blank = {f"{m}_CI95_{bound}": np.nan for m in BOOTSTRAP_METRICS
             for bound in ("lower", "upper")}
    blank.update({"Bootstrap_valid_PR_AUC": 0, "Bootstrap_valid_ROC_AUC": 0})
    draws = defaultdict(list)
    if n_draws <= 0 or frame.empty:
        return blank, draws

    y = frame["label"].to_numpy(dtype=np.int8)
    p = frame["pred_score"].to_numpy(dtype=float)
    parents, group = np.unique(frame["parent_accession"].astype(str), return_inverse=True)
    nparents = len(parents)
    # One parent cannot support meaningful between-protein uncertainty.
    if nparents < 2 or np.unique(group[y == 1]).size < 2 or np.sum(y) < 2 or np.sum(y == 0) < 2:
        return blank, draws

    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    sorted_p = p[order]
    tied_ends = np.flatnonzero(np.r_[sorted_p[1:] != sorted_p[:-1], True])
    predictions = p >= threshold
    bin_index = np.minimum((p * bins).astype(int), bins - 1)
    rng = np.random.default_rng(seed)

    for draw in range(n_draws):
        counts = np.bincount(rng.integers(0, nparents, size=nparents), minlength=nparents)
        w = counts[group]
        total = float(w.sum())
        pos = float(w[y == 1].sum())
        neg = total - pos
        if total <= 0:
            continue
        tp = float(w[(y == 1) & predictions].sum())
        fp = float(w[(y == 0) & predictions].sum())
        fn = pos - tp
        tn = neg - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / pos if pos else 0.0
        specificity = tn / neg if neg else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom else 0.0
        true_sum = np.bincount(bin_index, weights=w * y, minlength=bins)
        prob_sum = np.bincount(bin_index, weights=w * p, minlength=bins)
        base = {
            "Accuracy": (tp + tn) / total,
            "Balanced_accuracy": (recall + specificity) / 2 if pos and neg else np.nan,
            "Precision": precision, "Recall": recall, "Specificity": specificity,
            "F1": f1, "MCC": mcc,
            "Brier": float(np.dot(w, (p - y) ** 2) / total),
            "ECE": float(np.abs(true_sum - prob_sum).sum() / total),
        }
        if pos > 0:
            ww = w[order]
            cumulative_pos = np.cumsum(ww * sorted_y)[tied_ends]
            cumulative_neg = np.cumsum(ww * (1 - sorted_y))[tied_ends]
            increments = np.diff(np.r_[0.0, cumulative_pos])
            prec_at = np.divide(cumulative_pos, cumulative_pos + cumulative_neg,
                                out=np.zeros_like(cumulative_pos, dtype=float),
                                where=(cumulative_pos + cumulative_neg) > 0)
            base["PR_AUC_AP"] = float(np.dot(increments / pos, prec_at))
            base["ROC_AUC"] = (float(np.dot(np.diff(np.r_[0., cumulative_neg / neg]),
                              (np.r_[0., cumulative_pos[:-1] / pos] + cumulative_pos / pos) / 2))
                              if neg > 0 else np.nan)
        else:
            base["PR_AUC_AP"] = np.nan
            base["ROC_AUC"] = np.nan
        for metric, value in base.items():
            if np.isfinite(value):
                draws[metric].append((draw, float(value)))

    for metric in BOOTSTRAP_METRICS:
        values = [v for _, v in draws[metric]]
        if values:
            low, high = np.percentile(values, (2.5, 97.5))
            blank[f"{metric}_CI95_lower"] = float(low)
            blank[f"{metric}_CI95_upper"] = float(high)
    blank["Bootstrap_valid_PR_AUC"] = len(draws["PR_AUC_AP"])
    blank["Bootstrap_valid_ROC_AUC"] = len(draws["ROC_AUC"])
    return blank, draws


def full_comparison_value(rec, metric):
    return rec[dict(zip(COMPARISON_METRICS,
                        ("Accuracy", "Precision", "Recall", "PR_AUC_AP", "ROC_AUC")))[metric]]


def excel_comparison(path, full, balanced, classes):
    """Match the existing benchmark's two-sheet, five-metric comparison layout."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel export needs openpyxl. Install it or use --no-excel") from exc
    wb = Workbook()
    wb.active.title = "Full negatives"
    wb.create_sheet("Balanced negatives")
    for sheet, rows in ((wb.worksheets[0], full), (wb.worksheets[1], balanced)):
        sheet.merge_cells("A1:A2")
        sheet["A1"] = "PTM class"
        sheet.column_dimensions["A"].width = 25
        for im, name in enumerate(COMPARISON_METRICS):
            start = 2 + im * len(MODELS)
            sheet.merge_cells(start_row=1, start_column=start, end_row=1,
                              end_column=start + len(MODELS) - 1)
            sheet.cell(1, start, name)
            for ix, model in enumerate(MODELS):
                col = start + ix
                sheet.cell(2, col, DISPLAY[model])
                sheet.column_dimensions[get_column_letter(col)].width = 16
        for row_ix, ptm in enumerate(classes, start=3):
            sheet.cell(row_ix, 1, ptm)
            for mi, metric in enumerate(COMPARISON_METRICS):
                for ix, model in enumerate(MODELS):
                    value = rows.get((ptm, model), {}).get(metric, np.nan)
                    cell = sheet.cell(row_ix, 2 + 3 * mi + ix,
                                      float(value) if np.isfinite(value) else None)
                    cell.number_format = "0.000"
        sheet.freeze_panes = "B3"
        sheet.sheet_view.showGridLines = False
        for row in sheet.iter_rows(min_row=1, max_row=2, max_col=16):
            for c in row:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="27415C")
                c.alignment = Alignment(horizontal="center", vertical="center")
        sheet.row_dimensions[1].height = 24
        sheet.row_dimensions[2].height = 21
    wb.save(path)


def main():
    args = args_parser()
    paths = parse_prediction_paths(args.prediction)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train, test = read_dataset(args.train), read_dataset(args.test)
    classes = args.classes
    train_pos, test_pos = positives(train, classes), positives(test, classes)
    overlap_positive = train_pos & test_pos
    if overlap_positive:
        raise ValueError(f"Train/test shared {len(overlap_positive)} identical positive annotation triples")
    shared_uids = set(train) & set(test)
    for uid in shared_uids:
        if train[uid]["seq"] != test[uid]["seq"]:
            raise ValueError(f"Train/test sequences disagree for shared UID {uid}")
    train_parents = {uid.split("-")[0] for uid in train}
    test_parents = {uid.split("-")[0] for uid in test}
    overlapping_parents = train_parents & test_parents
    train_sequence_to_uid = defaultdict(list)
    for uid, record in train.items():
        train_sequence_to_uid[record["seq"]].append(uid)
    duplicate_sequences = sorted(uid for uid, rec in test.items()
                                 if rec["seq"] in train_sequence_to_uid)
    print(f"Train positives={len(train_pos)}, held-out positives={len(test_pos)}")
    print(f"Shared train/test UIDs={len(shared_uids)}, parents={len(overlapping_parents)}, "
          f"identical-sequence test UIDs={len(duplicate_sequences)}")
    if args.require_parent_disjoint and (overlapping_parents or duplicate_sequences):
        raise ValueError("--require-parent-disjoint failed: "
                         f"{len(overlapping_parents)} shared parent accessions and "
                         f"{len(duplicate_sequences)} test UIDs with identical training sequences")
    if shared_uids or overlapping_parents or duplicate_sequences:
        print("WARNING: test is NOT independent at the protein/parent-sequence level.")
    if not test_pos:
        raise ValueError("No supported held-out positives in test JSON")

    train_by_class = {p: sum(k[2] == p for k in train_pos) for p in classes}
    test_by_class = {p: sum(k[2] == p for k in test_pos) for p in classes}

    # Only test-side eligible residues are candidates. This is critical: using
    # the full curated-positive table would leak train labels into evaluation.
    aa_to_classes = defaultdict(list)
    for ptm in classes:
        aa_to_classes[PTM_AA[ptm]].append(ptm)
    rows = []
    for uid, rec in sorted(test.items()):
        parent = uid.split("-")[0]
        for site, aa in enumerate(rec["seq"], start=1):
            for ptm in aa_to_classes.get(aa, ()):  # same site can support several PTMs
                key = (uid, site, ptm)
                rows.append((uid, site, ptm, parent, aa, int(key in train_pos),
                             int(key in test_pos), int(key not in train_pos)))
    cols = KEY + ["parent_accession", "residue", "train_positive", "test_positive", "eval_eligible"]
    sites = pd.DataFrame.from_records(rows, columns=cols)
    if sites.empty:
        raise ValueError("No amino-acid-eligible test sites")
    if int(sites["test_positive"].sum()) != len(test_pos):
        raise ValueError("Some held-out positive annotations do not map to eligible sites")
    for model in MODELS:
        score_table = get_scores(model, paths[model], classes, test)
        sites = sites.merge(score_table, on=KEY, how="left", validate="one_to_one", sort=False)
        sites[f"has_{model}_score"] = sites[f"score_{model}"].notna().astype(np.int8)
        print(f"{DISPLAY[model]}: {int(sites[f'has_{model}_score'].sum())}/{len(sites)} candidate scores")
    score_cols = [f"score_{m}" for m in MODELS]
    sites["common_scored"] = (sites["eval_eligible"].eq(1) &
                              sites[score_cols].notna().all(axis=1)).astype(np.int8)
    eligible = int(sites["eval_eligible"].sum())
    scored = int(sites["common_scored"].sum())
    coverage = scored / eligible if eligible else 0
    print(f"Common scored: {scored}/{eligible} eligible test sites ({coverage:.1%})")
    if coverage < args.min_score_coverage:
        raise ValueError(f"Only {coverage:.1%} of eligible sites have scores from all models. "
                         "Provide full per-site CSVs, or deliberately lower --min-score-coverage.")
    sites["missing_models"] = [";".join(m for m in MODELS if pd.isna(row[f"score_{m}"]))
                               for _, row in sites.iterrows()]
    sites["exclusion_reason"] = np.where(sites["eval_eligible"].eq(0),
                                          "training_positive", np.where(
                                              sites["common_scored"].eq(0),
                                              "missing_model_score", ""))
    sites["sample_type"] = np.where(sites["train_positive"].eq(1),
                                     "excluded_training_positive", np.where(
                                         sites["test_positive"].eq(1),
                                         "heldout_positive", "presumed_negative"))
    sites.sort_values(KEY, inplace=True)
    sites.to_csv(output / "test_sites.csv", index=False)
    sites.loc[sites["common_scored"].eq(0)].to_csv(output / "omitted_sites.csv", index=False)

    all_metric_rows, balanced_rows, coverage_rows = [], [], []
    full_xlsx, balanced_xlsx = {}, {}
    paired_differences, bootstrap_distribution = [], []
    rng = np.random.default_rng(args.seed)

    for iclass, ptm in enumerate(classes):
        full = sites.loc[sites["PTM_type"].eq(ptm)]
        matched = full.loc[full["common_scored"].eq(1)].reset_index(drop=True)
        y = matched["test_positive"].to_numpy(dtype=np.int8)
        npos, nneg = int(y.sum()), int(len(y) - y.sum())
        coverage_rows.append({
            "PTM_type": ptm,
            "N_training_positive_annotations": train_by_class[ptm],
            "N_test_positive_annotations": test_by_class[ptm],
            "N_test_positives_with_common_scores": npos,
            "N_test_positives_missing_common_scores": test_by_class[ptm] - npos,
            "N_common_scored_negative": nneg,
            "N_common_scored_total": len(matched),
            "N_eligible_after_train_positive_removal": int(full["eval_eligible"].sum()),
            "N_missing_common_scores": int((full["eval_eligible"].eq(1) & full["common_scored"].eq(0)).sum()),
            "N_parent_accessions_evaluable": matched["parent_accession"].nunique(),
            "N_parent_accessions_with_positive": matched.loc[y == 1, "parent_accession"].nunique(),
        })
        print(f"{ptm:22} common: N+={npos:3}, N-={nneg:5}, "
              f"missing positives={test_by_class[ptm] - npos}")
        if npos == 0 or nneg == 0:
            print(f"WARNING: {ptm} missing one class in matched scores; metrics unavailable")
            for model in MODELS:
                full_xlsx[ptm, model] = {m: np.nan for m in COMPARISON_METRICS}
                balanced_xlsx[ptm, model] = {m: np.nan for m in COMPARISON_METRICS}
                all_metric_rows.append({"PTM_type": ptm, "Model": DISPLAY[model],
                                        "N_positive": npos, "N_negative": nneg})
                balanced_rows.append({"PTM_type": ptm, "Model": DISPLAY[model],
                                      "N_positive": npos, "N_negative_available": nneg,
                                      "Balanced_repeats": 0})
            continue

        # Matched full-negative metrics + parent-accession bootstrap. All model
        # calls use exactly the same parent draws for paired AP comparisons.
        draws_by_model = {}
        probabilities = {}
        for model in MODELS:
            s = matched[f"score_{model}"].to_numpy(dtype=np.float64)
            probabilities[model] = s
            result = compute_metrics(y, s, args.threshold, args.ece_bins)
            boot_frame = matched[["parent_accession", "test_positive"]].rename(
                columns={"test_positive": "label"}).copy()
            boot_frame["pred_score"] = s
            ci, draws = bootstrap_confidence_intervals(boot_frame, args.threshold,
                                                       args.ece_bins, args.bootstrap,
                                                       args.seed + iclass)
            draws_by_model[model] = dict(draws["PR_AUC_AP"])
            for draw_id, value in draws["PR_AUC_AP"]:
                bootstrap_distribution.append({"PTM_type": ptm, "Model": DISPLAY[model],
                                               "Bootstrap_draw": draw_id, "PR_AUC_AP": value})
            record = {"PTM_type": ptm, "Model": DISPLAY[model],
                      "N_positive": npos, "N_negative": nneg, "N_total": len(y),
                      "N_parent_accessions": matched["parent_accession"].nunique(),
                      "Threshold": args.threshold, "Bootstrap_requested": args.bootstrap,
                      **result, **ci}
            all_metric_rows.append(record)
            full_xlsx[ptm, model] = {metric: full_comparison_value(result, metric)
                                     for metric in COMPARISON_METRICS}

        for i, model_a in enumerate(MODELS):
            for model_b in MODELS[i + 1:]:
                shared_draws = sorted(set(draws_by_model[model_a]) & set(draws_by_model[model_b]))
                delta = np.asarray([draws_by_model[model_a][d] - draws_by_model[model_b][d]
                                    for d in shared_draws], dtype=float)
                low, high = np.percentile(delta, [2.5, 97.5]) if len(delta) else (np.nan, np.nan)
                paired_differences.append({
                    "PTM_type": ptm, "Model_A": DISPLAY[model_a], "Model_B": DISPLAY[model_b],
                    "AP_Difference_A_minus_B": full_xlsx[ptm, model_a]["PR-AUC"] -
                                               full_xlsx[ptm, model_b]["PR-AUC"],
                    "CI95_lower": float(low), "CI95_upper": float(high),
                    "Valid_paired_parent_bootstrap_draws": len(shared_draws)})

        # Repeat balanced negative sampling WITH replacement on TEST matched
        # sites; each draw is shared across ALL three prediction methods.
        positives_idx = np.flatnonzero(y == 1)
        negative_idx = np.flatnonzero(y == 0)
        balanced_samples = {m: [] for m in MODELS}
        for _ in range(args.balanced_repeats):
            picked = rng.choice(negative_idx, size=npos, replace=True)
            idx = np.r_[positives_idx, picked]
            yy = y[idx]
            for model in MODELS:
                balanced_samples[model].append(
                    five_metrics(yy, probabilities[model][idx], args.threshold))
        for model in MODELS:
            median = np.median(np.stack(balanced_samples[model]), axis=0)
            record = {"PTM_type": ptm, "Model": DISPLAY[model],
                      "N_positive": npos, "N_negative_available": nneg,
                      "N_negative_per_draw": npos, "Balanced_repeats": args.balanced_repeats,
                      **dict(zip(COMPARISON_METRICS, map(float, median)))}
            balanced_rows.append(record)
            balanced_xlsx[ptm, model] = {metric: float(value)
                                         for metric, value in zip(COMPARISON_METRICS, median)}

    # Audit files can be regenerated independently; use CSVs for source data.
    pd.DataFrame(all_metric_rows).to_csv(output / "test_metrics.csv", index=False)
    readable = []
    for row in all_metric_rows:
        def with_ci(metric):
            value = row.get(metric, np.nan)
            if not np.isfinite(value):
                return "NA"
            lo = row.get(f"{metric}_CI95_lower", np.nan)
            hi = row.get(f"{metric}_CI95_upper", np.nan)
            return (f"{value:.3f} ({lo:.3f}–{hi:.3f})" if np.isfinite(lo) and np.isfinite(hi)
                    else f"{value:.3f} (CI unavailable)")
        readable.append({"PTM_type": row["PTM_type"], "Model": row["Model"],
                         "N+": row["N_positive"], "N-": row["N_negative"],
                         "Accuracy": with_ci("Accuracy"),
                         "Precision": with_ci("Precision"), "Recall": with_ci("Recall"),
                         "ROC-AUC": with_ci("ROC_AUC"), "PR-AUC (AP)": with_ci("PR_AUC_AP")})
    pd.DataFrame(readable).to_csv(output / "test_metrics_readable.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "test_summary_by_ptm.csv", index=False)
    pd.DataFrame(balanced_rows).to_csv(output / "test_comparison_balanced.csv", index=False)
    pd.DataFrame(paired_differences, columns=["PTM_type", "Model_A", "Model_B",
        "AP_Difference_A_minus_B", "CI95_lower", "CI95_upper",
        "Valid_paired_parent_bootstrap_draws"]).to_csv(output / "paired_pr_auc_differences.csv", index=False)
    pd.DataFrame(bootstrap_distribution, columns=["PTM_type", "Model", "Bootstrap_draw", "PR_AUC_AP"]
                 ).to_csv(output / "bootstrap_pr_auc_distribution.csv", index=False)
    full_comparison_rows = [{"PTM_type": ptm, "Model": DISPLAY[model],
                             **full_xlsx[ptm, model]}
                            for ptm in classes for model in MODELS]
    pd.DataFrame(full_comparison_rows).to_csv(output / "test_comparison_full.csv", index=False)
    if not args.no_excel:
        excel_comparison(output / "test_model_comparison.xlsx", full_xlsx, balanced_xlsx, classes)

    # Independence: a parent-level split does not establish model-level
    # independence from external MIND-S pretraining or sequence homology.
    report = {
        "train_json": str(args.train), "test_json": str(args.test),
        "prediction_files": {model: str(path) for model, path in paths.items()},
        "comparison_policy": "only held-out test-side eligible sites scored by ALL three models",
        "train_positive_annotations": len(train_pos), "test_positive_annotations": len(test_pos),
        "train_test_positive_annotation_overlap": len(overlap_positive),
        "train_uids": len(train), "test_uids": len(test),
        "shared_train_test_uids": len(shared_uids),
        "shared_train_test_parent_accessions": len(overlapping_parents),
        "test_uids_with_identical_training_sequence": len(duplicate_sequences),
        "sequence_parent_level_independence_verified": not (
            shared_uids or overlapping_parents or duplicate_sequences),
        "test_parent_accessions": len(test_parents),
        "matched_test_positive_scores": int(sites.loc[sites["common_scored"].eq(1), "test_positive"].sum()),
        "matched_test_presumed_negatives": int((sites["common_scored"].eq(1) & sites["test_positive"].eq(0)).sum()),
        "candidate_test_sites_after_training_positive_exclusion": eligible,
        "candidate_test_sites_with_common_scores": scored,
        "common_model_score_coverage": coverage,
        "threshold_for_accuracy_precision_recall": args.threshold,
        "pr_auc_definition": "sklearn average_precision_score (not fixed threshold)",
        "roc_auc_definition": "continuous probabilities (not fixed threshold)",
        "balanced_policy": "all matched test positives + equally many matched negatives drawn with replacement per PTM; identical draws across models",
        "balanced_repeats": args.balanced_repeats,
        "balanced_aggregation": "median across balanced-negative draws for each model, PTM and metric",
        "bootstrap_unit": "parent UniProt accession (UID prefix before first hyphen)",
        "bootstrap_draws_requested": args.bootstrap,
        "confidence_intervals": "2.5th/97.5th percentiles of shared parent-accession resamples; suppressed if <2 positive parent groups",
        "seed": args.seed,
        "caveats": [
            "Unannotated eligible residues are presumed negatives, not experimentally verified negatives.",
            "Training-negative samples are not listed in source JSONs, so annotation splits may have used withheld positives as negatives.",
            "Protein/parent-disjoint results require zero parent accession overlap and retrained NR-PTMpred predictions.",
            "MIND-S pretraining and receptor-family homology independence are not established by this split.",
            "Missing model scores are excluded rather than filled with zero; this may bias comparisons.",
        ],
    }
    (output / "test_split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved matched three-model held-out comparison to: {output.resolve()}")
    print("Full negatives / balanced negatives; balanced medians use shared draws across models.")
    print("WARNING: only zero parent overlap establishes accession-disjoint test inputs; "
          "MIND-S pretraining independence is not verified.")


if __name__ == "__main__":
    main()
