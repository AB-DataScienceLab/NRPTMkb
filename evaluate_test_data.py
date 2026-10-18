#!/usr/bin/env python3
"""Audit test-only PTM metrics for ONE model (NRPTMKB).

Uses the PTM-specific positive labels in test_data_new.json, removes all
known training-positive (uid, residue, PTM) sites, and treats other eligible
residues as PRESUMED NEGATIVES. The training JSON does NOT identify sampled
training negatives; therefore a fully independent negative set is UNVERIFIED.

Input JSON positions are 0-based; model score CSV positions are 1-based.

Example:
  python evaluate_test_data.py \
      --train train_data_new.json --test test_data_new.json \
      --predictions results_newwwww/correct_predictions.csv \
      --output-dir test_only_results

Outputs: test_metrics.csv, test_metrics_readable.csv, test_sites.csv,
         omitted_sites.csv, test_split_summary.json, test_summary_by_ptm.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

KEY = ["uid", "site", "PTM_type"]
DEFAULT_CLASSES = [
    "Acetylation_K", "Methylation_K", "Phosphorylation_S",
    "Phosphorylation_T", "Phosphorylation_Y", "Sumoylation_K",
    "Ubiquitination_K",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="train_data_new.json", help="Training JSON with seq/label fields")
    p.add_argument("--test", default="test_data_new.json", help="Test JSON with seq/label fields")
    p.add_argument("--predictions", default="results_newwwww/correct_predictions.csv",
                   help="NRPTMKB per-site continuous prediction scores, INCLUDING scores below 0.5")
    p.add_argument("--output-dir", default="test_only_results")
    p.add_argument("--threshold", type=float, default=0.5, help="Fixed decision threshold; default 0.5")
    p.add_argument("--bootstrap", type=int, default=1000, help="Parent-protein bootstrap draws; default 1000")
    p.add_argument("--seed", type=int, default=252)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="PTM types to evaluate")
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--min-score-coverage", type=float, default=0.7,
                   help="Stop if scored fraction is below 0.7 by default. "
                        "Set 0 only if low model coverage is independently justified.")
    return p.parse_args()


def read_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    if not isinstance(d, dict):
        raise ValueError("Expected JSON mapping uid -> {'seq': str, 'label': list}")
    for uid, rec in d.items():
        if not isinstance(rec.get("seq"), str) or not isinstance(rec.get("label"), list):
            raise ValueError("Invalid seq/label data for " + uid)
    return d


def positives(dataset):
    """Translate zero-based JSON sites to one-based positions for matching scores."""
    pos = set()
    for uid, record in dataset.items():
        seq = record["seq"]
        for annotation in record["label"]:
            index = int(annotation["site"])
            ptm = str(annotation["ptm_type"])
            aa = ptm.rsplit("_", 1)[-1]
            if index < 0 or index >= len(seq):
                raise ValueError("Out-of-range annotation: %s %s %s" % (uid, index, ptm))
            if seq[index] not in aa:
                raise ValueError("PTM/residue mismatch: %s pos=%s PTM=%s found=%s" %
                                 (uid, index + 1, ptm, seq[index]))
            key = (uid, index + 1, ptm)
            if key in pos:
                raise ValueError("Duplicate positive annotation: %r" % (key,))
            pos.add(key)
    return pos


def get_scores(path):
    p = pd.read_csv(path, dtype={"uid": str, "PTM_type": str})
    if "pred_score" not in p.columns and "score" in p.columns:
        p = p.rename(columns={"score": "pred_score"})
    need = set(KEY + ["pred_score"])
    if not need.issubset(p.columns):
        raise ValueError("Predictions missing columns: %s. You need FULL site scores, not only predicted positives."
                         % sorted(need - set(p.columns)))
    p = p[KEY + ["pred_score"]].copy()
    p["site"] = pd.to_numeric(p["site"], errors="raise")
    if p["site"].isna().any() or not np.all(p["site"] == p["site"].astype(int)) or (p["site"] < 1).any():
        raise ValueError("Prediction site must contain positive, one-based integer positions")
    p["site"] = p["site"].astype(int)
    p["pred_score"] = pd.to_numeric(p["pred_score"], errors="raise")
    if p.duplicated(KEY).any():
        dup = p.loc[p.duplicated(KEY, keep=False), KEY].head(3).to_dict("records")
        raise ValueError("Duplicate model predictions at site/PTM: %s" % dup)
    bad = p["pred_score"].notna() & (~p["pred_score"].between(0, 1))
    if bad.any():
        raise ValueError("Predictions outside [0,1]. These must be probabilities, not logits.")
    return p


def ece_score(y, s, n_bins):
    if len(y) == 0:
        return np.nan
    bin_idx = np.minimum((s * n_bins).astype(int), n_bins - 1)
    return float(sum(np.mean(bin_idx == k) * abs(np.mean(y[bin_idx == k]) - np.mean(s[bin_idx == k]))
                     for k in range(n_bins) if (bin_idx == k).any()))


def compute_metrics(y, scores, threshold, n_bins):
    n = len(y)
    if n == 0:
        return {}
    yhat = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    values = dict(
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        Accuracy=float(accuracy_score(y, yhat)),
        All_negative_accuracy=float(np.mean(y == 0)),
        Precision=float(precision_score(y, yhat, zero_division=0)),
        Recall=float(recall_score(y, yhat, zero_division=0)),
        Specificity=float(tn / (tn + fp)) if tn + fp else np.nan,
        F1=float(f1_score(y, yhat, zero_division=0)),
        MCC=float(matthews_corrcoef(y, yhat)),
        Balanced_accuracy=float((tp / (tp + fn) + tn / (tn + fp)) / 2)
        if (tp + fn) and (tn + fp) else np.nan,
        ROC_AUC=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        PR_AUC_AP=float(average_precision_score(y, scores)) if np.any(y == 1) else np.nan,
        Brier=float(brier_score_loss(y, scores)),
        ECE=float(ece_score(y, scores, n_bins)),
    )
    return values


def bootstrap_confidence_intervals(frame, threshold, n_bins, n_draws, seed):
    """Exact cluster resampling by weights, avoiding expensive repeated sklearn calls.

    For each bootstrap draw, a parent accession may appear 0, 1, 2, ... times.
    Using that multiplicity as its site's sample weight is mathematically the
    same as physically repeating its rows. AP/ROC handle identical score ties.
    """
    metrics = ("Accuracy", "Balanced_accuracy", "Precision", "Recall", "Specificity",
               "F1", "MCC", "ROC_AUC", "PR_AUC_AP", "Brier", "ECE")
    blank = {m + "_CI95_lower": np.nan for m in metrics}
    blank.update({m + "_CI95_upper": np.nan for m in metrics})
    blank.update({"Bootstrap_valid_PR_AUC": 0, "Bootstrap_valid_ROC_AUC": 0})
    if n_draws <= 0 or frame.empty:
        return blank

    rng = np.random.default_rng(seed)
    y = frame["label"].to_numpy(dtype=np.int8)
    p = frame["pred_score"].to_numpy(dtype=float)
    ids, group_idx = np.unique(frame["parent_accession"].astype(str), return_inverse=True)
    ngroup = len(ids)
    sort_idx = np.argsort(-p, kind="mergesort")
    ysort = y[sort_idx]
    psort = p[sort_idx]
    gsort = group_idx[sort_idx]
    tie_ends = np.flatnonzero(np.r_[psort[1:] != psort[:-1], True])
    pred = p >= threshold
    bin_idx = np.minimum((p * n_bins).astype(int), n_bins - 1)

    samples = {name: [] for name in metrics}
    for _ in range(n_draws):
        multiplicity = np.bincount(rng.integers(0, ngroup, size=ngroup), minlength=ngroup)
        w = multiplicity[group_idx]
        n = float(np.sum(w))
        pos = float(np.sum(w[y == 1]))
        neg = n - pos
        if n <= 0:
            continue
        tp = float(np.sum(w[(y == 1) & pred]))
        fp = float(np.sum(w[(y == 0) & pred]))
        fn = pos - tp
        tn = neg - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / pos if pos else 0.0
        spec = tn / neg if neg else np.nan
        acc = (tp + tn) / n
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0
        base = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "Specificity": spec,
            "F1": f1, "MCC": mcc,
            "Balanced_accuracy": (rec + spec) / 2 if pos > 0 and neg > 0 else np.nan,
            "Brier": float(np.dot(w, (p - y) ** 2) / n),
        }
        weighted_bin = np.bincount(bin_idx, weights=w, minlength=n_bins)
        sum_true = np.bincount(bin_idx, weights=w * y, minlength=n_bins)
        sum_prob = np.bincount(bin_idx, weights=w * p, minlength=n_bins)
        base["ECE"] = float(np.abs(sum_true - sum_prob).sum() / n)
        if pos > 0:
            # Descending score thresholds and exact weighted tie grouping.
            ws = w[sort_idx]
            cum_pos = np.cumsum(ws * ysort)[tie_ends]
            cum_neg = np.cumsum(ws * (1 - ysort))[tie_ends]
            pos_change = np.diff(np.r_[0.0, cum_pos])
            precision_at_scores = np.divide(
                cum_pos, cum_pos + cum_neg,
                out=np.zeros_like(cum_pos, dtype=float), where=(cum_pos + cum_neg) > 0)
            base["PR_AUC_AP"] = float(np.dot(pos_change / pos, precision_at_scores))
            if neg > 0:
                tpr = np.r_[0.0, cum_pos / pos]
                fpr = np.r_[0.0, cum_neg / neg]
                base["ROC_AUC"] = float(np.dot(np.diff(fpr), (tpr[:-1] + tpr[1:]) / 2))
            else:
                base["ROC_AUC"] = np.nan
        else:
            base["PR_AUC_AP"] = np.nan
            base["ROC_AUC"] = np.nan
        for m in metrics:
            if np.isfinite(base[m]):
                samples[m].append(base[m])
    for m in metrics:
        if samples[m]:
            blank[m + "_CI95_lower"], blank[m + "_CI95_upper"] = map(
                float, np.percentile(samples[m], [2.5, 97.5]))
    blank["Bootstrap_valid_PR_AUC"] = len(samples["PR_AUC_AP"])
    blank["Bootstrap_valid_ROC_AUC"] = len(samples["ROC_AUC"])
    return blank


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1 or args.bootstrap < 0 or args.ece_bins < 2:
        raise ValueError("Threshold must be in [0,1], --bootstrap >= 0, and --ece-bins >= 2")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = read_dataset(args.train)
    test = read_dataset(args.test)
    train_pos = positives(train)
    test_pos = positives(test)
    dup_pos = train_pos & test_pos
    if dup_pos:
        raise ValueError("Train/test positive annotation leakage: %d identical uid/site/PTM labels" % len(dup_pos))

    shared_ids = set(train) & set(test)
    mismatched_sequences = sorted(u for u in shared_ids if train[u]["seq"] != test[u]["seq"])
    if mismatched_sequences:
        raise ValueError("Training and test sequences differ for same UID, e.g. %s" % mismatched_sequences[:3])

    scores = get_scores(args.predictions)
    present_classes = set(scores["PTM_type"])
    requested_classes = list(dict.fromkeys(args.classes))
    for ptm in requested_classes:
        if ptm not in present_classes:
            print("WARNING: no predictions at all for", ptm)

    train_by_ptm = {}
    test_by_ptm = {}
    for key in train_pos:
        train_by_ptm.setdefault(key[2], set()).add(key)
    for key in test_pos:
        test_by_ptm.setdefault(key[2], set()).add(key)

    # Generate ALL eligible positions from the test JSON first. Do not use the
    # pooled benchmark's labels: that would leak training-positive annotations.
    rows = []
    for uid, record in sorted(test.items()):
        seq = record["seq"]
        for ptm in requested_classes:
            aa = ptm.rsplit("_", 1)[-1]
            for index, char in enumerate(seq):
                if char not in aa:
                    continue
                key = (uid, index + 1, ptm)
                is_train_pos = key in train_pos
                is_test_pos = key in test_pos
                rows.append({
                    "uid": uid, "site": index + 1, "PTM_type": ptm,
                    "residue": char, "parent_accession": uid.split("-")[0],
                    "train_positive": int(is_train_pos),
                    "test_positive": int(is_test_pos),
                    "label": int(is_test_pos),
                    "eval_eligible": int(not is_train_pos),
                })
    universe = pd.DataFrame(rows)
    if universe.empty:
        raise ValueError("No eligible residues found in test JSON for requested classes")
    universe = universe.merge(scores, on=KEY, how="left", validate="one_to_one")
    universe["has_score"] = universe["pred_score"].notna().astype(int)
    universe["used_in_metrics"] = (universe["eval_eligible"].eq(1) & universe["has_score"].eq(1)).astype(int)
    universe["predicted_positive"] = pd.array([pd.NA] * len(universe), dtype="Int64")
    good = universe["has_score"].eq(1)
    universe.loc[good, "predicted_positive"] = (universe.loc[good, "pred_score"] >= args.threshold).astype(int).to_numpy()
    universe["sample_type"] = np.where(
        universe["train_positive"].eq(1), "excluded_training_positive",
        np.where(universe["test_positive"].eq(1), "heldout_positive", "presumed_negative"),
    )
    universe["exclusion_reason"] = np.where(
        universe["eval_eligible"].eq(0), "training_positive",
        np.where(universe["has_score"].eq(0), "missing_prediction_score", ""),
    )
    eligible_count = int(universe["eval_eligible"].sum())
    scored_count = int(universe["used_in_metrics"].sum())
    score_coverage = scored_count / eligible_count if eligible_count else 0
    if score_coverage < args.min_score_coverage:
        raise ValueError("Only %.1f%% of eligible test sites have scores (%d/%d). "
                         "Likely a threshold-filtered correct_predictions.csv rather than FULL "
                         "scores! Supply the complete per-site pred_score CSV. To bypass "
                         "the safety check explicitly, use --min-score-coverage 0."
                         % (100 * score_coverage, scored_count, eligible_count))
    universe.sort_values(KEY, inplace=True)
    universe.to_csv(output / "test_sites.csv", index=False)
    universe.loc[universe["used_in_metrics"].eq(0)].to_csv(output / "omitted_sites.csv", index=False)

    results = []
    summaries = []
    for offset, ptm in enumerate(requested_classes):
        full = universe[universe["PTM_type"] == ptm]
        sub = full[full["used_in_metrics"].eq(1)].reset_index(drop=True)
        n_test = len(test_by_ptm.get(ptm, set()))
        n_train = len(train_by_ptm.get(ptm, set()))
        n_scored_pos = int(sub["label"].sum())
        n_missing_test_pos = int(((full["test_positive"] == 1) & (full["has_score"] == 0)).sum())
        summary = {
            "PTM_type": ptm, "N_training_positive_annotations": n_train,
            "N_test_positive_annotations": n_test, "N_test_positives_with_scores": n_scored_pos,
            "N_test_positives_missing_scores": n_missing_test_pos,
            "N_training_positive_sites_excluded": int(full["train_positive"].sum()),
            "N_eligible_test_positions_after_train_positive_removal": int(full["eval_eligible"].sum()),
            "N_evaluation_sites_with_scores": len(sub),
            "N_presumed_negative_scored": int((sub["label"] == 0).sum()),
            "N_missing_scores_on_other_eligible_sites": int(((full["eval_eligible"] == 1) & (full["test_positive"] == 0) & (full["has_score"] == 0)).sum()),
            "N_test_uid_with_evaluable_sites": sub["uid"].nunique(),
            "N_parent_accessions_with_evaluable_sites": sub["parent_accession"].nunique(),
        }
        summaries.append(summary)
        rec = {"Model": "NRPTMKB", **summary, "Threshold": args.threshold, "Bootstrap_requested": args.bootstrap}
        if len(sub) and n_scored_pos:
            y = sub["label"].to_numpy(dtype=int)
            s = sub["pred_score"].to_numpy(dtype=float)
            rec.update(compute_metrics(y, s, args.threshold, args.ece_bins))
            rec.update(bootstrap_confidence_intervals(sub, args.threshold, args.ece_bins,
                                                     args.bootstrap, args.seed + offset))
        else:
            print("WARNING:", ptm, "has no scored test positives; no meaningful test metrics")
        results.append(rec)

    pd.DataFrame(results).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "test_summary_by_ptm.csv", index=False)

    # Easy-to-read table with the most useful metrics and CIs.
    compact = []
    for row in results:
        def fmt(metric):
            value = row.get(metric, np.nan)
            if not np.isfinite(value):
                return "NA"
            lo = row.get(metric + "_CI95_lower", np.nan)
            hi = row.get(metric + "_CI95_upper", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                return "%.3f (%.3f–%.3f)" % (value, lo, hi)
            return "%.3f" % value
        compact.append({
            "PTM_type": row["PTM_type"], "N+": row["N_test_positives_with_scores"],
            "N-": row["N_presumed_negative_scored"],
            "Accuracy": fmt("Accuracy"), "Balanced_accuracy": fmt("Balanced_accuracy"),
            "Precision": fmt("Precision"), "Recall": fmt("Recall"),
            "F1": fmt("F1"), "MCC": fmt("MCC"),
            "ROC_AUC": fmt("ROC_AUC"), "PR_AUC_AP": fmt("PR_AUC_AP"),
        })
    pd.DataFrame(compact).to_csv(output / "test_metrics_readable.csv", index=False)

    # Include simple identity/coverage checks so users can audit claims.
    report = {
        "evaluation_model": "NRPTMKB",
        "training_json": args.train, "test_json": args.test,
        "prediction_csv": args.predictions,
        "train_uid_count": len(train), "test_uid_count": len(test),
        "shared_train_test_uid_count": len(shared_ids),
        "unseen_test_uid_count": len(set(test) - set(train)),
        "unseen_test_parent_accessions": len({x.split("-")[0] for x in test} -
                                              {x.split("-")[0] for x in train}),
        "training_positive_annotation_count": len(train_pos),
        "test_positive_annotation_count": len(test_pos),
        "training_test_positive_annotation_overlap": len(dup_pos),
        "test_protein_sequences_equal_when_uid_shared": not bool(mismatched_sequences),
        "score_column": "pred_score", "fraction_eligible_sites_with_score": score_coverage,
        "test_json_position_base": 0,
        "predictions_csv_position_base": 1, "classification_threshold": args.threshold,
        "bootstrap_unit": "parent accession (prefix before first hyphen)",
        "bootstrap_draws_requested": args.bootstrap, "bootstrap_seed": args.seed,
        "ground_truth_definition": "test JSON positives; all other eligible non-training-positive residues are presumed negatives",
        "independence_warning": "All test UIDs may already occur in training; negative samples actually used during training are not recorded and may include test positives/negatives. This is NOT a verified independent-protein test.",
        "missing_scores_policy": "Never assign 0 to missing model scores; exclude and audit them",
        "ci_definition": "2.5 and 97.5 percentiles of parent-protein cluster bootstrap",
        "metric_definition": "PR_AUC_AP = sklearn average_precision_score (non-interpolated average precision)",
    }
    with open(output / "test_split_summary.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("\nNRPTMKB — HELD-OUT POSITIVE-SITE / PRESUMED-NEGATIVE EVALUATION")
    print("Training positives:", len(train_pos), "| Test positives:", len(test_pos),
          "| Positive overlap:", len(dup_pos))
    print("Shared protein UIDs:", len(shared_ids), "| Unseen test UIDs:", len(set(test)-set(train)))
    print("\n" + pd.DataFrame(compact).to_string(index=False))
    print("\nFiles saved to", output.resolve())
    print("\nCAUTION: presumed negatives may have been sampled during training; the supplied JSONs")
    print("do not record those sampled negatives. Test proteins also appear in training.")


if __name__ == "__main__":
    main()
