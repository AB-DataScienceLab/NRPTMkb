#!/usr/bin/env python3
r"""Plot per-PTM ROC curves, deriving benchmark labels DIRECTLY from curated CSVs.

Three-model comparison (directly from curated CSVs):
    python plot_ptm_roc.py \
      --sequences data/curated/NR_Sequences_New.csv \
      --curated data/curated/NR_InputData_New.csv \
      --prediction MIND=results_original/correct_predictions_newlabels.csv \
      --prediction NRPTMpred=results_new/correct_predictions.csv \
      --prediction MusiteDeep=results_musitedeep/correct_predictions.csv \
      --output-dir roc_comparison_plots

The program DOES NOT read benchmark_sites.csv, metrics.csv, or any previous
benchmark output. All amino-acid-eligible positions in the sequence CSV form
candidate sites. Only site/PTM keys explicitly annotated in NR_InputData_New.csv
are labelled positive; all other eligible sites are PRESUMED negatives. Each
PTM is evaluated on the intersection of sites having finite scores from ALL
three prediction files, so ROC comparisons use identical site populations.
This is a full-negative (NOT balanced-negative) evaluation. This benchmark may
contain training annotations and is NOT an independent held-out test.

Outputs: seven PTM-specific ROC figures (.png and .pdf), roc_summary.csv,
roc_curve_points.csv, and roc_site_coverage.csv plus
excluded_positive_sites.csv to audit missing model coverage.

Dependencies: numpy, pandas, matplotlib, scikit-learn.
"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe for headless/remote Python environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

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
MODELS = ("NRPTMpred", "MIND", "MusiteDeep")
LABELS = {"NRPTMpred": "NRPTMpred", "MIND": "MIND-S", "MusiteDeep": "MusiteDeep"}


def arguments():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", default="data/curated/NR_Sequences_New.csv",
                   help="Curated sequences CSV with UniprotID, Sequence")
    p.add_argument("--curated", default="data/curated/NR_InputData_New.csv",
                   help="Curated annotations CSV with UniprotID, Site, PTM_Type")
    p.add_argument("--prediction", action="append", default=[], metavar="MODEL=CSV",
                   help="Repeat for NRPTMpred, MIND, MusiteDeep")
    p.add_argument("--output-dir", default="roc_comparison_plots")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--strict-positive-coverage", action="store_true",
                   help="Stop if any curated positive site lacks a score from a model")
    a = p.parse_args()
    if a.dpi <= 0:
        p.error("--dpi must be positive")
    return a


def canonical_model(name):
    mapping = {
        "NRPTMpred": "NRPTMpred", "NRPTMKB": "NRPTMpred",
        "MIND": "MIND", "MIND-S": "MIND",
        "MusiteDeep": "MusiteDeep",
    }
    try:
        return mapping[name.strip()]
    except KeyError as exc:
        raise ValueError(f"Unknown prediction model {name!r}; use {MODELS}") from exc


def prediction_paths(specs):
    paths = {}
    for spec in specs:
        name, sep, filename = spec.partition("=")
        if not sep or not name.strip() or not filename.strip():
            raise ValueError(f"Invalid --prediction {spec!r}: expected MODEL=CSV")
        name = canonical_model(name)
        if name in paths:
            raise ValueError(f"Duplicate prediction for {name}")
        paths[name] = Path(filename.strip())
    if set(paths) != set(MODELS):
        raise ValueError(f"Exactly three predictions are required: {MODELS}; "
                         f"supplied {tuple(paths)}")
    return paths


def read_csv(path, columns):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def load_sequences(path):
    table = read_csv(path, ("UniprotID", "Sequence"))
    table = table[["UniprotID", "Sequence"]].copy()
    if table.isna().any().any():
        raise ValueError(f"{path}: missing UniProt accession or sequence")
    table["UniprotID"] = table["UniprotID"].astype(str).str.strip()
    table["Sequence"] = table["Sequence"].astype(str).str.strip().str.upper()
    if (table == "").any().any() or table["UniprotID"].duplicated().any():
        raise ValueError(f"{path}: empty fields or duplicate UniProt accessions")
    sequences = dict(zip(table["UniprotID"], table["Sequence"]))
    if not sequences:
        raise ValueError(f"{path}: no sequences")
    print(f"Curated sequence records: {len(sequences)}")
    return sequences


def load_positive_keys(path, sequences):
    table = read_csv(path, ("UniprotID", "Site", "PTM_Type"))
    positives = set()
    ignored = 0
    for uid, annotated_site, ptm_base in table[
            ["UniprotID", "Site", "PTM_Type"]].itertuples(index=False, name=None):
        uid = str(uid).strip()
        match = re.fullmatch(r"([A-Za-z])(\d+)", str(annotated_site).strip())
        if not match:
            raise ValueError(f"{path}: invalid site {annotated_site!r} for {uid}")
        aa, site = match.group(1).upper(), int(match.group(2))
        ptm = f"{str(ptm_base).strip()}_{aa}"
        if ptm not in PTM_AA:
            ignored += 1
            continue
        sequence = sequences.get(uid)
        if sequence is None:
            raise ValueError(f"{path}: curated positive for accession {uid!r}, "
                             "not present in curated sequences")
        if not 1 <= site <= len(sequence) or sequence[site - 1] != aa:
            raise ValueError(f"{path}: site {uid} {aa}{site} ({ptm}) "
                             "does not match its curated protein sequence")
        key = (uid, site, ptm)
        if key in positives:
            raise ValueError(f"{path}: duplicate positive annotation {key}")
        positives.add(key)
    if not positives:
        raise ValueError(f"{path}: no positives from supported PTMs")
    print(f"Curated positive annotations: {len(positives)} in seven PTM classes; "
          f"{ignored} other-class annotations ignored")
    return positives


def generate_candidates(sequences, positives):
    aa_to_ptms = {}
    for ptm, aa in PTM_AA.items():
        aa_to_ptms.setdefault(aa, []).append(ptm)
    records = []
    for uid, sequence in sequences.items():
        for site, aa in enumerate(sequence, start=1):
            for ptm in aa_to_ptms.get(aa, ()):
                records.append((uid, site, ptm, int((uid, site, ptm) in positives)))
    sites = pd.DataFrame(records, columns=KEY + ["label"])
    if len(sites) == 0:
        raise ValueError("No eligible PTM residues in curated sequences")
    if int(sites["label"].sum()) != len(positives):
        raise ValueError("Some curated positives are absent from generated candidate sites")
    if sites.duplicated(KEY).any():
        raise AssertionError("Candidate generator created duplicate site keys")
    print(f"Eligible PTM-residue candidate examples: {len(sites):,}")
    return sites


def read_model_scores(model, path, sequences):
    table = read_csv(path, KEY + ["pred_score"])[KEY + ["pred_score"]].copy()
    table["uid"] = table["uid"].astype(str).str.strip()
    table["PTM_type"] = table["PTM_type"].astype(str).str.strip()
    original_size = len(table)
    table = table.loc[table["PTM_type"].isin(PTM_AA)
                      & table["uid"].isin(sequences)].copy()
    excluded = original_size - len(table)
    positions = pd.to_numeric(table["site"], errors="raise")
    if positions.isna().any() or (positions % 1).ne(0).any() or (positions < 1).any():
        raise ValueError(f"{model}: invalid 1-based positions in {path}")
    table["site"] = positions.astype(np.int64)
    if table.duplicated(KEY).any():
        examples = table.loc[table.duplicated(KEY, keep=False), KEY].head(5)
        raise ValueError(f"{model}: duplicate prediction keys in {path}:\n{examples}")
    aa = table["PTM_type"].map(PTM_AA)
    residue = [sequences[uid][int(site) - 1]
               if int(site) <= len(sequences[uid]) else "?"
               for uid, site in zip(table["uid"], table["site"])]
    bad = np.asarray(residue) != aa.to_numpy()
    if bad.any():
        example = table.loc[bad, KEY].head(5)
        raise ValueError(f"{model}: predictions inconsistent with curated "
                         f"protein sequences (site/amino acid):\n{example}")
    # Blank scores are treated as missing coverage. All nonblank scored values
    # must be valid probabilities; no clipping or implicit imputation.
    table["pred_score"] = pd.to_numeric(table["pred_score"], errors="raise")
    finite = table["pred_score"].notna()
    nonfinite = finite & ~np.isfinite(table["pred_score"].to_numpy())
    out_of_range = finite & ~table["pred_score"].between(0, 1)
    if nonfinite.any() or out_of_range.any():
        raise ValueError(f"{model}: predictions must be finite probabilities in [0,1]")
    if excluded:
        print(f"  {model}: ignored {excluded:,} predictions outside curated "
              "accessions or selected PTM classes")
    print(f"  {model}: {int(finite.sum()):,} usable prediction scores")
    return table.rename(columns={"pred_score": f"_score_{model}"})


def load_curated_benchmark(a, output):
    sequences = load_sequences(a.sequences)
    positives = load_positive_keys(a.curated, sequences)
    sites = generate_candidates(sequences, positives)
    score_columns = {}
    for model, path in prediction_paths(a.prediction).items():
        col = f"_score_{model}"
        scores = read_model_scores(model, path, sequences)
        sites = sites.merge(scores, on=KEY, how="left", validate="one_to_one", sort=False)
        score_columns[LABELS[model]] = col
    all_scored = sites[list(score_columns.values())].notna().all(axis=1)
    missing_positives = sites.loc[(sites["label"] == 1) & ~all_scored].copy()
    missing_positives["Missing_models"] = missing_positives.apply(
        lambda row: ",".join(label for label, col in score_columns.items()
                             if pd.isna(row[col])), axis=1)
    missing_positives[KEY + ["Missing_models"]].to_csv(
        output / "excluded_positive_sites.csv", index=False)
    if len(missing_positives):
        print(f"WARNING: {len(missing_positives)} curated positive annotations "
              "lack one or more model scores; details: excluded_positive_sites.csv")
        if a.strict_positive_coverage:
            raise ValueError("Missing scores for curated positives; "
                             "see excluded_positive_sites.csv")
    used = sites.loc[all_scored].copy()
    if used.empty:
        raise ValueError("No common scored candidate sites across all three models")
    audit = []
    for ptm in PTM_AA:
        candidates = sites.loc[sites["PTM_type"] == ptm]
        evaluated = used.loc[used["PTM_type"] == ptm]
        row = {
            "PTM_type": ptm,
            "Candidate_positive": int(candidates["label"].sum()),
            "Candidate_negative": int((candidates["label"] == 0).sum()),
            "Scored_positive": int(evaluated["label"].sum()),
            "Scored_negative": int((evaluated["label"] == 0).sum()),
        }
        row["Missing_positive"] = row["Candidate_positive"] - row["Scored_positive"]
        row["Missing_negative"] = row["Candidate_negative"] - row["Scored_negative"]
        audit.append(row)
        print(f"{ptm:22s} common: N+={row['Scored_positive']:4d} "
              f"N-={row['Scored_negative']:5d}; "
              f"missing N+={row['Missing_positive']:3d} "
              f"N-={row['Missing_negative']:4d}")
    pd.DataFrame(audit).to_csv(output / "roc_site_coverage.csv", index=False)
    return used, score_columns


def plot_curves(df, score_columns, output, dpi, source):
    summaries = []
    points = []
    generated = 0
    for ptm in PTM_AA:
        sub = df.loc[df["PTM_type"] == ptm]
        y = sub["label"].to_numpy(dtype=np.int8)
        npos = int(y.sum())
        nneg = int(len(y) - npos)
        if npos == 0 or nneg == 0:
            warnings.warn(f"Skipping {ptm}: ROC undefined (N+={npos}, N-={nneg})")
            continue
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        for model, column in score_columns.items():
            prob = sub[column].to_numpy(dtype=np.float64)
            fpr, tpr, thresholds = roc_curve(y, prob, drop_intermediate=False)
            auc = float(roc_auc_score(y, prob))
            ap = float(average_precision_score(y, prob))
            ax.plot(fpr, tpr, linewidth=2, label=f"{model} (AUC = {auc:.3f})")
            summaries.append({
                "PTM_type": ptm, "Model": model, "N_positive": npos,
                "N_negative": nneg, "ROC_AUC": auc, "Average_Precision": ap,
                "Source": source,
            })
            points.extend({
                "PTM_type": ptm, "Model": model, "FPR": float(x),
                "TPR": float(z), "Threshold": float(threshold), "ROC_AUC": auc,
            } for x, z, threshold in zip(fpr, tpr, thresholds))
            print(f"{ptm:22s} {model:14s} N+={npos:4d} N-={nneg:5d} "
                  f"ROC-AUC={auc:.6f} AP={ap:.6f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2,
                color="0.5", label="Random classifier")
        ax.set(xlabel="False positive rate (1 - specificity)",
               ylabel="True positive rate (sensitivity)",
               xlim=(0, 1), ylim=(0, 1),
               title=f"{ptm.replace('_', ' ')}  |  N+ = {npos}, N- = {nneg}")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{ptm}_roc.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated += 1
    if not summaries:
        raise ValueError("No PTM classes have both positive and negative examples")
    pd.DataFrame(summaries).to_csv(output / "roc_summary.csv", index=False)
    pd.DataFrame(points).to_csv(output / "roc_curve_points.csv", index=False)
    print(f"\nSaved {generated} PTM-specific ROC plots (PNG + PDF) and "
          f"metric/curve CSVs to {output.resolve()}")


def main():
    a = arguments()
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df, score_columns = load_curated_benchmark(a, output)
    plot_curves(df, score_columns, output, a.dpi, "curated_full_negatives")


if __name__ == "__main__":
    main()
