#!/usr/bin/env python3
"""Fair PTM benchmark with protein-bootstrap PR-AUC confidence intervals.

Required input: continuous per-site prediction scores (uid, site, PTM_type,
pred_score). Sites are one-based; pred_score is a probability in [0, 1].
For each PTM, competing models are evaluated on shared scored sites only.
Missing sites are audited and never assigned invented scores.

Example:
    python evaluate_result.py \
        --prediction MIND=results_newwwww/all_site_scores.csv \
        --prediction Baseline=baseline_scores.csv

Optional:
    --sites-csv benchmark_sites.csv   # fixed test examples with labels
    --thresholds-json thresholds.json # validation-selected thresholds
    --output-dir results_newwwww

Other settings (input filenames, supported PTMs, training IDs, bootstrap size)
are in the CONFIG section below. If --sites-csv is absent, all eligible
residues are evaluated and unannotated sites are presumed negative.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
)


LABEL_TO_AA = {
    'Acetylation_K': 'K',
    'Methylation_K': 'K',
    'Phosphorylation_S': 'S',
    'Phosphorylation_T': 'T',
    'Phosphorylation_Y': 'Y',
    'Sumoylation_K': 'K',
    'Ubiquitination_K': 'K',
}
KEY = ['uid', 'site', 'PTM_type']


# -------------------- CONFIG (usually no CLI changes needed) --------------------
SEQUENCES_CSV = 'data/curated/NR_Sequences_New.csv'
CURATED_CSV = 'data/curated/NR_InputData_New.csv'
DEFAULT_OUTPUT_DIR = 'results_newwwww'
N_BOOTSTRAP = 1000
RANDOM_SEED = 252

# Optional checks/overrides. Set paths here when available.
SUPPORTED_CLASSES_JSON = None       # e.g. 'supported_classes.json'
TRAIN_UID_FILES = []                # e.g. ['train_uids.csv']

# Rules for reporting confidence intervals and calibration.
MIN_POSITIVE_FOR_CI = 2
MIN_POSITIVE_PROTEINS_FOR_CI = 2
LOW_SUPPORT_POSITIVE = 10
N_CALIBRATION_BINS = 10

# True: compare only identically scored (uid,site,PTM_type) examples; report omissions.
# False: require every supported model to score every benchmark candidate site.
MATCH_COMMON_SCORED_SITES = True
# -------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate PTM predictions with matched sites and bootstrap PR-AUC.'
    )
    parser.add_argument(
        '--prediction', action='append', required=True, metavar='MODEL=CSV',
        help='Model name and continuous per-site score CSV; repeat for each model.'
    )
    parser.add_argument(
        '--sites-csv', default=None, metavar='CSV',
        help='Optional fixed benchmark (uid,site,PTM_type,label).'
    )
    parser.add_argument(
        '--thresholds-json', default=None, metavar='JSON',
        help='Optional model/PTM thresholds fixed using validation data.'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR, metavar='DIR',
        help=f'Result folder (default: {DEFAULT_OUTPUT_DIR}).'
    )
    args = parser.parse_args()

    # Internal settings kept out of the command-line interface.
    args.sequences = SEQUENCES_CSV
    args.curated = CURATED_CSV
    args.supported_json = SUPPORTED_CLASSES_JSON
    args.train_uids = TRAIN_UID_FILES
    args.n_bootstrap = N_BOOTSTRAP
    args.seed = RANDOM_SEED
    args.min_positive_for_ci = MIN_POSITIVE_FOR_CI
    args.min_positive_proteins_for_ci = MIN_POSITIVE_PROTEINS_FOR_CI
    args.low_support_positive = LOW_SUPPORT_POSITIVE
    args.n_calibration_bins = N_CALIBRATION_BINS
    args.match_common_scored_sites = MATCH_COMMON_SCORED_SITES
    return args


def load_sequences(path):
    seq = pd.read_csv(path, dtype={'UniprotID': str, 'Sequence': str})
    required = {'UniprotID', 'Sequence'}
    if not required.issubset(seq.columns):
        raise ValueError(f'{path}: expected columns {sorted(required)}')
    seq = seq.dropna(subset=['UniprotID', 'Sequence']).copy()
    seq['UniprotID'] = seq['UniprotID'].str.strip()
    seq['Sequence'] = seq['Sequence'].str.upper().str.strip()
    if seq['UniprotID'].duplicated().any():
        raise ValueError('Test sequences contain duplicate UniProt IDs.')
    if seq['Sequence'].str.len().eq(0).any():
        raise ValueError('Empty protein sequences found.')
    return dict(zip(seq['UniprotID'], seq['Sequence']))


def load_curated(path, sequences):
    df = pd.read_csv(path, dtype={'UniprotID': str, 'PTM_Type': str, 'Site': str})
    required = {'UniprotID', 'PTM_Type', 'Site'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: expected columns {sorted(required)}')
    positive = set()
    n_not_in_test = 0
    for row in df.itertuples(index=False):
        uid = str(row.UniprotID).strip()
        m = re.fullmatch(r'([A-Za-z])(\d+)', str(row.Site).strip())
        if not m:
            raise ValueError(f'Invalid curated Site={row.Site!r} for {uid}; expected e.g. K42')
        aa, site = m.group(1).upper(), int(m.group(2))
        label = f'{str(row.PTM_Type).strip()}_{aa}'
        if label not in LABEL_TO_AA:
            continue  # Outside the seven PTM classes being benchmarked.
        if uid not in sequences:
            n_not_in_test += 1
            continue
        seq = sequences[uid]
        if not 1 <= site <= len(seq) or seq[site - 1] != aa:
            raise ValueError(f'Curated {uid} {label} at {site}: disagrees with test sequence.')
        positive.add((uid, site, label))
    if n_not_in_test:
        print(f'INFO: {n_not_in_test} curated annotations are outside supplied test proteins.')
    return positive


def validate_sites(frame, sequences, name):
    if frame.empty:
        raise ValueError(f'{name}: no candidate sites.')
    if frame[KEY].isna().any().any():
        raise ValueError(f'{name}: null uid/site/PTM_type found.')
    frame['uid'] = frame['uid'].astype(str).str.strip()
    frame['PTM_type'] = frame['PTM_type'].astype(str).str.strip()
    raw_site = pd.to_numeric(frame['site'], errors='raise')
    if (raw_site % 1 != 0).any() or (raw_site <= 0).any():
        raise ValueError(f'{name}: site must be a positive 1-based integer.')
    frame['site'] = raw_site.astype(np.int64)
    if frame.duplicated(KEY).any():
        example = frame.loc[frame.duplicated(KEY, keep=False), KEY].head(3)
        raise ValueError(f'{name}: duplicate (uid,site,PTM_type) entries:\n{example}')
    bad_classes = set(frame['PTM_type']) - set(LABEL_TO_AA)
    if bad_classes:
        raise ValueError(f'{name}: unsupported PTM labels: {sorted(bad_classes)}')
    for uid, site, label in frame[KEY].itertuples(index=False, name=None):
        seq = sequences.get(uid)
        if seq is None:
            raise ValueError(f'{name}: uid {uid!r} absent from test sequences.')
        if site > len(seq) or seq[site - 1] != LABEL_TO_AA[label]:
            raise ValueError(f'{name}: invalid {uid} residue {site} for {label}.')
    return frame


def build_benchmark(args, sequences, curated):
    if args.sites_csv:
        sites = pd.read_csv(args.sites_csv)
        required = set(KEY + ['label'])
        if not required.issubset(sites.columns):
            raise ValueError(f'{args.sites_csv}: expected columns {sorted(required)}')
        sites = validate_sites(sites[KEY + ['label']].copy(), sequences, args.sites_csv)
        if sites['label'].isna().any() or not sites['label'].isin([0, 1, 0.0, 1.0]).all():
            raise ValueError(f'{args.sites_csv}: label must be 0 or 1 at every site.')
        sites['label'] = sites['label'].astype(np.int8)
        # Ensure that labels never disagree with the independently supplied curated positives.
        curated_in_sites = {
            (uid, site, ptm)
            for uid, site, ptm, truth in sites[KEY + ['label']].itertuples(index=False, name=None)
            if truth == 1
        }
        # This explicit benchmark can have additional independently verified
        # positives not listed in the original curated file. However, any
        # known curated positive present in the benchmark must not be 0.
        present_keys = set(map(tuple, sites[KEY].to_numpy()))
        false_negs = [key for key in curated if key in present_keys
                      and key not in curated_in_sites]
        if false_negs:
            raise ValueError(f'{len(false_negs)} curated positives are marked negative in sites CSV.')
    else:
        aa_labels = {}
        for label, aa in LABEL_TO_AA.items():
            aa_labels.setdefault(aa, []).append(label)
        rows = []
        for uid, seq in sequences.items():
            for site, aa in enumerate(seq, start=1):
                for label in aa_labels.get(aa, ()):
                    rows.append((uid, site, label, int((uid, site, label) in curated)))
        sites = pd.DataFrame(rows, columns=KEY + ['label'])
        if sites.empty:
            raise ValueError('No candidate sites in test proteins.')
    return sites.sort_values(KEY).reset_index(drop=True)


def named_paths(specs):
    if not specs:
        raise ValueError('At least one --prediction MODEL=CSV is required. '
                         'correct_predictions.csv cannot supply continuous per-site scores. '
                         'Export uid,site,PTM_type,pred_score for ALL candidate sites.')
    output = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'Invalid --prediction {spec!r}; expected MODEL=CSV.')
        name, filename = spec.split('=', 1)
        if not name.strip() or not filename.strip() or name.strip() in output:
            raise ValueError(f'Invalid/duplicate model name in {spec!r}.')
        output[name.strip()] = filename.strip()
    return output


def load_json(path):
    if not path:
        return {}
    with open(path, encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'{path} must contain a JSON object.')
    return obj


def load_predictions(name, path, sequences):
    df = pd.read_csv(path)
    required = KEY + ['pred_score']
    if not set(required).issubset(df.columns):
        missing = sorted(set(required) - set(df.columns))
        raise ValueError(f'{name} ({path}): missing {missing}. '
                         'A binary/correct-predictions-only file is insufficient. '
                         'Export a continuous pred_score for EVERY eligible site.')
    df = df[required].copy()
    df['uid'] = df['uid'].astype(str).str.strip()
    df['PTM_type'] = df['PTM_type'].astype(str).str.strip()
    df['pred_score'] = pd.to_numeric(df['pred_score'], errors='raise')
    if df['pred_score'].isna().any() or not np.isfinite(df['pred_score']).all():
        raise ValueError(f'{name}: missing/non-finite prediction pred_score values.')
    if not df['pred_score'].between(0, 1).all():
        raise ValueError(f'{name}: pred_score values must be probabilities in [0,1].')
    # Predictions for proteins outside this test set can be ignored, but never
    # silently ignore missing scores for candidates in the test set.
    n_outside = (~df['uid'].isin(sequences)).sum()
    if n_outside:
        print(f'INFO: {name}: ignoring {n_outside} prediction rows outside test proteins.')
        df = df.loc[df['uid'].isin(sequences)].copy()

    # A model may support more PTM classes than this benchmark. Ignore those
    # classes instead of rejecting the entire prediction file. The comparison
    # still uses only classes in LABEL_TO_AA, with fixed test sites per class.
    out_of_scope = ~df['PTM_type'].isin(LABEL_TO_AA)
    if out_of_scope.any():
        extra = sorted(df.loc[out_of_scope, 'PTM_type'].unique().tolist())
        print(f'INFO: {name}: ignoring {int(out_of_scope.sum())} predictions '
              f'for PTM classes outside this benchmark: {extra}')
        df = df.loc[~out_of_scope].copy()

    if df.empty:
        raise ValueError(f'{name}: no predictions remain for benchmark PTM classes.')

    if df['pred_score'].min() >= 0.8:
        print(f'WARNING: {name}: all retained pred_score values are >= 0.8. '
              'This looks like an above-threshold-only output. Full PR-AUC '
              'requires scores for all evaluated sites, including low scores.')

    df = validate_sites(df, sequences, f'{name} prediction table')
    return df


def training_overlap_check(paths, test_uids):
    if not paths:
        print('WARNING: no TRAIN_UID_FILES configured; train/test independence NOT verified.')
        return False
    all_training = set()
    for path in paths:
        try:
            df = pd.read_csv(path, dtype=str)
            col = next((c for c in ('uid', 'UniprotID') if c in df.columns), None)
            if col:
                uids = df[col].dropna().astype(str).str.strip().tolist()
            else:
                uids = Path(path).read_text(encoding='utf-8').splitlines()
        except (pd.errors.ParserError, UnicodeDecodeError):
            uids = Path(path).read_text(encoding='utf-8').splitlines()
        all_training.update(uid.strip() for uid in uids if uid.strip())
    overlap = set(test_uids).intersection(all_training)
    if overlap:
        raise ValueError(f'TRAIN/TEST LEAKAGE: {len(overlap)} test proteins appear '
                         f'in training UID lists. Examples: {sorted(overlap)[:10]}')
    print('PASS: no exact protein UID overlap with supplied training UID lists. '
          '(Homology and any missing training sets still require separate checks.)')
    return True


def calibration(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, p, side='right') - 1, bins - 1)
    rows = []
    ece = 0.0
    for i in range(bins):
        mask = assignments == i
        count = int(mask.sum())
        confidence = float(np.mean(p[mask])) if count else np.nan
        frequency = float(np.mean(y[mask])) if count else np.nan
        if count:
            ece += count / len(y) * abs(frequency - confidence)
        rows.append({'bin': i, 'bin_lower': float(edges[i]),
                     'bin_upper': float(edges[i + 1]), 'count': count,
                     'mean_predicted_probability': confidence,
                     'observed_positive_fraction': frequency})
    return float(ece), rows


def bootstrap_joint(y, groups, scores, n_bootstrap, rng):
    """Protein resampling; one common index sample shared by all models."""
    # pandas.factorize keeps protein IDs stable across models for this PTM.
    group_codes, unique_groups = pd.factorize(groups, sort=True)
    members = [np.flatnonzero(group_codes == g) for g in range(len(unique_groups))]
    out = {name: [] for name in scores}
    attempts = 0
    max_attempts = max(n_bootstrap * 30, 200)
    while len(next(iter(out.values()))) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        drawn = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[g] for g in drawn])
        if np.unique(y[idx]).size < 2:
            continue
        for name, probability in scores.items():
            out[name].append(float(average_precision_score(y[idx], probability[idx])))
    n_valid = len(next(iter(out.values())))
    if n_valid < n_bootstrap:
        print(f'WARNING: only {n_valid}/{n_bootstrap} valid protein resamples; '
              'CI suppressed for this class.')
        return {}, attempts
    return {name: np.array(values) for name, values in out.items()}, attempts


def evaluate(args):
    if args.n_bootstrap < 2 or args.n_calibration_bins < 2:
        raise ValueError('--n-bootstrap and --n-calibration-bins must be >= 2.')
    paths = named_paths(args.prediction)
    supported_config = load_json(args.supported_json)
    threshold_config = load_json(args.thresholds_json)
    if (set(supported_config) - set(paths)) or (set(threshold_config) - set(paths)):
        raise ValueError('Model names in JSON must match --prediction model names.')

    sequences = load_sequences(args.sequences)
    independence_checked = training_overlap_check(args.train_uids, sequences.keys())
    if args.sites_csv and not Path(args.curated).exists():
        print('INFO: curated annotations file absent; using explicit --sites-csv labels.')
        positives = set()
    else:
        positives = load_curated(args.curated, sequences)
    benchmark = build_benchmark(args, sequences, positives)

    model_tables, supported = {}, {}
    for name, path in paths.items():
        table = load_predictions(name, path, sequences)
        model_tables[name] = table
        classes = set(table['PTM_type'])
        if name in supported_config:
            allowed = set(supported_config[name])
            unknown = allowed - set(LABEL_TO_AA)
            if unknown:
                raise ValueError(f'{name}: unknown supported labels: {sorted(unknown)}')
            if classes - allowed:
                raise ValueError(f'{name}: scores contain classes not declared supported: '
                                 f'{sorted(classes - allowed)}')
            classes = allowed
        supported[name] = classes
        print(f'{name}: {len(table)} scored sites; classes={sorted(classes)}')

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary, draws, differences, reliability, audit, eval_sites, excluded_sites = [], [], [], [], [], [], []

    for label in LABEL_TO_AA:
        relevant_models = [name for name, classes in supported.items() if label in classes]
        if not relevant_models:
            continue
        sites = benchmark.loc[benchmark['PTM_type'] == label].copy()
        if sites.empty:
            print(f'WARNING: {label}: no eligible test sites; skipping.')
            continue
        # Merge every model onto the fixed eligible benchmark. Select sites
        # covered by ALL compared models, never only predicted-positive sites.
        eligible_n = len(sites)
        joined = sites.copy()
        score_cols = {}
        for model_index, name in enumerate(relevant_models):
            column = f'_model_score_{model_index}'
            score_cols[name] = column
            pred = model_tables[name].loc[
                model_tables[name]['PTM_type'] == label, KEY + ['pred_score']
            ].rename(columns={'pred_score': column})
            joined = joined.merge(pred, on=KEY, how='left', validate='one_to_one', sort=False)

        missing_masks = {name: joined[column].isna()
                         for name, column in score_cols.items()}
        common_mask = joined[list(score_cols.values())].notna().all(axis=1)
        missing_total = int((~common_mask).sum())
        excluded_positive = int(joined.loc[~common_mask, 'label'].sum())
        excluded_negative = missing_total - excluded_positive
        if missing_total:
            counts = ', '.join(f'{name}={int(mask.sum())}'
                               for name, mask in missing_masks.items())
            examples = joined.loc[~common_mask, KEY].head(5).to_dict('records')
            if not args.match_common_scored_sites:
                raise ValueError(
                    f'{label}: {missing_total}/{eligible_n} benchmark sites lack scores '
                    f'({counts}). Examples: {examples}. Export missing scores or '
                    'set MATCH_COMMON_SCORED_SITES=True in CONFIG.')
            print(f'WARNING: {label}: using {int(common_mask.sum())}/{eligible_n} '
                  f'shared scored sites; {missing_total} excluded '
                  f'({excluded_positive} annotated positives, '
                  f'{excluded_negative} presumed negatives). '
                  f'Missing score counts: {counts}.')
            print('WARNING: Missing-score exclusion may bias results if model '
                  'coverage depends on site position, protein, or difficulty. '
                  'Recover those scores for the strongest benchmark.')
            omitted = joined.loc[~common_mask, KEY + ['label']].copy()
            omitted['Missing From Models'] = [
                ';'.join(name for name, mask in missing_masks.items() if mask.iat[idx])
                for idx in np.flatnonzero(~common_mask.to_numpy())
            ]
            excluded_sites.append(omitted)

        if not common_mask.any():
            print(f'WARNING: {label}: models have no common scored test sites; skipping.')
            continue

        joined = joined.loc[common_mask].copy()
        sites = joined[KEY + ['label']].copy()
        n = len(sites)
        probabilities = {}
        for name, column in score_cols.items():
            score_array = joined[column].to_numpy(dtype=np.float64)
            if np.isin(np.unique(score_array), [0.0, 1.0]).all():
                raise ValueError(f'{name}, {label}: only binary 0/1 pred_score values found. '
                                 'Export original continuous probabilities; PR-AUC '
                                 'cannot be benchmarked properly from binary decisions.')
            probabilities[name] = score_array

        y = sites['label'].to_numpy(dtype=np.int8)
        groups = sites['uid'].to_numpy(dtype=str)
        npos, nneg = int(np.sum(y)), int(np.sum(y == 0))
        nprot = int(np.unique(groups).size)
        nposprot = int(np.unique(groups[y == 1]).size)
        low_support = npos < args.low_support_positive
        possible_ci = (npos >= args.min_positive_for_ci and nneg >= 2 and
                       nposprot >= args.min_positive_proteins_for_ci and nprot >= 2)
        valid_ap = npos > 0 and nneg > 0
        boot, attempts = {}, 0
        if valid_ap and possible_ci:
            boot, attempts = bootstrap_joint(y, groups, probabilities,
                                            args.n_bootstrap, rng)
        if not valid_ap:
            print(f'WARNING: {label} has {npos} positives and {nneg} negatives; '
                  'PR-AUC undefined, skipping bootstrap.')
        elif not possible_ci:
            print(f'WARNING: {label}: too few independent positives/proteins for '
                  'a defensible bootstrap CI. CI set to NaN.')

        audit.append({'PTM': label, 'Models': ';'.join(relevant_models),
                      'Eligible Benchmark Sites': eligible_n,
                      'Excluded Missing Score Sites': missing_total,
                      'Excluded Positive Sites': excluded_positive,
                      'Excluded Negative Sites': excluded_negative,
                      'Class Size': n, 'Positive Class Size': npos,
                      'Negative Class Size': nneg, 'Number of Proteins': nprot,
                      'Positive Proteins': nposprot,
                      'Low Support': low_support, 'Valid Bootstrap Draws':
                          len(next(iter(boot.values()))) if boot else 0})
        eval_sites.append(sites.assign(models_compared=';'.join(relevant_models)))

        for name in relevant_models:
            p = probabilities[name]
            cfg = threshold_config.get(name, {})
            if not isinstance(cfg, dict):
                raise ValueError(f'Thresholds for model {name} must be a JSON object.')
            threshold = float(cfg.get(label, cfg.get('default', 0.5)))
            if not 0 <= threshold <= 1:
                raise ValueError(f'{name}, {label}: threshold must be in [0,1].')
            source = ('supplied class threshold (provenance unverified)' if label in cfg else
                      'supplied model threshold (provenance unverified)' if 'default' in cfg else
                      'assumed 0.5 (not optimised)')
            binary = (p >= threshold).astype(np.int8)
            tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
            ap = float(average_precision_score(y, p)) if valid_ap else np.nan
            ece, cal_rows = calibration(y, p, args.n_calibration_bins)
            reliability.extend({'PTM': label, 'Model': name, **r} for r in cal_rows)
            scores = boot.get(name, np.array([], dtype=float))
            ci_low, ci_high = (np.percentile(scores, [2.5, 97.5]) if len(scores)
                               else (np.nan, np.nan))
            summary.append({
                'PTM': label, 'Model': name,
                'Eligible Benchmark Sites': eligible_n,
                'Excluded Missing Score Sites': missing_total,
                'Class Size': n, 'Positive Class Size': npos,
                'Negative Class Size': nneg, 'Number of Proteins': nprot,
                'Positive Proteins': nposprot, 'Positive Prevalence': npos / n,
                'Threshold': threshold, 'Threshold Source': source,
                'True Positive': int(tp), 'False Positive': int(fp),
                'True Negative': int(tn), 'False Negative': int(fn),
                'Precision': float(precision_score(y, binary, zero_division=0)),
                'Recall': float(recall_score(y, binary, zero_division=0)),
                'F1': float(f1_score(y, binary, zero_division=0)),
                'Accuracy': float(accuracy_score(y, binary)),
                'PR-AUC (Average Precision)': ap,
                'PR-AUC Bootstrap Mean': float(np.mean(scores)) if len(scores) else np.nan,
                'PR-AUC Bootstrap Std': float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan,
                'PR-AUC CI95 Lower': float(ci_low),
                'PR-AUC CI95 Upper': float(ci_high),
                'Bootstrap Repetitions': int(len(scores)),
                'Low Support': bool(low_support),
                'Brier Score': float(brier_score_loss(y, p)),
                'ECE': ece,
                'Independence UID Check Supplied': independence_checked,
            })
            for i, ap_sample in enumerate(scores):
                draws.append({'PTM': label, 'Model': name,
                              'Bootstrap Sample': i, 'PR-AUC': float(ap_sample)})
            print(f'{label:20s} {name:18s} N+={npos:<6d} N-={nneg:<7d} '
                  f'AP={ap:.4f} CI=[{ci_low:.4f},{ci_high:.4f}] '
                  f'{"LOW SUPPORT" if low_support else ""}')

        if len(relevant_models) >= 2 and boot:
            for ia, model_a in enumerate(relevant_models):
                for model_b in relevant_models[ia + 1:]:
                    delta = boot[model_a] - boot[model_b]
                    low, high = np.percentile(delta, [2.5, 97.5])
                    differences.append({
                        'PTM': label, 'Model A': model_a, 'Model B': model_b,
                        'AP Difference A-B': float(average_precision_score(y, probabilities[model_a]) -
                                                   average_precision_score(y, probabilities[model_b])),
                        'Bootstrap Std': float(np.std(delta, ddof=1)),
                        'Difference CI95 Lower': float(low),
                        'Difference CI95 Upper': float(high),
                        'Shared Bootstrap Samples': len(delta),
                    })

    if not summary:
        raise RuntimeError('No supported classes could be evaluated.')
    pd.DataFrame(summary).to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(audit).to_csv(output / 'class_sizes.csv', index=False)
    omitted_cols = KEY + ['label', 'Missing From Models']
    omitted_frame = (pd.concat(excluded_sites, ignore_index=True)
                     if excluded_sites else pd.DataFrame(columns=omitted_cols))
    omitted_frame.to_csv(output / 'excluded_missing_score_sites.csv', index=False)
    pd.DataFrame(draws, columns=['PTM', 'Model', 'Bootstrap Sample', 'PR-AUC']).to_csv(
        output / 'bootstrap_pr_auc_distribution.csv', index=False)
    pd.DataFrame(differences, columns=['PTM', 'Model A', 'Model B', 'AP Difference A-B',
                                      'Bootstrap Std', 'Difference CI95 Lower',
                                      'Difference CI95 Upper', 'Shared Bootstrap Samples']).to_csv(
        output / 'paired_pr_auc_differences.csv', index=False)
    pd.DataFrame(reliability).to_csv(output / 'calibration_bins.csv', index=False)
    pd.concat(eval_sites, ignore_index=True).to_csv(output / 'benchmark_sites.csv', index=False)
    metadata = {
        'pr_auc_definition': 'sklearn.metrics.average_precision_score; non-interpolated average precision',
        'confidence_interval': '2.5/97.5 percentile of paired protein-cluster bootstrap',
        'bootstrap_seed': args.seed, 'bootstrap_requested': args.n_bootstrap,
        'class_size_definition': 'number of distinct tested protein/PTM/residue examples',
        'negative_label_definition': ('explicit labels from --sites-csv' if args.sites_csv else
                                      'eligible amino acids absent from curated positives: presumed negatives'),
        'independence_exact_uid_checked': independence_checked,
        'homology_independence_verified': False,
        'comparison_policy': ('intersection of sites with scores for every class-participating model'
                              if args.match_common_scored_sites else 'full site coverage required'),
        'coverage_caveat': ('Excluded sites are audited. Missingness can bias comparisons; '
                            'try to recover all scores if possible.'),
        'calibration_caveat': 'Brier/ECE on this evaluated site distribution; not evidence of improved calibration alone',
        'model_score_files': paths,
        'model_supported_classes': {k: sorted(v) for k, v in supported.items()},
        'thresholds_json': args.thresholds_json,
        'threshold_provenance_caveat': 'Supplied thresholds must have been fixed on independent validation data; this cannot be proved from their values.',
        'supported_classes_provenance': ('explicit supported-classes configuration' if args.supported_json
                                         else 'inferred from PTM_type values in score CSVs'),
    }
    (output / 'benchmark_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    try:
        evaluate(parse_args())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.exit(f'ERROR: {exc}')
