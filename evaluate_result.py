import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm
from pprint import pprint
import pdb

import json
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from os.path import exists

from src import utils
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token


label2aa = {
    'Acetylation_K': 'K', 'Acetylation_M': 'M', 'Glycosylation_S': 'S', 
    'Glycosylation_T': 'T', 'Methylation_G': 'G', 'Methylation_K': 'K', 'Methylation_R': 'R', 
    'Monomethylation_K': 'K', 'O-glcNAcylation_S': 'S', 'Palmitoylation_C': 'C', 
    'Phosphorylation_C': 'C', 'Phosphorylation_E': 'E', 'Phosphorylation_K': 'K', 
    'Phosphorylation_R': 'R', 'Phosphorylation_S': 'S', 'Phosphorylation_T': 'T', 
    'Phosphorylation_Y': 'Y', 'Sumoylation_K': 'K', 'Thiol oxidation_C': 'C', 'Ubiquitination_K': 'K',
    'Lipidation_C': 'C', 'Glycosylation_N': 'N'
}

labels = list(label2aa.keys())
# get unique labels
unique_labels = sorted(set(labels))
label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
random.seed(0)
chunk_size = 512
half_chunk_size = chunk_size//2


def cut_protein(sequence, chunk_size):
    # cut the protein if it is longer than chunk_size
    # only includes labels within middle chunk_size//2
    # during training, if no pos label exists, ignore the chunk
    # during eval, retain all chunks for multilabel; retain all chunks of protein have specific PTM for binary
    
    assert chunk_size%4 == 0
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    records = []
    if len(sequence) > chunk_size:
        for i in range((len(sequence)-1)//half_chunk_size):
            # the number of half chunks=(len(sequence)-1)//chunk_size+1,
            # minus one because the first chunks contains two halfchunks
            max_seq_ind = (i+2)*half_chunk_size
            if i==0:
                cover_range = (0,quar_chunk_size*3)
            elif i==((len(sequence)-1)//half_chunk_size):
                cover_range = (quar_chunk_size, len(sequence)-i*half_chunk_size)
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
            seq = sequence[i*half_chunk_size: max_seq_ind]
            # idx = [j for j in range(len((seq))) if (seq[j] in aa and j >= cover_range[0] and j < cover_range[1])]
            records.append({
                'chunk_id': i,
                'seq': seq,
                'cover_range': cover_range
                # 'idx': idx
            })
    else:
        records.append({
            'chunk_id': None,
            'seq': sequence,
            'cover_range': (0, len(sequence))
            # 'idx': [j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records


def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


def data():
    SEQ = pd.read_csv('210Merged.csv')
    CURATED = pd.read_csv('NRCoRP_TrainingSet.csv')
    PRED = pd.read_csv('results_musitedeep/correct_predictions.csv')

    RECs = []
    for _, r in SEQ.iterrows():
        uid = r['UniprotID']
        sequence = r['Sequence']

        curated_ptms = []
        for _, c in CURATED[CURATED['UniprotID'] == uid].iterrows():
            curated_ptms.append({
                'site': int(c['Site'][1:]) - 1,
                'type': c['PTM_Type']
            })
        pred_ptms = []
        for _, p in PRED[PRED['uid'] == uid].iterrows():
            pred_ptms.append({
                'site': p['site'] - 1,
                'type': p['PTM_type'].split('_')[0]
            })
        
        records = cut_protein(sequence, chunk_size=chunk_size)
        RECs.append((records, curated_ptms, pred_ptms))

    return RECs


def main(RECs):

    seqs = []
    chunk_ids = []

    X, Y, MASK, Y_PRED = [], [], [], []
    CFs = {label: [] for label in unique_labels}
    for records, curated_ptms, pred_ptms in RECs:
        for record in records:
            seq = record['seq']
            chunk_id = record['chunk_id']
            x = tokenize_seq(seq)

            seqs.append(seq)
            chunk_ids.append(chunk_id)
            X.append(x)

            # --------------------------------------------
            sub_labels_curated = [
                lbl for lbl in curated_ptms 
                if (chunk_id is None) or 
                    ((lbl['site'] >= record['cover_range'][0] + chunk_id * half_chunk_size) and 
                    (lbl['site'] < record['cover_range'][1] + chunk_id * half_chunk_size))
            ]
            sub_labels_pred = [
                lbl for lbl in pred_ptms
                if (chunk_id is None) or 
                    ((lbl['site'] >= record['cover_range'][0] + chunk_id * half_chunk_size) and 
                    (lbl['site'] < record['cover_range'][1] + chunk_id * half_chunk_size))
            ]
            y = np.zeros((len(x), len(unique_labels)))
            y_pred = np.zeros((len(x), len(unique_labels)))
            m = np.zeros((len(x), len(unique_labels)))
            unique_sublabels = {}
            for ptm in sub_labels_curated:
                if chunk_id is not None:
                    site_num = int(ptm['site']) - chunk_id * half_chunk_size + 1 # minus i * half chunk size
                else:
                    site_num = int(ptm['site']) + 1
                pt = ptm['type'] + '_' + seq[site_num - 1]
                assert x[site_num] == aa_to_token_index[label2aa[pt]]
                y[site_num, label_to_index[pt]] = 1
                m[site_num, label_to_index[pt]] = 1
                if pt in unique_sublabels:
                    unique_sublabels[pt].add(site_num)
                else:
                    unique_sublabels[pt] = {site_num}
            
            for ptm in sub_labels_pred:
                if chunk_id is not None:
                    site_num = int(ptm['site']) - chunk_id * half_chunk_size + 1
                else:
                    site_num = int(ptm['site']) + 1
                pt = ptm['type'] + '_' + seq[site_num - 1]
                assert x[site_num] == aa_to_token_index[label2aa[pt]]
                y_pred[site_num, label_to_index[pt]] = 1
            
            for pt, occ in unique_sublabels.items():
                neg_idx = [i + 1 for i, aa in enumerate(seq) if (i >= record['cover_range'][0]) and (i < record['cover_range'][1]) and (aa == label2aa[pt])]
                assert len(occ) == len(set(neg_idx).intersection(occ))
                neg_idx = list(set(neg_idx) - occ)
                if len(neg_idx) > len(occ):
                    neg_idx = random.sample(neg_idx, len(occ))
                m[neg_idx, label_to_index[pt]] = 1
            Y.append(y)
            MASK.append(m)
            Y_PRED.append(y_pred)
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    Y = np.vstack(Y)
    Y_PRED = np.vstack(Y_PRED)
    MASK = np.vstack(MASK)
    for label in unique_labels:
        msk = MASK[:, label_to_index[label]] == 1
        cf = confusion_matrix(Y[msk, label_to_index[label]], Y_PRED[msk, label_to_index[label]])
        CFs[label].append(cf)
    # print(Y.shape, Y_PRED.shape, MASK.shape)
    return CFs
            
    

if __name__ == '__main__':
    LL = {label: [] for label in unique_labels}
    D = data()
    CF = {}
    for k in range(1000):
        print(k)
        for lb, cf in main(D).items():
            LL[lb] += cf
    for lb, cf in LL.items():
        med_cf = np.median(cf, axis=0)
        if len(med_cf) == 0:
            continue
        print('---------------------------------')
        print(lb)
        print(med_cf)
        CF[lb] = med_cf
        print('---------------------------------')

    L = []
    for lbl, cf in CF.items():
        tn = cf[0][0]
        fp = cf[0][1]
        fn = cf[1][0]
        tp = cf[1][1]

        L.append({
            'PTM': lbl,
            'True Positive': tp,
            'False Positive': fp,
            'True Negative': tn,
            'False Negative': fn,
            'Precision': tp / (tp + fp) if tp + fp != 0 else np.nan,
            'Recall': tp / (tp + fn) if tp + fn != 0 else np.nan,
            'Accuracy': (tp + tn) / (tp + fp + tn + fn) if tp + fp + tn + fn != 0 else np.nan
        })
    pd.DataFrame(L).to_csv('results_musitedeep/metrics.csv', index=False)
