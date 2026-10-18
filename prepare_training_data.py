# coding: utf-8
import pandas as pd
import json
import random
import numpy as np

T1 = pd.read_csv("data/curated/NR_Sequences_New.csv")
T2 = pd.read_csv("data/curated/NR_InputData_New.csv")
T2['PTM_Label'] = T2['PTM_Type'] + '_' + T2['Site'].str[0]

print(T2['PTM_Label'].value_counts())

map2old = {
    'Methylation_K': 'Methy_K',
    'Methylation_R': 'Methy_R',
    'Palmitoylation_C': 'Palm_C',
    'Phosphorylation_S': 'Phos_ST',
    'Phosphorylation_T': 'Phos_ST',
    'Phosphorylation_Y': 'Phos_Y',
    'Sumoylation_K': 'SUMO_K',
    'Ubiquitination_K': 'Ubi_K',
    'Glycosylation_T': 'glyco_ST',
    'Glycosylation_S': 'glyco_ST',
    'Acetylation_K': 'N6-ace_K', 
}

TRAIN, TEST = {}, {}
for uid in T2.UniprotID.unique():
    seq = T1[T1['UniprotID'] == uid].iloc[0]['Sequence']
    TRAIN[uid] = {'seq': seq, 'label': []}
    TEST[uid] = {'seq': seq, 'label': []}

for ptm_label in T2.PTM_Label.unique():
    ptm_data = T2[T2['PTM_Label'] == ptm_label]
    if ptm_data.shape[0] < 20:
        continue
    
    train = ptm_data.sample(n=int(np.floor(0.9 * ptm_data.shape[0])), random_state=1, replace=False)
    test = ptm_data.drop(train.index)

    for dset, tget in zip([train, test], ['TRAIN', 'TEST']):
        for _, t in dset.iterrows():
            uid = t['UniprotID']
            seq = T1[T1['UniprotID'] == uid].iloc[0]['Sequence']
            site = int(t['Site'][1:]) - 1
            if site < 0 or site >= len(seq):
                print(uid, t['Site'], t['PTM_Type'], 'Sequence too short')
                continue
            ch = seq[site]
            if ch != t['Site'][0]:
                print(uid, t['Site'], t['PTM_Type'], ch, 'Site not matching')
                continue
            
            label = t['PTM_Label']
            if tget == 'TRAIN':
                TRAIN[uid]['label'].append({'site': site, 'ptm_type': label})
            else:
                # for setting old labels
                # if label in map2old:
                #     label = map2old[label]
                #     TEST[uid]['label'].append({'site': site, 'ptm_type': label})

                # for new labels
                TEST[uid]['label'].append({'site': site, 'ptm_type': label})


with open('train_data_new.json', 'w') as f:
    print(len(TRAIN))
    json.dump(TRAIN, f)
    f.close()

with open('test_data_new.json', 'w') as f:
    print(len(TEST))
    json.dump(TEST, f)
    f.close()
    
