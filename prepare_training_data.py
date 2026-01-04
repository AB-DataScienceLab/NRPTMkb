# coding: utf-8
import pandas as pd
import json
import random
import numpy as np

T1 = pd.read_csv("210Merged.csv")
T2 = pd.read_csv("NRCoRP_TrainingSet.csv")
T2['PTM_Label'] = T2['PTM_Type'] + '_' + T2['Site'].str[0]

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

UL = {}
TRAIN, TEST = {}, {}
for uid in T2.UniprotID.unique():
    tt = T2[T2['UniprotID'] == uid]
    seq = T1[T1['UniprotID'] == uid].iloc[0]['Sequence']
    group = tt.groupby('PTM_Label')
    TRAIN[uid] = {'seq': seq, 'label': []}
    TEST[uid] = {'seq': seq, 'label': []}
    for g in group.groups:
        dd = group.get_group(g)
        if dd.shape[0] >= 10:
            train = dd.sample(n=int(np.floor(0.9 * dd.shape[0])), random_state=1, replace=False)
            test = dd.drop(train.index)
        else:
            train = dd
            test = dd.sample(n=1, random_state=1, replace=False)
        for dset, tget in zip([train, test], ['TRAIN', 'TEST']):
            for _, t in dset.iterrows():
                site = int(t['Site'][1:]) - 1
                ch = seq[site]
                print(uid, t['Site'], t['PTM_Type'])
                assert ch == t['Site'][0]
                
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
    
