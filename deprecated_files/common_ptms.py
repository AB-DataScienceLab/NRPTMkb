import pandas as pd


t1 = pd.read_csv('NRCoRP_TrainingSet.csv')
t2 = pd.read_csv('results_original/correct_predictions_newlabels.csv')
t3 = pd.read_csv('results_new/correct_predictions.csv')
t4 = pd.read_csv('results_musitedeep/correct_predictions.csv')

t1.rename(columns={'UniprotID': 'uid', 'Site': 'site', 'PTM_Type': 'PTM_type'}, inplace=True)
t1['site'] = t1['site'].apply(lambda x: int(x[1:]))

for t in [t2, t3, t4]:
    t['PTM_type'] = t['PTM_type'].apply(lambda x: x.split('_')[0])
    t.drop(columns=['pred_score'], inplace=True)

for t, src in zip([t1, t2, t3, t4], ['curated', 'mind_original', 'mind_new', 'musitedeep']):
    t['src'] = src

D = pd.concat([t1, t2, t3, t4])

DD = D.groupby(by=['uid', 'site', 'PTM_type'])['src'].apply(lambda x: ';'.join(x)).reset_index()

for src in  ['curated', 'mind_original', 'mind_new', 'musitedeep']:
    DD[src] = 'NO'
    MASK = DD['src'].apply(lambda x: src in x.split(';'))
    DD.loc[MASK, src] = 'YES'

DD.sort_values(by=['uid', 'site', 'PTM_type'], inplace=True)
DD.to_csv('common_PTMS.csv', index=False)
