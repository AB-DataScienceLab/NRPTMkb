import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

output_dir = "results_musitedeep/cm"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("results_musitedeep/metrics.csv")

# Create and save a confusion matrix heatmap for each PTM
for _, row in df.iterrows():
    ptm = row['PTM']
    cm = np.array([
        [row['True Negative'], row['False Positive']],
        [row['False Negative'], row['True Positive']]
    ])
    labels = ['Negative', 'Positive']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels, linewidths=0.2, linecolor='black', annot_kws={"fontsize":14})
    plt.title(ptm)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"{ptm}.png")
    plt.savefig(filename)
    plt.close()

