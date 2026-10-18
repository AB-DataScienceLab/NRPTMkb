import pandas as pd

df = pd.read_csv("correct_predictions.csv")
seq = pd.read_csv("../data/curated/NR_Sequences_New.csv")

seq_map = dict(zip(seq["UniprotID"], seq["Sequence"]))

old_to_new = {
    "Hydro_K": None,
    "Hydro_P": None,
    "Methy_K": "Methylation_K",
    "Methy_R": "Methylation_R",
    "N6-ace_K": "Acetylation_K",
    "Palm_C": "Palmitoylation_C",
    "Phos_Y": "Phosphorylation_Y",
    "Pyro_Q": None,
    "SUMO_K": "Sumoylation_K",
    "Ubi_K": "Ubiquitination_K",
    "glyco_N": "Glycosylation_N",
}

def rename_ptm(row):
    ptm = row["PTM_type"]

    if ptm in ("Phos_ST", "glyco_ST"):
        aa = seq_map[row["uid"]][int(row["site"]) - 1]
        assert aa in ("S", "T")
        prefix = "Phosphorylation" if ptm == "Phos_ST" else "Glycosylation"
        return f"{prefix}_{aa}"

    return old_to_new[ptm]

df["PTM_type"] = df.apply(rename_ptm, axis=1)

# Remove classes unsupported by the new naming scheme
df = df.dropna(subset=["PTM_type"])

df.to_csv("correct_predictions_newlabels.csv", index=False)
