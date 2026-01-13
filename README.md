# AB-DataScienceLab – NR-PTMkb

This repository supports **NR-PTMkb (Nuclear Receptor Post-Translational Modification knowledgebase)**, a data-driven framework developed to curate, analyze, and predict post-translational modification (PTM) sites in nuclear receptors.

---

## 🧬 Project Background (NR-PTMkb)

Post-translational modifications (PTMs) play a central role in the regulation of nuclear receptors (NRs) and are crucial components of diverse biological process networks. Atypical PTMs or mutations at specific sites in nuclear receptors can contribute to a wide range of neurodegenerative, reproductive, and metabolic diseases. Therefore, in-depth investigation is required to understand how PTMs regulate nuclear receptor function and their interactions with coregulators.

**NR-PTMkb** is an open, comprehensive, and manually curated knowledgebase developed for the systematic study of post-translational modifications in nuclear receptors and their dynamic coregulator interactions. The current version of NR-PTMkb comprises **13 different types of PTMs**, with curated data covering **2,976 PTM sites** across **human, mouse, and rat nuclear receptors**. Additionally, information on **88 modifying enzymes**, their target domains in nuclear receptors, and associated signaling contexts has been manually compiled.

NR-PTMkb provides multiple interactive modules, including **NR Roots**, **NR Tabs**, **NR Dock**, and **PTM Profiler**, enabling users to explore nuclear receptor classification, detailed annotations, structural and functional impacts of PTMs, and prediction of potential PTM sites. The database also offers user-friendly search tools such as **simple search** and an **advanced hypothesis generator** to facilitate efficient exploration of curated and value-added data.

---

## 📁 Repository Structure

```
AB-DataScienceLab/
├── fig_manuscript/            # Figures generated for manuscript preparation
├── results_musitedeep/        # Results from MusiteDeep-based PTM prediction
├── results_new/               # Results from NR-PTMpred: trained specifically on NR-PTM models
├── results_original/          # Results from baseline models: MIND-S
├── roc_new/                   # ROC curves for NR-PTMpred
├── roc_original/              # ROC curves for MIND-S
├── src/                       # Source code directory (modules and utilities)
├── NR_InputData.csv           # Curated NR input features
├── NR_Sequences.csv           # Nuclear receptor protein sequences
├── batch_predict.py           # Batch PTM site prediction script
├── common_ptms.py             # Common PTMs
├── evaluate_result.py         # Model evaluation metrics and analysis
├── get_roc_testdata.py        # ROC data preparation script
├── plot_confusionmatrix.py    # Confusion matrix plotting utility
├── prepare_training_data.py   # Training data preparation pipeline
├── train_model.py             # Model training script
├── train_data_new.json        # Training dataset (new labels based on curated Nuclear Receptor data)
├── test_data_new.json         # Test dataset ((new labels based on curated Nuclear Receptor data)
├── test_data_new_oldlabel.json# Test dataset with original labels
```

---

## 📖 Citation

If you use this repository or NR-PTMkb in your research, please cite the corresponding NR-PTMkb publication (details to be updated).
