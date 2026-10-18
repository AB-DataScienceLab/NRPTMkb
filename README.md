# AB-DataScienceLab – NR-PTMkb

This repository supports **NR-PTMkb (Nuclear Receptor Post-Translational Modification knowledgebase)**, a data-driven framework developed to curate, analyze, and predict post-translational modification (PTM) sites in nuclear receptors.

---

## 🧬 Project Background (NR-PTMkb)

Post-translational modifications (PTMs) play a central role in the regulation of nuclear receptors (NRs) and are crucial components of diverse biological process networks. Atypical PTMs or mutations at specific sites in nuclear receptors can contribute to a wide range of neurodegenerative, reproductive, and metabolic diseases. Therefore, in-depth investigation is required to understand how PTMs regulate nuclear receptor function and their interactions with coregulators.

**NR-PTMkb** is an open, comprehensive, and manually curated knowledgebase developed for the systematic study of post-translational modifications in nuclear receptors and their dynamic coregulator interactions. The current version of NR-PTMkb comprises **13 different types of PTMs**, with curated data covering **2,782 PTM sites** across **human, mouse, and rat nuclear receptors**. Additionally, information on **88 modifying enzymes**, their target domains in nuclear receptors, and associated signaling contexts has been manually compiled.

NR-PTMkb provides multiple interactive modules, including **NR Roots**, **NR Tabs**, **NR Dock**, and **PTM Profiler**, enabling users to explore nuclear receptor classification, detailed annotations, structural and functional impacts of PTMs, and prediction of potential PTM sites. The database also offers user-friendly search tools such as **simple search** and an **advanced hypothesis generator** to facilitate efficient exploration of curated and value-added data.

---

**NR-PTMpred** is the NR-specific deep-learning PTM predictor developed using a curated subset of NR-PTMkb. It targets seven modification classes: `Acetylation_K`, `Methylation_K`, `Phosphorylation_S`, `Phosphorylation_T`, `Phosphorylation_Y`, `Sumoylation_K`, and `Ubiquitination_K`. The seven-class predictor dataset is distinct from the broader knowledgebase.

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── raw/                         # Original PTM source tables
│   │   ├── Human.csv
│   │   ├── Human_filtered.xlsx
│   │   ├── Human_tr.csv
│   │   └── LastRitesNRPTMdb.csv
│   └── curated/
│       ├── NR_InputData_New.csv      # Curated PTM annotations
│       ├── NR_Sequences_New.csv      # Nuclear receptor sequences
│       ├── protein_sequences_new.fasta
│       └── drop_PTM_sites.txt        # Excluded PTM sites
├── train_data_new.json              # Training annotation data
├── test_data_new.json               # Held-out annotation data
├── prepare_training_data.py         # Training-data preparation
├── train_model.py                   # Model training
├── batch_predict.py                 # Batch prediction
├── src/                             # Model architecture and supporting modules
├── model_desc.md                    # Model architecture description
├── new_model/                       # Newly trained NR-PTMpred ensemble/checkpoints
├── saved_model/                     # Original pretrained MIND-S model checkpoints
├── results_new/                     # NR-PTMpred site-level predictions
├── results_original/                # Original MIND-S predictions and label mapping
├── results_musitedeep/              # MusiteDeep site-level predictions
├── evaluate_test_data.py            # Held-out annotation evaluation
├── evaluate_result.py               # Three-model benchmark and uncertainty analyses
├── prepare_benchmark_comparison.py  # Full-/balanced-negative Excel comparisons
├── plot_ptm_performance.py          # Confusion matrices, ROC and PR curves
├── test_cut_protein_coverage.py     # Sequence-window coverage tests
├── test_only_results/              # Held-out evaluation results
├── benchmark_results/              # Benchmark metrics, scored sites and comparisons
├── plots/                           # PTM-specific performance figures (PNG)
├── fig_manuscript/                  # Manuscript figures
├── environment.yml                  # Conda environment specification
├── requirements-lock.txt           # Python dependency versions
├── deprecated_files/               # Historical data and scripts
└── README.md
```

### 🧠 Model Directories

The distinction between the model directories is important:

- **`new_model/`** holds the **newly trained NR-PTMpred models** developed using the curated nuclear receptor PTM dataset. The 15 ensemble checkpoints are stored as `MIND_fold_0` through `MIND_fold_14`, with ensemble metadata in `MIND_PRAU.json`. The `MIND` filename prefix is inherited from the underlying model implementation; these checkpoints represent the **NR-specific trained model**, not the original MIND-S baseline.
- **`saved_model/`** holds **original pretrained MIND-S models**, along with other original pretrained checkpoint families retained in the repository. These should not be confused with the newly trained NR-PTMpred ensemble.

The corresponding prediction outputs are kept separately in `results_new/` (NR-PTMpred), `results_original/` (MIND-S), and `results_musitedeep/` (MusiteDeep).

## 🗂️ Data and Annotation Provenance

The source tables are retained in `data/raw/`. The curated PTM annotations and sequences used by the predictor are in `data/curated/NR_InputData_New.csv` and `data/curated/NR_Sequences_New.csv`. The FASTA representation is `data/curated/protein_sequences_new.fasta`. The files `data/curated/drop_PTM_sites.txt` and `results_original/correct_ptm_labels.py` retain exclusion and label-mapping information, respectively. `prepare_training_data.py` processes the curated data for model training.

For evaluation, a **positive site** is an eligible amino-acid position with a matching curated PTM annotation. An eligible position without a matching annotation is treated as a **presumed negative**; absence of an annotation does not establish that a site is biologically unmodified. Residue positions in the prediction and evaluation CSVs are one-based.

### 🔢 Curated Predictor Dataset at a Glance

The updated NR-PTMpred annotation dataset contains **1,550 unique PTM sites** reported for human nuclear receptors in NR-PTMkb, 52 could not be matched to the canonical protein sequences due to the lack of isoform-level information, resulting in a final dataset of **1,498 PTM annotation records for NR-PTMpred** (UniProt identifier, residue position, and modification type), representing **87 protein-sequence identifiers** from **48 parent UniProt accessions**. The seven PTM classes with at least 20 annotated sites account for **1,448 records**; the remaining are excluded from the seven-class prediction task.

Among the retained annotations, **serine phosphorylation** accounts for **611** sites, **threonine phosphorylation** for **218**, **lysine methylation** for **205**, **lysine SUMOylation** for **156**, **tyrosine phosphorylation** for **96**, **lysine acetylation** for **91**, and **lysine ubiquitination** for **71**. These counts describe curated PTM annotations, rather than experimentally verified negative sites or a protein-independent test set.

## 🧪 Training and Data Splits

The seven retained PTM classes comprise **1,448 curated annotations** across **87 protein sequence identifiers** and **48 parent UniProt accessions**. These annotations were partitioned into **1,301 training annotations (89.85%)** and **147 held-out test annotations (10.15%)**. The partitions contain **no overlapping protein–residue–PTM annotation triples**.

`train_data_new.json` and `test_data_new.json` contain the training and held-out PTM annotations. The held-out test concerns annotations, **not entirely unseen nuclear receptor proteins**; receptors can occur in both training and test data. This distinction matters when interpreting the held-out evaluation and comparisons with other predictors.

`train_model.py` contains the model-training workflow, `batch_predict.py` performs batch inference, and `src/` contains model architecture, transformer, tokenization, and utility modules. NR-PTMpred is represented by the newly trained ensemble in `new_model/`.


## 🏋️ Training Methodology and MIND-S Transfer Learning

NR-PTMpred uses the **LSTM–Transformer architecture** implemented in `src/model.py` and `src/transformer.py`. Amino-acid sequences are tokenized with start/end tokens and processed in overlapping windows of up to **512 residues** (**514 tokens** including sequence delimiters). The model includes a **128-dimensional sequence embedding**, a **bidirectional LSTM** (64 units per direction), **three Transformer encoder layers** with **eight attention heads** and a **512-unit feed-forward subnetwork**, followed by a **64-unit dense layer** and **seven sigmoid prediction outputs**, one per retained PTM class.

Rather than training the sequence representation from scratch, `train_model.py` initializes each NR-PTMpred fold from the corresponding **original pretrained MIND-S checkpoint** in `saved_model/MIND_fifteenfold_fold_0` through `MIND_fifteenfold_fold_14`. It copies the pretrained weights of the embedding, LSTM, and Transformer encoder layers and **marks those transferred layers as non-trainable**. The original intermediate dense layer and final prediction layer are **not transferred**; the NR-specific 64-unit intermediate layer and seven-output prediction layer form the intended trainable head. The newly trained models are stored separately in `new_model/MIND_fold_0` through `MIND_fold_14`.

The **1,301 training annotations** are organized into **15 training/validation folds**, with the fold assignment designed to distribute less-represented PTM classes. Within each training window, eligible unannotated residues are randomly sampled as presumed negatives, generally targeting a **1:1 negative-to-positive ratio per PTM class** when sufficient candidates exist. The implementation uses **batch size 64**, **Adam with AMSGrad** at a learning rate of **0.001**, a maximum of **200 epochs**, and early stopping on training loss with **patience 2**; the random seed is **252**. The final ensemble combines predictions from the 15 folds using **PTM-specific weights normalized from their validation average-precision scores**. The validation folds are based on sequence windows and are **not protein-independent**.

## 🧪 Evaluation Methodology

The **held-out annotation evaluation** (`evaluate_test_data.py`) measures NR-PTMpred on positive PTM sites reserved in `test_data_new.json`, excluding known training-positive sites. Other eligible, unannotated residues are treated as presumed negatives, and positions without prediction scores are omitted and recorded. Metrics based on binary decisions use a probability threshold of **0.5**, while ROC-AUC and PR-AUC (average precision) use the continuous scores. **95% confidence intervals** are estimated with **1,000 parent-accession cluster-bootstrap resamples**: in each resample, parent accessions are drawn with replacement and their scored sites are retained together, without balancing the positive and negative counts.

The **three-model comparison** evaluates NR-PTMpred, MIND-S, and MusiteDeep on the same eligible PTM–residue positions for which all three models provide scores. It reports both a **full-negative evaluation** (all matched scored sites) and a **balanced-negative evaluation** (all matched positives plus an equal number of presumed negatives drawn with replacement over **1,000 repetitions**). Balanced-negative results are averages across repetitions and represent an artificial 1:1 class distribution. Because the benchmark includes annotations used during training and the held-out test shares receptors with training, neither analysis establishes performance on entirely unseen nuclear receptors.

## 🛠️ Software Environment

The repository includes `environment.yml` and `requirements-lock.txt` documenting the computational dependencies required for model training, inference, and evaluation.

## ⚖️ Benchmark Comparison: NR-PTMpred, MIND-S and MusiteDeep

The benchmarking analysis compares **NR-PTMpred**, the original **MIND-S** model, and **MusiteDeep** across the **same seven PTM–residue classes**. Positive labels are constructed from `data/curated/NR_InputData_New.csv`, and amino-acid-matched candidate positions are derived from `data/curated/NR_Sequences_New.csv`. Predictions are matched by **sequence identifier, one-based residue position, and PTM class**. For a fair comparison, each PTM-specific analysis includes only positions for which **all three models** have valid prediction scores. The common scored-site evaluation contains **16,557 PTM–residue examples** (**1,353 annotated positives** and **15,204 presumed negatives**); the 95 retained-class positive annotations without shared score coverage are not included in this comparison.

Across the seven PTM classes, the **unweighted mean AUC values** from the matched-site benchmark are summarized below. These are macro-averages of PTM-specific scores, not pooled scores across residues.

| Model | Full negatives: PR-AUC | Full negatives: ROC-AUC | Balanced negatives: PR-AUC | Balanced negatives: ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| **NR-PTMpred** | **0.580** | **0.942** | **0.931** | **0.941** |
| MIND-S | 0.192 | 0.651 | 0.677 | 0.652 |
| MusiteDeep | 0.196 | 0.654 | 0.683 | 0.655 |

Both class-distribution protocols are reported in `benchmark_results/NR_PTMpred_three_model_comparison.xlsx`. The **Full negatives** sheet uses all shared scored positions. The **Balanced negatives** sheet retains all shared scored positive sites and samples an equal number of presumed-negative sites **with replacement**, repeating this procedure **1,000 times with seed 252**; the same sites are used for all three models in each iteration, and the reported metrics are the means across iterations. Both sheets report **accuracy, precision, recall, PR-AUC (average precision), and ROC-AUC** for each model and PTM class. Accuracy, precision, and recall use a **0.5 prediction threshold**; PR-AUC and ROC-AUC use continuous prediction probabilities.

The benchmark scores are available in `results_new/correct_predictions.csv` (NR-PTMpred), `results_original/correct_predictions_newlabels.csv` (MIND-S), and `results_musitedeep/correct_predictions.csv` (MusiteDeep). `prepare_benchmark_comparison.py` generates both comparison sheets **directly from the curated annotations, sequences, and prediction files**, without relying on previously saved metric tables. `evaluate_result.py` and `benchmark_results/` also provide site-level matched evaluation records, class sizes, bootstrap results, and comparative statistics.

The balanced and full-negative PR-AUC and precision values represent **different positive prevalences** and should be interpreted separately. The benchmark includes PTM annotations represented in NR-PTMpred training, so it is a **matched-dataset model comparison**, not an independent assessment on previously unseen receptors.

## 📈 Figures and Evaluation Source Data

`plot_ptm_performance.py` generates **PNG figures** directly from the curated sequences, annotations, and per-site prediction scores. Under `plots/`, each of the seven PTM-specific folders contains `confusion_matrix.png`, `roc_curve.png`, and `pr_curve.png`, comparing all three models on the **full-negative matched-site dataset**. The confusion matrices apply the 0.5 classification threshold, while ROC and precision–recall curves use continuous probabilities.

The held-out annotation evaluation and its evaluated-site records are available under `test_only_results/` and are produced by `evaluate_test_data.py`. Model-comparison outputs and figure source data are retained in `benchmark_results/`; `test_cut_protein_coverage.py` checks the residue coverage of overlapping sequence windows.

## 🔍 Reproducibility and Source Data

Curated annotations, original source tables, training/test annotation files, model source code, the NR-PTMpred checkpoints, original pretrained MIND-S checkpoints, and model prediction scores are organized in the directories above. Site-level results in `test_only_results/` and `benchmark_results/` support inspection of evaluated labels, sample sizes, performance statistics, and uncertainty estimates. ROC and precision–recall figures can be regenerated directly from the curated annotations and per-site prediction files.

## 🔄 Updates and Error Corrections

Reports of incorrect PTM annotations, accession or isoform assignments, residue positions, duplicate records, or prediction issues can be submitted through the repository issue tracker, ideally with the affected accession, residue position, PTM class, and supporting reference. Reported discrepancies are checked against the original source tables, cited literature, and reference sequences. Accepted corrections are documented with the affected records and the reason for the change in version history or release notes. When corrections alter curated annotations, derived inputs and affected analyses are regenerated, and corrected releases are versioned so earlier results remain traceable.

## 📖 Citation

Please cite the corresponding NR-PTMkb / NR-PTMpred publication when using the resource.
