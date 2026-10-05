# Multi-Class-Object-Image-Classification
*CNN classifier trained to identify objects from images - This project compares custom CNNs trained from scratch to identify which of 73 class-collected objects appears in an image. A 74th class, no_object, is built from empty background photos. Everything runs in one Colab notebook, from raw data to evaluation.*


---

## Repository Contents

| Folder | Contents |
|---|---|
| `Notebook/` | `7615_Project_1_Notebook.ipynb` — full pipeline: data validation, train/val/test split, model training, evaluation |
| `Models/` | Trained `.keras` model checkpoints (multiple architectures/hyperparameter variants tested) |
| `Images/` | `confusion_cnn_4blocks_v2.png` (test-set confusion matrix) and `unseen_cnn_4blocks_v2.png` (predictions on held-out "unseen environment" images) |
| `Results/` | `results_v2.csv` (metrics per model), `eval_v2.csv` (final test evaluation), `per_class_cnn_4blocks_v2.csv` (per-class precision/recall/F1) |

`splits_v2.csv`, `registry_v2.csv`, and `class_names_v2.json` live at the **repo root** — these define the finalized dataset (train/val/test split, object ID to name mapping, and class label order) and must be used as is. If you're re running the notebook, update the file paths in **Cell 1** to point to wherever you've stored these three files locally/on Drive.

The datasets are **not** in this repo, because of their size. They live in the group's Google Drive folder (see below).

---

## Quick Summary of Results (validation set, `dataset_v2`)

| Model | Change from baseline | Params | Val acc |
|---|---|---|---|
| **cnn_4blocks** (selected) | 4 conv blocks | 407k | **63.3%** |
| cnn_combo_bn | 4 blocks × 2 convs + Dense(256) + BatchNorm | 1.26M | 59.2% |
| cnn_no_aug | no augmentation | 103k | 55.3% |
| cnn_baseline | 3 blocks, 32 filters, dropout 0.3, Adam | 103k | 49.3% |
| cnn_2blocks | 2 conv blocks | 24k | 26.7% |

Chance is 1.4% (1/74). The full table of 14 runs is in `results_v2.csv`. Test-set, background (scene-bias), and unseen-environment results are in the report.

Key findings: capacity was the main bottleneck. Batch norm made no difference at 3 blocks but was essential at 8 conv layers (without it, that model failed to train). The largest model overfit. A pilot on an earlier 38-class snapshot (`dataset_v1`) produced the same ranking.

## Re-running the experiments

### 1. Set up Google Drive

Create a Drive folder (default name `IE7615_project_1`) containing:

```
IE7615_project_1/
├── dataset_v2.zip        # final class dataset (zip of all images_OBJxxx folders)
├── registry_v2.csv       # class registry: OBJECT_ID → OBJECT_NAME
├── splits_v2.csv         # from this repo, so you reuse our exact split
└── class_names_v2.json   # from this repo
```

If the folder has a different name or location, edit `PROJECT` in **Cell 1** of the Notebook. All other paths are derived from it.

### 2. Open the notebook in Colab

Set **Runtime → Change runtime type → T4 GPU**, then use **Runtime → Run all**. The notebook:

1. unzips the dataset to local disk and automatically repairs known file problems (e.g. oversized images);
2. validates every file (naming, format, and a full decode check);
3. **loads the frozen split.** It doesn't regenerate it while `splits_v2.csv` exists (to regenerate, set `FORCE_RESPLIT = True`);
4. trains every config in `ALL_RUNS`, logging each to `results_v2.csv` and saving the best checkpoint to `models_v2/`;
5. evaluates the selected candidates on the test, background, and unseen sets.

**Runtime:** about 4–5 hours for all 14 experiments on a T4.

### 3. Useful options

**Train a single model:**
```python
model, history, row = run_experiment({"name": "cnn_4blocks", "blocks": 4})
```
Configs only list what differs from `DEFAULTS` in Section 5.1.

**Resume after a disconnect.** Re-run Cells 1–2, Section 4, and Sections 5.1–5.3, then the training loop. Experiments already in `results_v2.csv` are skipped.

**Re-train from scratch.** Delete (or rename) `results_v2.csv` and `models_v2/` on Drive first. Otherwise, logged runs are skipped.

**Evaluate only, without retraining.** Copy our `models_v2/` folder into the Drive folder and run Sections 1–4 and 6.

**Load a saved model elsewhere:**
```python
model = keras.models.load_model("models_v2/cnn_4blocks.keras")
```
The model takes raw 224×224 RGB images (0–255), since preprocessing is built in. Map output indices to class IDs with `class_names_v2.json`.

## Reproducibility notes

- The global seed is `SEED = 42`, and each class's split is seeded from its own ID.
- GPU training isn't bit-for-bit deterministic, so re-runs can differ from our numbers by a few percentage points. With ~15 validation images per class, differences under ~10 points between models shouldn't be treated as meaningful.
- The test set is used **only** in Section 6. Model selection used validation accuracy.

## Data handling summary

The class dataset had several quality issues, all handled in code with no manual edits to source files:

- one image far above the agreed 224×224 size, downscaled after unzipping;
- non-standard filenames (e.g. `BG001` for backgrounds), handled by a tolerant parser that takes class IDs from folder names;
- one duplicate download copy, excluded via `EXCLUDE` in Cell 1;
- uneven background counts (the `no_object` class is ~4× a typical class), handled with class weights.

The notebook's data-issues markdown cells document each one.
