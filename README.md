# Multi-Class-Object-Image-Classification

*CNN classifier trained to identify objects from images.*

---

## Where to Look

| Folder | Contents |
|---|---|
| `Notebook/` | `7615_Project_1_Notebook.ipynb` — full pipeline: data validation, train/val/test split, model training, evaluation |
| `Models/` | Trained `.keras` model checkpoints (multiple architectures/hyperparameter variants tested) |
| `Images/` | `confusion_cnn_4blocks_v2.png` (test-set confusion matrix) and `unseen_cnn_4blocks_v2.png` (predictions on unseen environment images) |
| `Results/` | `results_v2.csv` (metrics per model), `eval_v2.csv` (final test evaluation), `per_class_cnn_4blocks_v2.csv` (per-class precision/recall/F1), `class_names_v2.json` (class label list), `registry_v2.csv` (object ID → name mapping), `splits_v2.csv` (train/val/test split) |

---

## Quick Summary of Results

See `Results/results_v2.csv` for accuracy/loss across all model variants, and `Results/per_class_cnn_4blocks_v2.csv` for the per-class breakdown of the best model (`cnn_4blocks`). The confusion matrix and unseen-environment predictions in `Images/` give a visual sense of where the model struggles.

---

## How to Load a Model

```python
import json, tensorflow as tf
from pathlib import Path
from tensorflow import keras

MODELS_DIR = Path("Models")
RESULTS_DIR = Path("Results")

model = keras.models.load_model(MODELS_DIR / "cnn_4blocks.keras")

with open(RESULTS_DIR / "class_names_v2.json") as f:
    class_names = json.load(f)   # maps output index -> class ID (e.g. "OBJ001")

model.summary()  # layers, output shape, parameter counts
```

> `class_names_v2.json` is required — the model only outputs class probabilities by index; this file maps each index back to its object ID.
