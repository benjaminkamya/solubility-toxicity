# 🧠 ML for Drug Discovery: Solubility & Toxicity Prediction

A side-by-side intro to the two core machine learning task types used
constantly in computational pharmacology/chemistry:

1. **Regression** - predicting solubility (a continuous number)
2. **Classification** - predicting toxicity (a category: toxic / safe)

Built as a bridge project before starting an MSc - first proper
"real ML" project after two RDKit-based cheminformatics tools
([DrugLikenessEvaluator](https://github.com/benjaminkamya/DrugLikenessEvaluator),
[ChemSimilarity](https://github.com/benjaminkamya/ChemSimilarity)).

## ⚠️ A note on the data

The datasets used here are small, hand-picked, and **illustrative** -
built to teach the ML workflow, not the full official ESOL / Tox21
benchmark datasets, and the values are approximate. Good for learning
the pipeline, not for drawing real scientific conclusions.

For the real, full-size benchmark datasets, look up "MoleculeNet ESOL"
and "MoleculeNet Tox21" - DeepChem can load them directly:
```python
pip install deepchem
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_delaney()   # ESOL
tasks, datasets, transformers = dc.molnet.load_tox21()
```

## Setup

```bash
pip install rdkit scikit-learn matplotlib pandas
```

## Usage

```bash
python ml_intro.py
```

Runs both tasks and prints:
- Model performance metrics (RMSE/R² for regression, accuracy/precision/recall for classification)
- Predicted vs actual results for every test molecule
- Which molecular features mattered most for each task
- Saves a prediction scatter plot: `solubility_predictions.png`

Results:
Each point represents one test-set compound. The x-axis shows its
actual measured solubility (logS), the y-axis shows what the model
predicted. Points closer to the dashed diagonal line indicate more
accurate predictions.

## What this teaches

- **Featurization** - turning a SMILES string into numerical features
  a model can actually learn from (molecular weight, LogP, TPSA,
  H-bond donors/acceptors, rotatable bonds, ring counts)
- **Train/test splitting** - never evaluate a model on data it trained on
- **Regression vs classification** - two fundamentally different
  prediction problems needing different evaluation metrics
- **Why accuracy alone is misleading** - a model that always predicts
  "non-toxic" can score high accuracy on an imbalanced dataset while
  being completely useless. Precision and recall tell the real story.
- **Feature importance** - which molecular properties actually drove
  the model's predictions

## Built with

- [RDKit](https://www.rdkit.org/) - molecular featurization
- [scikit-learn](https://scikit-learn.org/) - Random Forest models & metrics
- [Matplotlib](https://matplotlib.org/) - plotting
- [Pandas](https://pandas.pydata.org/) - data handling
