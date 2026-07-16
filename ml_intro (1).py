"""
ML for Drug Discovery: Solubility (Regression) + Toxicity (Classification)
----------------------------------------------------------------------------
A side-by-side intro to the two core ML task types you'll run into
constantly in computational pharmacology/chemistry work:

    1. REGRESSION      -> predict a continuous number (solubility)
    2. CLASSIFICATION  -> predict a category (toxic / not toxic)

Same overall workflow both times:
    molecule (SMILES)
        -> descriptors (numeric features RDKit calculates)
        -> train/test split
        -> train a model
        -> evaluate it
        -> see which features mattered most

IMPORTANT NOTE ON THE DATA
----------------------------------------------------------------------------
The datasets below are small, hand-picked, ILLUSTRATIVE sets built to
teach the ML workflow — not the full official ESOL / Tox21 benchmark
datasets, and the values are approximate. They're realistic enough to
learn from, but don't treat them as authoritative scientific data.

If you want the real, full-size benchmark datasets for a more rigorous
version of this project (hundreds/thousands of compounds with measured
values), look up "MoleculeNet ESOL" and "MoleculeNet Tox21" — they're
free downloadable CSVs, and DeepChem can load them directly:
    pip install deepchem
    import deepchem as dc
    tasks, datasets, transformers = dc.molnet.load_delaney()   # ESOL
    tasks, datasets, transformers = dc.molnet.load_tox21()

Requirements:
    pip install rdkit scikit-learn matplotlib pandas
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt


# ============================================================
# PART 1: DATA
# ============================================================

# Illustrative solubility dataset (approx. logS — higher = more soluble)
SOLUBILITY_DATA = [
    ("Ethanol",       "CCO",                                   1.1),
    ("Glucose",       "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  0.8),
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O",                  -1.7),
    ("Caffeine",      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",              -0.9),
    ("Paracetamol",   "CC(=O)Nc1ccc(O)cc1",                     -1.1),
    ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              -3.6),
    ("Naphthalene",   "c1ccc2ccccc2c1",                         -3.6),
    ("Benzene",       "c1ccccc1",                                -1.6),
    ("Phenol",        "Oc1ccccc1",                                0.0),
    ("Acetone",       "CC(=O)C",                                  1.6),
    ("Toluene",       "Cc1ccccc1",                                -2.2),
    ("Warfarin",      "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",   -3.8),
    ("Diazepam",      "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",       -3.8),
    ("Nicotine",      "CN1CCC[C@H]1c1cccnc1",                     0.5),
    ("Testosterone",  "CC12CCC3c4ccc(=O)cc4CCC3C1CCC2O",         -3.7),
    ("Salicylic acid","OC(=O)c1ccccc1O",                         -1.9),
    ("Urea",          "NC(=O)N",                                  1.5),
    ("Methanol",      "CO",                                       1.3),
    ("Cholesterol",   "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C", -7.8),
    ("Sucrose",       "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H]1O", 1.0),
]

# Illustrative toxicity dataset (1 = flagged toxic/hazardous, 0 = generally safe)
TOXICITY_DATA = [
    ("Paracetamol",   "CC(=O)Nc1ccc(O)cc1",                     0),
    ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O",                  0),
    ("Ibuprofen",     "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              0),
    ("Caffeine",      "Cn1cnc2c1c(=O)n(C)c(=O)n2C",              0),
    ("Glucose",       "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  0),
    ("Vitamin C",     "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",          0),
    ("Ethanol",       "CCO",                                     0),
    ("Urea",          "NC(=O)N",                                 0),
    ("Benzene",       "c1ccccc1",                                1),
    ("Formaldehyde",  "C=O",                                     1),
    ("Warfarin",      "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",   1),
    ("Nicotine",      "CN1CCC[C@H]1c1cccnc1",                     1),
    ("Carbon tetrachloride", "ClC(Cl)(Cl)Cl",                    1),
    ("Chloroform",    "ClC(Cl)Cl",                                1),
    ("Acetaminophen overdose analog", "CC(=O)Nc1ccc(O)c(Cl)c1", 1),
    ("Naphthalene",   "c1ccc2ccccc2c1",                          1),
    ("Toluene",       "Cc1ccccc1",                                1),
    ("Methanol",      "CO",                                       1),
    ("Salicylic acid","OC(=O)c1ccccc1O",                         0),
    ("Acetone",       "CC(=O)C",                                  0),
]


# ============================================================
# PART 2: FEATURIZATION
# Turn each molecule into a row of numbers RDKit can calculate.
# This is the same idea as your drug-likeness project, just
# used as ML *input features* instead of pass/fail rules.
# ============================================================

def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
    }


def build_feature_table(data, label_name):
    rows = []
    for name, smiles, label in data:
        feats = featurize(smiles)
        if feats is None:
            print(f"⚠️  Skipping {name}, couldn't parse SMILES")
            continue
        feats["name"] = name
        feats[label_name] = label
        rows.append(feats)
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotatableBonds", "AromaticRings", "RingCount"
]


# ============================================================
# PART 3a: REGRESSION — predicting solubility (a number)
# ============================================================

def run_solubility_regression():
    print("\n" + "=" * 60)
    print("PART A: SOLUBILITY PREDICTION (Regression)")
    print("=" * 60)

    df = build_feature_table(SOLUBILITY_DATA, "logS")
    X = df[FEATURE_COLUMNS]
    y = df["logS"]

    # Small dataset -> small test split, just for demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\nTest set size: {len(y_test)} molecules")
    print(f"RMSE: {rmse:.2f}  (lower = better; this is in logS units)")
    print(f"R²:   {r2:.2f}  (closer to 1.0 = better fit)")

    print("\nPredicted vs Actual (test set):")
    for name, actual, pred in zip(df.loc[y_test.index, "name"], y_test, predictions):
        print(f"  {name:<20} actual: {actual:6.2f}   predicted: {pred:6.2f}")

    # Feature importance — which descriptors mattered most?
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nWhich features mattered most for solubility?")
    for feat, score in importances.items():
        print(f"  {feat:<18} {score:.3f}")

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, s=80, alpha=0.7)
    lims = [min(y.min(), predictions.min()) - 1, max(y.max(), predictions.max()) + 1]
    plt.plot(lims, lims, "k--", alpha=0.4, label="Perfect prediction")
    plt.xlabel("Actual logS")
    plt.ylabel("Predicted logS")
    plt.title("Solubility: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("solubility_predictions.png", dpi=120)
    print("\n📊 Saved plot: solubility_predictions.png")


# ============================================================
# PART 3b: CLASSIFICATION — predicting toxicity (a category)
# ============================================================

def run_toxicity_classification():
    print("\n" + "=" * 60)
    print("PART B: TOXICITY PREDICTION (Classification)")
    print("=" * 60)

    df = build_feature_table(TOXICITY_DATA, "toxic")
    X = df[FEATURE_COLUMNS]
    y = df["toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    # zero_division=0 avoids a warning if a class has no predicted samples
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nTest set size: {len(y_test)} molecules")
    print(f"Accuracy:  {acc:.2f}  (% correct overall — can be misleading!)")
    print(f"Precision: {prec:.2f}  (of predicted-toxic, how many really were)")
    print(f"Recall:    {rec:.2f}  (of actually-toxic, how many we caught)")
    print(f"\nConfusion matrix:\n{cm}")
    print("(rows = actual, columns = predicted | order: [safe, toxic])")

    print("\nPredicted vs Actual (test set):")
    label_map = {0: "safe", 1: "toxic"}
    for name, actual, pred in zip(df.loc[y_test.index, "name"], y_test, predictions):
        match = "✅" if actual == pred else "❌"
        print(f"  {name:<30} actual: {label_map[actual]:<6} predicted: {label_map[pred]:<6} {match}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    print("\nWhich features mattered most for toxicity?")
    for feat, score in importances.items():
        print(f"  {feat:<18} {score:.3f}")


if __name__ == "__main__":
    run_solubility_regression()
    run_toxicity_classification()

    print("\n" + "=" * 60)
    print("Why accuracy alone can be misleading:")
    print("=" * 60)
    print(
        "If 90% of a real dataset is 'non-toxic', a model that just\n"
        "predicts 'non-toxic' every single time gets 90% accuracy —\n"
        "while being completely useless. That's why precision and\n"
        "recall matter: they tell you how the model handles the class\n"
        "you actually care about catching (the toxic ones)."
    )
