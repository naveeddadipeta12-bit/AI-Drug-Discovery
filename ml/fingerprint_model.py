import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("ml/ml_dataset.csv")

print("Dataset loaded!")
print("Total records:", len(df))


# ==========================================
# 2. Remove missing SMILES and pIC50
# ==========================================

df = df.dropna(
    subset=["smiles", "pIC50"]
)

print("Records after cleaning:", len(df))


# ==========================================
# 3. Convert SMILES to RDKit molecules
# ==========================================

df["mol"] = df["smiles"].apply(
    lambda x: Chem.MolFromSmiles(str(x))
)

df = df.dropna(
    subset=["mol"]
)

print("Valid molecules:", len(df))


# ==========================================
# 4. Generate Morgan fingerprints
# ==========================================

def generate_fingerprint(molecule):

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        radius=2,
        nBits=2048
    )

    return np.array(fingerprint)


X = np.array(
    df["mol"].apply(generate_fingerprint).tolist()
)

y = df["pIC50"].values


print("Fingerprint shape:", X.shape)


# ==========================================
# 5. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. Create Random Forest
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 7. Train
# ==========================================

print("\nTraining fingerprint model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 8. Predict
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. Evaluate
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n===================================")
print("FINGERPRINT MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
# ==========================================
# 10. Actual vs Predicted pIC50
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")

plt.title("Actual vs Predicted pIC50")

plt.tight_layout()

plt.savefig(
    "ml/actual_vs_predicted.png",
    dpi=300
)

plt.show()

print("\nGraph saved to:")
print("ml/actual_vs_predicted.png")