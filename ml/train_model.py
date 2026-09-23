import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 1. Load ML dataset
# ==========================================

df = pd.read_csv("ml/ml_dataset.csv")

print("ML dataset loaded!")
print("Total compounds:", len(df))


# ==========================================
# 2. Select features
# ==========================================

features = [
    "MolecularWeight",
    "LogP",
    "HBD",
    "HBA",
    "RotatableBonds",
    "TPSA"
]

target = "pIC50"


# ==========================================
# 3. Prepare X and y
# ==========================================

X = df[features]
y = df[target]


# Remove missing values
valid = X.notna().all(axis=1) & y.notna()

X = X[valid]
y = y[valid]

print("Usable compounds:", len(X))


# ==========================================
# 4. Train/Test split
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
# 5. Create Random Forest model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. Train model
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 7. Make predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. Evaluate model
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
print("MODEL PERFORMANCE")
print("===================================")

print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")


# ==========================================
# 9. Feature importance
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ==========================================
# 10. Save model
# ==========================================

import joblib

joblib.dump(
    model,
    "ml/random_forest_model.pkl"
)

print("\nModel saved to:")
print("ml/random_forest_model.pkl")