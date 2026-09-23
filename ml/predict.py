import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors


# ==========================================
# 1. Load trained model
# ==========================================

model = joblib.load(
    "ml/random_forest_model.pkl"
)

print("AI model loaded successfully!")


# ==========================================
# 2. Get SMILES from user
# ==========================================

smiles = input("\nEnter SMILES: ")


# ==========================================
# 3. Convert SMILES to molecule
# ==========================================

mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES!")
    exit()


# ==========================================
# 4. Calculate the SAME 6 descriptors
#    used during model training
# ==========================================

molecular_weight = Descriptors.MolWt(mol)

logp = Descriptors.MolLogP(mol)

hbd = Descriptors.NumHDonors(mol)

hba = Descriptors.NumHAcceptors(mol)

rotatable_bonds = Descriptors.NumRotatableBonds(mol)

tpsa = Descriptors.TPSA(mol)


# ==========================================
# 5. Create feature array
# ==========================================

features = np.array([
    [
        molecular_weight,
        logp,
        hbd,
        hba,
        rotatable_bonds,
        tpsa
    ]
])


# ==========================================
# 6. Predict pIC50
# ==========================================

predicted_pIC50 = model.predict(features)[0]


# ==========================================
# 7. Convert pIC50 to estimated IC50
# ==========================================

predicted_ic50 = 10 ** (9 - predicted_pIC50)


# ==========================================
# 8. Display results
# ==========================================

print("\n===================================")
print("AI DRUG ACTIVITY PREDICTION")
print("===================================")

print("SMILES:", smiles)

print("\nMolecular Descriptors:")
print(f"Molecular Weight : {molecular_weight:.2f}")
print(f"LogP             : {logp:.2f}")
print(f"HBD              : {hbd}")
print(f"HBA              : {hba}")
print(f"Rotatable Bonds  : {rotatable_bonds}")
print(f"TPSA             : {tpsa:.2f}")

print("\nAI Prediction:")
print(f"Predicted pIC50  : {predicted_pIC50:.3f}")
print(f"Estimated IC50   : {predicted_ic50:.3f} nM")

print("===================================")