import pandas as pd
import psycopg2
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


# ==========================================
# 1. Connect to PostgreSQL
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="drug_discovery",
    user="postgres",
    password="Naveed*786*"
)

print("Connected to PostgreSQL!")


# ==========================================
# 2. Get Drug + EGFR + IC50 data
# ==========================================

query = """
SELECT
    d.drug_id,
    d.drug_name,
    d.chembl_id,
    d.smiles,
    b.activity_value,
    b.activity_unit
FROM drugs d
JOIN interactions i
    ON d.drug_id = i.drug_id
JOIN targets t
    ON i.target_id = t.target_id
JOIN bioactivities b
    ON i.interaction_id = b.interaction_id
WHERE t.uniprot_id = 'P00533'
AND b.activity_type = 'IC50'
AND LOWER(b.activity_unit) = 'nm'
AND b.activity_value > 0;
"""

df = pd.read_sql_query(query, conn)

conn.close()

print("Data extracted successfully!")
print("Number of records:", len(df))


# ==========================================
# 3. Remove missing SMILES
# ==========================================

df = df.dropna(subset=["smiles"])

print("Records after removing missing SMILES:", len(df))


# ==========================================
# 4. Convert SMILES to RDKit molecules
# ==========================================

df["molecule"] = df["smiles"].apply(
    lambda x: Chem.MolFromSmiles(str(x))
)

# Remove invalid molecules
df = df.dropna(subset=["molecule"])

print("Valid molecules:", len(df))


# ==========================================
# 5. Calculate molecular descriptors
# ==========================================

df["MolecularWeight"] = df["molecule"].apply(
    Descriptors.MolWt
)

df["LogP"] = df["molecule"].apply(
    Descriptors.MolLogP
)

df["HBD"] = df["molecule"].apply(
    Descriptors.NumHDonors
)

df["HBA"] = df["molecule"].apply(
    Descriptors.NumHAcceptors
)

df["RotatableBonds"] = df["molecule"].apply(
    Descriptors.NumRotatableBonds
)

df["TPSA"] = df["molecule"].apply(
    Descriptors.TPSA
)


# ==========================================
# 6. Convert IC50 nM → pIC50
# ==========================================

df["pIC50"] = -np.log10(
    df["activity_value"] * 1e-9
)


# ==========================================
# 7. Remove duplicate compounds
# ==========================================

df = df.drop_duplicates(
    subset=["chembl_id"]
)


# ==========================================
# 8. Select final ML columns
# ==========================================

final_columns = [
    "drug_name",
    "chembl_id",
    "smiles",
    "MolecularWeight",
    "LogP",
    "HBD",
    "HBA",
    "RotatableBonds",
    "TPSA",
    "activity_value",
    "pIC50"
]

ml_df = df[final_columns]


# ==========================================
# 9. Save ML dataset
# ==========================================

output_file = "ml/ml_dataset.csv"

ml_df.to_csv(
    output_file,
    index=False
)


# ==========================================
# 10. Display information
# ==========================================

print("\n===================================")
print("ML DATASET CREATED")
print("===================================")

print("Number of compounds:", len(ml_df))

print("\nColumns:")
print(ml_df.columns.tolist())

print("\nFirst 5 records:")
print(ml_df.head())

print("\nSaved to:")
print(output_file)