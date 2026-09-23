import pandas as pd

# --------------------------------
# 1. Read ChEMBL activity dataset
# --------------------------------

file_path = "data/bioactivity_raw.csv"

df = pd.read_csv(
    file_path,
    sep=";",
    quotechar='"',
    low_memory=False
)

print("Dataset loaded successfully!")
print("Total records:", len(df))


# --------------------------------
# 2. Select important columns
# --------------------------------

columns = [
    "Molecule ChEMBL ID",
    "Molecule Name",
    "Molecular Weight",
    "AlogP",
    "Smiles",
    "Standard Type",
    "Standard Relation",
    "Standard Value",
    "Standard Units",
    "pChEMBL Value",
    "Assay ChEMBL ID",
    "Assay Description",
    "Target ChEMBL ID",
    "Target Name",
    "Target Organism",
    "Document ChEMBL ID"
]

df = df[columns]


# --------------------------------
# 3. Keep IC50 measurements
# --------------------------------

df = df[
    df["Standard Type"].astype(str).str.upper() == "IC50"
]


# --------------------------------
# 4. Keep measurements in nM
# --------------------------------

df = df[
    df["Standard Units"].astype(str).str.lower() == "nm"
]


# --------------------------------
# 5. Keep numeric activity values
# --------------------------------

df["Standard Value"] = pd.to_numeric(
    df["Standard Value"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Molecule ChEMBL ID",
        "Standard Value"
    ]
)


# --------------------------------
# 6. Remove invalid/non-positive values
# --------------------------------

df = df[df["Standard Value"] > 0]


# --------------------------------
# 7. Keep exact measurements
# --------------------------------

df = df[
    df["Standard Relation"].astype(str).str.contains("=")
]


# --------------------------------
# 8. Remove duplicate records
# --------------------------------

df = df.drop_duplicates(
    subset=[
        "Molecule ChEMBL ID",
        "Assay ChEMBL ID",
        "Standard Type",
        "Standard Value"
    ]
)


# --------------------------------
# 9. Display results
# --------------------------------

print("\nCleaned dataset")
print("----------------")

print("Number of records:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 cleaned records:")
print(df.head())


# --------------------------------
# 10. Save cleaned dataset
# --------------------------------

output_file = "data/bioactivity_clean.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nClean dataset saved to:")
print(output_file)