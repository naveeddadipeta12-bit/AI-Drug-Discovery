import requests
import psycopg2


# -------------------------
# Get Imatinib from PubChem
# -------------------------

drug_name = "Imatinib"

url = (
    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    f"name/{drug_name}/property/"
    f"MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
)

response = requests.get(url, timeout=30)

if response.status_code != 200:
    print("Failed to fetch Imatinib from PubChem")
    print(response.text[:500])
    exit()


data = response.json()

properties = data["PropertyTable"]["Properties"][0]

pubchem_cid = properties.get("CID")
molecular_formula = properties.get("MolecularFormula")
molecular_weight = properties.get("MolecularWeight")
smiles = properties.get("ConnectivitySMILES")


print("Fetched from PubChem:")
print("Drug:", drug_name)
print("PubChem CID:", pubchem_cid)
print("Molecular Formula:", molecular_formula)
print("Molecular Weight:", molecular_weight)
print("SMILES:", smiles)


# -------------------------
# Connect to PostgreSQL
# -------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="drug_discovery",
    user="postgres",
    password="Naveed*786*"
)

cursor = conn.cursor()


# -------------------------
# Insert into drugs table
# -------------------------

cursor.execute(
    """
    INSERT INTO drugs
    (
        drug_name,
        pubchem_cid,
        molecular_formula,
        molecular_weight,
        smiles
    )
    VALUES (%s, %s, %s, %s, %s)
    """,
    (
        drug_name,
        pubchem_cid,
        molecular_formula,
        molecular_weight,
        smiles
    )
)

conn.commit()

cursor.close()
conn.close()

print("\nImatinib inserted into PostgreSQL successfully!")