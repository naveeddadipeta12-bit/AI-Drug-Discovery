import requests
import pandas as pd

drug_name = "Imatinib"

url = (
    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    f"name/{drug_name}/property/"
    f"MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
)

response = requests.get(url, timeout=30)

print("Status Code:", response.status_code)

if response.status_code != 200:
    print("PubChem request failed.")
    print(response.text[:500])
    exit()

data = response.json()

properties = data["PropertyTable"]["Properties"][0]

record = {
    "drug_name": drug_name,
    "pubchem_cid": properties.get("CID"),
    "molecular_formula": properties.get("MolecularFormula"),
    "molecular_weight": properties.get("MolecularWeight"),
    "smiles": properties.get("ConnectivitySMILES")
}

df = pd.DataFrame([record])

print("\n--- DRUG DATA ---")
print(df.to_string(index=False))

df.to_csv(
    "data/drugs.csv",
    index=False
)

print("\nDrug data saved to data/drugs.csv")