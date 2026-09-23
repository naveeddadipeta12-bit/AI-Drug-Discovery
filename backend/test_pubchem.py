import requests

drug_name = "Imatinib"

url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName/JSON"

response = requests.get(url, timeout=30)

print("Status Code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    properties = data["PropertyTable"]["Properties"][0]

    print("\n--- DRUG INFORMATION ---")
    print("Drug:", drug_name)
    print("PubChem CID:", properties.get("CID"))
    print("Molecular Formula:", properties.get("MolecularFormula"))
    print("Molecular Weight:", properties.get("MolecularWeight"))
    print("Canonical SMILES:", properties.get("ConnectivitySMILES"))
    print("IUPAC Name:", properties.get("IUPACName"))

else:
    print("Error:")
    print(response.text[:500])