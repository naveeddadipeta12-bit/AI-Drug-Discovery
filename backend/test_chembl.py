import requests

url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

params = {
    "limit": 1
}

headers = {
    "Accept": "application/json"
}

try:
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    print("Status Code:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))

    if response.status_code == 200:

        data = response.json()

        molecules = data.get("molecules", [])

        if molecules:
            molecule = molecules[0]

            print("\n--- CHEMBL API TEST ---")
            print("ChEMBL ID:", molecule.get("molecule_chembl_id"))
            print("Preferred Name:", molecule.get("pref_name"))
            print("Molecule Type:", molecule.get("molecule_type"))

        else:
            print("No molecule data found.")

    else:
        print("\nChEMBL server returned an error.")
        print(response.text[:300])

except requests.exceptions.RequestException as e:
    print("Connection error:", e)