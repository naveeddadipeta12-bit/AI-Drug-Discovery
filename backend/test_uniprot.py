import requests

url = "https://rest.uniprot.org/uniprotkb/search"

params = {
    "query": "accession:P00533",
    "format": "json",
    "size": 1
}

response = requests.get(url, params=params, timeout=30)

print("Status Code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    result = data["results"][0]

    print("\n--- EGFR INFORMATION ---")

    print("UniProt ID:", result.get("primaryAccession"))

    protein_name = (
        result["proteinDescription"]
        ["recommendedName"]
        ["fullName"]
        ["value"]
    )

    print("Protein:", protein_name)
    print("Organism:", result["organism"]["scientificName"])
    print("Length:", result["sequence"]["length"])

else:
    print("Error:", response.text)