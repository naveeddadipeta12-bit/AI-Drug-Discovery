import requests
import psycopg2

# -------------------------
# Get EGFR from UniProt
# -------------------------

url = "https://rest.uniprot.org/uniprotkb/P00533.json"

response = requests.get(url, timeout=30)

if response.status_code != 200:
    print("Failed to fetch UniProt data")
    exit()

data = response.json()

gene_name = "EGFR"
protein_name = data["proteinDescription"]["recommendedName"]["fullName"]["value"]
uniprot_id = data["primaryAccession"]
organism = data["organism"]["scientificName"]
sequence = data["sequence"]["value"]

print("Fetched from UniProt:")
print(gene_name)
print(protein_name)
print(uniprot_id)
print(organism)
print("Sequence length:", len(sequence))

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
# Insert into targets table
# -------------------------

cursor.execute(
    """
    INSERT INTO targets
    (gene_name, protein_name, uniprot_id, organism, sequence)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (uniprot_id) DO NOTHING
    """,
    (
        gene_name,
        protein_name,
        uniprot_id,
        organism,
        sequence
    )
)

conn.commit()

cursor.close()
conn.close()

print("\nEGFR inserted into PostgreSQL successfully!")