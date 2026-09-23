import pandas as pd
import psycopg2

# -------------------------
# Read CSV
# -------------------------

df = pd.read_csv("data/drugs.csv")

print("CSV loaded successfully!")
print(df)


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
# Insert data
# -------------------------

for _, row in df.iterrows():

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
        ON CONFLICT (pubchem_cid) DO NOTHING
        """,
        (
            row["drug_name"],
            int(row["pubchem_cid"]),
            row["molecular_formula"],
            float(row["molecular_weight"]),
            row["smiles"]
        )
    )


conn.commit()

cursor.close()
conn.close()

print("\nDrug CSV imported into PostgreSQL successfully!")