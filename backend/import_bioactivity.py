import pandas as pd
import psycopg2


# --------------------------------
# 1. Read cleaned bioactivity data
# --------------------------------

file_path = "data/bioactivity_clean.csv"

df = pd.read_csv(file_path)

print("Bioactivity dataset loaded!")
print("Total records:", len(df))


# --------------------------------
# 2. Connect to PostgreSQL
# --------------------------------

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="drug_discovery",
    user="postgres",
    password="Naveed*786*"
)

cursor = conn.cursor()

print("Connected to PostgreSQL!")


# --------------------------------
# 3. Process each bioactivity
# --------------------------------

inserted_interactions = 0
inserted_activities = 0


for _, row in df.iterrows():

    molecule_id = row["Molecule ChEMBL ID"]
    target_chembl_id = row["Target ChEMBL ID"]

    activity_type = row["Standard Type"]
    activity_value = row["Standard Value"]
    activity_unit = row["Standard Units"]
    assay_id = row["Assay ChEMBL ID"]


    # --------------------------------
    # Find the target in our database
    # --------------------------------

    cursor.execute(
        """
        SELECT target_id
        FROM targets
        WHERE uniprot_id = %s
        """,
        ("P00533",)
    )

    target_result = cursor.fetchone()

    if target_result is None:
        print("EGFR target not found!")
        continue

    target_id = target_result[0]


    # --------------------------------
    # Find or create the drug
    # --------------------------------

    cursor.execute(
        """
        SELECT drug_id
        FROM drugs
        WHERE chembl_id = %s
        """,
        (molecule_id,)
    )

    drug_result = cursor.fetchone()


    if drug_result:

        drug_id = drug_result[0]

    else:

        # Insert new drug
        cursor.execute(
            """
            INSERT INTO drugs
            (
                drug_name,
                chembl_id,
                molecular_formula,
                molecular_weight,
                smiles
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING drug_id
            """,
            (
                row["Molecule Name"],
                molecule_id,
                None,
                row["Molecular Weight"],
                row["Smiles"]
            )
        )

        drug_id = cursor.fetchone()[0]

    # --------------------------------
    # Create interaction
    # --------------------------------

    cursor.execute(
        """
        INSERT INTO interactions
        (
            drug_id,
            target_id,
            interaction_type,
            source
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (drug_id, target_id, source)
        DO NOTHING
        RETURNING interaction_id
        """,
        (
            drug_id,
            target_id,
            activity_type,
            "ChEMBL"
        )
    )

    interaction_result = cursor.fetchone()


    # If interaction already existed
    if interaction_result:

        interaction_id = interaction_result[0]
        inserted_interactions += 1

    else:

        cursor.execute(
            """
            SELECT interaction_id
            FROM interactions
            WHERE drug_id = %s
            AND target_id = %s
            AND source = %s
            """,
            (
                drug_id,
                target_id,
                "ChEMBL"
            )
        )

        interaction_id = cursor.fetchone()[0]


    # --------------------------------
    # Insert bioactivity
    # --------------------------------

    cursor.execute(
        """
        INSERT INTO bioactivities
        (
            interaction_id,
            activity_type,
            activity_value,
            activity_unit,
            assay_id,
            source
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            interaction_id,
            activity_type,
            activity_value,
            activity_unit,
            assay_id,
            "ChEMBL"
        )
    )

    inserted_activities += 1


# --------------------------------
# 4. Save changes
# --------------------------------

conn.commit()

cursor.close()
conn.close()


print("\n--------------------------------")
print("IMPORT COMPLETED")
print("--------------------------------")

print("Interactions processed:", inserted_interactions)
print("Bioactivities inserted:", inserted_activities)

print("\nData successfully imported into PostgreSQL!")