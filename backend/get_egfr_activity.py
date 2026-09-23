import requests
import pandas as pd

# ChEMBL target for EGFR
target_chembl_id = "CHEMBL203"

url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

params = {
    "target_chembl_id": target_chembl_id,
    "limit": 1000
}

headers = {
    "Accept": "application/json"
}

print("Requesting EGFR bioactivity data...")

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=60
)

print("Status Code:", response.status_code)

if response.status_code != 200:
    print("ChEMBL request failed.")
    print(response.text[:500])
    exit()

data = response.json()

activities = data.get("activities", [])

print("Number of records received:", len(activities))

if not activities:
    print("No activity records found.")
    exit()

# Convert to DataFrame
df = pd.DataFrame(activities)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

# Save raw data
df.to_csv(
    "data/egfr_activity_raw.csv",
    index=False
)

print("\nRaw EGFR activity data saved successfully!")