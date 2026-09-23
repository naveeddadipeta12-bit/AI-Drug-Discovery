from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rdkit import Chem
from rdkit.Chem import Descriptors

import numpy as np
import joblib
import requests

from urllib.parse import quote

from dotenv import load_dotenv
from google import genai

import os


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv("backend/.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# GEMINI CLIENT
# =========================================================

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini API key loaded successfully!")

    except Exception as e:

        print(
            "Gemini client error:",
            e
        )

else:

    print(
        "Gemini API key not found."
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AI Drug Discovery API",
    description="AI-powered drug-target activity prediction",
    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ml",
    "random_forest_model.pkl"
)

MODEL_PATH = os.path.abspath(
    MODEL_PATH
)


model = None


try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Machine learning model loaded successfully!"
    )

except Exception as e:

    print(
        "Model loading error:",
        e
    )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AI Drug Discovery API is running",
        "status": "success"
    }


# =========================================================
# COMPOUND NAME FUNCTION
# =========================================================

def get_compound_name(smiles):

    # -----------------------------------------------------
    # LOCAL KNOWN DRUG DATABASE
    # -----------------------------------------------------

    known_drugs = {

        "CC(=O)OC1=CC=CC=C1C(=O)O":
            "Aspirin",

        "CC(=O)NC1=CC=C(O)C=C1":
            "Paracetamol",

        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O":
            "Ibuprofen",

        "Cn1cnc2c1c(=O)n(C(=O)n2C)C":
            "Caffeine",

        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5":
            "Imatinib",

        "COC1=C(O)C=CC(=C1)NC2=NC=NC3=C2C=CC(=C3)F":
            "Gefitinib",

        "C#CCOC1=CC=CC(=C1)C#CC2=CN=C(N=C2N)C3=CC=CC=C3":
            "Erlotinib",

        "CC(C(=O)CC(C1=CC=CC=C1)O)C2=CC=CC=C2":
            "Warfarin",

        "COC1=CC2=C(C=C1)C(=CC=C2)C(C)C(=O)O":
            "Naproxen",

        "O=C(O)C1=C(N(C2=CC=CC=C2)C3=CC=C(Cl)C=C3)C=CC=C1":
            "Diclofenac"
    }


    # -----------------------------------------------------
    # CANONICALIZE INPUT SMILES
    # -----------------------------------------------------

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is not None:

            canonical_smiles = Chem.MolToSmiles(
                mol
            )

            # Check all known drugs

            for known_smiles, name in known_drugs.items():

                known_mol = Chem.MolFromSmiles(
                    known_smiles
                )

                if known_mol is None:
                    continue

                known_canonical = Chem.MolToSmiles(
                    known_mol
                )

                if canonical_smiles == known_canonical:

                    print(
                        "Local compound match:",
                        name
                    )

                    return name

    except Exception as e:

        print(
            "Local compound lookup error:",
            e
        )


    # -----------------------------------------------------
    # PUBCHEM BACKUP
    # -----------------------------------------------------

    try:

        encoded_smiles = quote(
            smiles,
            safe=""
        )

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/smiles/{encoded_smiles}/synonyms/JSON"
        )

        response = requests.get(
            url,
            timeout=15
        )

        print(
            "PubChem status:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "PubChem lookup failed"
            )

            return "Compound name not found"


        data = response.json()


        information = (
            data
            .get(
                "InformationList",
                {}
            )
            .get(
                "Information",
                []
            )
        )


        if not information:

            return "Compound name not found"


        synonyms = information[0].get(
            "Synonym",
            []
        )


        if not synonyms:

            return "Compound name not found"


        # Prefer common drug names

        preferred_names = [

            "Aspirin",
            "Paracetamol",
            "Acetaminophen",
            "Ibuprofen",
            "Caffeine",
            "Imatinib",
            "Gefitinib",
            "Erlotinib",
            "Warfarin",
            "Naproxen",
            "Diclofenac"

        ]


        for preferred in preferred_names:

            for synonym in synonyms:

                if synonym.lower() == preferred.lower():

                    return synonym


        # Otherwise return first available synonym

        for synonym in synonyms:

            if synonym.strip():

                return synonym


        return "Compound name not found"


    except Exception as e:

        print(
            "PubChem name lookup error:",
            e
        )

        return "Compound name not found"


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(smiles_data: dict):

    smiles = smiles_data.get(
        "smiles",
        ""
    ).strip()


    # -----------------------------------------------------
    # CHECK INPUT
    # -----------------------------------------------------

    if not smiles:

        return {

            "success": False,

            "error":
                "Please enter a SMILES string."

        }


    # -----------------------------------------------------
    # CONVERT SMILES TO MOLECULE
    # -----------------------------------------------------

    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:

        return {

            "success": False,

            "error":
                "Invalid SMILES string"

        }


    # -----------------------------------------------------
    # GET COMPOUND NAME
    # -----------------------------------------------------

    compound_name = get_compound_name(
        smiles
    )


    # -----------------------------------------------------
    # CALCULATE MOLECULAR PROPERTIES
    # -----------------------------------------------------

    molecular_weight = Descriptors.MolWt(
        mol
    )

    logp = Descriptors.MolLogP(
        mol
    )

    hbd = Descriptors.NumHDonors(
        mol
    )

    hba = Descriptors.NumHAcceptors(
        mol
    )

    rotatable_bonds = Descriptors.NumRotatableBonds(
        mol
    )

    tpsa = Descriptors.TPSA(
        mol
    )


    # -----------------------------------------------------
    # CHECK MODEL
    # -----------------------------------------------------

    if model is None:

        return {

            "success": False,

            "error":
                "Machine learning model could not be loaded."

        }


    # -----------------------------------------------------
    # CREATE FEATURES
    # -----------------------------------------------------

    features = np.array([[

        molecular_weight,

        logp,

        hbd,

        hba,

        rotatable_bonds,

        tpsa

    ]])


    # -----------------------------------------------------
    # MACHINE LEARNING PREDICTION
    # -----------------------------------------------------

    predicted_pIC50 = model.predict(
        features
    )[0]


    # Convert pIC50 to IC50 in nM

    estimated_ic50 = 10 ** (
        9 - predicted_pIC50
    )


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "success": True,

        "compound_name":
            compound_name,

        "smiles":
            smiles,

        "molecular_weight":
            round(
                molecular_weight,
                2
            ),

        "logp":
            round(
                logp,
                2
            ),

        "hbd":
            hbd,

        "hba":
            hba,

        "rotatable_bonds":
            rotatable_bonds,

        "tpsa":
            round(
                tpsa,
                2
            ),

        "predicted_pIC50":
            round(
                float(predicted_pIC50),
                3
            ),

        "estimated_ic50_nM":
            round(
                float(estimated_ic50),
                3
            )

    }


# =========================================================
# GEMINI EXPLANATION ENDPOINT
# =========================================================

@app.post("/explain")
def explain(data: dict):

    smiles = data.get(
        "smiles",
        ""
    ).strip()


    if not smiles:

        return {

            "success": False,

            "error":
                "SMILES is required."

        }


    mol = Chem.MolFromSmiles(
        smiles
    )


    if mol is None:

        return {

            "success": False,

            "error":
                "Invalid SMILES string."

        }


    if gemini_client is None:

        return {

            "success": False,

            "error":
                "Gemini API is not configured."

        }


    # -----------------------------------------------------
    # COMPOUND INFORMATION
    # -----------------------------------------------------

    compound_name = get_compound_name(
        smiles
    )


    molecular_weight = Descriptors.MolWt(
        mol
    )

    logp = Descriptors.MolLogP(
        mol
    )

    hbd = Descriptors.NumHDonors(
        mol
    )

    hba = Descriptors.NumHAcceptors(
        mol
    )

    rotatable_bonds = Descriptors.NumRotatableBonds(
        mol
    )

    tpsa = Descriptors.TPSA(
        mol
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    features = np.array([[

        molecular_weight,

        logp,

        hbd,

        hba,

        rotatable_bonds,

        tpsa

    ]])


    predicted_pIC50 = model.predict(
        features
    )[0]


    estimated_ic50 = 10 ** (
        9 - predicted_pIC50
    )


    # -----------------------------------------------------
    # GEMINI PROMPT
    # -----------------------------------------------------

    prompt = f"""

You are an educational AI assistant for a
bioinformatics drug discovery project.

Explain the following computational drug activity prediction
in simple scientific language suitable for a B.Tech Bioinformatics
student.

Compound name:
{compound_name}

SMILES:
{smiles}

Molecular weight:
{molecular_weight:.2f}

LogP:
{logp:.2f}

Hydrogen bond donors:
{hbd}

Hydrogen bond acceptors:
{hba}

Rotatable bonds:
{rotatable_bonds}

TPSA:
{tpsa:.2f}

Predicted pIC50:
{predicted_pIC50:.3f}

Estimated IC50:
{estimated_ic50:.3f} nM

Explain:

1. What the compound is, if its name is known.
2. What the molecular properties mean.
3. What the predicted pIC50 means.
4. What the estimated IC50 means.
5. Whether the predicted value indicates relatively stronger
   or weaker activity within this model.
6. Important limitations of this computational prediction.

Clearly state that this is a machine-learning prediction,
not experimental evidence or a clinical recommendation.

Keep the explanation easy to understand.
"""


    # -----------------------------------------------------
    # CALL GEMINI
    # -----------------------------------------------------

    try:

        response = gemini_client.models.generate_content(

            model="gemini-2.5-flash",

            contents=prompt

        )


        return {

            "success": True,

            "explanation":
                response.text

        }


    except Exception as e:

        print(
            "Gemini error:",
            e
        )

        return {

            "success": False,

            "error":
                str(e)

        }