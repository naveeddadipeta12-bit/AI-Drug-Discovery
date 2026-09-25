import os
import zipfile

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors

from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    print("Gemini API key loaded successfully!")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found!")
    gemini_client = None


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Drug Discovery API",
    description="AI-powered drug activity prediction and explanation API",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://localhost:5178",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "random_forest_model.pkl"
)

MODEL_ZIP_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "random_forest_model.zip"
)


# ============================================================
# EXTRACT COMPRESSED MODEL IF NEEDED
# ============================================================

if not os.path.exists(MODEL_PATH):

    if os.path.exists(MODEL_ZIP_PATH):

        try:
            print("Compressed ML model found.")
            print("Extracting ML model...")

            with zipfile.ZipFile(
                MODEL_ZIP_PATH,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    os.path.join(
                        BASE_DIR,
                        "ml"
                    )
                )

            print("ML model extracted successfully!")

        except Exception as e:

            print(
                "Model extraction error:",
                e
            )

    else:

        print(
            "ML model ZIP file not found."
        )


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

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


# ============================================================
# REQUEST MODELS
# ============================================================

class PredictionRequest(BaseModel):

    smiles: str


class ExplanationRequest(BaseModel):

    compound_name: str
    smiles: str
    predicted_pIC50: float
    estimated_ic50_nM: float


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Drug Discovery API is running",
        "status": "success"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_drug(request: PredictionRequest):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Machine learning model is not loaded."
        )

    smiles = request.smiles.strip()

    if not smiles:

        raise HTTPException(
            status_code=400,
            detail="SMILES cannot be empty."
        )


    # --------------------------------------------------------
    # Convert SMILES to molecule
    # --------------------------------------------------------

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        raise HTTPException(
            status_code=400,
            detail="Invalid SMILES string."
        )


    # --------------------------------------------------------
    # Calculate molecular descriptors
    # --------------------------------------------------------

    molecular_weight = Descriptors.MolWt(mol)

    logp = Descriptors.MolLogP(mol)

    hbd = Descriptors.NumHDonors(mol)

    hba = Descriptors.NumHAcceptors(mol)

    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    tpsa = Descriptors.TPSA(mol)


    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    features = np.array([
        [
            molecular_weight,
            logp,
            hbd,
            hba,
            rotatable_bonds,
            tpsa
        ]
    ])


    # --------------------------------------------------------
    # Predict pIC50
    # --------------------------------------------------------

    try:

        predicted_pIC50 = model.predict(
            features
        )[0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


    # --------------------------------------------------------
    # Convert pIC50 to IC50
    # --------------------------------------------------------

    estimated_ic50_nM = 10 ** (
        9 - predicted_pIC50
    )


    # --------------------------------------------------------
    # Identify compound
    # --------------------------------------------------------

    compound_name = "Unknown compound"


    # Known test compounds

    if smiles == (
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)"
        "CN3CCN(CC3)C)NC4=NC=CC(=N4)"
        "C5=CN=CC=C5"
    ):

        compound_name = "Imatinib"


    elif smiles == (
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    ):

        compound_name = "Aspirin"


    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {

        "compound_name": compound_name,

        "smiles": smiles,

        "molecular_weight": round(
            molecular_weight,
            2
        ),

        "logp": round(
            logp,
            2
        ),

        "hbd": int(hbd),

        "hba": int(hba),

        "rotatable_bonds": int(
            rotatable_bonds
        ),

        "tpsa": round(
            tpsa,
            2
        ),

        "predicted_pIC50": round(
            float(predicted_pIC50),
            3
        ),

        "estimated_ic50_nM": round(
            float(estimated_ic50_nM),
            3
        )
    }


# ============================================================
# GEMINI EXPLANATION ENDPOINT
# ============================================================

@app.post("/explain")
def explain_prediction(
    request: ExplanationRequest
):

    if gemini_client is None:

        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured."
        )


    prompt = f"""
You are an AI assistant for a computational
drug discovery project.

Explain the following predicted drug-target
activity result in simple scientific language.

Compound:
{request.compound_name}

SMILES:
{request.smiles}

Predicted pIC50:
{request.predicted_pIC50}

Estimated IC50:
{request.estimated_ic50_nM} nM

Explain:

1. What pIC50 means.
2. What the predicted value indicates.
3. What IC50 means.
4. What the molecular properties may indicate.
5. Important limitations of this prediction.

Important:
This is a computational prediction from a machine
learning model. It is NOT experimental evidence,
a clinical recommendation, or proof of drug efficacy.

Clearly mention that experimental validation is required.
"""


    try:

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        explanation = response.text

        return {
            "explanation": explanation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Gemini explanation failed: {str(e)}"
        )


# ============================================================
# END OF FILE
# ============================================================