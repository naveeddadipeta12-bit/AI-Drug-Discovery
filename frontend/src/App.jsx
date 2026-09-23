import { useState } from "react";
import "./App.css";

function App() {
  const [smiles, setSmiles] = useState("");
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [loading, setLoading] = useState(false);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [error, setError] = useState("");

  // =====================================================
  // PREDICT DRUG ACTIVITY
  // =====================================================

  const handlePredict = async () => {
    setError("");
    setResult(null);
    setExplanation("");

    if (!smiles.trim()) {
      setError("Please enter a SMILES string.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            smiles: smiles.trim(),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        setError(
          data.error || "Prediction failed. Please try again."
        );
        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };


  // =====================================================
  // GEMINI EXPLANATION
  // =====================================================

  const handleExplanation = async () => {
    if (!result) {
      return;
    }

    setError("");
    setExplanation("");
    setExplanationLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/explain",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            smiles: result.smiles,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok || !data.success) {
        setError(
          data.error ||
            "Unable to generate AI explanation."
        );
        return;
      }

      setExplanation(data.explanation);
    } catch (err) {
      console.error(err);

      setError(
        "Unable to connect to the Gemini explanation service."
      );
    } finally {
      setExplanationLoading(false);
    }
  };


  // =====================================================
  // MAIN UI
  // =====================================================

  return (
    <div className="app">

      {/* =================================================
          HEADER
          ================================================= */}

      <header className="header">

        <div className="header-content">

          <div className="brand-badge">
            🧬 Bioinformatics + AI
          </div>

          <h1>
            AI Drug Discovery
          </h1>

          <p className="header-subtitle">
            AI-Powered Drug–Target Activity Prediction
          </p>

          <div className="header-features">

            <div className="header-feature">
              <span>🧬</span>
              <strong>Bioinformatics</strong>
            </div>

            <div className="header-feature">
              <span>🤖</span>
              <strong>AI Prediction</strong>
            </div>

            <div className="header-feature">
              <span>💊</span>
              <strong>Drug Discovery</strong>
            </div>

            <div className="header-feature">
              <span>🔬</span>
              <strong>Molecular Analysis</strong>
            </div>

          </div>

          <div className="header-tagline">
            DISCOVER • ANALYZE • PREDICT
          </div>

        </div>

      </header>


      {/* =================================================
          MAIN CONTENT
          ================================================= */}

      <main className="container">


        {/* =================================================
            INPUT CARD
            ================================================= */}

        <section className="input-card">

          <div className="section-icon">
            🧪
          </div>

          <h2>
            Drug Activity Prediction
          </h2>

          <p>
            Enter a drug's SMILES notation to analyze
            its molecular properties and predict biological activity.
          </p>


          <div className="input-label-row">

            <label htmlFor="smiles">
              SMILES String
            </label>

            <span>
              Molecular structure input
            </span>

          </div>


          <textarea
            id="smiles"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
            placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O"
          />


          <div className="example-text">
            Example compound: <strong>Aspirin</strong>
          </div>


          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={loading}
          >

            {loading ? (
              <>
                <span className="loading-dot"></span>
                Analyzing Compound...
              </>
            ) : (
              <>
                🔬 Predict Drug Activity
              </>
            )}

          </button>


          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

        </section>



        {/* =================================================
            RESULTS
            ================================================= */}

        {result && (
          <section className="results-card">

            <div className="section-icon">
              📊
            </div>

            <h2>
              Prediction Results
            </h2>

            <p className="results-subtitle">
              Computational analysis of the submitted compound
            </p>


            {/* Compound Name */}

            <div className="compound-name-box">

              <div>

                <span className="result-label">
                  COMPOUND IDENTIFIED
                </span>

                <strong>
                  {result.compound_name ||
                    "Compound name not found"}
                </strong>

              </div>

              <div className="identified-icon">
                ✓
              </div>

            </div>


            {/* Molecular Properties */}

            <h3 className="subheading">
              Molecular Properties
            </h3>


            <div className="results-grid">

              <div className="result-item">
                <span>Molecular Weight</span>
                <strong>
                  {result.molecular_weight}
                  <small> Da</small>
                </strong>
              </div>


              <div className="result-item">
                <span>LogP</span>
                <strong>
                  {result.logp}
                </strong>
              </div>


              <div className="result-item">
                <span>H-Bond Donors</span>
                <strong>
                  {result.hbd}
                </strong>
              </div>


              <div className="result-item">
                <span>H-Bond Acceptors</span>
                <strong>
                  {result.hba}
                </strong>
              </div>


              <div className="result-item">
                <span>Rotatable Bonds</span>
                <strong>
                  {result.rotatable_bonds}
                </strong>
              </div>


              <div className="result-item">
                <span>TPSA</span>
                <strong>
                  {result.tpsa}
                  <small> Å²</small>
                </strong>
              </div>

            </div>


            {/* AI Prediction */}

            <h3 className="subheading prediction-heading">
              AI Activity Prediction
            </h3>


            <div className="prediction-box">

              <div className="prediction-card">

                <div className="prediction-icon">
                  🧠
                </div>

                <span>
                  Predicted pIC50
                </span>

                <strong>
                  {result.predicted_pIC50}
                </strong>

                <small>
                  Model prediction
                </small>

              </div>


              <div className="prediction-card">

                <div className="prediction-icon">
                  🎯
                </div>

                <span>
                  Estimated IC50
                </span>

                <strong>
                  {result.estimated_ic50_nM}
                  <small> nM</small>
                </strong>

                <small>
                  Estimated concentration
                </small>

              </div>

            </div>


            {/* Gemini */}

            <button
              className="explain-button"
              onClick={handleExplanation}
              disabled={explanationLoading}
            >

              {explanationLoading
                ? "🤖 Generating AI Explanation..."
                : "🤖 Explain Results with AI"}

            </button>

          </section>
        )}



        {/* =================================================
            GEMINI EXPLANATION
            ================================================= */}

        {explanation && (
          <section className="explanation-card">

            <div className="section-icon">
              🤖
            </div>

            <h2>
              AI-Powered Explanation
            </h2>

            <p className="results-subtitle">
              Gemini-generated interpretation of the prediction
            </p>

            <div className="explanation-content">
              {explanation}
            </div>

            <div className="disclaimer">
              <strong>Important:</strong> This is a
              computational prediction generated by a
              machine-learning model. It is not experimental
              evidence or a clinical recommendation.
            </div>

          </section>
        )}

      </main>


      {/* =================================================
          FOOTER
          ================================================= */}

      <footer className="footer">

        <div className="footer-title">
          🧬 AI Drug Discovery Platform
        </div>

        <div className="footer-text">
          AI • Bioinformatics • Molecular Analysis
        </div>

      </footer>

    </div>
  );
}

export default App;