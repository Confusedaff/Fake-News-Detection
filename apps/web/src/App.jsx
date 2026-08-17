import { useState } from "react";
import "./App.css";
import { mockResult } from "./services/mockData";

function App() {
  const [claim, setClaim] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const analyzeClaim = () => {
    if (!claim.trim()) {
      alert("Please enter a claim.");
      return;
    }

    setLoading(true);
    setResult(null);

    setTimeout(() => {
      setResult(mockResult);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="app">

      <header>
        <h1>Fake News Verifier</h1>
        <p>
          Verify claims using classification, evidence and verification.
        </p>
      </header>

      <main>

        {/* CLAIM INPUT */}
        <section className="card">

          <h2>Enter Claim</h2>

          <textarea
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            placeholder="Enter a claim you want to verify..."
            rows="6"
          />

          <h3>Optional Image</h3>

          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => setImage(e.target.files[0])}
          />

          {image && (
            <p>
              Selected image: {image.name}
            </p>
          )}

          <button onClick={analyzeClaim}>
            Analyze Claim
          </button>

        </section>

        {/* LOADING */}
        {loading && (
          <section className="card">
            <h2>Analyzing...</h2>

            <p>✓ Claim received</p>
            <p>⏳ Running classifier...</p>
            <p>○ Checking evidence...</p>
            <p>○ Generating verdict...</p>
          </section>
        )}

        {/* RESULT */}
        {result && (
          <section>

            {/* VERDICT */}
            <div className="card">
              <h2>Verdict</h2>

              <div className="verdict">
                {result.verdict}
              </div>

              <p>
                Heuristic Score:{" "}
                {result.decision.heuristic_score}
              </p>
            </div>

            {/* EVIDENCE */}
            <div className="card">

              <h2>Evidence</h2>

              {result.decision.contradicting_evidence.map(
                (evidence, index) => (
                  <div className="evidence" key={index}>

                    <h3>{evidence.source}</h3>

                    <p>
                      Evidence contradicts the submitted claim.
                    </p>

                    <p>
                      <strong>Result:</strong>{" "}
                      {evidence.nli_label}
                    </p>

                    <p>
                      <strong>Retrieval Score:</strong>{" "}
                      {evidence.retrieval_score}
                    </p>

                    <p>
                      <strong>Source Quality:</strong>{" "}
                      {evidence.source_quality}
                    </p>

                    <p>
                      <strong>NLI Confidence:</strong>{" "}
                      {evidence.nli_confidence}
                    </p>

                  </div>
                )
              )}

            </div>

            {/* DECISION TRAIL */}
            <div className="card">

              <h2>Decision Trail</h2>

              {result.decision_trail.map(
                (step, index) => (
                  <p key={index}>
                    ✓ {step}
                  </p>
                )
              )}

            </div>

          </section>
        )}

      </main>

    </div>
  );
}

export default App;