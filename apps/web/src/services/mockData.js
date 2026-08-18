export const mockResult = {
  claim_id: "demo-001",
  request_id: "request-001",

  status: "FINALIZED",

  verdict: "LIKELY_FALSE",

  decision: {
    verdict: "LIKELY_FALSE",

    heuristic_score: 0.23,

    calibrated_probability: null,

    support_mass: 0.15,

    contradict_mass: 0.82,

    supporting_evidence: [],

    contradicting_evidence: [
      {
        document_id: "doc-001",
        chunk_id: "chunk-001",
        source: "Example News Source",
        source_quality: 0.95,
        retrieval_score: 0.88,
        nli_label: "CONTRADICTS",
        nli_confidence: 0.92
      }
    ]
  },

  classifier: {
    label: "potentially_false",
    raw_confidence: 0.76,
    calibrated_probability: 0.31,
    ood_signal: 0.12,
    model_version: "classifier-demo-v1"
  },

  image_analysis: null,

  fusion: null,

  decision_trail: [
    "Claim received",
    "Classifier executed",
    "Verification required",
    "Evidence retrieved",
    "Evidence verified",
    "Evidence aggregated",
    "Final verdict generated"
  ],

  corpus_version: "demo-corpus-v1",

  latency_ms: 1200
};