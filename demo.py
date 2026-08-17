"""
Runnable demo matching the demo script in Section 25 of the architecture
doc. Uses the dependency-free mock orchestrator so this runs with nothing
but `pip install pydantic`.

Run: python3 demo.py
"""

from services.factory import build_mock_orchestrator


def show(title: str, result):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(f"status:   {result.status.value}")
    print(f"verdict:  {result.verdict.value if result.verdict else 'N/A'}")
    if result.decision:
        print(f"heuristic_score: {result.decision.heuristic_score}")
    print(f"latency:  {result.latency_ms}ms")
    print("decision trail:")
    for i, step in enumerate(result.decision_trail, 1):
        print(f"  {i}. {step}")


def main():
    orch = build_mock_orchestrator()
    try:
        # 1. Fast path — pin the classifier to be confident via a claim whose
        #    hash happens to clear the threshold most of the time; in a real
        #    demo you'd use the real trained classifier's actual output.
        show(
            "1) Fast path (classifier-confident, no verification triggered)",
            orch.run(claim_text="Water boils at 100 degrees Celsius at sea level.",
                      request_id="demo-1"),
        )

        # 2. Verification path — force it to show retrieval + NLI + aggregation.
        show(
            "2) Verification path (low confidence -> retrieval -> NLI -> aggregation)",
            orch.run(claim_text="The city council voted to ban plastic bags starting next year",
                      request_id="demo-2", force_verification=True),
        )

        # 3. Insufficient evidence — a claim with nothing in the fixture corpus.
        show(
            "3) INSUFFICIENT_EVIDENCE (honest 'we don't know', not a guess)",
            orch.run(claim_text="qqqq nonexistent topic zzzz not in corpus",
                      request_id="demo-3", force_verification=True),
        )

        # 4. Image + fabricated caption — real photo, wrong context.
        show(
            "4) Image submitted with claim (manipulation check + reverse search + fusion)",
            orch.run(
                claim_text="This photo shows today's flooding downtown",
                request_id="demo-4",
                image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 200,  # placeholder bytes
                image_mime_type="image/jpeg",
            ),
        )
    finally:
        orch.shutdown()


if __name__ == "__main__":
    main()
