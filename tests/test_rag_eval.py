import json
import os
import time

from deepeval.test_case import LLMTestCase
from interactive_eval import (
    JUDGE_MODEL,
    ask_backend,
    invoice_geval,
    check_answer_in_context,
    check_answer_correctness,
)

BACKEND_URL = "http://localhost:8000/query"
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "invoice_test_cases.json")

metric = invoice_geval


def run_test(tc, idx, total):
    question = tc["question"]
    expected = tc["expected"]

    print(f"\n[{idx}/{total}] {question}")

    try:
        result = ask_backend(question)
        answer = result["answer"]
        contexts = result["source_documents"]
    except Exception as e:
        print(f"  Backend error: {e}")
        return {"pass": False, "question": question, "answer": "ERROR", "error": str(e)}

    correct_answer = check_answer_correctness(expected, answer)
    in_context = check_answer_in_context(answer, contexts)
    if expected.upper() == "NOT_PRESENT":
        in_context = not in_context

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        retrieval_context=contexts,
    )

    metric.measure(test_case)
    score = metric.score
    rubric = round(score * 5) if score is not None else "?"

    overall_pass = correct_answer or in_context

    status = "PASS" if overall_pass else "FAIL"
    print(f"  Answer   : {answer}")
    print(f"  Expected : {expected}")
    score_str = f"{score:.2f}" if score is not None else "N/A"
    print(f"  Judge    : {rubric}/5 ({score_str})  (reference only)")
    print(f"  Correct  : {'YES' if correct_answer else 'NO'}")
    print(f"  Verdict  : {status}")

    return {
        "pass": overall_pass,
        "correct_answer": correct_answer,
        "in_context": in_context,
        "question": question,
        "answer": answer,
        "expected": expected,
        "rubric": rubric,
        "score": score,
        "reason": metric.reason,
    }


def main():
    with open(TEST_CASES_PATH) as f:
        test_cases = json.load(f)

    print("=" * 70)
    print("  Invoice RAG + LLM Judge — Full Evaluation Suite")
    print("=" * 70)
    print(f"  Backend  : {BACKEND_URL}")
    print(f"  Judge    : {JUDGE_MODEL} (reference only)")
    print(f"  Tests    : {len(test_cases)} invoice Q&A pairs")
    print("  Pass/Fail based on: answer correctness OR context match")
    print("=" * 70)

    results = []
    start = time.time()

    for i, tc in enumerate(test_cases, 1):
        r = run_test(tc, i, len(test_cases))
        results.append(r)
        time.sleep(0.5)

    elapsed = time.time() - start

    passed = sum(1 for r in results if r["pass"])
    failed = sum(1 for r in results if not r["pass"])
    correct_answers = sum(1 for r in results if r.get("correct_answer"))

    print("\n")
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total tests        : {len(results)}")
    print(f"  Correct answers    : {correct_answers}/{len(results)}")
    print(f"  Overall PASS       : {passed}/{len(results)}")
    print(f"  Overall FAIL       : {failed}/{len(results)}")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print("=" * 70)

    if failed > 0:
        print("\n  FAILED CASES:")
        print("-" * 70)
        for r in results:
            if not r["pass"]:
                print(f"  Q: {r['question']}")
                print(f"     A: {r.get('answer', 'N/A')}")
                print(f"     Expected: {r.get('expected', 'N/A')}")
                print(f"     Judge: {r.get('rubric', '?')}/5 (reference)")
                reason_short = (r.get("reason", "N/A") or "")[:150]
                print(f"     Reason: {reason_short}...")
                print()

    print("=" * 70)
    verdict = "ALL PASS" if failed == 0 else f"{failed} FAILED"
    print(f"  RESULT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
