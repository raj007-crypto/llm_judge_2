import json
import os
import time

from interactive_eval import (
    JUDGE_MODEL,
    ask_backend,
    judge_answer,
    judge_pass,
    check_answer_in_context,
    check_answer_correctness,
)

BACKEND_URL = "http://localhost:8000/query"
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "invoice_test_cases.json")

ADDITIONAL_QUESTIONS = [
    "Which date is the time delivery on?",
    "What is the bill number?",
    "What are the bank charges?",
    "Where is the shipment from?",
    "Where is the shipment going?",
    "What is the vessel name?",
    "What is the consignee address?",
    "What is the HS code?",
    "What is the container number?",
    "What are the shipping marks?",
]


def run_test(tc, idx, total):
    question = tc.get("question", "")
    expected = tc.get("expected", "")

    if not question:
        return {"pass": False, "question": "EMPTY", "answer": "N/A", "error": "No question"}

    print(f"\n[{idx}/{total}] {question}")

    try:
        result = ask_backend(question)
        answer = result.get("answer", "")
        contexts = result.get("source_documents", [])
    except Exception as e:
        print(f"  Backend error: {e}")
        return {"pass": False, "question": question, "answer": "ERROR", "error": str(e)}

    correct_answer = check_answer_correctness(expected, answer) if expected else False
    in_context = check_answer_in_context(answer, contexts)

    combined_context = "\n".join(contexts) if contexts else ""

    if expected:
        score, reason = judge_answer(question, answer, expected)
    else:
        score, reason = judge_answer(question, answer, "", combined_context)
    
    overall_pass = judge_pass(score, expected if expected else "CONTEXT_BASED")

    status = "PASS" if overall_pass else "FAIL"
    print(f"  Answer   : {answer}")
    if expected:
        print(f"  Expected : {expected}")
    score_str = f"{score}/5" if score is not None else "N/A"
    print(f"  Judge    : {score_str}")
    print(f"  Correct  : {'YES' if correct_answer else 'NO'}")
    print(f"  Verdict  : {status}")

    return {
        "pass": overall_pass,
        "correct_answer": correct_answer,
        "in_context": in_context,
        "question": question,
        "answer": answer,
        "expected": expected,
        "rubric": score,
        "score": score,
        "reason": reason,
    }


def main():
    try:
        with open(TEST_CASES_PATH) as f:
            test_cases = json.load(f)
            if not isinstance(test_cases, list):
                test_cases = []
    except (FileNotFoundError, json.JSONDecodeError):
        test_cases = []

    for q in ADDITIONAL_QUESTIONS:
        test_cases.append({"question": q, "expected": "", "field": "additional"})

    print("=" * 70)
    print("  Invoice RAG + LLM Judge — Full Evaluation Suite")
    print("=" * 70)
    print(f"  Backend  : {BACKEND_URL}")
    print(f"  Judge    : {JUDGE_MODEL}")
    print(f"  Tests    : {len(test_cases)} invoice Q&A pairs")
    print("  Pass/Fail based on: LLM judge evaluation (score >= 3 = PASS)")
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
