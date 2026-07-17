import json
import os
import re
import requests
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import OllamaModel

BACKEND_URL = "http://localhost:8000/query"
JUDGE_MODEL = "llama3.2:3b"
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "invoice_test_cases.json")

judge_llm = OllamaModel(
    model=JUDGE_MODEL,
    base_url="http://localhost:11434",
    temperature=0,
)

EVAL_STEPS = [
    "Read the retrieval context and list every fact, number, name, and code mentioned.",
    "Read the actual output and identify what specific information it provides.",
    "Check if the information in the actual output is supported by at least one fact in the retrieval context.",
    "Ignore formatting differences: currency symbols ($), letter case, commas in numbers, and trailing zeros do not matter.",
    "A short, direct answer that contains the correct value is a perfect answer. Do NOT penalize for being concise.",
    "If the actual output matches a fact from the context, score 5. If partially correct, score 3. If unsupported, score 1.",
]

from deepeval.metrics.g_eval import Rubric

invoice_geval = GEval(
    name="Invoice Faithfulness",
    criteria="Whether the actual output matches the expected output, using the retrieval context as evidence.",
    evaluation_steps=EVAL_STEPS,
    rubric=[
        Rubric(score_range=(1, 1), expected_outcome="Answer is unsupported by context or clearly wrong."),
        Rubric(score_range=(2, 2), expected_outcome="Answer is partially correct or vaguely related."),
        Rubric(score_range=(3, 3), expected_outcome="Answer is mostly correct with minor issues."),
        Rubric(score_range=(4, 4), expected_outcome="Answer is correct and supported, minor phrasing differences only."),
        Rubric(score_range=(5, 5), expected_outcome="Answer is fully correct and directly supported by context, even if concise."),
    ],
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    model=judge_llm,
    threshold=0.6,
)

_TEST_CASES = {}
if os.path.exists(TEST_CASES_PATH):
    with open(TEST_CASES_PATH) as _f:
        for _tc in json.load(_f):
            _TEST_CASES[_tc["question"].lower().strip()] = _tc["expected"]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def check_answer_correctness(expected: str, answer: str) -> bool:
    exp = normalize(expected)
    ans = normalize(answer)
    if exp in ans:
        return True
    try:
        return abs(float(exp) - float(ans)) < 0.01
    except ValueError:
        return False


def extract_key_value(answer: str) -> list[str]:
    val = re.sub(r"(?i)^the\s+.*?(?:is|was|are|were)[:\s]+", "", answer).strip()
    val = val.rstrip(".")
    if val:
        return [normalize(val)]
    tokens = normalize(answer).split()
    return [t for t in tokens if len(t) > 2]


def check_answer_in_context(answer: str, contexts: list[str]) -> bool:
    keys = extract_key_value(answer)
    combined_ctx = " ".join(normalize(c) for c in contexts)
    for k in keys:
        if k in combined_ctx:
            return True
    return False


def ask_backend(question: str) -> dict:
    resp = requests.post(BACKEND_URL, json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def run_eval(question: str, expected: str = ""):
    if not expected:
        expected = _TEST_CASES.get(question.lower().strip(), "")

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    print("\n[1/3] Retrieving answer from backend...")
    result = ask_backend(question)
    answer = result["answer"]
    contexts = result["source_documents"]

    print(f"\nAnswer    : {answer}")
    if expected:
        print(f"Expected  : {expected}")
        correct = check_answer_correctness(expected, answer)
        print(f"Correct   : {'YES' if correct else 'NO'}")
    else:
        correct = False

    in_ctx = check_answer_in_context(answer, contexts)
    print(f"In context: {'YES' if in_ctx else 'NO'}")

    print(f"\nRetrieved {len(contexts)} context chunk(s):")
    for i, ctx in enumerate(contexts, 1):
        preview = ctx[:120].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    overall = correct if expected else in_ctx

    if expected:
        print("\n[2/3] Running GEval judge (reference only, does not affect pass/fail)...")
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected,
            retrieval_context=contexts,
        )
        invoice_geval.measure(test_case)
        score = invoice_geval.score
        rubric = round(score * 5) if score is not None else "?"
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"  Judge     : {rubric}/5  ({score_str})")
        print(f"  Reason    : {invoice_geval.reason}")

    print(f"\n  Overall   : {'PASS' if overall else 'FAIL'}")
    print(f"  (correct={'YES' if correct else 'NO'}, in_context={'YES' if in_ctx else 'NO'})")
    print(f"{'─' * 60}\n")

    return overall


def main():
    print("Interactive RAG + LLM Judge Demo (Invoice Domain)")
    print("Backend:  ", BACKEND_URL)
    print("Judge:    ", JUDGE_MODEL)
    print("Tests:    ", len(_TEST_CASES), "questions in invoice_test_cases.json")
    print("Pass/Fail is based on answer correctness, NOT the LLM judge.\n")

    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        try:
            run_eval(q)
        except requests.ConnectionError as e:
            print(f"\n[ERROR] Connection failed — is the server running?\n  {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
