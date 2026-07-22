import json
import os
import re
import requests

BACKEND_URL = "http://localhost:8000/query"
JUDGE_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "invoice_test_cases.json")

JUDGE_PROMPT_TEMPLATE = """Compare these two values and decide if they match.

Expected: "{expected}"
Actual: "{actual}"

Rules:
- If the values are identical, answer: 5
- If one value is contained within the other, answer: 5
- If the values represent the same thing (e.g. "usd" and "united states dollar"), answer: 5
- If the values are different numbers, answer: 1
- If the values are completely different, answer: 1

Answer only a number (1-5):"""

JUDGE_PROMPT_NOT_PRESENT = """You are a strict judge. The expected answer is NOT_PRESENT, meaning this information is NOT in the document.

Actual: "{actual}"

Rules:
- If the actual output says it doesn't know, cannot find, or the information is not available → answer: 5
- If the actual output provides ANY specific name, number, or value → it is hallucinating → answer: 1

Your answer (only a number):"""

JUDGE_PROMPT_CONTEXT = """You are a strict judge evaluating whether an answer is correct based on the document context.

Question: "{question}"
Context: "{context}"
Answer: "{actual}"

Rules:
- Check if the answer is supported by the context
- Check if the answer is complete and relevant to the question
- If the answer is fully supported by context and answers the question → answer: 5
- If the answer is partially supported or incomplete → answer: 3
- If the answer is not supported by context or irrelevant → answer: 1
- If the answer says "I don't know" but context has the answer → answer: 1
- If the answer says "I don't know" and context doesn't have the answer → answer: 5

Your answer (only a number):"""

_TEST_CASES = {}
if os.path.exists(TEST_CASES_PATH):
    try:
        with open(TEST_CASES_PATH) as _f:
            data = json.load(_f)
            if isinstance(data, list):
                for _tc in data:
                    if isinstance(_tc, dict) and "question" in _tc and "expected" in _tc:
                        _TEST_CASES[_tc["question"].lower().strip()] = _tc["expected"]
    except (json.JSONDecodeError, KeyError):
        pass


def normalize_for_judge(s: str) -> str:
    """Normalize value for judge comparison - strip symbols, units, extra text."""
    s = s.lower().strip()
    s = re.sub(r"[$,\u20b9\u00a3\u20ac]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
    return s


def judge_answer(question: str, answer: str, expected: str, context: str = "") -> tuple[int, str]:
    if expected.upper() == "NOT_PRESENT":
        prompt = JUDGE_PROMPT_NOT_PRESENT.format(actual=answer)
    elif expected:
        norm_expected = normalize_for_judge(expected)
        norm_answer = normalize_for_judge(answer)
        prompt = JUDGE_PROMPT_TEMPLATE.format(expected=norm_expected, actual=norm_answer)
    else:
        prompt = JUDGE_PROMPT_CONTEXT.format(question=question, context=context[:1000], actual=answer)
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        score = None
        m = re.search(r"\b([1-5])\b", raw)
        if m:
            score = int(m.group(1))

        return score, raw
    except Exception as e:
        return None, str(e)


def judge_pass(score: int | None, expected: str) -> bool:
    """Determine pass/fail based on judge score."""
    if score is None:
        return False
    if expected.upper() == "NOT_PRESENT":
        return score >= 4
    if expected == "CONTEXT_BASED":
        return score >= 3
    return score >= 3


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def check_answer_correctness(expected: str, answer: str) -> bool:
    if expected.upper() == "NOT_PRESENT":
        ans = answer.lower()
        return any(phrase in ans for phrase in [
            "don't know", "not present", "not mentioned",
            "not found", "not available", "no information",
            "not available in", "cannot find", "no data",
            "does not mention", "not specified",
        ])
    exp = normalize(expected)
    ans = normalize(answer)

    if exp in ans:
        return True

    # strip common prefixes like "messrs.:", "m/s.", "the"
    exp_stripped = re.sub(r"^(messrs\.?:?\s*|m/s\.?\s*|the\s+|mr\.?\s*|mrs\.?\s*|ms\.?\s*|dr\.?\s*)", "", exp)
    if exp_stripped and exp_stripped in ans:
        return True

    # check if answer is contained in expected (reverse direction)
    ans_stripped = re.sub(r"^(messrs\.?:?\s*|m/s\.?\s*|the\s+|mr\.?\s*|mrs\.?\s*|ms\.?\s*|dr\.?\s*)", "", ans)
    if ans_stripped and ans_stripped in exp:
        return True
    if ans_stripped and ans_stripped in exp_stripped:
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

    combined_context = "\n".join(contexts)

    if expected:
        print("\n[2/3] Running LLM judge (test case mode) ....")
        score, reason = judge_answer(question, answer, expected)
    else:
        print("\n[2/3] Running LLM judge (context-based mode) ....")
        score, reason = judge_answer(question, answer, "", combined_context)
    
    overall = judge_pass(score, expected if expected else "CONTEXT_BASED")
    score_str = f"{score}/5" if score is not None else "N/A"
    print(f"  Judge     : {score_str}")
    print(f"  Reason    : {reason[:200]}")

    print(f"\n  Overall   : {'PASS' if overall else 'FAIL'}")
    print(f"  (correct={'YES' if correct else 'NO'}, in_context={'YES' if in_ctx else 'NO'})")
    print(f"{'─' * 60}\n")

    return overall


def main():
    print("Interactive RAG + LLM Judge Demo (Invoice Domain)")
    print("Backend:  ", BACKEND_URL)
    print("Judge:    ", JUDGE_MODEL)
    print("Tests:    ", len(_TEST_CASES), "questions in invoice_test_cases.json")
    print("Pass/Fail is based on LLM judge evaluation (score >= 3 = PASS).\n")

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
