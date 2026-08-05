import io
import json
import os
import re
import sys
import contextlib
import requests

from difflib import SequenceMatcher
from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import BaseMetric, FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, CacheConfig

BACKEND_URL = "http://localhost:8000/query"
JUDGE_MODEL = "llama3.1:8b"
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.5
TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "invoice_test_cases.json")


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        self.chat_model = ChatOllama(model=model, temperature=0, format="json")

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        raw = self.chat_model.invoke(prompt).content
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                json.loads(raw[start : end + 1])
                return raw[start : end + 1]
            except json.JSONDecodeError:
                pass
        return raw

    async def a_generate(self, prompt: str) -> str:
        res = await self.chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self.model_name


judge = OllamaJudge()


class _JudgeBinaryMetric(BaseMetric):
    _key = "verdict"
    _yes = "yes"

    def __init__(self, threshold=1.0, model=None, include_reason=True, async_mode=False):
        self.threshold = threshold
        self.model = model if model is not None else judge
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.error = None

    @property
    def __name__(self):
        return "Judge Binary Metric"

    def _build_prompt(self, test_case):
        raise NotImplementedError

    def measure(self, test_case, *args, **kwargs):
        data = judge_json(self._build_prompt(test_case))
        raw = str(data.get(self._key, "")).strip().lower()
        self.score = 1.0 if raw == self._yes else 0.0
        self.reason = str(data.get("reason", "") or "") if self.include_reason else None
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case, *args, **kwargs)


class ContextClarityMetric(_JudgeBinaryMetric):
    _key = "clear"
    _yes = "yes"

    @property
    def __name__(self):
        return "Context Clarity"

    def _build_prompt(self, test_case):
        question = test_case.input
        ctx = clean_context(test_case.retrieval_context or [])
        return (
            "You are evaluating a RAG system over invoice documents.\n"
            f'User question: "{question}"\n\n'
            "Retrieved context:\n"
            f"{ctx}\n\n"
            "Does the specific field this question asks about have a clear, readable value in the context?\n"
            "Field labels in OCR are often misspelled - match the asked field loosely "
            "(e.g. 'Addtional' = 'Additional', 'referrence'/'Refrence' = 'Reference').\n"
            "- Answer YES if the asked field has a value next to it in the context. Example: "
            "question 'What is the additional reference number?' and the context shows "
            "'Addtional Reference No. CO25/533004' -> YES, the value CO25/533004 belongs to the asked field.\n"
            "- Answer NO only if the asked field is followed by another field label instead of a value "
            "(e.g. 'Our Reference No Addtional Reference No. CO25/533004' with question 'our reference number' "
            "- 'Our Reference No' has no value of its own, the value CO25/533004 belongs to 'Additional "
            "Reference No'), if the field's text is garbled or missing, or if there are multiple conflicting "
            "values for the asked field.\n"
            "- If the asked field's value is clearly readable, answer YES even if a DIFFERENT value "
            "in the same line carries an [UNCLEAR:...] tag (e.g. in 'Currency & Amount USD "
            "[UNCLEAR:3,423.44]' the currency USD is clear even though the amount is not).\n"
            'Return only valid JSON: {"clear": "yes" or "no", "reason": "<one sentence>"}\nJSON:'
        )


class AnswerValidityMetric(_JudgeBinaryMetric):
    _key = "valid"
    _yes = "yes"

    @property
    def __name__(self):
        return "Answer Validity"

    def _build_prompt(self, test_case):
        question = test_case.input
        answer = test_case.actual_output
        ctx = clean_context(test_case.retrieval_context or [])
        return (
            "You are evaluating a RAG system over invoice documents.\n"
            f'Question: "{question}"\n'
            f'Answer given: "{answer}"\n\n'
            "Retrieved context:\n"
            f"{ctx}\n\n"
            "Is the answer correct? The value in an answer belongs to the field label immediately before "
            "it in the context. Field labels in OCR may be misspelled - match loosely "
            "('Addtional' = 'Additional', 'referrence'/'Refrence' = 'Reference').\n"
            "- Answer YES if the label immediately preceding the answer value is the field the question "
            "asks about. Example: question 'What is the additional reference number?', answer 'CO25/533004', "
            "and the context shows 'Addtional Reference No. CO25/533004' -> the value follows the asked "
            "field -> YES.\n"
            "- Answer NO only if the value follows a label for a DIFFERENT field than the one asked "
            "(e.g. the question asks for 'Our Reference No' but the value follows 'Additional Reference "
            "No'), or if the value is not present in the context.\n"
            'Return only valid JSON: {"valid": "yes" or "no", "reason": "<one sentence>"}\nJSON:'
        )


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

    exp_stripped = re.sub(r"^(messrs\.?:?\s*|m/s\.?\s*|the\s+|mr\.?\s*|mrs\.?\s*|ms\.?\s*|dr\.?\s*)", "", exp)
    if exp_stripped and exp_stripped in ans:
        return True

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
    val = re.sub(
        r"(?i)^(?:the|our|your|their|this)\s+.*?(?:is|was|are|were)[:\s]+",
        "", answer,
    ).strip()
    val = val.rstrip(".")
    if val:
        return [normalize(val)]
    tokens = normalize(answer).split()
    return [t for t in tokens if len(t) > 2]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in [
        "don't know", "not present", "not mentioned", "not found", "not available",
        "no information", "cannot", "unable", "not clearly visible",
        "could not be extracted", "no data", "not specified", "not visible",
    ])


def answer_grounded(answer: str, contexts: list[str]) -> bool:
    if is_refusal(answer):
        return True
    keys = extract_key_value(answer)
    combined = " ".join(normalize(c) for c in contexts)
    return any(k in combined for k in keys)


def check_answer_in_context(answer: str, contexts: list[str]) -> bool:
    keys = extract_key_value(answer)
    combined_ctx = " ".join(normalize(c) for c in contexts)
    for k in keys:
        if k in combined_ctx:
            return True
    return False


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fuzzy_contains(text_norm: str, small_norm: str, thresh: float = 0.85) -> bool:
    if small_norm in text_norm:
        return True
    if len(small_norm) < 6 or len(text_norm) < len(small_norm):
        return False
    for i in range(len(text_norm) - len(small_norm) + 1):
        if SequenceMatcher(None, small_norm, text_norm[i : i + len(small_norm)]).ratio() >= thresh:
            return True
    return False


def asked_field_labels(question: str) -> list[str]:
    q = question.lower()
    labels = []
    if "reference" in q or "referrence" in q or re.search(r"\bref\b", q):
        if "additional" in q:
            labels.append("additional reference")
        if "our reference" in q:
            labels.append("our reference")
        if not labels:
            labels.append("reference no")
    if "currency" in q:
        labels.append("currency")
    if "beneficiary" in q:
        labels.append("beneficiary")
    if "date" in q:
        labels.append("date")
    if "amount" in q or "total" in q or "invoice" in q:
        labels.append("amount")
    return labels


def answer_matches_asked_field(question: str, answer: str, contexts: list[str]) -> bool | None:
    keys = extract_key_value(answer)
    if not keys:
        return None
    asked = asked_field_labels(question)
    if not asked:
        return None
    combined = " ".join(contexts)
    lower = combined.lower()
    for k in keys:
        idx = lower.find(k)
        if idx == -1:
            continue
        candidates = []
        value_line_start = combined.rfind("\n", 0, idx)
        if value_line_start == -1:
            candidates.append(_norm_text(combined[:idx]))
        else:
            candidates.append(_norm_text(combined[value_line_start + 1 : idx]))
            prev_start = combined.rfind("\n", 0, value_line_start)
            if prev_start != -1:
                candidates.append(_norm_text(combined[prev_start + 1 : value_line_start]))
        for c in candidates:
            if not c:
                continue
            for a in asked:
                if _fuzzy_contains(c, _norm_text(a)):
                    return True
        return False
    return None


def clean_context(contexts):
    text = "\n".join(c.strip() for c in contexts)
    text = re.sub(r"\[UNCLEAR:[^\]]*\]", "[?]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ask_backend(question: str) -> dict:
    resp = requests.post(BACKEND_URL, json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def judge_json(prompt: str) -> dict:
    raw = judge.generate(prompt)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def run_deepeval_eval(question: str, answer: str, contexts: list[str]):
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
    os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=contexts,
    )
    metrics = [
        FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge,
                           include_reason=True, async_mode=False),
        AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge,
                              include_reason=True, async_mode=False),
        ContextClarityMetric(threshold=1.0, model=judge, include_reason=True, async_mode=False),
        AnswerValidityMetric(threshold=1.0, model=judge, include_reason=True, async_mode=False),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = evaluate(
            [test_case],
            metrics,
            async_config=AsyncConfig(run_async=False),
            display_config=DisplayConfig(show_indicator=False, print_results=False,
                                         inspect_after_run=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
    return result.test_results[0]


def deep_eval_reason(metrics_data, overall: bool) -> str:
    if overall:
        parts = [f"{md.name} passed (score={md.score})" for md in metrics_data]
        return "All DeepEval metrics passed: " + "; ".join(parts) + "."
    fails = [
        f"{md.name} failed (score={md.score})" + (f" - {md.reason}" if md.reason else "")
        for md in metrics_data if not md.success
    ]
    passes = [f"{md.name} passed (score={md.score})" for md in metrics_data if md.success]
    s = "DeepEval framework failed: " + "; ".join(fails) + "."
    if passes:
        s += " Passing: " + "; ".join(passes) + "."
    return s


def deterministic_diagnostics(question: str, answer: str, contexts: list[str]) -> list[str]:
    lines = [
        f"diagnostic - refusal={is_refusal(answer)} grounded={answer_grounded(answer, contexts)}"
    ]
    asked = asked_field_labels(question)
    if asked:
        lines.append(f"diagnostic - asked field labels: {asked}")
    match = answer_matches_asked_field(question, answer, contexts)
    if match is not None:
        lines.append(f"diagnostic - answer value belongs to asked field: {match}")
    return lines


def run_eval(question: str):
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    print("\n[1/2] Retrieving answer from backend ...")
    result = ask_backend(question)
    answer = result["answer"]
    contexts = result["source_documents"]
    print(f"  Answer    : {answer}")
    print(f"  Retrieved : {len(contexts)} context chunk(s)")

    print(f"\n[2/2] Running DeepEval framework (judge: {JUDGE_MODEL}) ...")
    test_result = run_deepeval_eval(question, answer, contexts)
    metrics_data = test_result.metrics_data or []

    print("\n  DeepEval metric results:")
    for md in metrics_data:
        print(f"    - {md.name:<18} score={md.score:<5} {'PASS' if md.success else 'FAIL'}")
        if md.reason:
            print(f"        reason: {md.reason}")
    for line in deterministic_diagnostics(question, answer, contexts):
        print("    " + line)

    overall = bool(test_result.success)
    print(f"\nScore: {'PASS' if overall else 'FAIL'}")
    print(f"Reason: {deep_eval_reason(metrics_data, overall)}")
    print(f"{'-' * 60}\n")

    return overall


def run_test_suite():
    cases = []
    if os.path.exists(TEST_CASES_PATH):
        try:
            with open(TEST_CASES_PATH, encoding="utf-8") as _f:
                data = json.load(_f)
            if isinstance(data, list):
                cases = [tc for tc in data
                         if isinstance(tc, dict) and tc.get("question") and tc.get("expected")]
        except (json.JSONDecodeError, OSError) as e:
            print("Failed to load test cases:", e)
    if not cases:
        print("No test cases found in", TEST_CASES_PATH)
        return
    print(f"Running {len(cases)} golden test cases (deterministic scoring, no LLM judge) ...\n")
    ok = 0
    for tc in cases:
        q, exp = tc["question"], tc["expected"]
        try:
            answer = ask_backend(q)["answer"]
            passed = check_answer_correctness(exp, answer)
            ok += int(passed)
            print(f"  [{'PASS' if passed else 'FAIL'}] {q}")
            print(f"        expected: {exp}")
            print(f"        answer  : {answer[:160]}")
        except Exception as e:
            print(f"  [ERROR] {q}: {e}")
    print(f"\nAccuracy: {ok}/{len(cases)} = {100.0 * ok / len(cases):.1f}%")


def main():
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if "--test" in sys.argv:
        run_test_suite()
        return

    print("Interactive RAG + DeepEval Demo (Invoice Domain)")
    print("Backend:  ", BACKEND_URL)
    print("Judge:    ", JUDGE_MODEL, "(DeepEval)")
    print("Tests:    ", len(_TEST_CASES), "questions in invoice_test_cases.json")
    print("Score     : DeepEval framework verdict only "
          "(Faithfulness + Answer Relevancy + Context Clarity + Answer Validity)\n")

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
