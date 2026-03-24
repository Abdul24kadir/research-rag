# eval/evaluate.py

import sys
import os
import json
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from retriever import load_vectorstore, build_qa_chain

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "false"


# Thresholds — CI will fail if scores drop below these

KEYWORD_OVERLAP_THRESHOLD = 0.3   # 30% of answer words must appear in context
MIN_ANSWER_LENGTH = 20            # answer must be at least 20 characters
MAX_REFUSAL_RATE = 0.3            # at most 30% of answers can be refusals


def load_golden_set(path: str):
    with open(path, "r") as f:
        return json.load(f)


def compute_keyword_overlap(answer: str, contexts: list) -> float:
    """
    Measures how many words in the answer
    also appear in the retrieved context chunks.
    
    High overlap = answer is grounded in retrieved docs
    Low overlap  = answer might be hallucinated
    
    This is a simple proxy for RAGAS faithfulness.
    """
    # combine all context chunks into one string
    full_context = " ".join(contexts).lower()

    # get unique meaningful words from answer
    # filter out short common words like "the", "is", "a"
    answer_words = set(
        word.strip(".,?!").lower()
        for word in answer.split()
        if len(word) > 3
    )

    if not answer_words:
        return 0.0

    # count how many answer words appear in context
    matched = sum(1 for word in answer_words if word in full_context)

    return matched / len(answer_words)


def is_refusal(answer: str) -> bool:
    """
    Checks if the model refused to answer.
    A refusal means the model said it couldn't find
    the answer in the document — which is correct behavior
    when context doesn't support the question,
    but too many refusals means retrieval is failing.
    """
    refusal_phrases = [
        "i cannot find",
        "not in the provided",
        "not mentioned",
        "cannot answer",
        "no information"
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in refusal_phrases)


def run_evaluation():

    print("Loading golden evaluation set...")
    golden_path = os.path.join(os.path.dirname(__file__), "golden_set.json")
    golden_set = load_golden_set(golden_path)
    print(f"Loaded {len(golden_set)} evaluation questions")

    print("\nLoading RAG pipeline...")
    vectorstore = load_vectorstore()
    chain, retriever = build_qa_chain(vectorstore)

    
    # Run each question and collect metrics
    
    print("\nRunning questions through RAG pipeline...")

    results = []

    for i, item in enumerate(golden_set):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1}/{len(golden_set)}] {question[:60]}...")

        # get answer
        answer = chain.invoke(question)

        # get retrieved chunks
        source_docs = retriever.invoke(question)
        context_texts = [doc.page_content for doc in source_docs]

        # compute metrics for this question
        overlap = compute_keyword_overlap(answer, context_texts)
        refused = is_refusal(answer)
        too_short = len(answer.strip()) < MIN_ANSWER_LENGTH

        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "keyword_overlap": overlap,
            "refused": refused,
            "too_short": too_short
        })

        # print per-question result
        status = "PASS" if (overlap >= KEYWORD_OVERLAP_THRESHOLD and not too_short) else "FAIL"
        print(f"         overlap={overlap:.2f} | refused={refused} | too_short={too_short} | {status}")

    
    # Compute overall scores
    
    total = len(results)
    avg_overlap = sum(r["keyword_overlap"] for r in results) / total
    refusal_rate = sum(1 for r in results if r["refused"]) / total
    too_short_count = sum(1 for r in results if r["too_short"])

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total questions:     {total}")
    print(f"Avg keyword overlap: {avg_overlap:.3f}  (threshold: {KEYWORD_OVERLAP_THRESHOLD})")
    print(f"Refusal rate:        {refusal_rate:.3f}  (max allowed: {MAX_REFUSAL_RATE})")
    print(f"Too short answers:   {too_short_count}/{total}")

    

    # Pass / Fail — exit code is what CI reads
    
    print("\n" + "=" * 50)
    passed = True

    if avg_overlap < KEYWORD_OVERLAP_THRESHOLD:
        print(f"FAILED — Keyword overlap {avg_overlap:.3f} below threshold {KEYWORD_OVERLAP_THRESHOLD}")
        passed = False

    if refusal_rate > MAX_REFUSAL_RATE:
        print(f"FAILED — Refusal rate {refusal_rate:.3f} above max {MAX_REFUSAL_RATE}")
        passed = False

    if too_short_count > total * 0.3:
        print(f"FAILED — Too many short answers: {too_short_count}/{total}")
        passed = False

    if passed:
        print("PASSED — All metrics above thresholds")

    print("=" * 50)
    return 0 if passed else 1


if __name__ == "__main__":
    exit_code = run_evaluation()
    sys.exit(exit_code)



