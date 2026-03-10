"""LLM-as-judge for semantic equivalence of answerable QA predictions."""

import time
from typing import Dict, List, Optional

from .config import NO_ANSWER_TOKEN


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating question-answering accuracy. "
    "Given a question, a gold (reference) answer, and a predicted answer, decide if the prediction is semantically correct. "
    "Two answers are correct if they convey the same meaning, even if worded differently "
    '(e.g. "executed" and "slaughtered them" for the same event are both correct). '
    "Respond with exactly CORRECT or INCORRECT."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Predicted answer: {prediction}\n\n"
    "Is the predicted answer semantically correct? Respond CORRECT or INCORRECT."
)


def _is_no_answer(value: str) -> bool:
    return (value or "").strip() == NO_ANSWER_TOKEN


def _call_judge(
    client,
    question: str,
    gold: str,
    prediction: str,
    model: str = "gpt-4o-mini",
) -> bool:
    """Call the LLM judge for a single prediction. Returns True if CORRECT."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=question, gold=gold, prediction=prediction,
            )},
        ],
        max_completion_tokens=64,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "CORRECT" in verdict


def judge_batch(
    records: List[Dict],
    dataset_path: Optional[str] = None,
    api_key: str = "YOUR_KEY_HERE",
    model: str = "gpt-4o-mini",
    requests_per_minute: int = 500,
) -> List[Dict]:
    """Run LLM-as-judge on answerable predictions.

    Only judges records where:
    - Gold is NOT <NO-ANSWER> (answerable questions)
    - Prediction is NOT <NO-ANSWER> (model attempted an answer)

    For abstention cases, llm_judge_correct is set deterministically:
    - Both abstain → True (correct abstention)
    - Pred abstains, gold doesn't → False (false abstention)
    - Pred answers, gold abstains → False (should have abstained)

    Returns records with added 'llm_judge_correct' field.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    delay = 60.0 / requests_per_minute

    for record in records:
        pred = record["prediction"]
        gold = record["gold"]
        pred_abstain = _is_no_answer(pred)
        gold_abstain = _is_no_answer(gold)

        if gold_abstain:
            # Unanswerable: correct only if model also abstained
            record["llm_judge_correct"] = pred_abstain
        elif pred_abstain:
            # Model abstained on answerable question
            record["llm_judge_correct"] = False
        else:
            # Both answered: use LLM judge
            question = record.get("input", record.get("id", ""))
            try:
                record["llm_judge_correct"] = _call_judge(
                    client, question=question, gold=gold, prediction=pred, model=model,
                )
            except Exception as e:
                print(f"Judge API error for {record.get('id')}: {e}")
                record["llm_judge_correct"] = False
            time.sleep(delay)

    return records
