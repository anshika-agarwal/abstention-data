import re
import string
from typing import Dict, List

from .config import NO_ANSWER_TOKEN


def _normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def _is_no_answer(value: str) -> bool:
    return (value or "").strip() == NO_ANSWER_TOKEN


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def compute_metrics(records: List[Dict]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {"n": 0}

    em_total = 0
    abstain_tp = 0
    abstain_fp = 0
    abstain_fn = 0
    answerable_em_hits = 0
    answerable_count = 0

    for row in records:
        pred = row["prediction"]
        gold = row["gold"]

        pred_norm = _normalize_text(pred)
        gold_norm = _normalize_text(gold)
        if pred_norm == gold_norm:
            em_total += 1

        pred_abstain = _is_no_answer(pred)
        gold_abstain = _is_no_answer(gold)
        if pred_abstain and gold_abstain:
            abstain_tp += 1
        elif pred_abstain and not gold_abstain:
            abstain_fp += 1
        elif not pred_abstain and gold_abstain:
            abstain_fn += 1

        if not gold_abstain:
            answerable_count += 1
            if pred_norm == gold_norm:
                answerable_em_hits += 1

    abstain_precision = _safe_div(abstain_tp, abstain_tp + abstain_fp)
    abstain_recall = _safe_div(abstain_tp, abstain_tp + abstain_fn)
    abstain_f1 = _safe_div(2 * abstain_precision * abstain_recall, abstain_precision + abstain_recall)

    # Per unanswerable-type breakdown
    type_metrics = {}
    type_buckets = {}
    for row in records:
        utype = row.get("unanswerable_type")
        if utype is None:
            continue
        type_buckets.setdefault(utype, []).append(row)

    for utype, rows in sorted(type_buckets.items()):
        n_type = len(rows)
        tp = sum(1 for r in rows if _is_no_answer(r["prediction"]) and _is_no_answer(r["gold"]))
        recall = _safe_div(tp, n_type)
        type_metrics[utype] = {"n": n_type, "abstain_recall": recall}

    result = {
        "n": total,
        "overall_exact_match": _safe_div(em_total, total),
        "answerable_exact_match": _safe_div(answerable_em_hits, answerable_count),
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "abstain_f1": abstain_f1,
        "pred_abstain_rate": _safe_div(sum(_is_no_answer(r["prediction"]) for r in records), total),
        "gold_abstain_rate": _safe_div(sum(_is_no_answer(r["gold"]) for r in records), total),
    }
    if type_metrics:
        result["per_unanswerable_type"] = type_metrics
    return result


def compute_metrics_with_judge(records: List[Dict]) -> Dict[str, float]:
    """Compute metrics using LLM judge scores for answerable accuracy.

    Expects records to have 'llm_judge_correct' boolean field (set by llm_judge module).
    Abstention metrics are identical to compute_metrics. Answerable accuracy uses
    llm_judge_correct instead of exact match.
    """
    base = compute_metrics(records)
    if base.get("n", 0) == 0:
        return base

    judge_hits = 0
    answerable_count = 0
    for row in records:
        gold = row["gold"]
        if not _is_no_answer(gold):
            answerable_count += 1
            if row.get("llm_judge_correct", False):
                judge_hits += 1

    base["answerable_llm_accuracy"] = _safe_div(judge_hits, answerable_count)

    # Overall LLM accuracy: abstention TP + answerable judge hits / total
    abstain_tp = sum(
        1 for r in records if _is_no_answer(r["prediction"]) and _is_no_answer(r["gold"])
    )
    base["overall_llm_accuracy"] = _safe_div(abstain_tp + judge_hits, len(records))

    return base

