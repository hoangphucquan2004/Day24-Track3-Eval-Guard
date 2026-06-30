from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
'''

    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
            {"role": "user",   "content": PROMPT_TEMPLATE.format(
                question=question, answer_a=answer_a, answer_b=answer_b)},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map[pass2_raw["winner"]]

    # Average: consensus only if both agree
    if pass1["winner"] == winner_pass2:
        final = pass1["winner"]
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1["winner"] == winner_pass2)

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
           judge_labels.count(0)/n * human_labels.count(0)/n)
    return (p_o - p_e) / (1 - p_e) if p_e != 1 else 0.0


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {"total_judged": 0, "position_bias_rate": 0.0, "verbosity_bias": 0.0}

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total, "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
                              "b_wins_b_longer": b_wins_b_longer,
                              "total_decisive": decisive},
        "interpretation": interpretation,
    }


def score_answer_quality(question: str, answer: str) -> int:
    """Dùng LLM để rate câu trả lời: 1 = đúng/đầy đủ, 0 = sai/thiếu."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": (
                "Bạn là expert đánh giá câu trả lời về HR policy. "
                "Trả về JSON: {\"label\": 0 hoặc 1}. "
                "1 = câu trả lời đúng và đầy đủ. "
                "0 = câu trả lời sai, thiếu thông tin quan trọng, hoặc dùng policy cũ."
            )},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nCâu trả lời: {answer}"},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return int(data.get("label", 0))


def save_judge_report(judge_results: list[JudgeResult], kappa: float,
                      bias: dict, path: str = "reports/judge_results.json") -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": (
            "almost perfect" if kappa > 0.8 else
            "substantial" if kappa > 0.6 else
            "moderate" if kappa > 0.4 else
            "fair" if kappa > 0.2 else "slight/poor"
        ),
        "bias_report": bias,
        "pairwise_results": [
            {
                "question": r.question[:80],
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    # --- Load human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"Human labels loaded: {len(human_labels)} questions")

    # --- Task 7: Tính κ thực sự bằng cách judge 10 câu ---
    print("\n[Task 7] Scoring 10 questions for Cohen's κ...")
    judge_labels = []
    for i, item in enumerate(human_data):
        label = score_answer_quality(item["question"], item["model_answer"])
        judge_labels.append(label)
        print(f"  [{i+1}/10] human={item['human_label']} judge={label}  {item['question'][:50]}...")

    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"\nCohen's κ = {kappa:.3f}")

    # --- Task 5+6: Pairwise + swap trên 3 cặp điển hình ---
    print("\n[Tasks 5+6] Running pairwise swap-and-average on sample pairs...")
    pairs = [
        (
            "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành.",
            "Theo quy định, nhân viên có 12 ngày phép hàng năm.",
        ),
        (
            "Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?",
            "Cần CEO phê duyệt vì vượt ngưỡng 50 triệu.",
            "Cần Giám đốc phòng ban phê duyệt.",
        ),
        (
            "Nhân viên tạm ứng 8 triệu, chưa thanh toán sau 30 ngày. Phí phạt là bao nhiêu?",
            "Phạt 2%/tháng tính pro-rata cho số ngày quá hạn.",
            "Phạt 2% tháng trên 8 triệu.",
        ),
    ]

    all_results: list[JudgeResult] = []
    for q, a, b in pairs:
        r = swap_and_average(q, a, b)
        all_results.append(r)
        print(f"  Q: {q[:50]}...")
        print(f"     Pass1={r.winner_pass1}  Pass2={r.winner_pass2}  Final={r.final_winner}  Consistent={r.position_consistent}")

    # --- Task 8: Bias report ---
    bias = bias_report(all_results)
    print(f"\n[Task 8] Bias report:")
    print(f"  Position bias rate: {bias['position_bias_rate']:.1%}")
    print(f"  Verbosity bias:     {bias['verbosity_bias']:.1%}")
    print(f"  Interpretation:     {bias['interpretation']}")

    # --- Save ---
    save_judge_report(all_results, kappa, bias)
    print("\nDone. Next: python src/phase_c_guard.py")
