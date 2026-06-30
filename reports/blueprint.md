# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Hoàng Phúc Quân - 2A202600560
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~9ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Đo thực tế từ Task 12 — measure_p95_latency(), warm run)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 1.80 | 9.19 | 9.19 | <10ms |
| NeMo Input Rail | 0.97 | 1.16 | 1.16 | <300ms |
| RAG Pipeline | — | — | — | <2000ms |
| NeMo Output Rail | — | — | — | <300ms |
| **Total Guard** | 2.77 | **10.31** | 10.31 | **<500ms** |

**Budget OK?** [x] Yes  
**Comment:** Total P95 là 10.31ms — jauh dưới budget 500ms. NeMo nhanh vì các adversarial inputs
match Colang keyword patterns (local matching, không cần LLM API call). Presidio cold-start
lần đầu ~2500ms do load spaCy model, nhưng sau khi warm thì chỉ ~9ms P95.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (factual) | 0.865 |
| RAGAS avg_score (multi_hop) | 0.676 |
| RAGAS avg_score (adversarial) | 0.663 |
| Worst metric | answer_relevancy (dominant failure cho factual) |
| Dominant failure distribution | multi_hop (faithfulness thấp nhất) |
| Cohen's κ | 0.167 (slight agreement) |
| Adversarial pass rate | 18 / 20 (90%) |
| Guard P95 latency (warm) | 10.31 ms |

---

## Nhận xét & Cải tiến

Pipeline hoạt động tốt ở tầng Presidio (detect CCCD/phone chính xác) và NeMo guardrails
(18/20 adversarial blocked đúng). Điểm yếu lớn nhất là **faithfulness trên multi_hop** —
pipeline hay hallucinate khi phải kết hợp nhiều tài liệu (avg faithfulness chỉ 0.396 vs
0.833 của factual). Nguyên nhân chủ yếu là context window bị giới hạn và enrichment API
bị lỗi JSON nên một số chunks không được enrich đầy đủ. Để deploy production, cần:
(1) retry enrichment với exponential backoff, (2) thêm chain-of-thought prompt cho multi_hop,
(3) pre-warm Presidio khi khởi động server để tránh cold-start 2500ms.
