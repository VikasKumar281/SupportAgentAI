# SupportAgentAI — Engineering Decision Log

**Author:** Vikas Kumar

This document records the main engineering decisions behind SupportAgentAI and the reasoning behind them.

## 1. Single Support Environment

**Decision:** Use AmazonHelp as the primary support environment.

**Reason:** A single support account keeps terminology, support style, and operational context consistent.

## 2. Customer/Support Conversation Pairs

**Decision:** Represent each historical interaction as a customer message paired with its support response.

**Reason:** Retrieval needs to answer: “What did support say when a similar customer request appeared?”

## 3. Compact Ten-Intent Taxonomy

The system uses:

1. `order_delivery`
2. `returns_refunds`
3. `payment_billing`
4. `account_security`
5. `subscription_prime`
6. `digital_content`
7. `device_technical`
8. `product_order_issue`
9. `customer_service_complaint`
10. `other_non_actionable`

Ten classes provide a practical balance between useful routing detail and classification ambiguity.

## 4. Separate Classification and Retrieval

Classification answers what the customer is asking. Retrieval answers which historical interaction is relevant.

Keeping them separate makes the system easier to evaluate and improve.

## 5. Majority Baseline

A majority-class baseline was implemented first.

| Metric | Result |
|---|---:|
| Accuracy | 36.00% |
| Macro F1 | 5.29% |

## 6. TF-IDF + Logistic Regression

The primary classifier uses TF-IDF with Logistic Regression because it is fast, interpretable, reproducible, and suitable for short text.

Development results:

| Metric | Result |
|---|---:|
| Validation Macro F1 | 33.79% |
| Test Accuracy | 45.33% |
| Test Macro F1 | 34.96% |

These are development results, separate from the final human-gold benchmark.

## 7. Confidence as a Safety Signal

The escalation policy uses classifier confidence because uncertain predictions should not be automatically acted upon.

Current threshold:

```text
Intent confidence < 0.20
```

## 8. Explicit Risk Detection

Deterministic detection covers security, financial, legal, and explicit human-agent requests.

Sensitive requests are routed to human review.

## 9. High-Risk Intents

`account_security` and `payment_billing` are treated conservatively because they can involve sensitive account or financial consequences.

## 10. Retrieval Similarity as a Safety Signal

A weak historical match provides weaker grounding evidence.

Current threshold:

```text
Retrieval similarity < 0.20
```

## 11. Retrieval Leakage Prevention

The 200 final human-gold examples are excluded from the evaluation retrieval corpus.

The leakage-free index contains:

```text
Documents: 152,813
Features: 200,000
```

This prevents direct self-retrieval from inflating evaluation similarity.

## 12. Historical Response Cleaning

Historical responses are cleaned to remove platform-specific artifacts such as handles, shortened URLs, signatures, and irrelevant trailing fragments.

## 13. Escalated Cases Remain Internal Drafts

When human review is required, the historical response is treated as an internal draft rather than an automatic customer-facing response.

```text
Escalated Request
      |
      v
Internal Draft
      |
      v
Human Review
```

## 14. Threshold Tuning on Development Data

The confidence and retrieval thresholds were selected using development data rather than the final human-gold labels.

The selected values are:

| Parameter | Value |
|---|---:|
| Intent confidence threshold | 0.20 |
| Retrieval similarity threshold | 0.20 |

On the earlier development benchmark, this operating point produced 95.52% escalation recall, 39.26% precision, and 55.65% F1. These are development-tuning results and are not the final human-gold metrics.

## 15. Per-Example Evaluation Data

Individual predictions, confidence values, retrieval scores, escalation decisions, and gold labels are preserved so that failures can be inspected rather than hidden by aggregate metrics.

## 16. Accuracy and Macro F1

Both Accuracy and Macro F1 are reported. Macro F1 is important because it gives equal weight to each intent.

## 17. Keep Generated Artifacts Outside Git

Raw data, generated CSVs, and serialized models are excluded from Git because they are large and reproducible.

## 18. Automated Regression Tests

The current test suite contains ten tests covering dataset/model availability and the main escalation and handling behaviors.

Current result:

```text
10 passed
```

## 19. Final Human-Gold Evaluation Set

The final benchmark contains 200 manually labeled examples covering all ten intents.

The author completed the human intent and escalation labels using the labeling guide.

Final results:

| Metric | Result |
|---|---:|
| Intent Accuracy | 49.00% |
| Intent Macro F1 | 49.75% |
| Escalation Precision | 15.95% |
| Escalation Recall | 100.00% |
| Escalation F1 | 27.51% |
| Auto-Handle Rate | 18.50% |

This is a single-annotator benchmark, so inter-annotator agreement is not claimed.

## 20. LLM-as-Judge Integrity

The LLM-as-judge harness is implemented, but an external evaluation run was not completed because the required API account lacked sufficient credits.

No fabricated judge scores or judge-human agreement values are reported.

# Decision Summary

```text
Historical Evidence
       +
Intent Understanding
       +
Risk Detection
       +
Confidence
       |
       v
Safe Automation Decision
```

The architecture favors traceability and conservative automation.

# Future Decisions to Revisit

1. TF-IDF versus semantic embeddings.
2. Flat versus hierarchical classification.
3. Single retrieval candidate versus multi-candidate reranking.
4. Static thresholds versus calibrated confidence.
5. Rule-based risk detection versus trained risk classification.
6. Larger human-gold coverage and multi-annotator agreement.
7. Direct response-quality evaluation.
