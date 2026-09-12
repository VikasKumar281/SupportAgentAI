# SupportAgentAI — Evaluation and Reporting Notes

**Author:** Vikas Kumar

## Purpose

This document records the evaluation methodology, interpretation of reported metrics, and important limitations of the current SupportAgentAI benchmark.

The purpose is to make the project measurable, reproducible, and easy to review.

---

## 1. Evaluation Philosophy

A support automation system should not be evaluated using a single number.

There are three separate questions:

1. Did the system understand the customer correctly?
2. Did the system make the correct automation or escalation decision?
3. Did the system prepare a useful and grounded response?

These dimensions are therefore evaluated separately.

```text
Intent Quality
      +
Decision Quality
      +
Response Quality
      =
Overall System Quality
```

---

## 2. Classification Evaluation

The intent classifier is evaluated using:

- Accuracy
- Macro F1
- Per-class performance
- Confusion matrix

Accuracy measures the proportion of correctly classified examples.

Macro F1 gives equal importance to every intent and is therefore particularly useful when the intent distribution is uneven.

The current development classifier results are:

| Metric | Result |
|---|---:|
| Validation Macro F1 | 0.3379 |
| Test Accuracy | 0.4533 |
| Test Macro F1 | 0.3496 |

The majority baseline results are:

| Metric | Result |
|---|---:|
| Accuracy | 0.3600 |
| Macro F1 | 0.0529 |

These development results show that the TF-IDF + Logistic Regression classifier provides a meaningful improvement over the majority baseline.

---

## 3. End-to-End Evaluation

The complete system was evaluated on 200 examples.

The current results are:

| Metric | Result |
|---|---:|
| Intent Accuracy | 0.5300 |
| Intent Macro F1 | 0.5281 |
| Escalation Precision | 0.3926 |
| Escalation Recall | 0.9552 |
| Escalation F1 | 0.5565 |
| Auto-Handle Rate | 0.1850 |
| Escalation Rate | 0.8150 |

The system escalated 163 of 200 examples and automatically handled 37.

---

## 4. Escalation Evaluation

Escalation is evaluated as a binary decision.

```text
Escalate
    vs
Auto Handle
```

The current system prioritizes escalation recall.

This is intentional because a missed high-risk request can be more costly than an unnecessary human review.

The current escalation metrics are:

```text
Precision: 39.26%
Recall:    95.52%
F1:        55.65%
```

The main trade-off is the high escalation rate.

---

## 5. Escalation Breakdown

The 163 escalated examples were distributed as follows:

| Reason | Count |
|---|---:|
| low_intent_confidence | 123 |
| high_risk_intent | 22 |
| low_historical_similarity | 11 |
| legal_or_high_risk | 4 |
| security_or_account_risk | 2 |
| payment_or_financial_risk | 1 |

The largest source of escalation is low intent confidence.

This indicates that classifier uncertainty is currently a larger bottleneck than explicit risk detection.

---

## 6. Retrieval Evaluation

Retrieval uses historical customer-support interactions.

A separate leakage-free retrieval index was created so that evaluation examples are not directly available to the retrieval system.

The leakage-free index contains:

```text
Documents: 152,813
Features: 200,000
```

Retrieval similarity is treated as a supporting signal rather than a complete retrieval-quality metric.

This is because lexical similarity does not necessarily represent semantic relevance.

---

## 7. Evaluation Leakage

A retrieval system can produce misleading results if evaluation examples remain in its retrieval corpus.

The problematic setup is:

```text
Evaluation Example
       |
       v
Same Example in Retrieval Index
       |
       v
Self Match
       |
       v
Artificially High Similarity
```

The project avoids this by excluding evaluation examples before building the leakage-free index.

This is an important evaluation safeguard.

---

## 8. Headline Number Interpretation

The current end-to-end intent accuracy is:

```text
53.0%
```

This should be interpreted as a development evaluation result.

It should not be treated as a production-level estimate.

The current 200-example benchmark uses assistant-assisted development labels and is not presented as independently hand-labeled gold data. Independent human annotation is still required for a strict gold benchmark.

The metric is useful for:

- comparing iterations
- identifying regressions
- measuring progress
- guiding future improvements

It should not be used to make unsupported claims about real-world production accuracy.

---

## 9. Human Evaluation Plan

A stronger benchmark should use independent reviewers.

The recommended process is:

```text
Evaluation Example
       |
       +-------------+
       |             |
       v             v
 Reviewer 1      Reviewer 2
       |             |
       +-------------+
               |
               v
       Agreement Analysis
               |
               v
        Final Gold Label
```

The process should be:

1. Select a representative evaluation sample.
2. Give reviewers the labeling guide.
3. Ask reviewers to label independently.
4. Compare their labels.
5. Investigate disagreements.
6. Clarify ambiguous definitions.
7. Re-label the calibration subset.
8. Measure agreement.
9. Freeze the labeling guide.
10. Build the final evaluation set.

---

## 10. Response Quality Evaluation

Intent accuracy does not guarantee response quality.

A system can correctly classify a request while producing an incomplete or unhelpful response.

Response evaluation should therefore measure:

| Dimension | Evaluation Question |
|---|---|
| Relevance | Does the response address the customer's actual request? |
| Groundedness | Is the response supported by historical evidence? |
| Helpfulness | Does it provide a useful next step? |
| Accuracy | Does it avoid unsupported claims? |
| Clarity | Is it easy to understand? |

---

## 11. Recommended Future Benchmark

A stronger benchmark should contain:

- 200-250 independently reviewed examples
- all ten intents
- common and minority intents
- ambiguous requests
- sensitive requests
- short messages
- longer messages
- escalation labels
- response-quality labels

The evaluation set should be frozen before final model tuning.

---

## 12. Reproducibility

The project keeps the evaluation scripts separate from the generated artifacts.

Important evaluation outputs include:

```text
data/processed/agent_evaluation_examples.csv
data/processed/agent_evaluation_results.txt
data/processed/intent_confusion_matrix.csv
data/processed/failure_analysis.csv
data/processed/top_failure_examples.csv
data/processed/escalation_threshold_results.csv
```

These outputs allow individual examples and aggregate metrics to be inspected.

---

## 13. Interpretation Summary

The current system demonstrates:

- useful intent classification
- strong escalation recall
- historical grounding
- leakage-aware retrieval evaluation
- explicit risk handling
- automated regression tests

The most important current weakness is excessive escalation.

The main improvement path is:

```text
More Reliable Labels
        |
        v
Better Intent Model
        |
        v
Better Confidence Calibration
        |
        v
Fewer Unnecessary Escalations
```

---

# Conclusion

The evaluation framework is designed to measure not only whether the model is correct, but also whether the system knows when it should not act automatically.

The most important evaluation limitation is that independent human gold labeling and external LLM-judge execution were not completed. The project does not report fabricated agreement numbers.
