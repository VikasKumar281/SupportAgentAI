# SupportAgentAI — Evaluation Integrity & Honesty Note

**Author:** Vikas Kumar

## Purpose

This document defines the boundaries of the reported evaluation so that the project is measurable without overstating its capability.

## 1. Final Human-Gold Benchmark

The final end-to-end benchmark contains 200 manually labeled examples.

Each example has a customer message, historical support response, intent label, and escalation label.

The labels were completed by the author using the project labeling guide. All ten intents are represented.

## 2. Final Results

| Metric | Result |
|---|---:|
| Intent Accuracy | 49.00% |
| Intent Macro F1 | 49.75% |
| Escalation Precision | 15.95% |
| Escalation Recall | 100.00% |
| Escalation F1 | 27.51% |
| Auto-Handle Rate | 18.50% |
| Escalation Rate | 81.50% |

The system correctly classified 98 of 200 intent labels, auto-handled 37 examples, and escalated 163.

## 3. What the 49% Number Means

The 49.00% figure is intent accuracy on this 200-example benchmark.

It does not mean:

- 49% of real-world conversations would be solved correctly
- 49% of responses are safe to send automatically
- 49% of generated replies are correct
- production performance is known

The benchmark is limited in size and uses one human annotator.

## 4. What Is Misleading About the Headline Number?

The most misleading interpretation would be to present 49.00% intent accuracy as an automation success rate.

Only 18.50% of the final benchmark was auto-handled.

The escalation policy achieved 100.00% recall but only 15.95% precision, showing that the current system is strongly conservative and over-escalates.

The correct description is:

> SupportAgentAI is a measurable support triage and response-drafting prototype, not a production-ready autonomous support system.

## 5. Development Results Versus Final Evaluation

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 36.00% | 5.29% |
| TF-IDF + Logistic Regression development test | 45.33% | 34.96% |
| Final human-gold end-to-end evaluation | 49.00% | 49.75% |

These are different stages and should not be presented as one controlled comparison.

The escalation thresholds were selected on development data and then applied to the final human-gold benchmark without retuning on those labels.

## 6. Retrieval Limitation

The 200 final evaluation examples are excluded from the leakage-free retrieval index.

This prevents direct self-retrieval.

However, TF-IDF cosine similarity is only a supporting signal. High lexical similarity does not prove semantic relevance or response correctness.

## 7. LLM-as-Judge Status

The repository contains an LLM-as-judge harness.

The external judge run was not completed because the required API account lacked sufficient credits.

Therefore:

- no judge score is reported
- no judge-human agreement score is reported
- no unsupported response-quality metric is claimed

## 8. Human Agreement Limitation

The final benchmark is single-annotator human gold.

The project therefore does not claim Cohen's kappa, Krippendorff's alpha, or any other inter-annotator agreement statistic.

A stronger future benchmark should use at least two independent reviewers.

## 9. Main Evaluation Risks

- 200-example benchmark size
- single-annotator gold labels
- overlapping intent boundaries
- short and ambiguous customer messages
- conservative escalation
- lexical retrieval limitations
- small development training set

## 10. Honest Conclusion

The strongest evidence is that the pipeline is runnable, the classifier improves over the trivial development baseline, retrieval leakage was addressed, a 200-example human-gold benchmark exists, and the system explicitly handles risk and uncertainty.

The dominant current weakness is excessive escalation and intent confusion.

The next improvement should focus on better labeled data, confidence calibration, semantic classification, retrieval relevance, and response-quality evaluation.
