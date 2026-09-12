# SupportAgentAI — Technical Report

**Author:** Vikas Kumar

## Executive Summary

SupportAgentAI is an end-to-end customer-support automation system built around historical customer-support conversations.

It addresses four connected problems:

1. identify the customer's primary support intent
2. retrieve relevant historical support interactions
3. decide whether the request can be auto-handled or should receive human review
4. prepare a grounded response or internal draft

The implementation uses a ten-intent taxonomy, TF-IDF + Logistic Regression classification, TF-IDF cosine retrieval, deterministic risk detection, confidence-based escalation, retrieval-based escalation, leakage-aware evaluation, automated tests, and per-example failure analysis.

The final human-gold benchmark contains 200 manually labeled examples.

## 1. Problem Framing

Customer-support requests repeat operational patterns such as delivery delays, returns, refunds, billing problems, account issues, subscriptions, digital content, technical issues, and support complaints.

The intended workflow is:

```text
Customer Message
      |
      v
Intent Classification
      |
      v
Intent + Confidence
      |
      v
Historical Retrieval
      |
      v
Risk Detection
      |
      v
Escalation Decision
      |
      +----------------------+
      |                      |
      v                      v
Human Review          Auto Handling
      |                      |
      v                      v
Internal Draft         Grounded Reply
```

The system is designed to decide what to do before deciding what to say.

## 2. Dataset and Scope

The project uses the Twitter Customer Support Conversations dataset.

The raw source contains approximately 2.8 million tweets.

The selected support environment is AmazonHelp.

The extraction produced approximately:

| Stage | Records |
|---|---:|
| Direct customer/support pairs before final cleaning | 168,823 |
| Usable original pairs after cleaning/deduplication | 149,680 |

The processed records contain customer text, historical support response, identifiers, timestamps, and support-account information.

## 3. Intent Taxonomy

| Intent | Meaning |
|---|---|
| `order_delivery` | Delivery, tracking, delayed or missing packages |
| `returns_refunds` | Returns, cancellations and refunds |
| `payment_billing` | Charges, billing and payment problems |
| `account_security` | Account access and security concerns |
| `subscription_prime` | Prime and subscription issues |
| `digital_content` | Digital books, video, music and digital content |
| `device_technical` | Device and technical troubleshooting |
| `product_order_issue` | Wrong, damaged or defective products |
| `customer_service_complaint` | Complaints about support experience |
| `other_non_actionable` | Unclear or non-actionable messages |

The taxonomy is intentionally compact so that each class remains operationally meaningful.

## 4. Classification and Baselines

The classifier is:

```text
Customer Message
      |
      v
TF-IDF
      |
      v
Logistic Regression
      |
      v
Intent + Confidence
```

Development configuration:

```text
N-gram range: (1, 1)
Minimum document frequency: 1
C: 0.5
```

Development split:

```text
Training: 350
Validation: 75
Test: 75
```

### Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 36.00% | 5.29% |
| TF-IDF + Logistic Regression | 45.33% | 34.96% |

The learned classifier improves over the trivial development baseline.

These are development-stage results and are separate from the final human-gold evaluation.

## 5. Historical Retrieval

For a new customer message, the system searches historical customer messages using TF-IDF cosine similarity.

The associated historical support response becomes the grounding source for the draft.

```text
New Customer Message
      |
      v
TF-IDF Representation
      |
      v
Cosine Similarity
      |
      v
Best Historical Interaction
      |
      v
Historical Support Response
```

Historical responses are cleaned to remove unnecessary platform-specific artifacts such as handles, shortened URLs, signatures, and irrelevant trailing fragments.

## 6. Retrieval Leakage Prevention

Evaluation examples are excluded from the retrieval corpus before constructing the final evaluation index.

The leakage-free index contains:

```text
Documents: 152,813
Features: 200,000
```

This prevents direct self-retrieval from artificially inflating similarity.

Retrieval similarity remains only a supporting signal because lexical similarity does not guarantee semantic relevance.

## 7. Escalation Design

The decision layer combines:

```text
Risk Signals
     +
Intent
     +
Intent Confidence
     +
Retrieval Similarity
     |
     v
Escalation Decision
```

The system escalates when:

1. a strong risk pattern is detected
2. the predicted intent is `account_security` or `payment_billing`
3. intent confidence is below `0.20`
4. retrieval similarity is below `0.20`

Risk detection covers security, financial, legal, and explicit human-agent requests.

Escalated responses are internal drafts for human review.

## 8. Final Human-Gold Evaluation

A fixed benchmark of 200 examples was manually labeled by the author using the labeling guide.

All ten intents are represented.

### Final Results

| Metric | Result |
|---|---:|
| Intent Accuracy | 49.00% |
| Intent Macro F1 | 49.75% |
| Escalation Precision | 15.95% |
| Escalation Recall | 100.00% |
| Escalation F1 | 27.51% |
| Auto-Handle Rate | 18.50% |
| Escalation Rate | 81.50% |

The system correctly classified 98 of 200 intent labels.

It auto-handled 37 examples and escalated 163.

### Escalation Breakdown

| Reason | Count |
|---|---:|
| `low_intent_confidence` | 123 |
| `high_risk_intent` | 22 |
| `low_historical_similarity` | 11 |
| `legal_or_high_risk` | 4 |
| `security_or_account_risk` | 2 |
| `payment_or_financial_risk` | 1 |

The dominant escalation driver is low classifier confidence.

## 9. Representative End-to-End Example

Input:

```text
Where is my order? It was supposed to arrive yesterday.
```

System output:

```text
Intent: order_delivery
Intent Confidence: 0.1388
Retrieval Similarity: 0.6653
Escalate: True
Reason: low_intent_confidence
```

Draft:

```text
Oh no! I'm sorry to hear that. What is the last tracking update on the order?
```

The historical match is reasonably strong, but the system still escalates because intent confidence is below the safety threshold.

## 10. Top Five Failure Modes

### 1. Excessive Escalation

123 of 163 final escalations were caused by low intent confidence.

**Hypothesis:** the classifier has limited labeled training data relative to the diversity of the support corpus, especially for short and ambiguous messages.

**Next step:** increase labeled data and calibrate confidence before relaxing thresholds.

### 2. Delivery vs Product Confusion

`order_delivery` and `product_order_issue` share vocabulary such as order, item, package, and product.

The key distinction is whether the customer is waiting for delivery or reporting a problem with the product.

**Next step:** add boundary examples and consider hierarchical classification.

### 3. Returns/Refunds vs Payment/Billing

Money-related language can make `returns_refunds` and `payment_billing` difficult to separate.

**Next step:** add more outcome-focused boundary examples.

### 4. Other/Non-Actionable vs Actionable Intents

Short or conversational messages can lack enough information for reliable routing.

**Next step:** expand representative examples and improve short-message handling.

### 5. Lexical Retrieval Is Not Semantic Relevance

High TF-IDF similarity can still produce a superficially similar historical interaction that is not the best evidence.

**Next step:** use semantic embeddings, multiple candidates, and reranking with intent compatibility.

## 11. What Is Misleading About My Headline Number?

The headline **49.00% intent accuracy** is not a 49% automation success rate.

Only 18.50% of the final benchmark was auto-handled.

The escalation layer achieved 100.00% recall but only 15.95% precision, showing that the current system is highly conservative and over-escalates.

The benchmark contains 200 examples and uses one human annotator, so the result should not be presented as a production estimate.

A more accurate description is:

> SupportAgentAI is a measurable support triage and response-drafting prototype, not a production-ready autonomous support system.

## 12. Evaluation Limitations

The main limitations are:

- 200-example benchmark size
- single-annotator human gold
- no inter-annotator agreement measurement
- no completed external LLM-as-judge run
- lexical rather than semantic retrieval
- conservative escalation policy
- small development training set

The LLM-as-judge harness is implemented, but the external run was not completed because the required API account lacked sufficient credits.

No judge score or judge-human agreement number is reported.

## 13. Engineering Validation

The repository contains automated regression tests for dataset/model availability and the main escalation and handling behaviors.

Current result:

```text
10 passed
```

The project also preserves per-example evaluation outputs for confusion and failure analysis.

## 14. One-Week Next Plan

### Day 1 — Improve Labels

Add more manually reviewed examples around minority classes and major confusion boundaries.

### Day 2 — Improve Classification

Compare TF-IDF with semantic embeddings and hierarchical routing.

### Day 3 — Calibrate Confidence

Calibrate intent probabilities and choose an operating point based on the cost of false escalation versus unsafe automation.

### Day 4 — Improve Retrieval

Retrieve multiple candidates and rerank them using semantic and intent compatibility.

### Day 5 — Improve Response Evaluation

Run the existing LLM-as-judge harness with an available external evaluation account and compute judge-human agreement.

### Day 6 — Regression and Error Review

Re-run the frozen human-gold benchmark, inspect failure clusters, and add regression tests for important cases.

### Day 7 — Final Validation

Freeze the taxonomy, evaluation set, thresholds, and report, then compare the next system version against the current baselines.

## 15. Reproducibility

Expected environment:

```text
Python 3.13+
Windows PowerShell
```

Dataset:

```text
data/raw/twcs.csv
```

Install:

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run:

```powershell
python .\scripts\prepare_training_data.py
python .\scripts	rain_intent_classifier.py
python .\scriptsuild_retrieval_index.py
python .\scripts\evaluate_agent.py
python .\scriptsnalyze_failures.py
python .\scripts\generate_failure_report.py
```

Tests:

```powershell
pytest -q
```

## 16. Conclusion

SupportAgentAI demonstrates a complete support-automation workflow rather than a standalone classifier.

Its strongest engineering properties are modular architecture, historical grounding, leakage-aware evaluation, explicit risk handling, conservative escalation, a human-gold benchmark, per-example failure analysis, automated tests, and transparent evaluation limitations.

The main improvement target is reducing unnecessary escalation without weakening safety. That requires better labeled data, confidence calibration, stronger semantic classification, and better retrieval relevance.
