# SupportAgentAI

**Author:** Vikas Kumar

SupportAgentAI is an end-to-end customer-support automation system built around historical customer-support conversations.

It performs four main tasks:

1. classify the customer's primary support intent
2. retrieve relevant historical support interactions
3. decide whether the request should be auto-handled or escalated
4. prepare a grounded response or an internal draft

The design prioritizes **traceability, conservative automation, and measurable evaluation**.

## 1. System Overview

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

## 2. Dataset

The project uses the Twitter Customer Support Conversations dataset.

The raw source contains approximately 2.8 million tweets.

The selected support environment is:

```text
AmazonHelp
```

The extraction produced approximately:

```text
168,823 direct customer/support pairs
149,680 usable original pairs after cleaning and deduplication
```

Raw dataset path:

```text
data/raw/twcs.csv
```

Processed dataset:

```text
data/processed/amazonhelp_conversations.csv
```

Raw and generated artifacts are excluded from Git.

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

Full labeling rules are in `LABELING_GUIDE.md`.

## 4. Classification

Development configuration:

```text
TF-IDF
Logistic Regression
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

### Baseline Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 36.00% | 5.29% |
| TF-IDF + Logistic Regression | 45.33% | 34.96% |

These are development-stage results.

## 5. Historical Retrieval

The system searches historical customer messages using TF-IDF cosine similarity and uses the associated support response as grounding evidence.

A leakage-free evaluation index excludes the 200 final human-gold examples.

```text
Documents: 152,813
Features: 200,000
```

Similarity is used as a supporting signal, not as a standalone retrieval-quality claim.

## 6. Escalation Policy

Current thresholds:

```text
Intent confidence < 0.20
Retrieval similarity < 0.20
```

The system also escalates for:

- account security risk
- payment or financial risk
- legal/high-risk language
- explicit human-agent requests
- high-risk intents

Escalated responses remain internal drafts for human review.

## 7. Final Human-Gold Evaluation

The final benchmark contains:

```text
200 manually labeled examples
10 intents
Complete human intent labels
Complete human escalation labels
```

The labels were completed by the author using the labeling guide.

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

The system auto-handled 37 examples and escalated 163.

### Escalation Reasons

| Reason | Count |
|---|---:|
| `low_intent_confidence` | 123 |
| `high_risk_intent` | 22 |
| `low_historical_similarity` | 11 |
| `legal_or_high_risk` | 4 |
| `security_or_account_risk` | 2 |
| `payment_or_financial_risk` | 1 |

## 8. What Is Misleading About the Headline Number?

The 49.00% intent accuracy is not a 49% automation rate.

Only 18.50% of the final benchmark was auto-handled.

The escalation policy achieved 100.00% recall but only 15.95% precision, showing that the current system is strongly conservative and over-escalates.

The benchmark contains 200 examples and uses one human annotator, so it should be treated as a benchmark result rather than a production estimate.

## 9. Main Failure Modes

### Excessive Escalation

Low intent confidence causes the majority of escalations.

### Delivery vs Product

`order_delivery` and `product_order_issue` share vocabulary and require understanding the customer's actual problem.

### Returns vs Payment

`returns_refunds` and `payment_billing` can overlap around money-related language.

### Other vs Actionable

Short or vague messages can be difficult to route.

### Lexical Retrieval

High lexical similarity does not guarantee semantic relevance.

## 10. Evaluation Integrity

The project does not claim metrics that were not actually produced.

The LLM-as-judge harness is implemented, but the external evaluation run was not completed because the required API account lacked sufficient credits.

Therefore:

- no fabricated judge scores
- no fabricated judge-human agreement
- no unsupported response-quality metric

The final human-gold benchmark is single-annotator, so inter-annotator agreement is not claimed.

## 11. Repository Structure

```text
SupportAgentAI/
|
├── data/
│   ├── raw/
│   └── processed/
|
├── models/
├── scripts/
├── tests/
|
├── DECISION_LOG.md
├── HONESTY_NOTE.md
├── LABELING_GUIDE.md
├── README.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

## 12. Installation

Recommended environment:

```text
Python 3.13+
Windows PowerShell
```

Create and activate the environment:

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the raw dataset at:

```text
data/raw/twcs.csv
```

## 13. Run the Pipeline

```powershell
python .\scripts\prepare_training_data.py
python .\scripts	rain_intent_classifier.py
python .\scriptsuild_retrieval_index.py
python .\scripts\evaluate_agent.py
python .\scriptsnalyze_failures.py
python .\scripts\generate_failure_report.py
```

Run tests:

```powershell
pytest -q
```

Current regression result:

```text
10 passed
```

## 14. LLM-as-Judge

The repository includes:

```text
scripts/run_llm_judge.py
scripts/calculate_judge_agreement.py
```

The judge harness can evaluate response quality dimensions such as relevance, groundedness, helpfulness, accuracy, and clarity when a valid external evaluation account is available.

No judge scores are reported in the current submission because the external run was not completed.

## 15. Important Generated Artifacts

```text
data/processed/agent_evaluation_human_gold_200.csv
data/processed/intent_confusion_matrix.csv
data/processed/failure_analysis.csv
data/processed/top_failure_examples.csv
data/processed/escalation_threshold_results.csv
data/processed/majority_baseline_results.txt
data/processed/tfidf_logistic_regression_results.txt
```

These files are generated locally and excluded from Git.

## 16. One-Week Next Plan

### Day 1
Expand manually labeled data around minority classes and confusion boundaries.

### Day 2
Compare lexical classification with semantic embeddings and hierarchical routing.

### Day 3
Calibrate intent probabilities and re-evaluate the escalation operating point.

### Day 4
Retrieve multiple historical candidates and rerank them using semantic and intent compatibility.

### Day 5
Run the LLM-as-judge harness with a valid external evaluation account and measure judge-human agreement.

### Day 6
Run regression analysis and inspect new failure clusters.

### Day 7
Freeze the taxonomy, evaluation set, thresholds, and final report.

## 17. Documentation

| File | Purpose |
|---|---|
| `README.md` | Project overview and reproducibility |
| `REPORT.md` | Technical report and final results |
| `LABELING_GUIDE.md` | Intent and escalation rules |
| `DECISION_LOG.md` | Engineering decisions and trade-offs |
| `HONESTY_NOTE.md` | Evaluation integrity and limitations |

## 18. Conclusion

SupportAgentAI is a measurable support-triage and response-drafting prototype.

Its strongest properties are modular architecture, historical grounding, leakage-aware evaluation, explicit risk handling, conservative escalation, a human-gold benchmark, per-example failure analysis, automated regression tests, and transparent evaluation limitations.

The main limitation is excessive escalation caused by low classifier confidence.

The next iteration should focus on better labeled data, confidence calibration, semantic classification, and retrieval relevance.
