# SupportAgentAI — Technical Report

**Author:** Vikas Kumar

## Executive Summary

SupportAgentAI is an end-to-end customer-support automation system built around historical customer-support conversations.

The system is designed to solve four connected problems:

1. Understand the customer's primary support intent.
2. Retrieve relevant historical support interactions.
3. Decide whether the request is safe to handle automatically.
4. Prepare a grounded support response or an internal draft for human review.

The implementation uses a compact ten-intent taxonomy, TF-IDF with Logistic Regression for intent classification, TF-IDF cosine-similarity retrieval for historical grounding, deterministic risk detection, confidence-based escalation, retrieval-quality thresholds, leakage-aware evaluation, automated tests, and per-example failure analysis.

The central design principle is conservative automation:

> The system should automate when evidence and confidence are sufficient, and prefer human review when the request is sensitive or uncertain.

The current end-to-end evaluation reports:

| Metric | Result |
|---|---:|
| Intent Accuracy | 53.00% |
| Intent Macro F1 | 52.81% |
| Escalation Precision | 39.26% |
| Escalation Recall | 95.52% |
| Escalation F1 | 55.65% |
| Auto-Handle Rate | 18.50% |
| Escalation Rate | 81.50% |

The most important finding is that the current system is conservative but over-escalates. Most escalations are caused by low classifier confidence, which makes better labeled data and stronger confidence calibration the highest-value next improvements.

---

# 1. Problem Framing

Customer-support systems receive many requests that are variations of previously seen problems.

Typical requests include:

- delivery delays
- tracking questions
- missing packages
- returns
- refunds
- billing problems
- account access
- security concerns
- subscription questions
- digital-content problems
- device issues
- incorrect or damaged products
- customer-service complaints

A useful automation system should not simply generate a response based on the wording of a new message.

The system should first determine what the customer needs.

The problem can therefore be represented as:

```text
Customer Message
       |
       v
What is the customer asking?
       |
       v
Can historical evidence be found?
       |
       v
Is the request safe to automate?
       |
       v
What response should be prepared?
```

This leads to a controlled pipeline rather than an unconstrained response-generation system.

---

# 2. Dataset and Scope

The project uses the Twitter Customer Support Conversations dataset.

The raw source contains approximately 2.8 million tweets and includes conversation relationship fields that make it possible to connect customer messages with support responses.

The source schema includes:

| Field | Purpose |
|---|---|
| `tweet_id` | Unique tweet identifier |
| `author_id` | Author identifier |
| `inbound` | Message direction |
| `created_at` | Timestamp |
| `text` | Tweet text |
| `response_tweet_id` | Response relationship |
| `in_response_to_tweet_id` | Parent-message relationship |

A single support environment, AmazonHelp, was selected for the main project.

This keeps the retrieval corpus internally consistent and avoids mixing response styles and operational contexts from unrelated support accounts.

---

# 3. Data Preparation

The raw dataset was transformed into direct customer-to-support response pairs.

The processing flow was:

```text
Raw Tweets
    |
    v
Support Account Filtering
    |
    v
Conversation Relationship Resolution
    |
    v
Customer/Support Pair Extraction
    |
    v
Incomplete Record Removal
    |
    v
Deduplication
    |
    v
Processed Conversation Corpus
```

The extraction produced approximately:

```text
168,823
```

direct customer/support pairs before final cleaning and deduplication.

After cleaning and deduplication, approximately:

```text
149,680
```

usable original pairs remained.

The processed dataset is stored as:

```text
data/processed/amazonhelp_conversations.csv
```

Each record contains the customer message and the corresponding historical support response.

This structure is particularly useful because it supports both classification and retrieval.

---

# 4. Intent Taxonomy

A ten-class operational taxonomy was created.

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
| `customer_service_complaint` | Complaints about the support experience |
| `other_non_actionable` | Unclear or non-actionable messages |

The taxonomy was intentionally kept compact.

The objective was to create categories that are useful for:

- routing
- retrieval
- escalation
- response handling
- evaluation

A smaller operational taxonomy also makes classification errors easier to interpret.

---

# 5. Intent Classification

The intent classifier uses a conventional text-classification pipeline:

```text
Customer Message
       |
       v
TF-IDF Vectorization
       |
       v
Logistic Regression
       |
       v
Intent Prediction
       |
       v
Confidence Score
```

The development data was split into:

```text
Training:      350
Validation:     75
Test:           75
```

The classifier configuration selected during development was:

```text
Algorithm: Logistic Regression
Features: TF-IDF
N-gram range: (1, 1)
Minimum document frequency: 1
C: 0.5
```

The validation Macro F1 was:

```text
0.3379
```

The held-out development test results were:

| Metric | Score |
|---|---:|
| Accuracy | 45.33% |
| Macro F1 | 34.96% |

The classifier is useful as a baseline and as a component of the larger pipeline, but the results also show that the taxonomy contains several overlapping categories that are difficult to distinguish with lexical features alone.

---

# 6. Baseline Comparison

Two baselines were used to establish the expected performance range.

## 6.1 Majority Baseline

The trivial baseline always predicts the most frequent intent:

```text
order_delivery
```

Results:

| Metric | Score |
|---|---:|
| Accuracy | 36.00% |
| Macro F1 | 5.29% |

This establishes a lower bound.

The extremely low Macro F1 demonstrates that a majority-only strategy does not provide balanced performance across the taxonomy.

---

## 6.2 TF-IDF + Logistic Regression

The simple learned baseline uses TF-IDF with Logistic Regression.

Results:

| Metric | Score |
|---|---:|
| Accuracy | 45.33% |
| Macro F1 | 34.96% |

The learned classifier improves substantially over the majority baseline.

However, the two baselines and the end-to-end benchmark use different development/evaluation stages, so their numbers should not be interpreted as a single controlled benchmark comparison.

---

# 7. Historical Retrieval

The response-drafting system uses historical support conversations as evidence.

The retrieval pipeline is:

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
Historical Customer Messages
       |
       v
Best Matching Interaction
       |
       v
Historical Support Response
```

The retrieval corpus contains approximately 168k historical customer messages.

For every incoming request, the system finds the most similar historical customer message.

The support response associated with that historical interaction is then used as the grounding source for the response draft.

This provides traceability because a developer can inspect the historical interaction behind a generated draft.

---

# 8. Retrieval Leakage Prevention

Retrieval evaluation has a specific leakage risk.

If an evaluation example is present in the retrieval corpus, the system may retrieve the same example.

That produces an artificially strong similarity score.

The problematic setup is:

```text
Evaluation Message
       |
       v
Same Message in Retrieval Index
       |
       v
Self Match
       |
       v
Artificially High Similarity
```

To prevent this, a separate leakage-free retrieval index was created.

The evaluation examples are removed before constructing the evaluation retrieval corpus.

The resulting leakage-free index contains:

```text
Documents: 152,813
Features: 200,000
```

This is an important evaluation safeguard.

---

# 9. Response Drafting

The system does not generate a response independently of historical evidence.

Instead:

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Retrieval
       |
       v
Relevant Historical Response
       |
       v
Response Cleaning
       |
       v
Draft
```

Historical social-media support messages may contain platform-specific artifacts.

The response-cleaning stage removes unnecessary elements such as:

- handles
- shortened URLs
- historical signatures
- irrelevant trailing fragments

This makes the resulting draft more suitable for review.

When the escalation gate decides that human review is required, the draft is explicitly treated as an internal draft rather than an automatic customer-facing response.

---

# 10. Escalation Design

The escalation system is one of the most important components of the project.

It combines four signals:

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

1. A high-risk pattern is detected.
2. The predicted intent is high risk.
3. Intent confidence is too low.
4. Historical retrieval similarity is too low.

This separates:

```text
Can I classify this?
```

from:

```text
Should I automate this?
```

That distinction is important for support systems.

---

# 11. Risk Detection

The risk detector explicitly handles several sensitive situations.

## Security Risk

Examples:

- hacked account
- compromised account
- unauthorized access
- identity theft
- suspicious account activity

## Financial Risk

Examples:

- unauthorized charge
- unknown charge
- fraudulent charge
- duplicate charge

## Legal Risk

Examples:

- lawyer
- attorney
- lawsuit
- legal action
- court

## Explicit Human Request

Examples:

- speak to a human
- speak to an agent
- talk to a representative
- contact a supervisor

These requests are escalated independently of normal classification confidence.

---

# 12. High-Risk Intents

The following intents are treated conservatively:

```text
account_security
payment_billing
```

This reflects the potential sensitivity of account and financial requests.

A historically similar response does not automatically make a sensitive request safe to automate.

The escalation layer therefore acts as a policy boundary around the classifier and retrieval system.

---

# 13. Confidence Threshold

The selected intent-confidence threshold is:

```text
0.20
```

Requests below this confidence are escalated.

The reasoning is straightforward:

```text
Low confidence
      |
      v
Higher uncertainty
      |
      v
Human review
```

This is especially important for short, ambiguous, or linguistically unusual customer messages.

---

# 14. Retrieval Threshold

The selected retrieval-similarity threshold is:

```text
0.20
```

If the best historical match falls below this threshold, the request is escalated.

The reasoning is:

```text
Weak historical evidence
          |
          v
Uncertain grounding
          |
          v
Human review
```

Retrieval similarity is not treated as a standalone quality metric.

It is used as one signal in the broader automation decision.

---

# 15. Threshold Tuning

Multiple confidence and retrieval thresholds were evaluated.

The selected configuration is:

| Parameter | Value |
|---|---:|
| Intent confidence threshold | 0.20 |
| Retrieval similarity threshold | 0.20 |

The resulting escalation performance is:

| Metric | Result |
|---|---:|
| Precision | 39.26% |
| Recall | 95.52% |
| F1 | 55.65% |
| Escalation Rate | 81.50% |

The selected policy intentionally favors recall.

This means the system is willing to send more requests to human review in order to reduce the chance of automatically handling a request that should have been escalated.

---

# 16. End-to-End Evaluation

The complete system was evaluated on 200 examples.

Results:

| Metric | Result |
|---|---:|
| Intent Accuracy | 53.00% |
| Intent Macro F1 | 52.81% |
| Escalation Precision | 39.26% |
| Escalation Recall | 95.52% |
| Escalation F1 | 55.65% |
| Auto-Handle Rate | 18.50% |
| Escalation Rate | 81.50% |

The system made:

```text
163 escalation decisions
37 automatic-handling decisions
```

The current operating point is therefore strongly conservative.

---

# 17. Escalation Breakdown

The 163 escalations were caused by:

| Reason | Count |
|---|---:|
| Low intent confidence | 123 |
| High-risk intent | 22 |
| Low historical similarity | 11 |
| Legal/high-risk signal | 4 |
| Security/account risk | 2 |
| Payment/financial risk | 1 |

The most important observation is that:

```text
123 / 163
```

escalations were caused by low intent confidence.

This means classifier uncertainty is currently the dominant factor limiting automatic handling.

---

# 18. Representative End-to-End Example

Input:

```text
Where is my order? It was supposed to arrive yesterday.
```

The system produced:

```text
Intent:
order_delivery

Intent Confidence:
0.1388

Retrieval Similarity:
0.6653

Escalate:
True

Reason:
low_intent_confidence
```

The retrieved historical response produced a cleaned internal draft:

```text
Oh no! I'm sorry to hear that. What is the last tracking update on the order?
```

The important behavior is that the historical match was reasonably strong, but the request was still escalated because intent confidence was below the configured threshold.

This demonstrates the safety-first decision structure.

---

# 19. Failure Mode 1 — Excessive Escalation

### Observation

The largest current weakness is excessive escalation.

Most escalations are triggered by:

```text
low_intent_confidence
```

### Evidence

```text
123 of 163 escalations
```

were caused by low classifier confidence.

### Hypothesis

The classifier has limited labeled training data relative to the diversity of the support corpus.

Short customer messages are particularly difficult because they may contain very few discriminative words.

### Improvement

Increase labeled training data, especially for:

- minority intents
- ambiguous cases
- short messages
- commonly confused categories

Confidence calibration should also be added before lowering the escalation threshold.

---

# 20. Failure Mode 2 — Delivery vs Product Issues

The classifier can confuse:

```text
order_delivery
```

with:

```text
product_order_issue
```

Both can contain terms such as:

- order
- package
- item
- product
- delivery

The important distinction is the customer's underlying problem.

If the issue is:

```text
Where is my package?
```

the intent is delivery.

If the package arrived but:

```text
The product is damaged.
```

the intent is a product issue.

### Improvement

Add more boundary examples and consider hierarchical classification.

A broad routing stage could first determine:

```text
Shipping
Product
Financial
Account
Digital
Technical
```

and a second stage could classify the specific intent.

---

# 21. Failure Mode 3 — Refund vs Billing

The classifier can confuse:

```text
returns_refunds
```

with:

```text
payment_billing
```

because both categories can contain financial language.

The key distinction is:

```text
Returns/refunds
=
money expected back
```

versus:

```text
Payment/billing
=
money charged or payment processing problem
```

### Improvement

Increase examples covering:

- unexpected charges
- duplicate charges
- refund requests
- refund delays
- return-related refunds
- payment failures

---

# 22. Failure Mode 4 — Device vs Digital Content

Technical vocabulary can occur in both:

```text
device_technical
```

and:

```text
digital_content
```

A customer may describe a content problem through the device being used.

### Improvement

Use semantic representations and product/entity context so that the system can distinguish:

```text
Problem with device
```

from:

```text
Problem with content accessed through device
```

---

# 23. Failure Mode 5 — Generic Historical Responses

Some historical support responses are short and repetitive.

This creates a retrieval limitation.

A high lexical similarity score does not necessarily mean the historical response is the best response for the new customer.

Therefore:

```text
High lexical similarity
```

does not guarantee:

```text
High semantic relevance
```

### Improvement

Use a two-stage retrieval architecture:

```text
Semantic Retrieval
       |
       v
Top K Candidates
       |
       v
Reranking
       |
       v
Best Grounding Example
```

The reranker should consider the relationship between:

- customer problem
- predicted intent
- historical customer message
- historical support response

---

# 24. Confusion Matrix Insights

The current evaluation shows that some categories are substantially easier than others.

Examples from the 200-example evaluation include:

| Intent | Correct |
|---|---:|
| `order_delivery` | 17 / 35 |
| `customer_service_complaint` | 16 / 28 |
| `returns_refunds` | 13 / 19 |
| `digital_content` | 11 / 21 |
| `device_technical` | 9 / 19 |
| `subscription_prime` | 9 / 13 |
| `account_security` | 8 / 13 |
| `payment_billing` | 7 / 15 |
| `product_order_issue` | 5 / 18 |

These results show that the most difficult categories are not necessarily the smallest categories.

The main issue is overlap in language and the limited amount of labeled data available for learning fine distinctions.

---

# 25. What Is Misleading About the Headline Number?

The headline end-to-end intent accuracy is:

```text
53.0%
```

It is useful, but it should not be interpreted as a production accuracy estimate.

The current 200-example evaluation set was prepared and reviewed during development.

The classifier development labels were also created as part of the development process.

Therefore, the result should primarily be used to:

- compare future versions
- detect regressions
- identify weaknesses
- guide engineering decisions

A stronger benchmark would require an independently hand-labeled and frozen evaluation set.

This distinction is important because a benchmark is only as reliable as its labels and evaluation protocol.

---

# 26. Why Accuracy Alone Is Not Enough

Customer-support intent data is not perfectly balanced.

A system can obtain reasonable accuracy while performing poorly on minority classes.

For this reason, the project reports:

```text
Accuracy
Macro F1
Per-class performance
Confusion matrix
```

Macro F1 is especially useful because it gives every intent equal importance.

For example, a model that performs very well on `order_delivery` but poorly on `payment_billing` should not appear strong merely because delivery examples are frequent.

---

# 27. Why Escalation Recall Matters

Escalation has a different risk profile from ordinary classification.

A false negative means:

```text
Request should have been reviewed
                |
                v
Request was automatically handled
```

A false positive means:

```text
Normal request
      |
      v
Human review unnecessarily
```

The first error can be more serious.

Therefore, the system currently favors escalation recall.

The measured recall is:

```text
95.52%
```

The trade-off is lower precision:

```text
39.26%
```

This explains the high escalation rate.

---

# 28. Engineering Assessment

The current architecture has several strong properties.

## Strong Separation of Responsibilities

Classification, retrieval, risk detection and escalation are independent components.

## Grounded Response Drafting

Historical interactions provide concrete response evidence.

## Conservative Decision Policy

Sensitive and uncertain requests are not automatically treated as safe.

## Leakage-Aware Retrieval

Evaluation examples are excluded from the leakage-free retrieval index.

## Reproducible Scripts

Major stages are implemented as executable scripts.

## Automated Regression Tests

The core decision logic is covered by ten tests.

---

# 29. Current Limitations

The main limitations are:

### Limited Labeled Dataset

More labeled examples are needed for robust classification.

### Lexical Retrieval

TF-IDF is sensitive to word overlap and does not fully capture semantic similarity.

### High Escalation Rate

The current safety policy escalates 81.5% of evaluated examples.

### Confidence Calibration

Raw classifier confidence is useful as a signal but should be calibrated before being treated as a probability of correctness.

### Evaluation Maturity

The evaluation benchmark should be strengthened through independent human annotation and agreement measurement.

### No Live Customer Context

The system does not directly query:

- live order status
- customer accounts
- payment systems
- inventory
- shipping APIs

Therefore, it prepares grounded responses from historical evidence rather than resolving live transactional issues.

---

# 30. Recommended Next Architecture

The next version should evolve from:

```text
TF-IDF Classification
+
TF-IDF Retrieval
+
Rules
```

toward:

```text
Semantic Intent Model
        |
        v
Calibrated Confidence
        |
        v
Semantic Retrieval
        |
        v
Intent-Aware Reranking
        |
        v
Grounded Response
        |
        v
Policy Validation
        |
        v
Automation / Human Review
```

The current modular architecture makes this migration possible without rewriting the entire system.

---

# 31. One-Week Improvement Plan

## Day 1 — Stronger Evaluation Set

Create a larger independently reviewed evaluation set.

Target:

```text
200-250 examples
```

Ensure coverage across all ten intents.

Include both common and difficult examples.

---

## Day 2 — Human Agreement

Have two reviewers independently label a calibration subset.

Measure agreement.

Investigate disagreements.

Update the labeling guide where necessary.

---

## Day 3 — Semantic Retrieval

Introduce sentence embeddings.

Compare:

```text
TF-IDF Retrieval
```

with:

```text
Semantic Retrieval
```

using the same frozen evaluation set.

---

## Day 4 — Intent-Aware Retrieval

Use predicted intent to restrict or rerank historical candidates.

This should reduce cross-category matches.

---

## Day 5 — Response Evaluation

Evaluate response drafts on:

- relevance
- groundedness
- helpfulness
- factual support
- clarity

---

## Day 6 — Confidence Calibration

Calibrate classifier probabilities and re-evaluate escalation thresholds.

The objective is:

```text
Lower unnecessary escalation
+
Maintain high escalation recall
```

---

## Day 7 — Final Benchmark

Freeze:

- taxonomy
- training data
- evaluation data
- model
- retrieval system
- escalation policy
- evaluation scripts

Then run one final benchmark and record the results.

---

# 32. Productionization Roadmap

A production system would require additional infrastructure.

Important future components include:

- authenticated customer context
- order-status APIs
- account verification
- payment-status APIs
- policy validation
- audit logging
- rate limiting
- monitoring
- alerting
- model versioning
- data drift monitoring
- response approval workflows

The current project focuses on the core intelligence and decision layer.

---

# 33. Final Recommendation

The highest-value improvement is not simply replacing the classifier with a larger model.

The current results indicate that the primary bottleneck is the quality and quantity of labeled data.

The recommended order of work is:

```text
Better Evaluation Labels
          |
          v
More Training Examples
          |
          v
Better Intent Model
          |
          v
Confidence Calibration
          |
          v
Semantic Retrieval
          |
          v
Better Response Reranking
          |
          v
Lower Unnecessary Escalation
```

This sequence improves the measurement foundation before increasing system complexity.

---

# 34. Conclusion

SupportAgentAI demonstrates a complete architecture for grounded customer-support automation.

The project combines:

- real customer-support conversation data
- structured preprocessing
- operational intent classification
- historical retrieval
- leakage-aware evaluation
- response drafting
- risk detection
- confidence-based escalation
- retrieval-based escalation
- automated tests
- failure analysis

The current system achieves:

```text
53.00% Intent Accuracy
52.81% Intent Macro F1
95.52% Escalation Recall
55.65% Escalation F1
18.50% Auto-Handle Rate
```

The strongest property is the conservative escalation behavior, particularly the high escalation recall.

The largest weakness is excessive escalation caused by classifier uncertainty.

The next iteration should therefore focus on improving labeled data, confidence calibration, semantic retrieval, and response reranking.

The long-term objective is not maximum automation at any cost.

The objective is:

> Safe, grounded and measurable automation with clear human oversight whenever the system is uncertain.

---

## Author

**Vikas Kumar**

Software Engineer | Full Stack Developer | Python | Machine Learning | Backend Systems

SupportAgentAI represents a practical engineering approach to building reliable, interpretable and evaluation-driven customer-support automation.
