# SupportAgentAI — Engineering Decision Log

**Author:** Vikas Kumar

This document records the major engineering decisions made while building SupportAgentAI.

The purpose of the decision log is to explain not only what was implemented, but why each architectural choice was made.

---

## Decision 1 — Use a Single Support Environment

### Decision

Use AmazonHelp as the primary support environment.

### Reason

The system needs historical responses that are relevant to the same support context.

Using one support account reduces variation in:

- terminology
- support style
- operational context
- response patterns

### Result

The historical retrieval corpus remains internally consistent.

---

## Decision 2 — Build Customer/Support Conversation Pairs

### Decision

Represent each historical interaction as a customer message paired with its corresponding support response.

### Reason

The retrieval system needs to answer:

> What did support say when a similar customer request appeared?

### Result

Each historical record becomes:

```text
Customer Message
       +
Support Response
```

This structure is directly useful for retrieval and response drafting.

---

## Decision 3 — Use a Compact Intent Taxonomy

### Decision

Use ten operational intents.

### Reason

A taxonomy with too many classes would increase ambiguity and make training difficult.

A very small taxonomy would lose useful routing information.

Ten categories provide a practical middle ground.

### Result

The final intents are:

```text
order_delivery
returns_refunds
payment_billing
account_security
subscription_prime
digital_content
device_technical
product_order_issue
customer_service_complaint
other_non_actionable
```

---

## Decision 4 — Separate Classification from Retrieval

### Decision

Keep intent classification and historical retrieval as separate components.

### Reason

Classification answers:

> What is the customer asking?

Retrieval answers:

> What similar support interaction exists?

Keeping them separate makes the system easier to debug and improve.

### Result

The classifier and retrieval index can be upgraded independently.

---

## Decision 5 — Use a Majority Baseline

### Decision

Implement a majority-class baseline before training a real classifier.

### Reason

A simple baseline establishes the minimum expected performance.

Without a baseline, model performance is difficult to interpret.

### Result

The majority baseline achieved:

```text
Accuracy: 36.00%
Macro F1: 5.29%
```

---

## Decision 6 — Use TF-IDF + Logistic Regression

### Decision

Use TF-IDF features with Logistic Regression as the primary classifier.

### Reason

The approach is:

- fast
- transparent
- reproducible
- inexpensive
- suitable for short text

It provides a strong conventional baseline before moving to more complex semantic models.

### Result

Development test performance:

```text
Accuracy: 45.33%
Macro F1: 34.96%
```

---

## Decision 7 — Use Confidence as a Safety Signal

### Decision

Use classifier confidence in the escalation policy.

### Reason

A classification system should not automatically act when it is uncertain.

### Result

Requests with confidence below 0.20 are escalated.

---

## Decision 8 — Add Explicit Risk Detection

### Decision

Add deterministic detection for security, financial, legal and explicit-human-request signals.

### Reason

Certain requests require conservative handling even when a classifier is confident.

### Result

Sensitive requests are routed to human review.

---

## Decision 9 — Treat Account Security and Payment as High-Risk Intents

### Decision

Escalate `account_security` and `payment_billing` conservatively.

### Reason

These categories may involve:

- account verification
- sensitive customer information
- financial consequences

Historical response similarity alone is not enough to safely resolve such cases.

---

## Decision 10 — Add Retrieval Similarity to Escalation

### Decision

Use retrieval similarity as another safety signal.

### Reason

A weak historical match should not be treated as strong grounding evidence.

### Result

Requests with retrieval similarity below 0.20 are escalated.

---

## Decision 11 — Prevent Retrieval Leakage

### Decision

Create a leakage-free retrieval index for evaluation.

### Reason

If evaluation examples remain inside the retrieval corpus, the system may retrieve itself.

That would inflate retrieval similarity and produce misleading evaluation results.

### Result

Evaluation examples are excluded from the leakage-free retrieval index.

---

## Decision 12 — Clean Historical Responses

### Decision

Remove platform-specific artifacts from historical responses before displaying them as drafts.

### Reason

Historical support messages can contain:

- handles
- shortened URLs
- signatures
- irrelevant trailing fragments

These should not be copied directly into a new response.

### Result

The generated draft is cleaner and easier to review.

---

## Decision 13 — Keep Escalated Responses as Internal Drafts

### Decision

When a request is escalated, show the response as an internal draft rather than an automatic customer response.

### Reason

The system has already determined that human review is required.

### Result

The workflow becomes:

```text
Escalated Request
       |
       v
Internal Draft
       |
       v
Human Review
```

---

## Decision 14 — Prioritize Escalation Recall

### Decision

Tune escalation thresholds with recall as an important objective.

### Reason

A missed sensitive request can be more costly than unnecessary human review.

### Result

The selected policy achieves:

```text
Escalation Recall: 95.52%
```

while accepting a lower precision.

---

## Decision 15 — Preserve Per-Example Evaluation Data

### Decision

Store individual predictions and decisions.

### Reason

Aggregate metrics do not reveal why individual cases fail.

Per-example results make it possible to inspect:

- wrong intents
- low confidence
- weak retrieval
- unnecessary escalation
- missed escalation

### Result

Failure-analysis artifacts are generated under:

```text
data/processed/
```

---

## Decision 16 — Use Macro F1 Alongside Accuracy

### Decision

Report both Accuracy and Macro F1.

### Reason

Accuracy can hide poor performance on less frequent intents.

Macro F1 gives each class equal importance.

### Result

The evaluation better reflects class-level weaknesses.

---

## Decision 17 — Keep Generated Artifacts Outside Git

### Decision

Do not commit the raw dataset, generated CSVs or serialized model files.

### Reason

These artifacts can be large and are reproducible from the pipeline.

### Result

The repository remains focused on:

- source code
- tests
- configuration
- documentation

---

## Decision 18 — Add Automated Regression Tests

### Decision

Create tests for the main escalation and pipeline behavior.

### Reason

Changes to the decision logic should not silently break safety behavior.

### Result

The current test suite contains ten tests covering:

- dataset availability
- model availability
- security escalation
- payment escalation
- explicit human requests
- low confidence
- low retrieval similarity
- safe handling
- normal messages

Current result:

```text
10 passed
```

---

# Decision Summary

The overall architecture reflects the following priorities:

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

The project deliberately favors a system that can explain why it acted or escalated.

---

# Future Decisions to Revisit

The following decisions should be revisited as the system grows:

1. TF-IDF versus semantic embeddings.
2. Flat intent classification versus hierarchical classification.
3. Single retrieval candidate versus multi-candidate reranking.
4. Static thresholds versus calibrated confidence.
5. Rule-based risk detection versus trained risk classification.
6. Development evaluation versus independently reviewed benchmark.

These are the highest-value architectural decisions for the next iteration.
