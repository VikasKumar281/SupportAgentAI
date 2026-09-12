# SupportAgentAI — Labeling Guide

**Author:** Vikas Kumar

This document defines the labeling framework used to organize customer-support requests into operational intent categories.

The objective is to make labels consistent, understandable, and useful for both model training and evaluation.

---

# 1. Labeling Principles

Each customer message should receive one primary intent.

The primary intent should represent the customer's main support problem rather than every topic mentioned in the message.

When a message contains multiple issues, choose the issue that appears to be the main reason for contacting support.

---

# 2. General Labeling Procedure

For every message:

1. Read the complete customer message.
2. Identify the customer's main requested outcome.
3. Ignore unnecessary emotional or conversational text.
4. Determine the operational support problem.
5. Select exactly one intent.
6. If no intent is sufficiently clear, use `other_non_actionable`.
7. Separately determine whether the request should be escalated.

The intent and escalation decision are separate labels.

---

# 3. Intent Taxonomy

The available intents are:

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

# 4. order_delivery

## Definition

The customer is asking about shipment or delivery.

## Include

- package tracking
- delayed delivery
- missing package
- expected delivery
- delivery status
- shipment status
- package arrival

## Exclude

If the package arrived but the product is damaged or incorrect, use:

```text
product_order_issue
```

## Decision Question

> Is the main problem where or when the package will arrive?

If yes, use `order_delivery`.

---

# 5. returns_refunds

## Definition

The customer wants to return something or receive money back.

## Include

- return requests
- refund requests
- refund status
- return eligibility
- cancellation-related refund questions

## Exclude

If the main problem is an unexpected charge or payment failure, use:

```text
payment_billing
```

## Decision Question

> Is the customer primarily trying to return an item or recover money?

If yes, use `returns_refunds`.

---

# 6. payment_billing

## Definition

The customer's main issue concerns a payment or charge.

## Include

- payment failures
- billing problems
- unexpected charges
- duplicate charges
- payment methods
- card-related payment issues

## Decision Question

> Is the customer primarily concerned about money being charged or a payment being processed?

If yes, use `payment_billing`.

---

# 7. account_security

## Definition

The customer has an account-access or security problem.

## Include

- hacked account
- compromised account
- unauthorized access
- suspicious account activity
- stolen account
- identity-related account concerns

## Decision Question

> Is account access or account security the main issue?

If yes, use `account_security`.

---

# 8. subscription_prime

## Definition

The request is primarily about a subscription or Prime membership.

## Include

- Prime membership
- subscription status
- subscription cancellation
- Prime benefits
- membership questions

---

# 9. digital_content

## Definition

The customer is having an issue with digital content.

## Include

- digital books
- digital video
- digital music
- digital purchases
- digital-content access

## Decision Question

> Is the customer's main problem with digital content rather than the physical device?

If yes, use `digital_content`.

---

# 10. device_technical

## Definition

The customer's main problem concerns a device or technical behavior.

## Include

- device setup
- device malfunction
- technical troubleshooting
- hardware problems
- device connectivity
- device-specific issues

## Decision Question

> Is the device itself the main source of the problem?

If yes, use `device_technical`.

---

# 11. product_order_issue

## Definition

The customer has a problem with the actual product or item associated with an order.

## Include

- wrong product
- damaged product
- defective product
- incorrect item
- product quality issue

## Exclude

If the customer is simply waiting for delivery, use:

```text
order_delivery
```

---

# 12. customer_service_complaint

## Definition

The customer is primarily complaining about the support experience.

## Include

- poor support experience
- unresolved support
- complaints about representatives
- dissatisfaction with customer service
- repeated unresolved contact

The important characteristic is that the support experience itself is the main subject.

---

# 13. other_non_actionable

## Definition

Use this category when no operational intent can be determined confidently.

## Include

- informational messages
- ambiguous messages
- non-actionable statements
- messages outside the defined taxonomy
- messages without a clear support request

This category should not be used simply because a message is difficult.

It should be used when the message genuinely lacks a sufficiently clear operational intent.

---

# 14. Multi-Intent Messages

Some customer messages contain more than one issue.

Use the primary customer need.

For example:

```text
My package is late and I want a refund.
```

The correct label depends on the customer's primary requested outcome.

If the main issue is the missing delivery:

```text
order_delivery
```

If the customer is primarily asking to recover money:

```text
returns_refunds
```

The goal is to capture the main operational reason for the interaction.

---

# 15. Escalation Labeling

Escalation is separate from intent.

A message can be:

```text
Intent:
order_delivery

Escalation:
True
```

or:

```text
Intent:
order_delivery

Escalation:
False
```

The escalation label answers:

> Should this request receive human review rather than automatic handling?

---

# 16. Escalation Conditions

Escalate when the request contains a strong safety or operational signal.

## Security

Examples:

- hacked account
- compromised account
- unauthorized access
- identity theft

## Financial

Examples:

- unauthorized charge
- fraudulent charge
- unknown charge
- duplicate charge

## Legal

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

---

# 17. Uncertainty-Based Escalation

Even when no explicit risk phrase exists, a request may require human review if the automated system is uncertain.

The current system uses:

```text
Intent Confidence < 0.20
```

as an escalation condition.

It also uses:

```text
Retrieval Similarity < 0.20
```

as an escalation condition.

These thresholds are system-level decision rules rather than human intent labels.

---

# 18. Annotation Consistency

Reviewers should avoid guessing.

If two categories appear possible:

1. Identify the customer's requested outcome.
2. Determine which problem is central.
3. Apply the category definitions.
4. Use `other_non_actionable` only when the intent remains genuinely unclear.

The goal is consistency rather than forcing every message into a highly specific category.

---

# 19. Recommended Review Sheet

A final human-review sheet should contain:

| Column | Description |
|---|---|
| customer_tweet_id | Original customer identifier |
| customer_text | Customer message |
| predicted_intent | System prediction |
| gold_intent | Reviewer label |
| predicted_escalate | System escalation |
| gold_escalate | Reviewer escalation label |
| human_review_status | Review state |
| reviewer_notes | Explanation for difficult cases |

---

# 20. Human Review Workflow

The recommended workflow is:

```text
Sample Examples
       |
       v
First Independent Review
       |
       v
Second Independent Review
       |
       v
Compare Labels
       |
       v
Resolve Disagreements
       |
       v
Freeze Gold Labels
```

Reviewers should label independently before seeing the model prediction whenever possible.

This reduces confirmation bias.

---

# 21. Calibration

Before labeling the final benchmark, reviewers should jointly review a small calibration subset.

The calibration process should identify:

- ambiguous intent definitions
- common confusion pairs
- unclear escalation situations
- taxonomy gaps

The guide should be updated before the final benchmark is frozen.

---

# 22. Common Confusion Pairs

The following categories require particular attention.

## order_delivery vs product_order_issue

Focus on whether the customer is waiting for delivery or reporting a product problem.

## returns_refunds vs payment_billing

Focus on whether the customer wants money back or is reporting a charge/payment issue.

## device_technical vs digital_content

Focus on whether the device or the content is the main source of the problem.

## customer_service_complaint vs other_non_actionable

Focus on whether the customer is actively expressing dissatisfaction with the support experience.

---

# 23. Label Quality Checklist

Before finalizing a label, ask:

- Did I read the complete message?
- Did I identify the main requested outcome?
- Did I choose one primary intent?
- Did I distinguish delivery from product problems?
- Did I distinguish refunds from charges?
- Did I separately consider escalation?
- Did I avoid guessing when the intent is unclear?

---

# 24. Final Labeling Principle

The most important rule is:

> Label the customer's primary operational need, not simply the words that appear most frequently in the message.

Consistent labels are essential for meaningful model evaluation.

