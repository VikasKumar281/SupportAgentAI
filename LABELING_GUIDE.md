# SupportAgentAI — Labeling Guide

**Author:** Vikas Kumar

This guide defines the intent and escalation labels used by SupportAgentAI.

## 1. Core Principle

Each customer message receives one primary intent.

Choose the customer's main operational need, not every topic mentioned in the message.

Intent and escalation are separate labels.

## 2. Labeling Procedure

1. Read the complete customer message.
2. Identify the main requested outcome.
3. Ignore unnecessary conversational or emotional text.
4. Determine the operational support problem.
5. Select exactly one intent.
6. Use `other_non_actionable` only when no defined operational intent is sufficiently clear.
7. Separately decide whether human review is required.

## 3. Intent Taxonomy

| Intent | Definition |
|---|---|
| `order_delivery` | Delivery status, tracking, delayed or missing packages |
| `returns_refunds` | Returns, cancellations and refunds |
| `payment_billing` | Charges, billing and payment problems |
| `account_security` | Account access and security concerns |
| `subscription_prime` | Prime and subscription issues |
| `digital_content` | Digital books, video, music and digital-content issues |
| `device_technical` | Device problems and technical troubleshooting |
| `product_order_issue` | Wrong, damaged, defective or problematic products |
| `customer_service_complaint` | Complaints about the support experience |
| `other_non_actionable` | Unclear, informational or non-actionable messages |

## 4. Intent Definitions

### `order_delivery`

Use when the main issue is where or when a package will arrive.

Include tracking, delayed delivery, missing packages, expected delivery, shipment status, and package arrival.

If the package arrived but the product is damaged, defective, or incorrect, use `product_order_issue`.

### `returns_refunds`

Use when the customer primarily wants to return an item, cancel for a refund, or recover money.

If the main issue is an unexpected charge or payment failure, use `payment_billing`.

### `payment_billing`

Use for charges, billing, payment methods, unexpected charges, duplicate charges, or payment failures.

### `account_security`

Use for account access or security concerns such as hacked accounts, compromised accounts, unauthorized access, suspicious activity, or identity-related concerns.

### `subscription_prime`

Use for Prime membership, subscription status, subscription cancellation, benefits, or membership questions.

### `digital_content`

Use for digital books, video, music, digital purchases, or digital-content access.

### `device_technical`

Use when the device itself is the source of the problem, including setup, hardware, malfunction, connectivity, or troubleshooting.

### `product_order_issue`

Use for wrong, damaged, defective, incorrect, or poor-quality products.

If the only problem is that the package has not arrived, use `order_delivery`.

### `customer_service_complaint`

Use when the support experience itself is the main subject, such as poor support, unresolved support, complaints about representatives, or repeated unresolved contact.

### `other_non_actionable`

Use when there is no sufficiently clear operational support intent.

Examples include informational messages, genuinely ambiguous messages, non-actionable statements, or messages outside the taxonomy.

Do not use this category merely because a message is difficult.

## 5. Multi-Intent Messages

Choose the primary requested outcome.

Example:

```text
My package is late and I want a refund.
```

If the main issue is the missing delivery, use `order_delivery`.

If the primary request is recovering money, use `returns_refunds`.

## 6. Escalation Label

Escalation answers:

> Should this request receive human review rather than automatic handling?

A message can have any intent with either `True` or `False` escalation.

## 7. Escalation Conditions

Escalate when there is a strong safety or operational signal.

### Security

- hacked account
- compromised account
- unauthorized access
- identity theft

### Financial

- unauthorized charge
- fraudulent charge
- unknown charge
- duplicate charge

### Legal

- lawyer
- attorney
- lawsuit
- legal action
- court

### Explicit Human Request

- speak to a human
- speak to an agent
- talk to a representative
- contact a supervisor

## 8. Uncertainty-Based Escalation

The current system also escalates when automated evidence is weak:

```text
Intent Confidence < 0.20
Retrieval Similarity < 0.20
```

These are system decision thresholds, not human labels.

## 9. Important Confusion Boundaries

### `order_delivery` vs `product_order_issue`

Ask whether the customer is waiting for delivery or reporting a problem with the product itself.

### `returns_refunds` vs `payment_billing`

Ask whether the customer wants money back or is reporting a charge/payment problem.

### `device_technical` vs `digital_content`

Ask whether the device or the content is the main source of the problem.

### `customer_service_complaint` vs `other_non_actionable`

Ask whether the customer is actively complaining about the support experience.

## 10. Final Human-Gold Benchmark

The project contains 200 manually labeled evaluation examples covering all ten intents.

The author completed the human intent and escalation labels using this guide.

Because the benchmark has one annotator, inter-annotator agreement is not claimed.

## 11. Review Checklist

Before finalizing a label:

- Did I read the complete message?
- Did I identify the main requested outcome?
- Did I choose one primary intent?
- Did I distinguish delivery from product problems?
- Did I distinguish refunds from charges?
- Did I separately consider escalation?
- Did I avoid guessing when the intent is unclear?

## 12. Final Principle

> Label the customer's primary operational need, not simply the words that appear most frequently in the message.
