# AmazonHelp Intent Labeling Guide

## Purpose

This guide defines the intent taxonomy used to manually label the golden evaluation set for the AmazonHelp support agent.

## Labels

### order_delivery

Use when the customer is asking about delivery, shipment, tracking, delayed packages, missing packages, delivery attempts, estimated delivery dates, or delivery location.

Examples:
- My package has not arrived.
- Tracking says delivered but I don't have it.
- Why is my delivery late?
- When will my order arrive?

Do not use for refund requests unless the primary request is about the refund.

### returns_refunds

Use when the primary issue concerns returning an item, receiving a refund, refund eligibility, or refund status.

Examples:
- Where is my refund?
- Can I return this item?
- I returned the product but haven't received my refund.
- Will you refund the shipping charge?

### payment_billing

Use when the primary issue concerns charges, payment methods, cards, billing, unexpected transactions, or payment failures.

Examples:
- I was charged twice.
- I don't recognize this Amazon charge.
- Why was my card charged?
- My payment failed.

### account_security

Use for account access, account closure, unauthorized activity, suspicious messages, security concerns, or suspected fraud involving the account.

Examples:
- I want to close my account.
- Someone used my account.
- I received a suspicious Amazon email.
- I cannot access my account.

### subscription_prime

Use for Amazon Prime membership, Prime benefits, Prime subscription charges, or Prime membership cancellation.

Examples:
- Why is my Prime membership charged?
- How do I cancel Prime?
- My Prime benefits aren't working.

### digital_content

Use for Prime Video, Kindle content, streaming, digital media, playback, or digital-content availability issues.

Examples:
- Prime Video keeps showing an error.
- My video won't play.
- I cannot access a Kindle book.

### device_technical

Use for Amazon devices or technical problems involving devices, apps, or hardware.

Examples:
- My Fire TV Stick isn't working.
- My Kindle screen is broken.
- The Amazon app keeps crashing.

### product_order_issue

Use for product availability, product information, seller-related questions, product compatibility, or order-level issues that are not primarily delivery, payment, or returns.

Examples:
- Is this product available?
- Can I exchange this item?
- Why is this product showing the wrong information?
- Is this seller legitimate?

### customer_service_complaint

Use when the primary purpose is complaining about Amazon/customer support, repeated failed resolution, poor service, or requesting human intervention because previous support failed.

Examples:
- I've contacted support three times and nobody helped.
- Your customer service keeps giving me different answers.
- This issue has been unresolved for weeks.

### other_non_actionable

Use for messages that do not contain a clear support request or actionable customer issue.

Examples:
- Thanks for the information.
- Okay, understood.
- General praise.
- General product enthusiasm.
- Ambiguous messages with insufficient information.

## Annotation Rules

1. Label the customer's primary intent.
2. Do not infer an intent that is unsupported by the message.
3. If multiple issues are present, choose the issue that requires the customer's primary resolution.
4. If the message is primarily a complaint about unresolved support, use customer_service_complaint.
5. If a delivery problem also asks for a refund, choose returns_refunds when the refund is the primary request.
6. Use other_non_actionable when no reliable intent can be assigned.
7. Preserve uncertainty in the notes field rather than inventing information.