# SupportAgentAI

**Author:** Vikas Kumar

SupportAgentAI is a customer-support automation system designed to
understand customer messages, identify the underlying support intent,
retrieve relevant historical support interactions, prepare a grounded
response, and determine whether the request can be handled automatically
or should be reviewed by a human.

The system is designed around a simple principle:

> Understand the customer's request first, find relevant historical
> evidence, evaluate confidence and risk, and only then decide whether
> automatic handling is appropriate.

The project combines intent classification, historical retrieval,
response drafting, confidence estimation, risk detection, escalation
logic, evaluation, automated testing, and failure analysis into a single
end-to-end workflow.

------------------------------------------------------------------------

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Problem Statement](#2-problem-statement)
3.  [Project Goals](#3-project-goals)
4.  [System Architecture](#4-system-architecture)
5.  [Dataset](#5-dataset)
6.  [Data Extraction](#6-data-extraction)
7.  [Data Preprocessing](#7-data-preprocessing)
8.  [Processed Dataset](#8-processed-dataset)
9.  [Intent Taxonomy](#9-intent-taxonomy)
10. [Intent Definitions](#10-intent-definitions)
11. [Training Data](#11-training-data)
12. [Intent Classification](#12-intent-classification)
13. [Majority Baseline](#13-majority-baseline)
14. [TF-IDF + Logistic Regression](#14-tf-idf--logistic-regression)
15. [Historical Retrieval](#15-historical-retrieval)
16. [Retrieval Leakage Prevention](#16-retrieval-leakage-prevention)
17. [Response Drafting](#17-response-drafting)
18. [Escalation System](#18-escalation-system)
19. [Risk Detection](#19-risk-detection)
20. [Confidence-Based Escalation](#20-confidence-based-escalation)
21. [Retrieval-Based Escalation](#21-retrieval-based-escalation)
22. [End-to-End Pipeline](#22-end-to-end-pipeline)
23. [Evaluation Methodology](#23-evaluation-methodology)
24. [Evaluation Results](#24-evaluation-results)
25. [Escalation Results](#25-escalation-results)
26. [End-to-End Example](#26-end-to-end-example)
27. [Failure Analysis](#27-failure-analysis)
28. [What Is Misleading About the Headline
    Number](#28-what-is-misleading-about-the-headline-number)
29. [Why Accuracy Alone Is Not
    Enough](#29-why-accuracy-alone-is-not-enough)
30. [Why Escalation Recall Matters](#30-why-escalation-recall-matters)
31. [Engineering Decisions](#31-engineering-decisions)
32. [Repository Structure](#32-repository-structure)
33. [Installation](#33-installation)
34. [Dataset Setup](#34-dataset-setup)
35. [Running the Data Pipeline](#35-running-the-data-pipeline)
36. [Training the Classifier](#36-training-the-classifier)
37. [Building the Retrieval Index](#37-building-the-retrieval-index)
38. [Running Evaluation](#38-running-evaluation)
39. [Interactive Usage](#39-interactive-usage)
40. [Testing](#40-testing)
41. [Generated Artifacts](#41-generated-artifacts)
42. [Current Strengths](#42-current-strengths)
43. [Current Limitations](#43-current-limitations)
44. [Future Improvements](#44-future-improvements)
45. [One-Week Improvement Plan](#45-one-week-improvement-plan)
46. [Final Architecture](#46-final-architecture)
47. [Conclusion](#47-conclusion)
48. [Author](#48-author)

------------------------------------------------------------------------

# 1. Project Overview

SupportAgentAI is an end-to-end customer-support system built around
historical customer-support conversations.

The system is designed to process a customer message and determine:

1.  What the customer is asking about.
2.  Whether there is relevant historical evidence.
3.  Whether the request is safe to handle automatically.
4.  What response should be prepared.
5.  Whether human review is required.

The system does not treat customer support as a simple
response-generation problem.

Instead, it separates the workflow into multiple stages:

``` text
Customer Message
       |
       v
Text Processing
       |
       v
Intent Classification
       |
       v
Intent Confidence
       |
       v
Historical Retrieval
       |
       v
Retrieval Similarity
       |
       v
Risk Detection
       |
       v
Escalation Decision
       |
       +-----------------------------+
       |                             |
       v                             v
Human Review                  Automatic Handling
       |                             |
       v                             v
Internal Draft                 Grounded Reply
```

This architecture makes the system easier to understand, evaluate,
debug, and improve.

------------------------------------------------------------------------

# 2. Problem Statement

Customer-support teams receive a large volume of repetitive requests.

Many requests fall into recurring operational categories such as:

-   delivery delays
-   package tracking
-   missing orders
-   product returns
-   refund requests
-   payment problems
-   account-access issues
-   security concerns
-   subscription questions
-   digital-content problems
-   device troubleshooting
-   incorrect products
-   damaged products
-   customer-service complaints

A support automation system needs to do more than produce fluent text.

A useful system needs to understand the customer correctly before
deciding what to do.

The core problem can therefore be divided into four questions.

### 2.1 What is the customer asking?

This is the intent-classification problem.

### 2.2 Has a similar problem been handled before?

This is the historical retrieval problem.

### 2.3 Is the request safe to automate?

This is the risk and escalation problem.

### 2.4 What response should be prepared?

This is the response-drafting problem.

SupportAgentAI addresses all four stages as one pipeline.

------------------------------------------------------------------------

# 3. Project Goals

The main goals of the project are:

-   Build a practical customer-support intent classifier.
-   Create a compact operational support taxonomy.
-   Use historical support interactions as evidence.
-   Retrieve relevant previous support responses.
-   Prevent direct retrieval leakage during evaluation.
-   Detect sensitive support requests.
-   Escalate uncertain requests.
-   Prepare grounded support-response drafts.
-   Measure classification performance.
-   Measure escalation performance.
-   Analyze individual failure cases.
-   Keep the architecture simple and interpretable.
-   Provide a clear foundation for future improvements.

The project prioritizes safe and traceable automation over unrestricted
response generation.

------------------------------------------------------------------------

# 4. System Architecture

The system consists of five major layers.

## Layer 1 --- Data Layer

Historical customer-support conversations are extracted and converted
into structured customer/support pairs.

## Layer 2 --- Understanding Layer

A text classifier predicts the customer's primary support intent.

## Layer 3 --- Evidence Layer

Historical customer messages are searched to find similar support
interactions.

## Layer 4 --- Decision Layer

The system checks risk signals, intent confidence, and retrieval
quality.

## Layer 5 --- Response Layer

The system prepares either:

-   an automatic grounded reply
-   or an internal response draft for human review

The complete flow is:

``` text
                    Customer Message
                           |
                           v
                  Message Processing
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
                Retrieval Similarity
                           |
                           v
                    Risk Detection
                           |
                           v
                  Escalation Decision
                           |
              +------------+------------+
              |                         |
              v                         v
        Human Review              Auto Handling
              |                         |
              v                         v
       Internal Draft             Grounded Reply
```

Each stage produces an interpretable output that can be inspected
independently.

------------------------------------------------------------------------

# 5. Dataset

The project uses the Twitter Customer Support Conversations dataset.

The dataset contains customer-support interactions involving multiple
companies and support accounts.

The raw dataset contains approximately 2.8 million tweets.

The source data contains fields including:

  Field                     Description
  ------------------------- -----------------------------
  tweet_id                  Unique tweet identifier
  author_id                 Author identifier
  inbound                   Message direction
  created_at                Message timestamp
  text                      Original tweet text
  response_tweet_id         Response relationship
  in_response_to_tweet_id   Parent-message relationship

For this project, a single support environment was selected:

``` text
AmazonHelp
```

Using a single support account keeps the historical response corpus
consistent.

This is important because a response from the same support environment
is generally more relevant than a response from an unrelated
organization.

------------------------------------------------------------------------

# 6. Data Extraction

The raw dataset contains conversations from many different support
accounts.

The first stage therefore isolates the conversations associated with the
selected support account.

The extraction process follows these steps.

## Step 1 --- Load the raw dataset

The source dataset is placed at:

``` text
data/raw/twcs.csv
```

The raw dataset is large, so it is processed programmatically instead of
being manually opened in spreadsheet software.

## Step 2 --- Identify support messages

AmazonHelp messages are identified from the source data.

## Step 3 --- Resolve response relationships

The relationship fields are used to connect customer messages with
corresponding support responses.

The desired relationship is:

``` text
Customer Message
       |
       v
Support Response
```

## Step 4 --- Keep direct customer/support interactions

The project focuses on direct customer-to-support pairs.

## Step 5 --- Remove incomplete records

Records without usable customer messages or support responses are
removed.

## Step 6 --- Normalize the structure

The extracted conversations are converted into a consistent schema.

## Step 7 --- Remove duplicates

Duplicate records are removed so that repeated interactions do not
disproportionately influence training and retrieval.

------------------------------------------------------------------------

# 7. Data Preprocessing

The preprocessing stage converts raw conversation data into structured
examples.

Each usable record contains:

-   customer message
-   support response
-   customer identifier
-   customer tweet identifier
-   support tweet identifier
-   timestamps
-   support account

The preprocessing pipeline is:

``` text
Raw Twitter Data
       |
       v
Support Account Filtering
       |
       v
Response Relationship Matching
       |
       v
Customer/Support Pairing
       |
       v
Missing Data Filtering
       |
       v
Deduplication
       |
       v
Processed Conversation Dataset
```

The resulting dataset is suitable for both retrieval and downstream
analysis.

------------------------------------------------------------------------

# 8. Processed Dataset

The extraction process produced approximately 168,823 direct
customer/support response pairs before final cleaning and deduplication.

After cleaning and deduplication, approximately 149,680 usable original
pairs remained.

The processed dataset is stored at:

``` text
data/processed/amazonhelp_conversations.csv
```

The processed schema is:

  Column                Description
  --------------------- -----------------------------
  customer_tweet_id     Customer tweet identifier
  brand_tweet_id        Support response identifier
  customer_author_id    Customer identifier
  brand                 Support account
  customer_created_at   Customer message timestamp
  brand_created_at      Support response timestamp
  customer_text         Customer's original message
  brand_response        Historical support response

The fundamental historical unit is:

``` text
Customer Message
       +
Historical Support Response
```

------------------------------------------------------------------------

# 9. Intent Taxonomy

The project uses ten operational support intents.

  -----------------------------------------------------------------------
  Intent                              Description
  ----------------------------------- -----------------------------------
  `order_delivery`                    Delivery status, tracking, delayed
                                      shipments and missing packages

  `returns_refunds`                   Returns, cancellations and refunds

  `payment_billing`                   Charges, billing and
                                      payment-related problems

  `account_security`                  Account access, compromised
                                      accounts and security issues

  `subscription_prime`                Prime membership and
                                      subscription-related requests

  `digital_content`                   Digital books, video, music and
                                      digital-content issues

  `device_technical`                  Device problems and technical
                                      troubleshooting

  `product_order_issue`               Incorrect, damaged, defective or
                                      problematic products

  `customer_service_complaint`        Complaints about customer service
                                      or support experience

  `other_non_actionable`              Messages without a sufficiently
                                      actionable support request
  -----------------------------------------------------------------------

The taxonomy is intentionally compact.

The objective is not to create a separate category for every linguistic
variation.

Instead, the categories are designed to be useful for:

-   routing
-   retrieval
-   escalation
-   response selection
-   evaluation

------------------------------------------------------------------------

# 10. Intent Definitions

## 10.1 order_delivery

Use this category when the customer's main problem concerns:

-   delivery status
-   tracking
-   delayed shipments
-   missing packages
-   expected delivery
-   package arrival

The key question is:

> Is the customer primarily asking where or when the package will
> arrive?

------------------------------------------------------------------------

## 10.2 returns_refunds

Use this category when the customer is primarily asking about:

-   returning a product
-   requesting a refund
-   refund status
-   return eligibility
-   cancellation-related refunds

The main issue is the return or recovery of money.

------------------------------------------------------------------------

## 10.3 payment_billing

Use this category for:

-   charges
-   billing
-   payment methods
-   unexpected charges
-   duplicate charges
-   payment failures

Financially sensitive requests are additionally handled by the
escalation system.

------------------------------------------------------------------------

## 10.4 account_security

Use this category for:

-   account access
-   compromised accounts
-   hacked accounts
-   unauthorized access
-   suspicious activity
-   security concerns

These requests are treated conservatively.

------------------------------------------------------------------------

## 10.5 subscription_prime

Use this category for:

-   Prime membership
-   subscription status
-   Prime benefits
-   subscription cancellation
-   subscription-related issues

------------------------------------------------------------------------

## 10.6 digital_content

Use this category for:

-   digital books
-   digital video
-   digital music
-   digital purchases
-   digital-content access

------------------------------------------------------------------------

## 10.7 device_technical

Use this category for:

-   device setup
-   hardware problems
-   device malfunction
-   technical troubleshooting
-   device-related problems

------------------------------------------------------------------------

## 10.8 product_order_issue

Use this category when the problem concerns the product itself.

Examples include:

-   damaged product
-   defective product
-   wrong item
-   incorrect product

If the only problem is that the package has not arrived, use:

``` text
order_delivery
```

------------------------------------------------------------------------

## 10.9 customer_service_complaint

Use this category for:

-   complaints about support
-   poor customer-service experience
-   unresolved support
-   dissatisfaction with previous support
-   service-quality complaints

------------------------------------------------------------------------

## 10.10 other_non_actionable

Use this category when:

-   there is no clear support request
-   the message is informational
-   the message is too ambiguous
-   the message does not fit the operational categories

This prevents unclear messages from being forced into an unrelated
intent.

------------------------------------------------------------------------

# 11. Training Data

A development sample was created from the processed support
conversations.

The classifier development data was divided into:

``` text
Training:     350 examples
Validation:    75 examples
Test:          75 examples
```

The purpose of each split was:

### Training

Used to fit the classifier.

### Validation

Used to compare configurations and select the model settings.

### Test

Used as a held-out development evaluation split.

The larger historical corpus is used separately for retrieval.

------------------------------------------------------------------------

# 12. Intent Classification

The intent-classification component predicts the primary support
category.

The classification pipeline is:

``` text
Customer Message
       |
       v
TF-IDF Vectorization
       |
       v
Sparse Text Features
       |
       v
Logistic Regression
       |
       v
Predicted Intent
       |
       v
Confidence Score
```

The confidence score is important because it is also used by the
escalation layer.

A prediction with low confidence should not automatically be treated as
reliable.

------------------------------------------------------------------------

# 13. Majority Baseline

The first baseline is deliberately simple.

The classifier always predicts the most common intent.

The majority intent is:

``` text
order_delivery
```

Results on the 75-example development test split:

  Metric        Score
  ---------- --------
  Accuracy     0.3600
  Macro F1     0.0529

The baseline establishes a lower bound.

The low Macro F1 shows that predicting only the most common class
performs poorly across the complete taxonomy.

------------------------------------------------------------------------

# 14. TF-IDF + Logistic Regression

The primary development classifier uses TF-IDF features with Logistic
Regression.

The selected configuration is:

``` text
Model:
Logistic Regression

Vectorizer:
TF-IDF

N-gram range:
(1, 1)

Minimum document frequency:
1

C:
0.5
```

The configuration was selected using the validation split.

Validation Macro F1:

``` text
0.3379
```

Development test results:

  Metric       Result
  ---------- --------
  Accuracy     0.4533
  Macro F1     0.3496

The model provides a meaningful improvement over the majority baseline.

However, several intents remain difficult because customer-support
messages frequently share vocabulary.

For example:

``` text
order_delivery
product_order_issue
```

can both contain:

-   order
-   package
-   item
-   product
-   delivery

Similarly:

``` text
returns_refunds
payment_billing
```

can both contain financial terminology.

------------------------------------------------------------------------

# 15. Historical Retrieval

Historical retrieval is used to find previous customer-support
interactions that resemble the incoming request.

The retrieval architecture is:

``` text
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
Best Historical Match
       |
       v
Historical Support Response
```

The retrieval index contains approximately 168k historical customer
messages.

For a new request, the system calculates similarity against historical
customer messages and selects the most similar interaction.

The support response associated with that interaction becomes the
grounding source for the response draft.

------------------------------------------------------------------------

# 16. Why Historical Retrieval

Historical retrieval provides several useful properties.

## Traceability

The response can be connected to a previous support interaction.

## Consistency

The system follows previously used support language.

## Interpretability

A developer can inspect the retrieved example.

## Simplicity

The retrieval system is straightforward to reproduce and debug.

The approach also has an important limitation.

Lexical similarity does not always represent semantic similarity.

Two messages can share many words while describing different problems.

This is why retrieval similarity is treated as a supporting signal
rather than a complete quality measurement.

------------------------------------------------------------------------

# 17. Retrieval Leakage Prevention

Evaluation leakage is an important concern in retrieval systems.

If an evaluation example is still present in the retrieval index, the
system may retrieve the exact same example.

That would make the retrieval result look artificially strong.

The undesirable workflow would be:

``` text
Evaluation Message
       |
       v
Same Message Already in Index
       |
       v
Self Retrieval
       |
       v
Artificially High Similarity
```

To prevent this, a separate leakage-free retrieval index was created.

The evaluation examples are excluded before constructing the retrieval
index.

------------------------------------------------------------------------

# 18. Leakage-Free Retrieval Index

The leakage-free index contains:

``` text
Documents: 152,813
Features: 200,000
```

The evaluation examples are removed from the retrieval corpus.

This prevents direct self-retrieval during evaluation.

The leakage-free index is used by the response-generation and evaluation
workflows.

This separation makes retrieval evaluation more meaningful.

------------------------------------------------------------------------

# 19. Response Drafting

The response-generation component uses historical support responses as
the basis for the draft.

The response workflow is:

``` text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Retrieval
       |
       v
Best Historical Response
       |
       v
Response Cleaning
       |
       v
Draft Response
```

Historical Twitter responses can contain platform-specific artifacts.

The cleaning stage removes unnecessary elements such as:

-   Twitter handles
-   shortened URLs
-   historical signatures
-   irrelevant trailing fragments

The goal is to produce a cleaner support response.

The final treatment depends on the escalation decision.

If the request is safe to handle, the result can be used as a grounded
response.

If escalation is required, the result is treated as an internal draft
for human review.

------------------------------------------------------------------------

# 20. Escalation System

The system does not automatically handle every customer message.

The escalation layer acts as a safety mechanism around classification
and retrieval.

The decision considers:

1.  Explicit risk signals
2.  Predicted intent
3.  Intent confidence
4.  Historical retrieval similarity

The basic logic is:

``` text
Customer Message
       |
       v
Intent + Confidence
       |
       v
Historical Match + Similarity
       |
       v
Risk Detection
       |
       v
Escalation Gate
       |
       +----------------------+
       |                      |
       v                      v
Escalate                Auto Handle
```

------------------------------------------------------------------------

# 21. Risk Detection

The system explicitly handles several categories of sensitive requests.

## Account and Security Risk

Examples include:

-   hacked account
-   compromised account
-   stolen account
-   unauthorized access
-   identity theft
-   suspicious account activity

These requests are escalated.

## Financial Risk

Examples include:

-   unauthorized charge
-   unknown charge
-   fraudulent charge
-   duplicate charge
-   credit-card fraud

These requests are escalated.

## Legal Risk

Examples include:

-   lawyer
-   attorney
-   lawsuit
-   legal action
-   court

These requests are routed to human review.

## Explicit Human Request

Customers may explicitly request human assistance.

Examples include:

-   speak to a human
-   speak to an agent
-   talk to a representative
-   contact a supervisor

Such requests are escalated even if the classifier is otherwise
confident.

------------------------------------------------------------------------

# 22. Confidence-Based Escalation

The classifier provides a confidence score for its predicted intent.

The selected threshold is:

``` text
0.20
```

The system escalates when:

``` text
intent_confidence < 0.20
```

The reason is that low classifier confidence indicates uncertainty about
the customer's actual problem.

This prevents the system from automatically responding to a message when
it may have misunderstood the request.

------------------------------------------------------------------------

# 23. Retrieval-Based Escalation

The retrieval component produces a similarity score.

The selected threshold is:

``` text
0.20
```

The system escalates when:

``` text
retrieval_similarity < 0.20
```

This prevents weak historical evidence from being treated as a reliable
basis for automatic handling.

------------------------------------------------------------------------

# 24. High-Risk Intents

Two intent categories receive additional conservative handling:

``` text
account_security
payment_billing
```

The reason is that these categories can involve sensitive account or
financial information.

The system therefore prefers human review for these requests.

------------------------------------------------------------------------

# 25. Threshold Selection

Multiple combinations of intent-confidence and retrieval-similarity
thresholds were evaluated.

The selected configuration is:

  Parameter                          Value
  -------------------------------- -------
  Intent confidence threshold         0.20
  Retrieval similarity threshold      0.20

The selected configuration produces:

  Metric                   Result
  ---------------------- --------
  Escalation Precision     0.3926
  Escalation Recall        0.9552
  Escalation F1            0.5565
  Escalation Rate          0.8150

The policy prioritizes escalation recall.

The reason is that missing a request that should have received human
attention is considered more serious than unnecessarily escalating a
normal request.

------------------------------------------------------------------------

# 26. End-to-End Pipeline

The final system can be represented as:

``` text
                    Customer Message
                           |
                           v
                   Text Representation
                           |
                           v
                   Intent Classifier
                           |
                           v
                  Intent + Confidence
                           |
                           v
                  Historical Retrieval
                           |
                           v
                  Similarity Score
                           |
                           v
                    Risk Detection
                           |
                           v
                   Escalation Gate
                           |
                +----------+----------+
                |                     |
                v                     v
          Human Review          Auto Handling
                |                     |
                v                     v
        Internal Draft          Grounded Reply
```

Each component produces a signal that can be inspected independently.

This makes debugging significantly easier.

------------------------------------------------------------------------

# 27. Evaluation Methodology

The project evaluates three important areas.

## Intent Evaluation

The classifier is evaluated using:

-   accuracy
-   Macro F1
-   per-class performance
-   confusion matrix

## Escalation Evaluation

The escalation layer is evaluated using:

-   precision
-   recall
-   F1
-   escalation rate
-   false-negative cases

## Response Evaluation

Response quality should be evaluated using:

-   relevance
-   groundedness
-   helpfulness
-   unsupported claims
-   clarity

Separating these dimensions prevents one metric from hiding weaknesses
in another part of the system.

------------------------------------------------------------------------

# 28. Evaluation Results

The complete system was evaluated on 200 examples.

The current measured results are:

  Metric                   Result
  ---------------------- --------
  Intent Accuracy          0.5300
  Intent Macro F1          0.5281
  Escalation Precision     0.3926
  Escalation Recall        0.9552
  Escalation F1            0.5565
  Auto-Handle Rate         0.1850
  Escalation Rate          0.8150

The system escalated:

``` text
163 / 200 examples
```

and automatically handled:

``` text
37 / 200 examples
```

The current behavior is therefore conservative.

------------------------------------------------------------------------

# 29. Escalation Results

The escalation breakdown is:

  Escalation Reason             Count
  --------------------------- -------
  low_intent_confidence           123
  high_risk_intent                 22
  low_historical_similarity        11
  legal_or_high_risk                4
  security_or_account_risk          2
  payment_or_financial_risk         1

The most significant contributor is low intent confidence.

123 of the 163 escalated examples were routed to human review because
the classifier was uncertain.

This indicates that improving intent classification should be the first
major optimization target.

------------------------------------------------------------------------

# 30. End-to-End Example

Customer message:

> Where is my order? It was supposed to arrive yesterday.

The system produces:

``` text
Intent:
order_delivery

Intent Confidence:
0.1388

Retrieval Similarity:
0.6653

Escalate:
True

Escalation Reason:
low_intent_confidence
```

The historical response used for the draft is:

> Oh no! I'm sorry to hear that. What is the last tracking update on the
> order?

The response is not automatically sent because the classifier confidence
is below the configured threshold.

Instead, it is presented as an internal draft for human review.

This example demonstrates an important design property:

> A strong historical match does not override uncertainty in the intent
> classifier.

------------------------------------------------------------------------

# 31. Failure Analysis

The project stores per-example evaluation information so that failures
can be investigated individually.

Aggregate accuracy alone does not explain:

-   which intents are confused
-   why escalation happens
-   when retrieval fails
-   where the classifier is uncertain

The main failure patterns are described below.

## Failure Mode 1 --- Excessive Escalation

The largest failure mode is excessive escalation caused by low
classifier confidence.

Out of 163 escalated examples:

``` text
123
```

were escalated because of low intent confidence.

### Likely Cause

The classifier is trained on a relatively small labeled dataset compared
with the diversity of the full support corpus.

Customer messages also contain:

-   spelling variations
-   short phrases
-   incomplete context
-   overlapping terminology
-   informal language

### Improvement

Increase the number of labeled examples.

Priority should be given to:

-   minority classes
-   ambiguous examples
-   commonly confused intents
-   short customer messages

------------------------------------------------------------------------

## Failure Mode 2 --- Delivery vs Product Issues

The classifier can confuse:

``` text
order_delivery
product_order_issue
```

Both categories can contain words such as:

-   order
-   package
-   item
-   product
-   delivery

The difference depends on the customer's underlying problem.

A message asking where a package is belongs to delivery.

A message reporting a damaged or incorrect product belongs to the
product category.

### Improvement

A hierarchical classification strategy could first determine the broad
problem family and then classify the specific issue.

------------------------------------------------------------------------

## Failure Mode 3 --- Refund vs Billing

The classifier can confuse:

``` text
returns_refunds
payment_billing
```

because both categories contain financial vocabulary.

The distinction is:

``` text
Returns/refunds
=
money expected back
```

while:

``` text
Payment/billing
=
money charged or payment problem
```

### Improvement

Training data should contain more explicit examples separating:

-   charges
-   refunds
-   return requests
-   refund delays
-   payment failures

------------------------------------------------------------------------

## Failure Mode 4 --- Device vs Digital Content

Technical language can occur in both:

``` text
device_technical
digital_content
```

A customer may describe a content problem through the device they are
using.

### Improvement

The next version should use semantic embeddings and product/entity
signals to distinguish device-level problems from content-level
problems.

------------------------------------------------------------------------

## Failure Mode 5 --- Generic Historical Responses

Historical responses can sometimes contain generic support language.

A response may receive high lexical similarity because it contains many
words that are common across support conversations.

Therefore:

``` text
High Similarity
```

does not always mean:

``` text
High Relevance
```

### Improvement

The retrieval pipeline should eventually become:

``` text
Semantic Retrieval
       |
       v
Top Candidate Responses
       |
       v
Relevance Reranker
       |
       v
Best Grounding Response
```

The reranker should evaluate whether the historical response actually
addresses the current customer's underlying problem.

------------------------------------------------------------------------

# 32. What Is Misleading About the Headline Number?

The current end-to-end intent accuracy is:

``` text
53.0%
```

This should be interpreted as a development evaluation result.

It should not be presented as a production-level performance estimate.

The current 200-example evaluation set was reviewed during development
and is not an independently audited production benchmark.

The classifier development data was also created as part of the
development process.

Therefore, the 53.0% result is useful for measuring the current
implementation, but it should not be treated as definitive evidence of
real-world performance.

A stronger evaluation would require:

-   independently hand-labeled examples
-   multiple reviewers
-   explicit annotation guidelines
-   inter-annotator agreement
-   a frozen evaluation set
-   structured response-quality evaluation

The important lesson is that benchmark quality is as important as model
quality.

------------------------------------------------------------------------

# 33. Why Accuracy Alone Is Not Enough

The intent distribution is not perfectly balanced.

A classifier can obtain reasonable accuracy while performing poorly on
less frequent intents.

For this reason, the project reports:

-   Accuracy
-   Macro F1
-   per-class results
-   confusion matrix

Macro F1 gives each intent equal importance.

This makes it easier to identify weaknesses in minority categories.

------------------------------------------------------------------------

# 34. Why Escalation Recall Matters

Escalation is a safety-sensitive part of the system.

A false negative means:

``` text
Request should have gone to a human
                 |
                 v
System handled it automatically
```

This can be more problematic than a false positive:

``` text
Normal request
      |
      v
Sent to human review unnecessarily
```

Therefore, the current escalation policy intentionally prioritizes
recall.

The current result is:

``` text
Escalation Recall = 95.52%
```

The trade-off is lower precision:

``` text
Escalation Precision = 39.26%
```

This trade-off explains the high escalation rate.

------------------------------------------------------------------------

# 35. Engineering Decisions

Several engineering decisions shaped the project.

## Single Support Environment

A single support account was selected so that historical responses
remain contextually consistent.

## Compact Taxonomy

Ten operational intents provide a balance between useful detail and
manageable classification complexity.

## Conventional Classifier

TF-IDF with Logistic Regression provides a transparent and reproducible
starting point.

## Historical Retrieval

Previous support interactions provide concrete evidence for response
drafting.

## Leakage-Free Retrieval

Evaluation examples are removed from the retrieval corpus.

## Conservative Escalation

Sensitive and uncertain requests are routed to human review.

## Macro F1 Reporting

Macro F1 prevents frequent categories from dominating the evaluation.

## Per-Example Analysis

Individual failures are preserved so that model weaknesses can be
investigated.

------------------------------------------------------------------------

# 36. Repository Structure

``` text
SupportAgentAI/
|
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   |
│   └── processed/
│
├── models/
│
├── scripts/
│   ├── inspect_brands.py
│   ├── extract_brand_data.py
│   ├── build_intent_sample.py
│   ├── suggest_intents.py
│   ├── prepare_training_data.py
│   ├── train_classifier.py
│   ├── evaluate_majority_baseline.py
│   ├── build_retrieval_index.py
│   ├── build_leakage_free_index.py
│   ├── evaluate_retrieval_leakage_free.py
│   ├── escalation_gate.py
│   ├── tune_escalation_thresholds.py
│   ├── generate_reply.py
│   ├── evaluate_agent.py
│   ├── analyze_failures.py
│   └── generate_failure_report.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── REPORT.md
├── HONESTY_NOTE.md
├── DECISION_LOG.md
└── LABELING_GUIDE.md
```

------------------------------------------------------------------------

# 37. Installation

Create a virtual environment:

``` powershell
python -m venv venv
```

Activate the environment:

``` powershell
.\venv\Scripts\Activate.ps1
```

Install project dependencies:

``` powershell
pip install -r requirements.txt
```

Verify Python:

``` powershell
python --version
```

Verify the environment:

``` powershell
python -m pip --version
```

------------------------------------------------------------------------

# 38. Dataset Setup

Place the source dataset at:

``` text
data/raw/twcs.csv
```

The raw dataset should not be committed to Git because of its large
size.

The `.gitignore` configuration excludes the raw dataset and generated
model/data artifacts.

The expected input structure is:

``` text
SupportAgentAI/
|
└── data/
    └── raw/
        └── twcs.csv
```

------------------------------------------------------------------------

# 39. Running the Data Pipeline

Inspect available support accounts:

``` powershell
python scripts\inspect_brands.py
```

Extract the selected support conversations:

``` powershell
python scripts\extract_brand_data.py
```

Build the intent development sample:

``` powershell
python scripts\build_intent_sample.py
```

Prepare training, validation and test data:

``` powershell
python scripts\prepare_training_data.py
```

After these steps, the processed conversation dataset and development
datasets are available under:

``` text
data/processed/
```

------------------------------------------------------------------------

# 40. Training the Classifier

Run:

``` powershell
python scripts\train_classifier.py
```

The trained classifier is saved to:

``` text
models/intent_classifier.joblib
```

The training process also records evaluation results for the validation
and development test splits.

------------------------------------------------------------------------

# 41. Building the Retrieval Index

Build the historical retrieval index:

``` powershell
python scripts\build_retrieval_index.py
```

Build the leakage-free retrieval index:

``` powershell
python scripts\build_leakage_free_index.py
```

The leakage-free index excludes the evaluation examples before indexing
the historical conversations.

------------------------------------------------------------------------

# 42. Running Retrieval Evaluation

Run:

``` powershell
python scripts\evaluate_retrieval_leakage_free.py
```

This evaluates retrieval behavior without directly including the
evaluation examples in the retrieval corpus.

The retrieval results are stored in:

``` text
data/processed/retrieval_leakage_free_results.csv
```

------------------------------------------------------------------------

# 43. Running the Escalation Threshold Tuning

Run:

``` powershell
python scripts\tune_escalation_thresholds.py
```

This evaluates different combinations of:

-   intent-confidence thresholds
-   retrieval-similarity thresholds

The resulting threshold analysis is stored in:

``` text
data/processed/escalation_threshold_results.csv
```

------------------------------------------------------------------------

# 44. Running Complete Evaluation

Run:

``` powershell
python scripts\evaluate_agent.py
```

The evaluation produces:

-   intent predictions
-   intent confidence
-   retrieval similarity
-   escalation decision
-   escalation reason
-   per-example evaluation information

The results are stored under:

``` text
data/processed/
```

------------------------------------------------------------------------

# 45. Failure Analysis

Run:

``` powershell
python scripts\analyze_failures.py
```

This identifies classification and escalation failures.

Generate the summarized failure report:

``` powershell
python scripts\generate_failure_report.py
```

Important output files include:

``` text
data/processed/failure_analysis.csv
data/processed/top_failure_examples.csv
data/processed/intent_confusion_matrix.csv
data/processed/agent_evaluation_examples.csv
data/processed/agent_evaluation_results.txt
```

These files make it possible to inspect individual model failures
instead of relying only on aggregate metrics.

------------------------------------------------------------------------

# 46. Interactive Usage

Run:

``` powershell
python scripts\generate_reply.py
```

The system reports:

``` text
Intent
Intent Confidence
Retrieval Similarity
Escalation Decision
Escalation Reason
Response Draft
```

For escalated requests, the generated text is presented as an internal
draft.

For requests that satisfy the automatic-handling conditions, the system
produces a grounded response.

------------------------------------------------------------------------

# 47. Example Interactive Output

For a customer message such as:

``` text
Where is my order? It was supposed to arrive yesterday.
```

the system can produce:

``` text
Intent: order_delivery
Intent Confidence: 0.1388
Retrieval Similarity: 0.6653
Escalate: True
Escalation Reason: low_intent_confidence
Recommended Action: HUMAN REVIEW
```

The retrieved historical support response is cleaned before being
displayed as the internal draft.

------------------------------------------------------------------------

# 48. Testing

The project includes automated tests covering the main pipeline
behavior.

Run:

``` powershell
python -m pytest tests/ -v
```

The test suite covers:

-   conversation dataset availability
-   evaluation dataset availability
-   trained classifier availability
-   security-risk escalation
-   payment-risk escalation
-   explicit human-agent requests
-   low-confidence escalation
-   low-retrieval escalation
-   safe auto-handling
-   normal low-risk messages

Current test result:

``` text
10 passed
```

The tests provide a basic regression layer for the core decision logic.

------------------------------------------------------------------------

# 49. Generated Artifacts

The main generated artifacts include:

``` text
data/processed/amazonhelp_conversations.csv
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
data/processed/golden_set.csv
data/processed/golden_set_review_ready.csv
data/processed/golden_set_human_review.csv
data/processed/golden_set_reviewed.csv
data/processed/agent_evaluation_examples.csv
data/processed/agent_evaluation_results.txt
data/processed/failure_analysis.csv
data/processed/top_failure_examples.csv
data/processed/intent_confusion_matrix.csv
data/processed/retrieval_leakage_free_results.csv
data/processed/escalation_threshold_results.csv
```

Model artifacts include:

``` text
models/intent_classifier.joblib
models/retrieval_vectorizer.joblib
models/retrieval_matrix.joblib
models/retrieval_documents.csv
models/leakage_free_vectorizer.joblib
models/leakage_free_matrix.joblib
models/leakage_free_documents.csv
```

These generated files are intentionally excluded from source control
because they are reproducible artifacts.

------------------------------------------------------------------------

# 50. Current Strengths

## Clear Architecture

The system separates classification, retrieval, response drafting and
escalation.

## Historical Grounding

Response drafts are based on previous support interactions.

## Conservative Escalation

Sensitive and uncertain requests are routed to human review.

## Leakage-Aware Evaluation

Evaluation examples are excluded from the leakage-free retrieval corpus.

## Reproducibility

The main workflow is represented through executable scripts.

## Test Coverage

The main escalation and decision logic is covered by automated tests.

## Failure Analysis

The project preserves per-example results for deeper debugging.

------------------------------------------------------------------------

# 51. Current Limitations

The current system is a working prototype and has several limitations.

## Limited Labeled Training Data

The classifier needs more labeled examples to cover the diversity of
customer language.

## Lexical Retrieval

TF-IDF retrieval depends heavily on word overlap.

## Generic Historical Responses

Some historical support responses are short or repetitive.

## High Escalation Rate

The current policy escalates 81.5% of evaluated requests.

## No Live Account Information

The system does not directly query customer orders, accounts or
payments.

## Evaluation Maturity

The evaluation process should be strengthened with independent
annotation and human agreement.

------------------------------------------------------------------------

# 52. Future Improvements

## Semantic Retrieval

Replace TF-IDF retrieval with sentence embeddings.

This should improve matching when two requests have similar meaning but
different wording.

## Intent-Aware Retrieval

Use the predicted intent as an additional retrieval signal.

This should reduce unrelated historical matches.

## Response Reranking

Retrieve multiple candidates and rank them based on:

-   semantic relevance
-   intent consistency
-   response usefulness
-   grounding quality

## Better Confidence Estimation

Classifier probabilities should be calibrated so that confidence values
better represent actual prediction reliability.

## Hierarchical Classification

Use a broad-to-specific classification structure for overlapping
categories.

## Stronger Evaluation

Increase the independently reviewed evaluation set and freeze it before
model tuning.

------------------------------------------------------------------------

# 53. One-Week Improvement Plan

## Day 1 --- Expand the Evaluation Set

Build a stronger evaluation set of approximately 200-250 independently
reviewed examples.

Ensure that every intent is represented.

Include:

-   common requests
-   minority intents
-   ambiguous examples
-   sensitive examples
-   short messages
-   long messages

## Day 2 --- Human Agreement

Use two independent reviewers on a calibration subset.

Compare disagreements.

Refine the labeling guide where necessary.

Measure inter-annotator agreement.

## Day 3 --- Semantic Retrieval

Replace lexical TF-IDF retrieval with sentence embeddings.

Compare:

``` text
TF-IDF Retrieval
```

against:

``` text
Semantic Retrieval
```

using the same evaluation examples.

## Day 4 --- Intent-Aware Retrieval

Use the predicted intent as an additional retrieval signal.

A future pipeline can be:

``` text
Predicted Intent
       |
       v
Relevant Historical Candidates
       |
       v
Semantic Ranking
```

## Day 5 --- Response Quality Evaluation

Introduce structured evaluation of:

-   relevance
-   groundedness
-   helpfulness
-   unsupported claims
-   clarity

## Day 6 --- Escalation Optimization

Tune escalation thresholds using the stronger evaluation set.

The goal is:

``` text
Lower unnecessary escalation
+
Maintain high escalation recall
```

## Day 7 --- Final Benchmark

Freeze:

-   taxonomy
-   training data
-   evaluation set
-   classifier
-   retrieval system
-   escalation policy
-   evaluation scripts

Run the final benchmark and document the final results.

------------------------------------------------------------------------

# 54. Final Architecture

The target architecture for the next version is:

``` text
Customer Message
       |
       v
Intent + Risk Understanding
       |
       v
Semantic Retrieval
       |
       v
Intent-Aware Filtering
       |
       v
Candidate Reranking
       |
       v
Grounded Response Draft
       |
       v
Policy Validation
       |
       +----------------------+
       |                      |
       v                      v
Human Review            Automatic Reply
```

The current implementation already establishes the major interfaces
required for this evolution.

The classifier can be upgraded independently.

The retrieval component can be upgraded independently.

The escalation policy can be tuned independently.

The evaluation framework can remain the central measurement layer.

------------------------------------------------------------------------

# 55. Engineering Principles

The project follows several engineering principles.

## Grounded Responses

Responses should be based on relevant historical support evidence
whenever possible.

## Conservative Automation

Uncertain requests should be reviewed instead of being handled
automatically.

## Sensitive-Request Protection

Security, financial and legal requests receive conservative treatment.

## Evaluation Leakage Prevention

Evaluation examples are removed from the leakage-free retrieval corpus.

## Interpretability

The current system uses transparent components that can be inspected and
debugged.

## Failure-Driven Development

Individual failures are analyzed rather than relying only on a single
aggregate score.

## Reproducibility

The major stages of the project are implemented as independent scripts
that can be rerun.

------------------------------------------------------------------------

# 56. Design Trade-offs

Every component involves a trade-off.

### TF-IDF vs Semantic Models

TF-IDF is simpler and faster, while semantic models are likely to
improve meaning-based matching.

### Automation vs Safety

Aggressive automation increases coverage but can increase the risk of
incorrect handling.

### Precision vs Recall

The current escalation policy prioritizes recall, which increases the
number of requests sent for human review.

### Compact Taxonomy vs Fine-Grained Taxonomy

A compact taxonomy is easier to train and operate, while a fine-grained
taxonomy can provide more routing detail.

### Retrieval Simplicity vs Retrieval Quality

Lexical retrieval is easy to debug, while semantic retrieval is more
capable of handling paraphrases.

------------------------------------------------------------------------

# 57. Why the Current System Is Conservative

The system intentionally avoids treating every confident-looking
prediction as automatically safe.

A support message can be difficult for several reasons:

-   incomplete context
-   ambiguous wording
-   multiple issues in one message
-   unusual terminology
-   sensitive account information
-   financial consequences
-   explicit human requests

The escalation layer provides a second decision stage after
classification.

This gives the architecture:

``` text
Prediction
    +
Evidence
    +
Risk
    =
Automation Decision
```

This separation is important because classification confidence alone is
not enough to determine whether a support response should be automated.

------------------------------------------------------------------------

# 58. Production Considerations

A production version would require additional infrastructure.

Potential components include:

-   live order-status APIs
-   account verification
-   payment-status APIs
-   authenticated customer context
-   policy validation
-   rate limiting
-   audit logging
-   response approval workflows
-   monitoring
-   alerting
-   model versioning
-   data-quality monitoring

The current system intentionally focuses on the machine-learning and
decision-making layer.

------------------------------------------------------------------------

# 59. Monitoring Metrics for a Production System

A production deployment should continuously monitor:

## Classification

-   intent accuracy
-   Macro F1
-   confidence distribution
-   class distribution drift

## Retrieval

-   retrieval similarity distribution
-   retrieval failure rate
-   duplicate retrieval rate
-   candidate relevance

## Escalation

-   escalation rate
-   escalation precision
-   escalation recall
-   false-negative rate
-   high-risk routing rate

## Response Quality

-   customer satisfaction
-   response acceptance
-   correction rate
-   human-edited response rate
-   unsupported-response rate

Monitoring these metrics would make it possible to detect degradation
after deployment.

------------------------------------------------------------------------

# 60. Final Results Summary

## Majority Baseline

``` text
Accuracy: 36.00%
Macro F1: 5.29%
```

## TF-IDF + Logistic Regression

``` text
Validation Macro F1: 33.79%
Test Accuracy: 45.33%
Test Macro F1: 34.96%
```

## End-to-End System

``` text
Intent Accuracy: 53.00%
Intent Macro F1: 52.81%

Escalation Precision: 39.26%
Escalation Recall: 95.52%
Escalation F1: 55.65%

Auto-Handle Rate: 18.50%
Escalation Rate: 81.50%
```

These results come from their respective development and end-to-end
evaluation splits and should not be treated as directly comparable
benchmark measurements.

------------------------------------------------------------------------

# 61. Final Conclusion

SupportAgentAI demonstrates an end-to-end approach to customer-support
automation using historical customer-support conversations.

The system combines:

-   structured data extraction
-   intent classification
-   historical retrieval
-   response drafting
-   risk detection
-   confidence-based escalation
-   leakage-aware evaluation
-   automated testing
-   failure analysis

The current architecture intentionally favors reliability over
aggressive automation.

The strongest part of the system is the conservative escalation policy.

The largest weakness is the high number of requests escalated because of
classifier uncertainty.

The most valuable next improvement is better labeled data combined with
stronger semantic retrieval and confidence estimation.

The long-term objective is not maximum automation.

The objective is:

> Maximum safe automation with clear human oversight whenever the system
> is uncertain.

------------------------------------------------------------------------

# 62. Author

**Vikas Kumar**

Software Engineer \| Full Stack Developer \| Python \| Machine Learning
\| Backend Systems

SupportAgentAI represents my engineering work focused on building
practical, interpretable, reliability-focused automation systems for
customer-support workflows.
