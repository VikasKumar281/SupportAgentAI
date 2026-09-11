# Honesty Note — What's Real vs. Simulated in This Build

Being upfront about this in one place, since it affects how every number in
`REPORT.md` should be read.

## The dataset is synthetic, not the real Kaggle file

The build environment used to produce this submission has network egress
restricted to package registries (pypi/npm/github) — it cannot reach
`kaggle.com`, which also requires an authenticated API token to download the
~1.7GB `twcs.csv`. Rather than fabricate having used it, `data/generate_synthetic_data.py`
generates a dataset that:
- Uses the **identical column schema** as the real file (`tweet_id`,
  `author_id`, `inbound`, `created_at`, `text`, `response_tweet_id`,
  `in_response_to_tweet_id`), so `src/data_prep.py` and everything downstream
  works unmodified on the real file — just replace `data/raw_tweets.csv`.
- Reproduces realistic noise: typos, emoji, hashtags, @mentions, multi-turn
  follow-ups, varied phrasing per intent.
- Is templated, which means it is **easier** than real tweets in one
  specific way: real customer language is far more varied than our ~3
  phrasing variants per intent. Our classification accuracy numbers should
  be read as an **upper bound** relative to what the same architecture would
  score on real data — this is explicitly called out again in
  `REPORT.md`'s misleading-number section.

## The golden set's "hand labeling" is a documented deterministic process, not a human

`eval/build_golden_set.py` implements the intent- and escalation-labeling
rules as code (see `LABELING_GUIDE.md` for the exact rules), including
deliberately injecting ~5% label noise on genuinely confusable examples to
avoid an unrealistically clean eval set. This is a real limitation: it means
our "human" labels share some logic with parts of the system under test
(both are informed by the same domain knowledge), which is a form of
evaluation leakage a fully independent human annotator would not have. We
mitigate this by making the **escalation** gold-label rules deliberately
different in structure and stricter in policy than `src/escalation.py`'s
rules (see `LABELING_GUIDE.md`), so at least that half of the eval isn't
circular — but it's a mitigation, not a fix.

## The "human" in judge-vs-human agreement is simulated

`src/llm_judge.py::simulate_human_score()` is a clearly-labeled stand-in for
a human rater, implemented with different scoring logic than the judge
(different features/weights) so agreement isn't trivially 1.0. It proves the
**agreement-measurement harness works end-to-end** — `compute_agreement()`
takes two same-shaped score lists and works identically whether they came
from a real human CSV or this simulation. It should not be read as evidence
that a real human would agree with the judge at the reported rate.

## What would change with real data / a real human

1. Real tweets: far more spelling variation, sarcasm, multi-issue messages,
   non-English text, and off-topic replies (spam, other customers jumping
   into a thread) — expect meaningfully lower classification accuracy and
   more `no_grounded_precedent` escalations.
2. Real human labeling: would surface genuinely ambiguous cases we can't
   anticipate by writing rules for the ambiguity we expect — the actual
   confusable-pair rate is very likely higher than our injected 5%.
3. Real human judge-agreement: would very likely be lower on `groundedness`
   specifically, since a human catches subtle invented details our
   heuristic groundedness check (presence of unseen numbers) misses
   entirely (e.g. an invented policy promise with no numbers in it).
