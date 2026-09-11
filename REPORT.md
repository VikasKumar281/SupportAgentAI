# Report — AI Support Agent for AmazonHelp

*Numbers below are from a fresh run of `bash scripts/run_all.sh` in
no-API-key mode (see README "Two modes"). Re-run to reproduce exactly —
seeded with `RANDOM_SEED=42` throughout.*

**Read `HONESTY_NOTE.md` before this report.** The dataset is a
schema-faithful synthetic generator (no Kaggle access in this build
environment), and parts of the "human" evaluation are documented simulations.
Every number below should be read with that context.

---

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp on Twitter is
high-volume and mostly routine: "where's my order", "cancel this", "refund
that". The cost structure is asymmetric — a wrong auto-reply on a routine
question costs a little brand goodwill; a wrong auto-reply (or wrong
auto-*handle* decision) on an account-security or billing-fraud message
costs real money, trust, or a PR incident. So "good" here means:

1. **High recall on the escalation gate for high-risk categories**, even at
   the cost of over-escalating some routine ones. A missed escalation is
   much more expensive than an unnecessary one.
2. **Replies must never invent facts** (order numbers, refund amounts,
   promises) not present in the customer's message or in a real historical
   precedent — grounding is a hard requirement, not a nice-to-have.
3. **Auditable decisions.** A support lead should be able to look at any
   auto-handled message and see *why* it wasn't escalated in one sentence,
   not reverse-engineer an LLM's reasoning.

**What I chose not to build:**
- **No embeddings/vector DB for retrieval.** TF-IDF cosine similarity is
  fully offline, deterministic, and the match is literally inspectable
  (shared words) — good enough at this scale (hundreds of historical
  threads) and avoids adding an embedding-model dependency + cost to a
  system whose grounding step is safety-critical. Would revisit at real
  scale (see §5).
- **No multi-turn conversation state.** Each message is scored independently
  even though ~30% of threads have a customer follow-up. A production
  version would use thread history to detect "this is the 2nd unresolved
  contact" directly instead of inferring it from phrasing.
- **No sentiment model.** Escalation uses keyword/punctuation heuristics
  instead of a trained sentiment classifier — again for auditability, and
  because a wrong sentiment score with no visible reason is worse than a
  crude keyword rule everyone can read.
- **No fine-tuning.** Out of scope for a take-home; TF-IDF+LogReg and
  prompted LLM classification are both zero-training-infra options.

## 2. Results vs. two baselines

Intent classification, on the 185-example golden set:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **Trivial** (always predict majority class `general_feedback`) | 0.097 | 0.022 |
| **Simple** (TF-IDF + Logistic Regression, 1-2 grams) | 0.881 | 0.870 |
| **Full system** (same classifier + retrieval + escalation logic) | 0.881 | 0.870 |

The full system's classification numbers match the simple baseline exactly
in no-API-key mode because the full system's classifier *is* the TF-IDF
baseline when no LLM key is present (see README "Two modes") — the "full
system" column exists to show that retrieval/escalation logic doesn't
silently change the classification numbers, and to have the LLM-mode column
be a true apples-to-apples comparison when a key is supplied.

Escalation decision, vs. independent human-reviewer gold labels (see
`LABELING_GUIDE.md` for why these are independently defined, not copied from
`src/escalation.py`):

| Metric | Value |
|---|---|
| Precision | 0.683 |
| Recall | 0.757 |
| F1 | 0.718 |
| **Unsafe auto-handle rate** (should've escalated, didn't) | **0.243** (9 of 37) |
| Unnecessary escalation rate (escalated, didn't need to) | 0.088 |

Reply quality (40 sampled replies, heuristic judge — no API key in this run):

| Metric | Value |
|---|---|
| Mean judge score | 4.44 / 5 |
| Mean simulated-human score | 4.03 / 5 |
| Judge-vs-human Pearson r | 0.472 |
| Judge-vs-human within-1-point agreement | 100% |

## 3. Failure analysis — top 5, with real examples

All examples below are pulled directly from `outputs/predictions.csv` on the
run that produced the numbers above.

**1. Weak-supervision blind spot in the training-label generator, not the
classifier.** 12 of 22 (55%) misclassifications are the exact same
phrasing pattern: *"order #041583 was supposed to be here already. where is
it?"* — gold intent `delivery_delay`, predicted `general_feedback`. Root
cause: `src/intents.py`'s silver labeler (used to generate training labels,
since we don't have gold labels for the training pool either — see its
docstring) has no keyword pattern matching "supposed to be here already",
so every training example with this exact phrasing was silver-labeled
`general_feedback`, and the classifier faithfully learned that wrong
association. **This is a labeling-function coverage gap, not a model
capacity problem** — more data or a bigger model would not fix it; better
keyword/regex coverage (or switching that slice to LLM-based silver
labeling) would.

**2. Cascading failure: misclassification → wrong retrieval pool → false
escalation.** 13 of 22 escalation mismatches are `no_grounded_precedent`
triggered on messages misclassified per failure #1 above. Because retrieval
is filtered by *predicted* intent, a message wrongly classified as
`general_feedback` searches the general-feedback precedent pool instead of
delivery precedents, finds a poor match, and the low-similarity escalation
rule fires. **One upstream classification bug produces two downstream
symptoms** (wrong intent shown *and* an unnecessary escalation) that look
unrelated in a metrics dashboard but share one root cause. Lesson: failure
analysis needs to trace metric regressions back through the pipeline, not
just categorize each metric's errors independently.

**3. Policy gap on `account_access`, by design, but under-mitigated.** In 9
cases, a confidently-classified, well-grounded `account_access` message
(e.g. *"I've been locked out of my account for 2 days now, password reset
isn't working"*) was auto-handled by the system, but the independent human
policy says account-access issues should **always** escalate regardless of
model confidence (see `LABELING_GUIDE.md`). `src/escalation.py` only
escalates `account_access` when confidence/grounding is weak or an explicit
"hacked" keyword fires — it doesn't treat the intent itself as an automatic
trigger. This is the single largest chunk of the 24.3% unsafe-auto-handle
rate and is a straightforward one-line policy fix (add `account_access` to
an always-escalate intent set), which we deliberately did **not** apply
before reporting the number, so it's visible here rather than fixed in code
right before submission.

**4. Regex word-boundary bug caught during development (already fixed, kept
here for the record).** An earlier version of the risk/repeated-contact
keyword regexes had no `\b` word boundaries, so the substring "again" inside
the hashtag `#neveragain` matched the "repeated contact" rule and silently
escalated unrelated refund requests. Fixed in `src/escalation.py` (see
inline comment) once found by inspecting real `predictions.csv` rows —
included here as a concrete example of why row-level inspection of
predictions, not just aggregate metrics, is necessary before trusting a
rule-based safety gate.

**5. Reply-drafting fallback isn't tailored, only grounded.** In
no-API-key mode, `reply_generator.py` returns the single closest historical
reply verbatim. This is *always* grounded (it's a real past reply) but
sometimes references specifics that don't fit — e.g. a reply that says "have
you tried the app?" for a customer who never mentioned the app. The judge
heuristic doesn't penalize this much (it checks for invented facts, not
irrelevant-but-true statements), which is itself a judge limitation worth
noting (see §4).

## 4. What is misleading about my headline number

The 88.1% classification accuracy and 4.44/5 reply-quality score are both
individually true and collectively give an inflated picture of how "done"
this is. Specifically:

- **The dataset is synthetic and templated** (see `HONESTY_NOTE.md`). Real
  tweets have far more lexical variety, sarcasm, and off-topic noise; 88.1%
  on ~3 phrasing variants per intent is not evidence of 88.1% on real
  traffic. This is the single biggest caveat and it isn't a minor one.
- **The golden set's escalation labels share a keyword vocabulary with the
  system under test.** Both `src/escalation.py` and the independent
  "human" rules in `eval/build_golden_set.py` were written by the same
  person with the same domain assumptions about what "risky" language looks
  like (chargeback, lawyer, fraud, etc.). A real human annotator, or a
  message using risk language we didn't anticipate, would likely reveal a
  bigger gap than the 0.718 F1 suggests.
- **The reply-quality judge in this run is the heuristic fallback, not an
  LLM.** It checks for a narrow set of things (word overlap, invented
  numbers, apology words, "DM us" phrasing) and would miss subtler quality
  problems (a factually-fine reply that's tonally cold, or one that answers
  the wrong sub-question in a multi-part message). The 4.44/5 mean score
  measures "hits our checklist," not "a customer would be happy."
- **`unsafe_auto_handle_rate = 0.243` undersells the concentration of
  risk.** It's not 24.3% of *arbitrary* messages that get mishandled — it's
  concentrated almost entirely in one category (`account_access`, failure
  #3) with a known, cheap fix. Reporting one blended rate hides that the
  actual remaining risk surface, post-fix, would be much smaller. Conversely,
  reporting it as "mostly fixed" would hide that we haven't actually applied
  the fix or re-measured.
- **Escalation "recall" is computed on a golden set where 20% of examples
  (37/185) are positive (should-escalate) by stratified construction, not
  by natural traffic proportion.** Real AmazonHelp traffic is probably
  >90% routine; a system tuned to this golden set's balance could behave
  differently (likely more conservative than necessary) on real traffic
  where risky messages are rarer and the trivial "escalate everything
  billing-related" instinct is cheaper to sustain.

## 5. What I'd do with one more week

1. **Fix the account_access policy gap and the silver-labeling blind spot**
   identified in failure analysis, then re-run the full harness — I
   deliberately left both unfixed so the report reflects what testing
   actually found, but shipping the fixes is the obvious next step.
2. **Get real human labels** on at least the 40-example judge-calibration
   subset (real annotator, not the simulated proxy) — this is the highest-
   value fix for the eval harness's credibility specifically.
3. **Swap in the real Kaggle dataset** and re-measure; expect classification
   accuracy to drop meaningfully and treat that drop as the real baseline,
   not a regression to "fix."
4. **Add thread-level state** so "this is the 3rd time I've asked" is read
   from actual thread history instead of inferred from phrasing like "again."
5. **Move retrieval to embeddings** once the historical-resolution corpus is
   large enough that TF-IDF's exact-word-overlap requirement starts missing
   paraphrased-but-equivalent precedents — with an explicit before/after
   comparison on the golden set to justify the added complexity/cost.
6. **Expand the judge rubric's groundedness check** beyond "did it invent a
   number" to catch invented *qualitative* claims (policies, promises,
   timelines) — the current heuristic judge would miss those entirely, and
   even an LLM judge should be explicitly prompted to look for them.
