# AI Support Agent — AmazonHelp (Hiver SDE Take-Home)

An AI support agent for one brand (`AmazonHelp`) from the Customer Support
on Twitter dataset that (1) classifies incoming messages into 8 intents,
(2) drafts a reply grounded in how the brand has historically resolved
similar issues, and (3) decides auto-handle vs. escalate-to-human, with a
stated reason.

**Read this first:** [`HONESTY_NOTE.md`](HONESTY_NOTE.md) — this sandboxed
build environment cannot reach kaggle.com, so the dataset here is a
schema-faithful **synthetic generator**, not the real 3M-tweet file. Every
script is written to be dataset-agnostic — swap in the real `twcs.csv` and
nothing else changes. Full details on what's real vs. simulated (and why)
are there and in `REPORT.md`'s "what's misleading about my headline number"
section. This is not a footnote — it materially affects how to read the
numbers in this repo, so please don't skip it.

## Quickstart (reproduces headline results in ~1-3 minutes)

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
bash scripts/run_all.sh
```

That single command:
1. Generates the synthetic Kaggle-schema dataset (`data/raw_tweets.csv`)
2. Builds customer↔agent conversation pairs for the brand (`data/conversations.csv`)
3. Builds the 185-example golden evaluation set (`data/golden_eval_set.csv`)
4. Trains the baseline intent classifier
5. Runs the test suite
6. Runs the full evaluation harness and prints a results summary

Outputs land in `outputs/`:
- `predictions.csv` — every golden example, every system's prediction, side by side
- `metrics.json` — everything printed to stdout, machine-readable
- `judge_scores.csv` — LLM-judge vs. simulated-human scores on 40 sampled replies

## Optional: enable real LLM classification / generation / judging

```bash
cp .env.example .env
# edit .env, set ANTHROPIC_API_KEY=sk-...
bash scripts/run_all.sh
```

Without a key, the system runs on fully offline fallbacks (TF-IDF classifier,
retrieval-template replies, heuristic judge) — see "Two modes" below. With a
key, `src/classify_llm.py`, `src/reply_generator.py`, and `src/llm_judge.py`
each call the Anthropic API and fall back automatically (never crash) if the
call fails for any reason (no network, bad key, rate limit).

## Two modes, by design

| Component | No API key | With API key |
|---|---|---|
| Intent classification | TF-IDF + Logistic Regression | LLM few-shot classification (falls back to TF-IDF on any error) |
| Reply drafting | Closest historical reply, verbatim (always grounded, not tailored) | LLM drafts a new reply using top-3 retrieved examples as few-shot grounding |
| Reply judging | Feature-based heuristic scorer | LLM-as-judge on a 4-dimension rubric |

This was a deliberate choice, not laziness: a take-home graded by "can we
run this in 15 minutes" should not silently fail or hang if the grader
doesn't have an API key handy, or if network egress is restricted (as it
was for us, building this). Every numeric result in `REPORT.md` was
produced in **no-API-key mode**; the LLM-enhanced path is there and tested,
but its outputs are not what the headline numbers describe unless you
supply a key.

## Repo layout

```
data/
  generate_synthetic_data.py   # synthetic Kaggle-schema dataset generator
  raw_tweets.csv                # generated (Kaggle twcs.csv schema)
  conversations.csv             # generated (customer<->agent pairs)
  golden_eval_set.csv           # generated (185 hand-labeled-methodology examples)
src/
  config.py           # brand, paths, intent taxonomy, thresholds
  data_prep.py         # raw tweets -> conversation pairs
  intents.py            # rule-based silver labeler (weak supervision for training)
  classify_baseline.py  # trivial baseline + TF-IDF/LogReg baseline
  classify_llm.py        # optional LLM classifier w/ fallback
  retrieval.py            # TF-IDF nearest-neighbor retrieval (grounding)
  reply_generator.py      # template fallback + LLM-grounded generation
  escalation.py            # rule-based auto-handle/escalate decision + reason
  pipeline.py               # ties it all together: AgentPipeline.handle(text)
  evaluate.py                # classification + escalation metrics
  llm_judge.py                # LLM-as-judge + human-agreement calibration
eval/
  build_golden_set.py   # builds the golden eval set (see methodology inside)
  run_eval.py             # the evaluation harness — run this to reproduce results
  human_calibration_labels.csv  # generated: judge vs. simulated-human scores
tests/
  test_pipeline.py    # sanity tests (not the eval harness)
scripts/run_all.sh    # one-command reproduction
REPORT.md             # written analysis (problem framing, baselines, failures, next steps)
DECISION_LOG.md        # 13 non-obvious decisions and why
LABELING_GUIDE.md       # exact rules used to hand-label the golden set
HONESTY_NOTE.md          # what's real vs. simulated in this build, and why
```

## Try a single message interactively

```bash
python -c "
from src.pipeline import AgentPipeline
p = AgentPipeline()
print(p.handle(\"where is my package, ordered 8 days ago and nothing\"))
"
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Citations / borrowed material

- Dataset schema and brand-handle concept: Kaggle "Customer Support on
  Twitter" (`thoughtvector/customer-support-on-twitter`) — schema only, no
  data was downloaded from it (see `HONESTY_NOTE.md`).
- Standard library usage: scikit-learn (TF-IDF, LogisticRegression, metrics),
  scipy (correlation), pandas/numpy. No copied code from external repos or
  tutorials. Anthropic Python SDK used per its public docs for API calls.
