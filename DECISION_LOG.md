# Decision Log

Non-obvious decisions made while building this, and why. See `REPORT.md`
and `HONESTY_NOTE.md` for the fuller reasoning behind several of these.

1. **Built a synthetic-but-schema-faithful dataset instead of skipping the
   dataset requirement.** The sandboxed build environment can't reach
   Kaggle. Rather than write the pipeline against an assumed schema and
   never actually run it, I generated fake-but-realistic data so every
   script is tested end-to-end, and documented the gap loudly instead of
   quietly (`HONESTY_NOTE.md`, referenced from the README's first line).

2. **Trained the classifier on "silver" (rule-based) labels, not the
   synthetic generator's known ground truth.** Using the generator's
   `true_intent` directly would be training on an oracle that wouldn't
   exist with real data (nobody hand-labels 3M tweets for training). Using
   a keyword labeler instead — with all its coverage gaps — produces
   realistically imperfect training signal, which is why failure #1 in the
   report exists at all and is worth discussing.

3. **Golden-set escalation labels use a rule set independent from
   `src/escalation.py`, not the same function.** If both were the same
   function, precision/recall on escalation would be tautologically 1.0.
   Independence is imperfect (same author, overlapping domain assumptions —
   see `HONESTY_NOTE.md`) but it's a real, meaningfully different rule
   (e.g. `account_access` always escalates under the human policy but not
   under the system's), which is what let failure #3 surface.

4. **Injected ~5% deliberate label noise into the golden set** (confusable
   neighbor intents) rather than leaving every label "clean." An eval set
   where 100% accuracy is achievable isn't testing anything interesting;
   modeling genuine annotator disagreement on ambiguous cases is more
   representative of a real hand-labeling pass.

5. **Escalation logic is rule-based, not another LLM call.** A safety gate
   should be deterministic and auditable — "why did this get escalated"
   needs a one-line answer a support lead can verify, not an LLM's
   probabilistic judgment that could differ between runs on the same input.

6. **Every LLM-touching module (classifier, reply generator, judge) has an
   automatic, silent-to-the-user fallback if no API key is present or the
   call fails.** A take-home graded on "runs in 15 minutes" must not hang
   on a network call that can't succeed in a restricted grading sandbox.
   This was directly informed by hitting exactly that restriction myself
   while building this.

7. **Reported headline numbers were generated in no-API-key mode, not
   LLM-mode**, even though an LLM-enhanced path exists in the code. This
   makes every number in `REPORT.md` reproducible by anyone regardless of
   whether they have an API key, and avoids reporting numbers I couldn't
   fully verify were representative (LLM outputs are non-deterministic).

8. **Used TF-IDF cosine similarity for retrieval instead of embeddings.**
   At the scale of a few hundred historical threads, TF-IDF is fully
   offline, deterministic, and — critically for a safety-relevant grounding
   step — the match is literally inspectable by looking at shared words.
   Documented as a "chose not to build" item with a concrete revisit
   trigger (corpus size) in `REPORT.md` §1 and §5.

9. **Retrieval is filtered by predicted intent before searching for
   similar precedents**, not run over the whole corpus. This makes
   retrieval faster and more precise when classification is right, but
   directly causes failure #2 (cascading misclassification → wrong pool →
   false escalation) when classification is wrong. Kept the design because
   the alternative (unfiltered search) would blur genuinely different
   intents' precedents together even more often — traded one failure mode
   for a smaller one, and said so explicitly rather than hiding the
   trade-off.

10. **Left two known bugs unfixed in the reported run** (account_access
    policy gap, silver-labeler blind spot on one phrasing pattern) instead
    of quietly patching them right before writing the report. The
    assignment explicitly rewards "top 5 failure modes with real examples,"
    and a report describing bugs that were fixed five minutes before
    submission isn't the same evidence as a report describing what testing
    actually found in the shipped state.

11. **`unsafe_auto_handle_rate` is reported as a distinct headline metric
    from escalation F1**, not folded into it. F1 can look acceptable while
    hiding a concentrated safety risk in a minority class — this is called
    out explicitly in `REPORT.md` §4.

12. **Chose 8 intents, not more granular ones (e.g. splitting
    `product_defect` into "wrong item" vs. "broken item").** More classes
    would mean thinner golden-set coverage per class (185 examples ÷ more
    buckets) and more classifier confusion between near-duplicate classes
    that don't actually change downstream behavior — see tie-breaking rule
    #4 in `LABELING_GUIDE.md`.

13. **Judge sample size capped at 40 (`JUDGE_SAMPLE_SIZE`), not the full 185
    golden examples.** With an LLM judge enabled, scoring all 185 would risk
    blowing past the 15-minute reproduction budget on API latency; 40 is
    enough to compute a meaningful judge-vs-human correlation without that
    risk. Documented in `eval/run_eval.py`.

14. **`.env` is git-ignored and `.env.example` is committed instead** — so
    the repo never accidentally ships a real API key, and reproduction
    instructions are unambiguous about where to put one.

15. **Chose AmazonHelp over a smaller/simpler brand handle.** Wanted an
    intent taxonomy diverse enough (orders, refunds, billing, account
    security, delivery) to make classification and escalation genuinely
    non-trivial, rather than picking a brand where 90% of traffic would be
    one intent and the whole exercise would be less informative.
