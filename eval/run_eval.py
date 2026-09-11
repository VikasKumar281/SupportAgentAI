"""
The evaluation harness. Run after data generation + prep + golden-set
build + baseline training (scripts/run_all.sh does all of this in order).

Produces:
  outputs/predictions.csv   - row-per-golden-example, every system's prediction
  outputs/metrics.json      - all metrics below, machine-readable
  stdout                    - human-readable summary table
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.config import (
    GOLDEN_EVAL_CSV, CONVERSATIONS_CSV, OUTPUTS_DIR, HUMAN_CALIBRATION_CSV, INTENTS,
)
from src.classify_baseline import TfidfLogRegBaseline, TrivialBaseline, make_silver_training_labels
from src.pipeline import AgentPipeline
from src.evaluate import classification_metrics, escalation_metrics, save_metrics
from src.llm_judge import score_reply, simulate_human_score, compute_agreement

JUDGE_SAMPLE_SIZE = 40  # keep LLM-judge calls (if key present) fast for the <15min repro goal


def main():
    golden = pd.read_csv(GOLDEN_EVAL_CSV)
    conversations = pd.read_csv(CONVERSATIONS_CSV)
    print(f"Loaded golden set: {len(golden)} examples | training pool: {len(conversations)} conversations")

    # ---- Baselines (trained fresh here on silver labels, same data the
    # main system's classifier was trained on, for a fair comparison) ----
    silver_labels = make_silver_training_labels(conversations)
    trivial = TrivialBaseline().fit(conversations["customer_text"].tolist(), silver_labels.tolist())
    simple = TfidfLogRegBaseline.load()  # trained by scripts/run_all.sh via classify_baseline.py

    trivial_preds = trivial.predict(golden["customer_text"].tolist())
    simple_preds = simple.predict(golden["customer_text"].tolist())

    # ---- Main system (full pipeline: classify + retrieve + draft + escalate) ----
    pipeline = AgentPipeline(conversations_df=conversations, baseline_model=simple)
    system_results = [pipeline.handle(t) for t in golden["customer_text"].tolist()]
    system_preds = [r["predicted_intent"] for r in system_results]
    system_escalate = [r["escalate"] for r in system_results]

    predictions_df = golden.copy()
    predictions_df["trivial_pred_intent"] = trivial_preds
    predictions_df["simple_pred_intent"] = simple_preds
    predictions_df["system_pred_intent"] = system_preds
    predictions_df["system_confidence"] = [r["intent_confidence"] for r in system_results]
    predictions_df["classifier_source"] = [r["classifier_source"] for r in system_results]
    predictions_df["system_top_similarity"] = [r["top_similarity"] for r in system_results]
    predictions_df["system_draft_reply"] = [r["draft_reply"] for r in system_results]
    predictions_df["reply_source"] = [r["reply_source"] for r in system_results]
    predictions_df["system_escalate"] = system_escalate
    predictions_df["escalation_rule"] = [r["escalation_rule"] for r in system_results]
    predictions_df["escalation_reason"] = [r["escalation_reason"] for r in system_results]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    # ---- Classification metrics: trivial vs simple vs system ----
    y_true = golden["gold_intent"].tolist()
    metrics = {
        "n_golden_examples": len(golden),
        "classification": {
            "trivial_baseline": classification_metrics(y_true, trivial_preds),
            "simple_baseline_tfidf_logreg": classification_metrics(y_true, simple_preds),
            "full_system": classification_metrics(y_true, system_preds),
        },
    }

    # ---- Escalation metrics: system vs. independent human-reviewer gold labels ----
    metrics["escalation"] = escalation_metrics(golden["gold_escalate"].tolist(), system_escalate)

    # ---- Reply quality: LLM-as-judge + agreement with simulated human ----
    judge_sample = predictions_df.head(JUDGE_SAMPLE_SIZE).copy()
    judge_rows = []
    for _, row in judge_sample.iterrows():
        raw_grounded = row.get("grounded_on", "[]")
        try:
            grounded = ast.literal_eval(raw_grounded) if isinstance(raw_grounded, str) else (raw_grounded or [])
        except (ValueError, SyntaxError):
            grounded = []
        j = score_reply(row["customer_text"], row["system_draft_reply"], grounded)
        h = simulate_human_score(row["customer_text"], row["system_draft_reply"])
        judge_rows.append({
            "customer_text": row["customer_text"],
            "system_draft_reply": row["system_draft_reply"],
            "overall_judge_score": j["overall"],
            "judge_source": j["judge_source"],
            "judge_rationale": j.get("rationale", ""),
            "simulated_human_score": h,
        })

    judge_df = pd.DataFrame(judge_rows)
    judge_df.to_csv(OUTPUTS_DIR / "judge_scores.csv", index=False)
    HUMAN_CALIBRATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    judge_df.to_csv(HUMAN_CALIBRATION_CSV, index=False)

    agreement = compute_agreement(judge_df["overall_judge_score"].tolist(), judge_df["simulated_human_score"].tolist())
    metrics["reply_quality"] = {
        "n_scored": len(judge_df),
        "judge_source_used": judge_df["judge_source"].mode().iloc[0] if len(judge_df) else None,
        "mean_judge_score": round(float(judge_df["overall_judge_score"].mean()), 3),
        "mean_simulated_human_score": round(float(judge_df["simulated_human_score"].mean()), 3),
        "judge_vs_human_agreement": agreement,
    }

    save_metrics(metrics, OUTPUTS_DIR / "metrics.json")

    # ---- Print human-readable summary ----
    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION — accuracy / macro-F1")
    print("=" * 70)
    for name, key in [("Trivial (majority class)", "trivial_baseline"),
                       ("Simple (TF-IDF + LogReg)", "simple_baseline_tfidf_logreg"),
                       ("Full system", "full_system")]:
        m = metrics["classification"][key]
        print(f"  {name:30s} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    print("\n" + "=" * 70)
    print("ESCALATION DECISION — vs. independent human-reviewer labels")
    print("=" * 70)
    em = metrics["escalation"]
    print(f"  precision={em['precision']:.3f}  recall={em['recall']:.3f}  f1={em['f1']:.3f}")
    print(f"  ** unsafe_auto_handle_rate={em['unsafe_auto_handle_rate']:.3f} "
          f"({em['n_should_escalate']} examples should've escalated) — THIS is the number that matters most for safety.")
    print(f"  unnecessary_escalation_rate={em['unnecessary_escalation_rate']:.3f} (cost, not safety)")

    print("\n" + "=" * 70)
    print(f"REPLY QUALITY (LLM-as-judge, source={metrics['reply_quality']['judge_source_used']}, n={metrics['reply_quality']['n_scored']})")
    print("=" * 70)
    print(f"  mean judge score:            {metrics['reply_quality']['mean_judge_score']:.2f} / 5")
    print(f"  mean simulated-human score:  {metrics['reply_quality']['mean_simulated_human_score']:.2f} / 5")
    a = agreement
    print(f"  judge-vs-human agreement:    pearson_r={a['pearson_r']}  within_1_point={a['within_1_point_agreement_rate']}")
    print(f"  (see REPORT.md: simulated-human is a documented proxy, not a real annotator)")
    print("\nFull metrics written to outputs/metrics.json, row-level predictions to outputs/predictions.csv")


if __name__ == "__main__":
    main()
