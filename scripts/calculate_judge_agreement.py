import pandas as pd
from sklearn.metrics import cohen_kappa_score

JUDGE = "data/processed/llm_judge_scores.csv"
HUMAN = "data/processed/llm_judge_human_calibration.csv"
OUTPUT = "data/processed/llm_judge_agreement.txt"

def main():
    judge = pd.read_csv(JUDGE)
    human = pd.read_csv(HUMAN)
    df = judge.merge(human, on="sample_id", suffixes=("_judge", "_human"))
    metrics = {}
    for field in ["grounded", "relevant", "helpful", "overall"]:
        metrics[field] = cohen_kappa_score(df[f"{field}_human"], df[f"{field}_judge"])
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(f"Calibration examples: {len(df)}\n")
        for key, value in metrics.items():
            f.write(f"{key}_cohen_kappa: {value:.4f}\n")
    print(open(OUTPUT, encoding="utf-8").read())

if __name__ == "__main__":
    main()
