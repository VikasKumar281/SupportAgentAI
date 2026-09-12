import os
import json
import time
import pandas as pd
from anthropic import Anthropic, BadRequestError

INPUT = "data/processed/agent_evaluation_200.csv"
OUTPUT = "data/processed/llm_judge_scores.csv"
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

def judge_row(client, row):
    prompt = f"""Evaluate this customer-support draft against the retrieved historical support response.

Customer message:
{row['customer_text']}

Retrieved historical customer message:
{row['retrieved_customer_text']}

Retrieved historical support response:
{row['retrieved_brand_response']}

Draft support response:
{row['draft_reply']}

Return JSON only with:
grounded: integer 1-5
relevant: integer 1-5
helpful: integer 1-5
unsupported_claim: integer 0 or 1
overall: integer 1-5
reason: string

Score groundedness by whether the draft stays supported by the retrieved historical response.
Score relevance by whether it addresses the customer's actual request.
Score helpfulness by whether it gives a useful next step without inventing facts.
Set unsupported_claim to 1 if the draft introduces a material unsupported fact or promise."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    text = "".join(block.text for block in message.content if hasattr(block, "text"))
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Judge did not return valid JSON")
    return json.loads(text[start:end])

def main():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("LLM judge skipped: ANTHROPIC_API_KEY is not set.")
        print("Core evaluation remains reproducible without an external judge.")
        return
    df = pd.read_csv(INPUT)
    client = Anthropic(api_key=key)
    rows = []
    try:
        for idx, row in df.iterrows():
            result = judge_row(client, row)
            result["sample_id"] = int(row["sample_id"])
            rows.append(result)
            print(f"Judged {idx + 1}/{len(df)}")
            time.sleep(0.1)
    except BadRequestError as exc:
        message = str(exc)
        if "credit balance is too low" in message.lower():
            print("LLM judge skipped: Anthropic API credit balance is too low.")
            print("No fabricated judge scores were generated.")
            return
        raise
    scores = pd.DataFrame(rows)
    scores = scores[["sample_id", "grounded", "relevant", "helpful", "unsupported_claim", "overall", "reason"]]
    scores.to_csv(OUTPUT, index=False)
    print(f"Saved to: {os.path.abspath(OUTPUT)}")
    print(f"Mean grounded: {scores['grounded'].mean():.3f}")
    print(f"Mean relevant: {scores['relevant'].mean():.3f}")
    print(f"Mean helpful: {scores['helpful'].mean():.3f}")
    print(f"Unsupported claim rate: {scores['unsupported_claim'].mean():.3f}")
    print(f"Mean overall: {scores['overall'].mean():.3f}")

if __name__ == "__main__":
    main()
