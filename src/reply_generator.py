"""
Draft a reply "grounded in how this brand has historically resolved similar
issues" (assignment requirement #2).

Two modes:
  - No API key: return the closest historically-similar agent reply
    (retrieval's #1 match) as-is. This is a legitimate, honest baseline:
    it is ALWAYS grounded (it's literally a real past reply) but can't adapt
    wording to the new message's specifics.
  - API key present: pass the top-k retrieved (customer_text, agent_reply)
    pairs to the LLM as few-shot grounding and ask it to draft a NEW reply
    tailored to the current message, explicitly instructed to stay
    consistent with the tone/policy shown in the examples and to not invent
    facts (order numbers, refund amounts) not present in the message.
"""
import os
from src.config import ANTHROPIC_MODEL

_GEN_PROMPT = """You are a customer support agent for an e-commerce brand, replying on Twitter.
Below are real past examples of how this brand resolved similar customer issues.
Write a NEW reply to the current customer message that matches the brand's tone
and policy shown in the examples. Keep it under 280 characters, no hashtags.
Do NOT invent an order number, dollar amount, or promise not shown in the
customer's message or the examples — if information is missing, ask the
customer to DM it (as the examples do).

Past resolved examples:
{examples}

Current customer message: "{message}"

Reply:"""


def _client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def draft_reply(customer_text: str, retrieved: list):
    """
    retrieved: output of HistoricalResolutionRetriever.top_k(...)
    Returns dict: {reply, grounded_on: [customer_text,...], source}
    """
    if not retrieved:
        return {
            "reply": "Thanks for reaching out — could you DM us more details so we can help?",
            "grounded_on": [],
            "source": "no_grounding_fallback",
        }

    grounded_examples = [r["customer_text"] for r in retrieved]

    client = _client()
    if client is not None:
        examples_block = "\n".join(
            f'- Customer: "{r["customer_text"]}" -> Agent: "{r["agent_reply"]}"' for r in retrieved
        )
        prompt = _GEN_PROMPT.format(examples=examples_block, message=customer_text)
        try:
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            if text:
                return {"reply": text, "grounded_on": grounded_examples, "source": "llm_grounded"}
        except Exception:
            pass  # fall through to template

    # Fallback: reuse the closest historical reply verbatim. Always grounded,
    # never hallucinates, but not tailored — this trade-off is discussed in
    # REPORT.md.
    return {
        "reply": retrieved[0]["agent_reply"],
        "grounded_on": grounded_examples,
        "source": "retrieval_template",
    }
