import os
import re
import json
from typing import List, Dict, Any
from openai import OpenAI


CHUNK_PROMPT = """You are a privacy-conscious relationship assistant.
Summarize the following WhatsApp chat chunk for the USER (not for the other person).
Focus only on relationship-relevant memory.

Return STRICT JSON with keys:
- "key_topics": array of short strings
- "notable_facts": array of short strings (only if clearly in text; no guessing)
- "open_loops": array of short strings (promises, follow-ups, unanswered questions)
- "tone": one of ["warm","neutral","tense","mixed","unknown"]
- "time_signals": array of short strings

Rules:
- Do NOT include sensitive content unless necessary for follow-up.
- If the chunk is mostly noise, return empty arrays and tone="unknown".
- No extra keys. No markdown. JSON only.
"""

REDUCE_PROMPT = """You are a privacy-conscious relationship assistant.
You will receive multiple JSON chunk summaries of a WhatsApp chat.
Merge them into a final, actionable relationship summary.

Return STRICT JSON with keys:
- "relationship_summary": array of 2-4 bullet strings (facts only, no guessing)
- "key_topics": array of up to 8 short strings
- "open_loops": array of up to 6 short strings
- "suggested_next_action": array of 1-3 bullet strings (timing + what to do)
- "message_drafts": object with keys "whatsapp", "linkedin", "email" (each max 70 words)
- "confidence_note": one short string explaining any missing context

Rules:
- Be warm, human, not salesy.
- If information is insufficient, ask 1 clarifying question inside "confidence_note".
- No extra keys. No markdown. JSON only.
"""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _chunk_text(text: str, max_chars: int = 9000) -> List[str]:
    """
    Simple character-based chunking for MVP.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start + 1000:
            cut = end
        chunks.append(text[start:cut])
        start = cut

    return chunks


def _call_openai_json(system_prompt: str, user_content: str, model: str) -> Dict[str, Any]:
    client = _get_client()

    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    txt = resp.choices[0].message.content.strip()

    # Best-effort strict JSON parsing
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return valid JSON:\n{txt}")


def summarize_whatsapp_hybrid(
    clean_text: str,
    person_name: str,
    purpose: str,
    tone: str,
    channel: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Step B: AI summarization
    - Chunk summaries (Map)
    - Final merge into one JSON (Reduce)
    """
    chunks = _chunk_text(clean_text, max_chars=9000)

    chunk_summaries: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks, start=1):
        payload = f"""Context:
- Person name: {person_name}
- Purpose: {purpose}
- Desired tone for drafts: {tone}
- Primary channel: {channel}

Chat chunk {i}/{len(chunks)}:
{ch}
"""
        chunk_summaries.append(_call_openai_json(CHUNK_PROMPT, payload, model=model))

    reduce_payload = json.dumps({"chunk_summaries": chunk_summaries}, ensure_ascii=False)
    final = _call_openai_json(REDUCE_PROMPT, reduce_payload, model=model)
    return final
