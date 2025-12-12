import re
from typing import Tuple, Dict, Any

SYSTEM_NOISE_PHRASES = [
    "Messages and calls are end-to-end encrypted",
    "You changed your phone number",
    "joined using this group's invite link",
    "created group",
    "added",
    "left",
    "changed the subject",
    "changed this group's icon",
    "This message was deleted",
    "Missed voice call",
    "Missed video call",
]

ATTACHMENT_MARKERS = [
    "<Media omitted>",
    "image omitted",
    "video omitted",
    "GIF omitted",
    "audio omitted",
    "document omitted",
    "sticker omitted",
    "attached",
]

def mask_pii(text: str) -> str:
    """
    Light PII masking: emails + phone-ish strings.
    """
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "[PHONE]", text)
    return text

def clean_whatsapp_text(
    raw_text: str,
    keep_last_lines: int = 800,
    remove_system: bool = True,
    remove_attachments: bool = True,
    pii_mask: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Step A: preprocessing only (NOT summarization).
    - Keep last N lines (privacy/cost control)
    - Remove obvious system/noise lines
    - Remove attachment markers
    - Optional PII masking
    """
    lines = raw_text.splitlines()
    original_lines = len(lines)

    if keep_last_lines and keep_last_lines > 0:
        lines = lines[-keep_last_lines:]

    cleaned = []
    removed = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue

        if remove_system and any(p.lower() in s.lower() for p in SYSTEM_NOISE_PHRASES):
            removed += 1
            continue

        if remove_attachments and any(m.lower() in s.lower() for m in ATTACHMENT_MARKERS):
            removed += 1
            continue

        cleaned.append(s)

    text = "\n".join(cleaned)
    if pii_mask:
        text = mask_pii(text)

    stats = {
        "original_lines": original_lines,
        "kept_lines_window": len(lines),
        "removed_noise_lines": removed,
        "final_chars": len(text),
    }
    return text, stats
