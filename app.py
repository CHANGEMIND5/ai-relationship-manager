import os
import streamlit as st
from dotenv import load_dotenv

# ✅ Load environment variables FIRST (before importing modules that might use them)
load_dotenv()

from parsers.whatsapp_txt import read_txt_file
from core.preprocess import clean_whatsapp_text
from core.summarizer import summarize_whatsapp_hybrid


st.set_page_config(page_title="AI Relationship Manager (MVP)", layout="wide")
st.title("💬 AI Relationship Manager — WhatsApp Summarizer (MVP)")
st.caption(
    "Upload a WhatsApp TXT export, run light preprocessing (noise removal + optional masking), "
    "then generate an actionable relationship summary with OpenAI."
)

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OPENAI_API_KEY is not set. Add it to your .env file or environment variables to enable AI summarization."
    )

with st.expander("🔐 Consent / Privacy"):
    st.write("Use only WhatsApp TXT exports that the user exported themselves.")
    st.write("By default, we process only the last N lines. You can also mask phone numbers and emails.")
    st.write("Tip: Keep this privacy-first. Summarize one chat at a time and avoid storing raw transcripts.")

# -------- Upload + optional inputs --------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("1) Upload")
    txt_file = st.file_uploader("WhatsApp export (.txt)", type=["txt"])

with col2:
    st.subheader("2) Optional inputs (recommended)")
    person_name = st.text_input("Person name (helps message drafts)", value="")
    purpose = st.selectbox("Purpose", ["Catch up", "Follow-up", "Thank you", "Introduction", "Career advice"])
    tone = st.selectbox("Tone", ["Warm", "Professional", "Casual"])
    channel = st.selectbox("Primary channel", ["WhatsApp", "LinkedIn", "Email"])

st.divider()
st.subheader("3) Preprocess options (Step A)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    keep_last_lines = st.selectbox("Summarize range (lines)", [200, 500, 800, 1500, 3000], index=2)
with c2:
    remove_system = st.checkbox("Remove system/noise lines", value=True)
with c3:
    remove_attachments = st.checkbox("Remove attachment markers", value=True)
with c4:
    pii_mask = st.checkbox("Mask phone/email", value=True)

if txt_file is None:
    st.info("Upload a WhatsApp TXT file to begin.")
    st.stop()

raw_text = read_txt_file(txt_file)
clean_text, stats = clean_whatsapp_text(
    raw_text,
    keep_last_lines=keep_last_lines,
    remove_system=remove_system,
    remove_attachments=remove_attachments,
    pii_mask=pii_mask,
)

st.markdown("### Preprocess stats")
st.json(stats)

with st.expander("Preview cleaned text (first 60 lines)"):
    st.code("\n".join(clean_text.splitlines()[:60]), language="text")

st.divider()
st.subheader("4) Generate with AI (Step B)")

model = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4.1"], index=0)

if st.button("✨ Summarize with AI"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing.")
        st.stop()

    if not person_name.strip():
        person_name = "Unknown"

    with st.spinner("Generating summary..."):
        result = summarize_whatsapp_hybrid(
            clean_text=clean_text,
            person_name=person_name,
            purpose=purpose,
            tone=tone,
            channel=channel,
            model=model,
        )

    st.success("Done.")
    st.markdown("## ✅ Final JSON output")
    st.json(result)

    st.markdown("## 📌 Outputs")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Relationship summary")
        for b in result.get("relationship_summary", []):
            st.write(f"- {b}")

        st.markdown("### Key topics")
        for t in result.get("key_topics", []):
            st.write(f"- {t}")

        st.markdown("### Open loops")
        for o in result.get("open_loops", []):
            st.write(f"- {o}")

    with right:
        st.markdown("### Suggested next action")
        for a in result.get("suggested_next_action", []):
            st.write(f"- {a}")

        drafts = result.get("message_drafts", {})
        st.markdown("### Message drafts (copy/paste)")
        st.text_area("WhatsApp", value=drafts.get("whatsapp", ""), height=120)
        st.text_area("LinkedIn", value=drafts.get("linkedin", ""), height=120)
        st.text_area("Email", value=drafts.get("email", ""), height=120)

    st.caption(f"Confidence: {result.get('confidence_note', '')}")
