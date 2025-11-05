# app.py — AdVeritas LITE (Streamlit)
import streamlit as st
from PIL import Image
from ad_audit_lite import AdVeritasLiteModels, audit_ad_lite

st.set_page_config(page_title="AdVeritas — LITE", page_icon="✅", layout="wide")
st.title("AdVeritas — AI Ad Transparency Auditor (LITE)")

@st.cache_resource
def get_models():
    return AdVeritasLiteModels(use_toxicity=False)

models = get_models()

with st.sidebar:
    st.header("Upload Ad")
    img_file = st.file_uploader("Ad image", type=["png","jpg","jpeg"])
    ad_copy  = st.text_area("Ad copy (optional)", height=120)
    run_btn  = st.button("Run Audit")

def risk_label(x: float) -> str:
    return "LOW" if x < 0.33 else ("MEDIUM" if x < 0.66 else "HIGH")

if run_btn:
    if not img_file:
        st.warning("Please upload an image.")
        st.stop()
    image = Image.open(img_file).convert("RGB")
    result = audit_ad_lite(models, image, ad_copy or "")
    c1, c2 = st.columns([1,1])
    with c1:
        st.image(image, caption="Ad", use_container_width=True)
        st.subheader("Audit Results")
        st.metric("Overall Risk (0=low, 1=high)", f"{result.overall_risk:.2f}")
        lvl = risk_label(result.overall_risk)
        color = {"LOW":"🟢","MEDIUM":"🟠","HIGH":"🔴"}[lvl]
        st.markdown(f"### {color} **{lvl} RISK LEVEL**")
        st.markdown("**OCR Extracted Text**")
        st.write(result.ocr_text or "—")
        st.markdown("**Notes**")
        for n in result.notes:
            st.write("• " + n)
    with c2:
        st.markdown("### Sentiment")
        st.json(result.sentiment)
        st.markdown("### Pattern hits by category")
        st.json({
            "sensitive": result.sensitive_hits,
            "health": result.health_hits,
            "finance": result.finance_hits,
            "unsafe": result.unsafe_hits,
            "sexual": result.sexual_hits,
            "absolute_claims": result.claim_hits,
            "greenwashing": result.greenwashing_hits,
            "evidence_terms": result.evidence_hits,
        })
        st.markdown("### Risk contributions (calibrated)")
        st.json(result.contributions)
