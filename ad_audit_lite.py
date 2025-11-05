# ad_audit_lite.py — AdVeritas LITE (OCR + règles + sentiment, scoring calibré)
from dataclasses import dataclass
from typing import Dict, List
import re, numpy as np
from PIL import Image
import torch
from transformers import pipeline
import easyocr

ABSOLUTE_CLAIMS = [
    r"\b(best|number\s*1|#1|ultimate|perfect|always|never|guarantee(d)?|proof|proven)\b",
    r"\b(0\s*%|zero|no\s*emissions|no\s*impact)\b",
]
GREENWASH_FLAGS = [
    r"\b(carbon\s*neutral|net\s*zero|100%\s*sustainable|eco[-\s]*friendly|green\s*energy|planet[-\s]*safe)\b",
]
EVIDENCE_TERMS = [
    r"\b(source|peer[-\s]*review|study|methodology|lca|life\s*cycle|certificate|iso\s*1400\d|certified|audit(ed)?)\b"
]
SENSITIVE_FLAGS = [
    r"\b(white|purity|pure|fair|whitening|lighten|skin\s*whitening|fair\s*skin|colorism)\b",
    r"\b(superior|inferior|race|racial|ethnic|caste|tribe)\b",
    r"\b(stereotype|obedient\s*wife|boys\s*don’t\s*cry|man\s*up|ladylike)\b",
]
HEALTH_RISK_FLAGS = [
    r"\b(miracle|cure|instant\s*results|detox|cleanse|rapid\s*weight\s*loss|burn\s*fat|fat[-\s]*burner)\b",
    r"\b(no\s*side\s*effects|100%\s*safe)\b",
]
FINANCE_DECEPTIVE_FLAGS = [
    r"\b(guaranteed\s*returns?|risk[-\s]*free\s*profits?|get\s*rich\s*quick|make\s*\$?\d+\s*(per|a)\s*(day|week|month))\b",
    r"\b(no\s*risk|zero\s*risk|sure\s*profit|lifetime\s*income)\b",
]
UNSAFE_BEHAVIOR_FLAGS = [
    r"\b(drink\s*and\s*drive|text\s*while\s*driving|no\s*helmet|speed\s*racing|illegal\s*download|pirated)\b",
]
SEXUAL_CONTENT_FLAGS = [
    r"\b(18\+|nsfw|explicit\s*content|onlyfans|adult\s*content)\b"
]
def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.I) for p in patterns]
ABS_PAT    = _compile(ABSOLUTE_CLAIMS)
GW_PAT     = _compile(GREENWASH_FLAGS)
EVID_PAT   = _compile(EVIDENCE_TERMS)
SENS_PAT   = _compile(SENSITIVE_FLAGS)
HEALTH_PAT = _compile(HEALTH_RISK_FLAGS)
FIN_PAT    = _compile(FINANCE_DECEPTIVE_FLAGS)
UNSAFE_PAT = _compile(UNSAFE_BEHAVIOR_FLAGS)
SEXY_PAT   = _compile(SEXUAL_CONTENT_FLAGS)

@dataclass
class AuditResult:
    ocr_text: str
    sentiment: Dict[str, float]
    claim_hits: List[str]
    greenwashing_hits: List[str]
    evidence_hits: List[str]
    sensitive_hits: List[str]
    health_hits: List[str]
    finance_hits: List[str]
    unsafe_hits: List[str]
    sexual_hits: List[str]
    overall_risk: float
    contributions: Dict[str, float]
    notes: List[str]

class AdVeritasLiteModels:
    def __init__(self, device: str = None, use_toxicity: bool=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if self.device == "cuda" else -1,
        )
        self.use_toxicity = use_toxicity
        if self.use_toxicity:
            self.toxicity = pipeline(
                "text-classification",
                model="unitary/unbiased-toxic-roberta",
                device=0 if self.device == "cuda" else -1,
                top_k=None
            )
        else:
            self.toxicity = None
        self.ocr = easyocr.Reader(["en"], gpu=(self.device == "cuda"))

    def ocr_text(self, pil_img: Image.Image) -> str:
        import numpy as np
        result = self.ocr.readtext(np.array(pil_img), detail=0, paragraph=True)
        return " ".join(result).strip()

    def sentiment_scores(self, text: str) -> Dict[str, float]:
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.0}
        res = self.sentiment(text[:500])[0]
        return {"label": res["label"].upper(), "score": float(res["score"])}

    def toxicity_score(self, text: str) -> float:
        if not self.toxicity or not text.strip():
            return 0.0
        preds = self.toxicity(text[:500])[0]
        score = 0.0
        for p in preds:
            if p["label"].lower() in {"toxicity","insult","hate","threat","identity_attack","sexual_explicit"}:
                score += float(p["score"])
        return float(np.clip(score, 0.0, 1.0))

def find_hits(text: str, pats: List[re.Pattern]) -> List[str]:
    return [p.pattern for p in pats if p.search(text)]

def score_contributions(claim_hits, gw_hits, evidence_hits, sentiment_dict,
                        sensitive_hits, health_hits, finance_hits, unsafe_hits,
                        sexual_hits, tox) -> Dict[str, float]:
    label = (sentiment_dict or {}).get("label", "NEUTRAL").upper()
    conf  = float((sentiment_dict or {}).get("score", 0.0))
    contrib = {
        "sensitive": 0.35 * min(1.0, len(sensitive_hits) / 2),
        "health":    0.30 * min(1.0, len(health_hits) / 2),
        "finance":   0.30 * min(1.0, len(finance_hits) / 2),
        "unsafe":    0.25 * min(1.0, len(unsafe_hits) / 2),
        "sexual":    0.20 * min(1.0, len(sexual_hits) / 2),
        "claims":    0.25 * min(1.0, len(claim_hits) / 2),
        "green":     0.30 * min(1.0, len(gw_hits) / 2),
        "toxicity":  0.25 * tox,
        "evidence":  -0.12 if evidence_hits else 0.0,
        "sentiment": (0.10 * conf if label == "NEGATIVE" else (-0.05 * conf if label == "POSITIVE" else 0.0)),
    }
    return contrib

def risk_score(contrib: Dict[str, float], claim_hits, gw_hits,
               evidence_hits, sensitive_hits, health_hits, finance_hits,
               unsafe_hits, sexual_hits) -> float:
    base = sum(contrib.values())
    major_hits = any([sensitive_hits, health_hits, finance_hits, unsafe_hits, sexual_hits])
    greenish   = bool(gw_hits)
    absolutish = bool(claim_hits)
    if major_hits and base < 0.55: base = 0.55
    elif (greenish or absolutish) and not evidence_hits and base < 0.40: base = 0.40
    return float(np.clip(base, 0.0, 1.0))

def audit_ad_lite(models: AdVeritasLiteModels, pil_img: Image.Image, ad_copy: str = "") -> AuditResult:
    ocr_txt = models.ocr_text(pil_img)
    full_text = " ".join([ocr_txt, ad_copy]).lower()
    sent_text = ad_copy if ad_copy.strip() else ocr_txt
    sentiment = models.sentiment_scores(sent_text)
    tox       = models.toxicity_score(full_text)
    claim_hits     = find_hits(full_text, ABS_PAT)
    gw_hits        = find_hits(full_text, GW_PAT)
    evidence_hits  = find_hits(full_text, EVID_PAT)
    sensitive_hits = find_hits(full_text, SENS_PAT)
    health_hits    = find_hits(full_text, HEALTH_PAT)
    finance_hits   = find_hits(full_text, FIN_PAT)
    unsafe_hits    = find_hits(full_text, UNSAFE_PAT)
    sexual_hits    = find_hits(full_text, SEXY_PAT)
    contrib = score_contributions(
        claim_hits, gw_hits, evidence_hits, sentiment,
        sensitive_hits, health_hits, finance_hits, unsafe_hits,
        sexual_hits, tox
    )
    score = risk_score(
        contrib, claim_hits, gw_hits, evidence_hits,
        sensitive_hits, health_hits, finance_hits, unsafe_hits, sexual_hits
    )
    notes = []
    if sensitive_hits: notes.append("Sensitive / biased / stereotyping language detected.")
    if health_hits:    notes.append("Potentially harmful health claims detected.")
    if finance_hits:   notes.append("Potential deceptive financial claims detected.")
    if unsafe_hits:    notes.append("Unsafe/illegal behavior cues detected.")
    if sexual_hits:    notes.append("Sexualized content cues present.")
    if claim_hits:     notes.append("Absolute/superlative claims detected.")
    if gw_hits:        notes.append("Environmental (green) claims detected.")
    if not evidence_hits and (claim_hits or gw_hits):
        notes.append("No evidence terms found; consider adding sources/certifications.")
    if evidence_hits:  notes.append("Evidence/verification terms present.")
    if tox > 0.25:     notes.append(f"Toxic language score is elevated (toxicity={tox:.2f}).")
    return AuditResult(
        ocr_text=ocr_txt,
        sentiment=sentiment,
        claim_hits=claim_hits,
        greenwashing_hits=gw_hits,
        evidence_hits=evidence_hits,
        sensitive_hits=sensitive_hits,
        health_hits=health_hits,
        finance_hits=finance_hits,
        unsafe_hits=unsafe_hits,
        sexual_hits=sexual_hits,
        overall_risk=score,
        contributions=contrib,
        notes=notes,
    )
