import os
from pathlib import Path
from typing import Dict, Optional # <-- FINAL FIX: Dict and Optional are now correctly imported
import random

# --- Import the Google Gemini Client ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    genai = None
    APIError = None

BASE = Path(os.getenv("SMARTSEG_BASE", Path.cwd()))
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def _build_prompt(cluster_id: int, cluster_kpi: Dict[str, float], persona_text: str) -> str:
    lines = [f"Cluster {cluster_id} KPIs:"]
    for k, v in cluster_kpi.items():
        lines.append(f"- {k}: {v}")
    if persona_text:
        lines.append("Persona summary:")
        lines.append(persona_text)
    lines.append("")
    lines.append(
        "Task: Draft a short marketing email (start with 'Subject: ' on the first line, then a 2-4 sentence body). "
        "Keep the tone friendly, concise, and include a single clear CTA. Avoid any PII. Only output the subject and body."
    )
    return "\n".join(lines)

def generate_email_via_api(cluster_id: int, cluster_kpi: Dict[str, float], persona_text: str,
                           model: str = "gemini-2.5-flash", max_tokens: int = 200, temperature: float = 0.7) -> Dict:
    """
    Call the Gemini LLM using the Google GenAI SDK with defensive checks.
    """
    api_key = os.getenv("GEMINI_API_KEY") # Check for the new key name
    
    if not api_key:
        return {"error": "no_api", "message": "GEMINI_API_KEY not found in .env"}

    if genai is None:
         return {"error": "import_error", "message": "google-genai library not installed"}

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(cluster_id, cluster_kpi, persona_text)

        system_instruction = (
            "You are a helpful marketing copywriter. Only produce safe, neutral promotional text. "
            "Your response must start with 'Subject: ' on the first line, then a blank line, then the email body. "
            "Avoid personal data, sensational claims, or unsafe content."
        )

        resp = client.models.generate_content(
            model=model,
            contents=[
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            config={
                "system_instruction": system_instruction,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )

        # Robust extraction: some responses are blocked or only available in candidates
        content = (getattr(resp, "text", None) or "").strip()
        if not content:
            # Try extracting from candidates.parts
            try:
                candidates = getattr(resp, "candidates", []) or []
                parts_text = []
                for cand in candidates:
                    content_obj = getattr(cand, "content", None)
                    if content_obj is None:
                        continue
                    parts = getattr(content_obj, "parts", []) or []
                    for p in parts:
                        t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                        if t:
                            parts_text.append(t)
                content = "\n".join(parts_text).strip()
            except Exception:
                content = ""

        # If still empty (likely safety filter), synthesize a safe fallback without error
        if not content:
            try:
                avg_m = float(cluster_kpi.get("avg_monetary", 0))
                avg_f = float(cluster_kpi.get("avg_frequency", 0))
                avg_s = float(cluster_kpi.get("avg_sentiment", 0))
            except Exception:
                avg_m, avg_f, avg_s = 0.0, 0.0, 0.0

            if avg_s < -0.05 and avg_m > 100:
                offer = "20% off + priority support"
                intro = "We value your experience and want to make it right."
            elif avg_f < 1:
                offer = "10% off your next order"
                intro = "It’s been a while — let’s pick up where we left off."
            else:
                offer = "special picks + 5% off"
                intro = "Here are fresh recommendations tailored to your tastes."

            subject = f"{'We miss you' if avg_f < 1 else 'A special offer for you'} — {offer.split('+')[0].strip()}"
            body = (
                f"{intro} Curated selections with {offer}. "
                "See what's new and enjoy the savings while it lasts.\n\nGrab your offer."
            ).strip()
            return {"subject": subject, "body": body, "provider_response": "gemini_fallback", "raw": "blocked"}
        
        # --- Parsing Logic ---
        if "Subject:" in content:
            parts = content.split("Subject:", 1)
            subject_body = parts[1].strip() if len(parts) > 1 else ""
            
            lines = subject_body.splitlines()
            subject = lines[0].strip() if lines else "Special offer"
            
            # Find the first non-empty line after the subject line as the start of the body
            body_lines = [l.strip() for l in lines[1:] if l.strip()]
            body = "\n".join(body_lines).strip()
            
            if not body:
                # Fallback if parsing results in empty body
                subject = subject or "Special offer"
                body = "\n".join(lines[1:]).strip()
        else:
            # Fallback if model doesn't use the 'Subject:' prefix
            lines = content.splitlines()
            subject = (lines[0][:80]) if lines else "Special offer"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            
        return {"subject": subject, "body": body, "provider_response": content, "raw": str(resp)}

    except Exception as e:
        # If specific APIError type is available, try to detect and label; else provide a safe fallback
        if APIError and isinstance(e, APIError):
            return {"error": "api_error", "message": str(e)}
        # Generic runtime: provide a synthesized safe fallback without surfacing an error upstream
        subject = "Special offer — welcome back"
        body = (
            f"Here’s a friendly offer to re-engage. {persona_text if persona_text else ''} "
            "Click to claim your discount and see recommended picks."
        )
        return {"subject": subject, "body": body, "provider_response": "runtime_fallback"}


def generate_email_fallback(cluster_id: int, cluster_kpi: Dict[str, float], persona_text: str) -> Dict:
    """
    Deterministic fallback generator.
    """
    try:
        avg_m = float(cluster_kpi.get("avg_monetary", 0))
        avg_f = float(cluster_kpi.get("avg_frequency", 0))
        avg_s = float(cluster_kpi.get("avg_sentiment", 0))
    except Exception:
        avg_m, avg_f, avg_s = 0.0, 0.0, 0.0

    if avg_s < -0.05 and avg_m > 100:
        offer = "20% off + priority support"
        intro = "We value your experience and want to make it right."
    elif avg_f < 1:
        offer = "10% off your next order"
        intro = "It’s been a while — let’s pick up where we left off."
    else:
        offer = "special picks + 5% off"
        intro = "Here are fresh recommendations tailored to your tastes."

    subject = f"{'We miss you' if avg_f < 1 else 'A special offer for you'} — {offer.split('+')[0].strip()}"
    body = f"{intro} Curated selections with {offer}. See what's new and enjoy the savings while it lasts.\n\nGrab your offer."
    return {"subject": subject, "body": body, "provider_response": "fallback"}

def save_generated_email(out_path: Optional[Path] = None, cluster_id: Optional[int] = None,
                         subject: str = "", body: str = "", metadata: Optional[dict] = None) -> Path:
    """
    Append generated email to outputs/generated_emails.csv.
    """
    import pandas as pd
    p = out_path if out_path is not None else (OUT / "generated_emails.csv")
    row = {"cluster_id": cluster_id, "subject": subject, "body": body}
    if metadata:
        row.update(metadata)
    if p.exists():
        df = pd.read_csv(p)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(p, index=False)
    return p


def generate_email_variants(cluster_id: int, cluster_kpi: Dict[str, float], persona_text: str,
                            provider: str = "gemini", count: int = 3) -> Dict:
    """
    Generate multiple email drafts. If provider is 'gemini', call the API with varied tones.
    If provider is 'offline', create safe variations using the fallback logic.

    Returns a dict with key 'drafts': List[Dict(subject, body, provider_response)].
    """
    tones = ["friendly", "urgent", "informative", "playful", "premium"]
    # Style buckets to encourage alternation per draft
    subject_styles = ["benefit-led", "curiosity", "question", "actionable", "bracketed"]
    cta_styles = ["direct", "urgent", "conversational", "vip"]
    body_formats = [
        "short paragraph",
        "bullet list (3 bullets)",
        "PAS (Problem-Agitate-Solve)",
        "AIDA (Attention-Interest-Desire-Action)"
    ]
    subject_templates = [
        "Unlock extra {discount}% on picks you'll love",
        "Your next favorite is waiting — enjoy {discount}% off",
        "Ready for more? Here's {discount}% off curated for you",
        "For you: handpicked picks with {discount}% savings",
        "A little extra goes a long way — {discount}% off"
    ]
    cta_templates = [
        "Grab your offer",
        "Shop now",
        "See your picks",
        "Start saving",
        "Claim your upgrade"
    ]
    discount_val = int(cluster_kpi.get("recommended_discount", 5))

    drafts = []
    n = max(1, min(5, int(count)))

    for _ in range(n):
        # Randomize core style elements per draft to ensure alternation
        tone = random.choice(tones)
        subj_style = random.choice(subject_styles)
        cta_style = random.choice(cta_styles)
        body_style = random.choice(body_formats)
        subj_template = random.choice(subject_templates)
        cta_line = random.choice(cta_templates)

        tone_hint = f" Use a {tone} tone."
        style_hint = (
            f" Vary the writing with a {subj_style} subject, a {cta_style} CTA, and a {body_style} body."
            " Avoid repeating phrases from prior drafts."
            " Do not use the phrases 'We thought you’d like these picks', 'Based on what customers like you do', or 'Click here to claim your offer'."
            " Keep it under 150 words, and keep the offer as provided if mentioned."
        )
        subject_hint = subj_template.format(discount=discount_val)

        if provider.lower().startswith("gemini"):
            res = generate_email_via_api(
                cluster_id,
                cluster_kpi,
                (persona_text or "") + " " + tone_hint + " " + style_hint + f" Suggested subject: '{subject_hint}'. CTA: '{cta_line}'."
            )
            drafts.append(res)
        else:
            # Offline variations: rewrite with selected styles
            base = generate_email_fallback(cluster_id, cluster_kpi, persona_text or "")
            subj = subject_hint
            body = base.get("body", "")
            # Build variant body with chosen body style and CTA
            if body_style.startswith("bullet"):
                lines = [
                    "• Fresh picks tailored to your tastes",
                    f"• {discount_val}% off applied at checkout",
                    "• Free returns on eligible items"
                ]
                body = "\n".join(lines)
            elif "PAS" in body_style:
                body = (
                    "Missing out on tailored deals? \n"
                    "We get it — browsing takes time. \n"
                    f"Solve it with curated picks and {discount_val}% off today."
                )
            elif "AIDA" in body_style:
                body = (
                    "Attention: new picks match your taste. \n"
                    "Interest: curated from your recent favorites. \n"
                    f"Desire: save {discount_val}% on select items. \n"
                    f"Action: {cta_line}."
                )
            else:
                body = (
                    f"We handpicked new items for you — and added {discount_val}% off. "
                    "Take a look, see what fits, and enjoy the savings while it lasts."
                )
            body = body + f"\n\n{cta_line}."
            drafts.append({"subject": subj, "body": body, "provider_response": "fallback_variant"})

    return {"drafts": drafts}