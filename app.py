from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from groq import Groq
import json
import os
import sqlite3
from dotenv import load_dotenv
import traceback
import re

load_dotenv()

app = Flask(__name__)

# ── SQLite feedback DB setup ──────────────────────────────────────────────
def init_feedback_db():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT,
            variation TEXT,
            approved INTEGER,
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_feedback_db()

# ── Load ChromaDB ─────────────────────────────────────────────────────────
try:
    chroma_client = chromadb.PersistentClient(path="./whatsapp_template_db")
    onnx_ef = ONNXMiniLM_L6_V2()
    utility_coll = chroma_client.get_collection(
        "utility_templates",
        embedding_function=onnx_ef
    )
    marketing_coll = chroma_client.get_collection(
        "marketing_templates",
        embedding_function=onnx_ef
    )
    print("ChromaDB collections loaded successfully")
except Exception as e:
    print(f"Failed to load ChromaDB: {e}")
    utility_coll = None
    marketing_coll = None

# ── Groq client ───────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=api_key)
        print("Groq client initialized")
    except Exception as e:
        print(f"Groq initialization failed: {e}")
        groq_client = None

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Meta WhatsApp Business API template compliance specialist for The Sleep Company.

Your job is to take ANY user input — transactional, promotional, vague, or mixed — and rewrite it into a Meta-approved WhatsApp UTILITY template.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You ALWAYS output Utility. Never Marketing. Never Mixed.
classification must always be "Utility" in your JSON output.

Your intelligence is used to figure out HOW to reframe the input as Utility —
not whether to do it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO REFRAME ANY INPUT AS UTILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read the user input and decide which reframe path fits:

PATH A — IMAGE NOTIFICATION (use when input is promotional or campaign-based)
The customer is being informed that an image has been shared with them.
The actual offer or campaign details live inside the image — not in the template text.
This is what makes it Utility: it is a notification that information has been sent,
not a direct promotional message.

CRITICAL FOR PATH A:
You must read the input carefully and extract the SPECIFIC subject matter.
Do NOT write generic shells like "details regarding the product" or
"information about the update". That is lazy and will be rejected.

Instead, derive the specific neutral topic from the input:
- Input about sofa/chair marketing  -> {{2}} = "SmartGRID seating range overview"
- Input about a sale campaign        -> {{2}} = "seasonal pricing update"
- Input about a review request       -> {{2}} = "post-purchase feedback program"
- Input about a product launch       -> {{2}} = "new product availability details"
- Input about a festive campaign     -> {{2}} = "festive period product information"
- Input about a massager/accessory   -> {{2}} = "comfort accessories range update"

The LLM must use its understanding of the input to write specific,
meaningful placeholder labels — never filler text like "the product" or "the update".

Structure for Path A:
Hi {{1}},
We have shared an image with details regarding the {{2}}.
The image contains information about the {{3}}.
[one optional factual line about what to do next]

Topic: {{2}}
Reference: {{5}}

If you have any questions, please contact us at {{4}}.

{{2}} must be a specific neutral topic name derived from the actual input.
{{3}} must describe what the image specifically contains — no filler, no hype.

PATH B — STANDARD 4-PART UTILITY (use when input is transactional)
Order confirmations, delivery updates, returns, exchanges, appointments,
store visits, custom orders, trial updates — anything triggered by a customer action.

Structure:
PART 1 — What happened (1-2 sentences, factual)
PART 2 — Current status or timeline (optional)
PART 3 — Details block with Label: {{N}} lines
PART 4 — One clear action or closing instruction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILS BLOCK FORMAT — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The details block MUST follow this exact format for every variation:

SomeLabel: {{N}}
AnotherLabel: {{N}}

Rules:
- The placeholder {{N}} must come DIRECTLY after the colon and space
- Do NOT write "Label: SomeName - {{N}}" — the dash format is WRONG
- Do NOT write "Label: {{N}} (description)" — nothing after the placeholder
- Correct:   Product Name: {{2}}
- Correct:   Order ID: {{3}}
- Wrong:     Product Name - {{2}}
- Wrong:     Label: Product Name - {{2}}
- Wrong:     Product Name: SmartGRID Chair {{2}}

Every single variation MUST contain at least one line in this exact format.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLACEHOLDER LANGUAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rules for Path A placeholders:
- {{1}} = customer identifier (name, number)
- {{2}} = the sale/campaign name variable — the placeholder REPRESENTS the name,
           so the label can be "Sale Name: {{2}}" which is perfectly valid Utility.
           The actual value (e.g. "End of Season Sale") goes in when sending.
           Never hardcode "45% off" or any offer amount as literal body text.
- {{3}} = a factual description of what the image contains — no hype, no claims
- {{4}} = contact detail or store name

Rules for Path B placeholders:
- Use whatever labels fit the transaction
- Always sequential: {{1}}, {{2}}, {{3}} — never skip numbers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
META UTILITY HARD RULES — apply to ALL output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No promotional language anywhere in the template text
- No discount amounts, no percentages, no urgency phrases
- No emojis
- No exclamation marks
- No bullet lists
- No vague closings like "feel free", "don't hesitate", "we're here for you"
- One clear purpose per template
- Tone: factual, professional, like an official notification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BODY TEXT WORD BAN — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These words must NEVER appear as literal words in the template body text:
"sale", "offer", "discount", "deal", "off", "% off"

The ONLY exception: these words may appear as PART OF A PLACEHOLDER LABEL LINE.
A placeholder label line is any line containing {{N}}.

CORRECT — these are placeholder label lines (allowed):
  Campaign Name: {{2}}
  Offer Reference: {{3}}

WRONG — these are body text sentences (never allowed):
  "We have shared an image about the Sale."
  "The image contains information about the sale details."
  "Details regarding the Offer are in the image."

The {{2}} placeholder IS the sale name — the body text describes it neutrally:
  RIGHT: "We have shared an image with details regarding the {{2}}."
  WRONG: "We have shared an image with details regarding the Sale."

If you catch yourself writing "sale", "offer", or "discount" in a body sentence,
STOP and replace it with {{2}} or a neutral phrase like "campaign update" or "pricing information".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIATION TYPES — generate EXACTLY 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Minimal — fewest fields, shortest version
2. Specific — adds date, store, reference number
3. Action-oriented — strong single CTA
4. Confirmatory — asks customer to confirm receipt or action
5. Informational — most complete version

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After generating variations, rate your confidence (0-100) that these templates
will be approved by Meta as Utility. Be honest — if the input was borderline
or the reframe was difficult, score lower. Output this as "confidence" in JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — valid JSON only, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "classification": "Utility",
  "confidence": 0-100,
  "warning": null,
  "variations": [
    {
      "id": 1,
      "type": "Minimal|Specific|Action-oriented|Confirmatory|Informational",
      "template": "full template text with {{1}} {{2}} etc.",
      "why": "1-2 sentences explaining what makes this Meta-approvable"
    }
  ]
}"""

# ── Forbidden terms for utility sanitization ──────────────────────────────
# Removed broad terms ("% off", "discount", "sale", "promo") that false-flag
# valid placeholder labels like "Sale Name: {{2}}". Promo content is handled
# by the LLM system prompt and validator Check 8 instead.
FORBIDDEN_UTILITY_TERMS = [
    "don't hesitate",
    "feel free to",
    "we'd love to",
    "we would love to",
    "hope you had a great",
    "hope you enjoyed",
    "glad you visited",
    "we're here for you",
    "anything else we can help",
    "hurry",
    "limited time offer",
    "flash sale",
    "exclusive offer",
    "don't miss out",
    "last chance",
    "grab now",
    "best deal",
    "we appreciate your",
    "thank you for choosing",
    "valued customer",
]

# ── Generic shell phrases — indicate lazy Path A generation ───────────────
GENERIC_SHELL_PHRASES = [
    "details regarding the product",
    "details regarding the update",
    "image with details regarding the product",
    "information about the product",
    "information about the update",
    "details about the product",
    # Sale-specific generic shells the model falls back to
    "details regarding the sale",
    "information about the sale",
    "details regarding the offer",
    "information about the offer",
    "details regarding the discount",
]

# ── Placeholder renumbering ───────────────────────────────────────────────
def renumber_placeholders(template: str) -> str:
    """
    Renumbers all {{N}} placeholders in order of first appearance.
    Ensures sequential numbering with no gaps.
    e.g. {{1}}, {{3}}, {{5}} -> {{1}}, {{2}}, {{3}}
    """
    found = re.findall(r'\{\{(\d+)\}\}', template)
    if not found:
        return template

    seen = {}
    counter = 1
    for num in found:
        if num not in seen:
            seen[num] = counter
            counter += 1

    def replacer(match):
        original = match.group(1)
        return "{{" + str(seen[original]) + "}}"

    return re.sub(r'\{\{(\d+)\}\}', replacer, template)

# ── Structure validator ───────────────────────────────────────────────────
def validate_structure(template: str) -> dict:
    """
    Validates a generated template against Meta's structural requirements.
    Returns { "valid": bool, "errors": [str] }
    """
    errors = []

    # ── 1. Character limit ────────────────────────────────────────
    if len(template) > 1024:
        errors.append(f"Exceeds 1024 character limit ({len(template)} chars)")

    # ── 2. Exclamation marks ──────────────────────────────────────
    if "!" in template:
        errors.append("Contains exclamation mark(s) — not allowed in Utility templates")

    # ── 3. Placeholder sequencing ─────────────────────────────────
    found = list(map(int, re.findall(r'\{\{(\d+)\}\}', template)))
    if found:
        expected = list(range(1, len(set(found)) + 1))
        actual_unique = sorted(set(found))
        if actual_unique != expected:
            errors.append(
                f"Placeholders are not sequential — found {actual_unique}, expected {expected}"
            )

    # ── 4. Details block check ────────────────────────────────────
    # Fix: loosened regex — matches any line that contains a {{N}} placeholder
    # Handles: "Product Name: {{2}}", "Topic: {{2}}", "Label: Name - {{2}}" etc.
    label_line = re.search(r'^.+\{\{\d+\}\}', template, re.MULTILINE)
    if not label_line:
        errors.append(
            "Missing details block — no label lines with {{N}} placeholders found. "
            "Every template must have at least one line in format: 'Label: {{N}}'"
        )

    # ── 5. Minimum structure check ────────────────────────────────
    lines = [l.strip() for l in template.strip().split('\n') if l.strip()]
    if len(lines) < 3:
        errors.append(
            "Template too short — likely missing structural parts (Context / Details / Action)"
        )

    # ── 6. Action sentence at end ─────────────────────────────────
    last_para = template.strip().split('\n\n')[-1].strip()
    label_only = re.match(r'^(\w[\w\s]+\s*\{\{\d+\}\}\s*\n?)+$', last_para)
    if label_only:
        errors.append(
            "Template ends with a details block — missing closing action sentence"
        )

    # ── 7. Generic shell detection ────────────────────────────────
    # Catches templates that wrap without reading the actual input
    template_lower = template.lower()
    placeholder_count = len(re.findall(r'\{\{\d+\}\}', template))
    is_generic = any(p in template_lower for p in GENERIC_SHELL_PHRASES)

    if is_generic and placeholder_count < 3:
        errors.append(
            "Template is a generic shell — {{2}} and {{3}} must contain "
            "meaningful specific labels derived from the actual input, not filler "
            "like 'the product' or 'the update'"
        )

    # ── 8. Raw promotional content in body text ─────────────────
    # Checks non-placeholder body lines only.
    # "Sale Name: {{2}}" is a placeholder line — never flagged.
    # "We have shared an image about the Sale." is body text — always flagged.
    promo_body_patterns = [
        r'\d+\s*%\s*(off|discount)',   # "45% off", "30% discount"
        r'rs\.?\s*\d{3,}',             # "Rs. 5000"
        r'\u20b9\s*\d+',               # "₹499"
        r'\bflash sale\b',
        r'\blast chance\b',
        r'\blimited time\b',
    ]
    # Hardcoded promo words in body sentences (not label lines)
    promo_body_words = [r'\bsale\b', r'\boffer\b', r'\bdiscount\b', r'\bdeal\b']

    body_lines = [
        l for l in template.split('\n')
        if l.strip() and not re.search(r'\{\{\d+\}\}', l)
    ]
    body_text = ' '.join(body_lines)

    promo_found = False
    for pattern in promo_body_patterns:
        if re.search(pattern, body_text, re.IGNORECASE):
            promo_found = True
            break
    if not promo_found:
        for pattern in promo_body_words:
            if re.search(pattern, body_text, re.IGNORECASE):
                promo_found = True
                break

    if promo_found:
        errors.append(
            "Template body contains hardcoded promotional word (sale/offer/discount/deal) "
            "or raw promotional content. Use {{2}} as a variable instead. "
            "WRONG: \'details regarding the Sale\' — "
            "RIGHT: \'details regarding the {{2}}\'"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

# ── Sanitize utility ──────────────────────────────────────────────────────
def sanitize_utility(data):
    """
    Line-level sanitization — splits on newlines, not sentences.
    Has a safety guard: if sanitization empties the template, restores original.
    Removes tone-violation lines only — does NOT remove "sale" or "% off" since
    those are legitimate placeholder label values after LLM reframing.
    """
    for var in data.get("variations", []):
        original_template = var.get("template", "")
        template = original_template
        changed = False

        # ── Line-level forbidden term removal ─────────────────────
        lines = template.split('\n')
        cleaned_lines = []
        for line in lines:
            flagged = any(term.lower() in line.lower() for term in FORBIDDEN_UTILITY_TERMS)
            if flagged:
                changed = True
                print(f"[SANITIZE] Removed line: {line[:80]}")
            else:
                cleaned_lines.append(line)
        template = '\n'.join(cleaned_lines)

        # ── Safety: if sanitization emptied the template, restore ─
        if not template.strip():
            print("[SANITIZE] Template would be empty — restoring original")
            var["template"] = original_template
            var["why"] = (var.get("why", "") + " [sanitize skipped — review manually]").strip()
            continue

        # ── Hard exclamation mark removal ─────────────────────────
        if "!" in template:
            template = template.replace("!", "")
            changed = True

        # ── Remove decorative emojis ──────────────────────────────
        emoji_pattern = re.compile(
            "[\U00010000-\U0010ffff"
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U00002600-\U000027BF"
            "\u2714\u2705\u274C\u2713\u2022"
            "]+", flags=re.UNICODE
        )
        cleaned = emoji_pattern.sub("", template).strip()

        # ── Remove bullet/checkmark lines ─────────────────────────
        cleaned = re.sub(r'^\s*[✔✓•\-\*]\s*.+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        if cleaned != template:
            changed = True

        if changed:
            var["template"] = cleaned
            var["why"] = (
                var.get("why", "") + " [auto-sanitized for Meta compliance]"
            ).strip()

    return data

# ── Intent detection (LLM-based with keyword fallback) ───────────────────
def detect_intent(user_input: str) -> str:
    """
    Uses llama-3.1-8b-instant to detect intent.
    Falls back to keyword matching if LLM call fails.
    """
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": f"""Classify the intent of this WhatsApp template request into exactly one of these labels:
review_request, image_notification, order_update, return_refund,
store_visit_followup, trial_update, appointment, custom_order, general

Input: \"{user_input}\"

Rules:
- Use image_notification for any promotional, campaign, sale, offer, or marketing-style input
- Use the most specific label that fits
- Output ONLY the label, nothing else"""
                }],
                temperature=0,
                max_tokens=20,
            )
            intent = response.choices[0].message.content.strip().lower()
            valid_intents = [
                "review_request", "image_notification", "order_update",
                "return_refund", "store_visit_followup", "trial_update",
                "appointment", "custom_order", "general"
            ]
            if intent in valid_intents:
                return intent
        except Exception as e:
            print(f"[INTENT LLM ERROR] Falling back to keywords: {e}")

    # ── Keyword fallback ──────────────────────────────────────────
    lower = user_input.lower()
    if any(k in lower for k in ["review", "feedback", "rate", "rating", "screenshot", "google review"]):
        return "review_request"
    if any(k in lower for k in ["order", "dispatch", "shipped", "delivery", "track"]):
        return "order_update"
    if any(k in lower for k in ["return", "refund", "exchange", "replace"]):
        return "return_refund"
    if any(k in lower for k in ["visit", "store", "walked in", "came in", "in-store", "checking in"]):
        return "store_visit_followup"
    if any(k in lower for k in ["trial", "100 night", "sleep trial"]):
        return "trial_update"
    if any(k in lower for k in ["appointment", "booking", "schedule", "slot"]):
        return "appointment"
    if any(k in lower for k in ["custom", "size", "measurement", "dimension"]):
        return "custom_order"
    return "general"

# ── Two-stage generation: Stage 1 ─────────────────────────────────────────
def extract_intent_facts(user_input: str) -> dict:
    """
    Stage 1: Extract structured intent and facts from raw user input.
    Returns clean JSON so Stage 2 never sees raw promotional language.
    """
    if not groq_client:
        raise ValueError("Groq client not initialized")

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": f"""Extract AND NEUTRALIZE the intent from this WhatsApp template request.
Output ONLY valid JSON, no markdown:

{{
  "intent": "one sentence describing what this template is for — no promo language",
  "path": "A or B — A for promotional/campaign, B for transactional",
  "key_facts": ["list", "of", "neutral", "facts"],
  "dynamic_fields": ["field1", "field2"],
  "neutral_topic": "NEUTRALIZED topic for {{2}} — MUST NOT contain: sale, off, discount, offer, deal, %. Rephrase as an information update. Examples: seasonal pricing update, limited period campaign details, product pricing information, comfort range overview",
  "image_contents": "NEUTRALIZED description of image contents — MUST NOT contain: sale, off, discount, offer, deal. Describe factually. Examples: applicable products and revised pricing, product specifications and pricing details, comfort product range and availability",
  "tone_notes": "any tone or context notes"
}}

NEUTRALIZATION RULES — CRITICAL:
- neutral_topic MUST NOT contain the words: sale, off, discount, offer, deal, %, price drop
- image_contents MUST NOT contain the words: sale, off, discount, deal, offer
- If input mentions a sale -> rephrase as: pricing update, campaign information, product pricing details
- If input mentions a specific % like 45% -> do NOT include that number anywhere in your output
- The actual sale name will be passed as runtime variable {{2}} — write the LABEL, not the VALUE
- Wrong neutral_topic: "Sale", "45% off sale", "discount offer"
- Correct neutral_topic: "seasonal pricing update", "limited period campaign details"

Input: \"{user_input}\""""
        }],
        temperature=0,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
    return json.loads(content)

# ── Generate variations ───────────────────────────────────────────────────
def generate_variations(user_input: str):
    if not groq_client:
        raise ValueError("Groq client not initialized")
    if not utility_coll or not marketing_coll:
        raise ValueError("ChromaDB collections not loaded")

    # ── Stage 1: Extract structured intent ───────────────────────
    extracted = None
    try:
        extracted = extract_intent_facts(user_input)
        print(f"[DEBUG] Extracted: {extracted}")
    except Exception as e:
        print(f"[STAGE 1 ERROR] Falling back to raw input: {e}")

    # ── RAG: fetch approved examples ──────────────────────────────
    util_res = utility_coll.query(query_texts=[user_input], n_results=6)
    approved_examples = util_res['documents'][0] if util_res['documents'] else []

    mark_res = marketing_coll.query(query_texts=[user_input], n_results=3)
    marketing_examples = mark_res['documents'][0] if mark_res['documents'] else []

    detected_intent = detect_intent(user_input)

    sep = "\n---\n"
    approved_block = sep.join(approved_examples) if approved_examples else "None available"
    marketing_block = (
        "\n".join([f"{i+1}. {ex}" for i, ex in enumerate(marketing_examples)])
        if marketing_examples else "None available"
    )

    # ── Stage 2: Build prompt from extracted facts only ───────────
    if extracted:
        path_label = (
            "PATH A (Image Notification)"
            if extracted.get("path") == "A"
            else "PATH B (Standard 4-Part Utility)"
        )
        generation_context = f"""EXTRACTED INTENT SUMMARY — use this, NOT the raw input:
Intent: {extracted.get('intent', '')}
Path: {path_label}
Key Facts: {', '.join(extracted.get('key_facts', []))}
Dynamic Fields: {', '.join(extracted.get('dynamic_fields', []))}
Neutral Topic for {{{{2}}}}: {extracted.get('neutral_topic', '')}
Image Contents for {{{{3}}}}: {extracted.get('image_contents', '')}
Tone Notes: {extracted.get('tone_notes', '')}

IMPORTANT: Use the Neutral Topic above as the specific value context for {{{{2}}}} — never write "the product" or "the update"."""
    else:
        generation_context = f'USER INPUT (direct fallback): "{user_input}"'

    base_prompt = f"""DETECTED INTENT: {detected_intent}

{generation_context}

APPROVED TEMPLATES FROM DATABASE — study their structure:
{approved_block}

MARKETING EXAMPLES — reference for what NOT to do:
{marketing_block}

TASK:
- Generate EXACTLY 5 Utility variations
- classification must always be "Utility"
- Choose path from extracted intent above
- Every variation MUST have a details block with lines in format: SomeLabel: {{{{N}}}}
  The placeholder must come DIRECTLY after the colon — no dashes between label and placeholder
- Do NOT collapse into one paragraph — preserve line-break structure
- Output ONLY valid JSON with confidence score"""

    # ── JSON newline fixer (used across all calls) ────────────────
    def fix_json_newlines(s):
        result = []
        in_string = False
        escape_next = False
        for ch in s:
            if escape_next:
                result.append(ch)
                escape_next = False
            elif ch == '\\':
                result.append(ch)
                escape_next = True
            elif ch == '"':
                in_string = not in_string
                result.append(ch)
            elif ch == '\n' and in_string:
                result.append('\\n')
            elif ch == '\r' and in_string:
                result.append('\\r')
            elif ch == '\t' and in_string:
                result.append('\\t')
            else:
                result.append(ch)
        return ''.join(result)

    def parse_and_clean(raw_content):
        """Parse JSON, restore newlines, renumber placeholders, force Utility."""
        raw_content = re.sub(r'^```json\s*|\s*```$', '', raw_content, flags=re.MULTILINE).strip()
        raw_content = fix_json_newlines(raw_content)
        data = json.loads(raw_content)
        for var in data.get("variations", []):
            if "template" in var:
                var["template"] = var["template"].replace('\\n', '\n')
                var["template"] = renumber_placeholders(var["template"])
        data["classification"] = "Utility"
        if data.get("warning") in ["null", "none", "None", ""]:
            data["warning"] = None
        return data

    def get_failures(data):
        """Return list of failure dicts for any variation that fails validation."""
        failures = []
        for var in data.get("variations", []):
            r = validate_structure(var.get("template", ""))
            if not r["valid"]:
                failures.append({
                    "id": var.get("id", "?"),
                    "type": var.get("type", ""),
                    "template_preview": var.get("template", "")[:150],
                    "errors": r["errors"]
                })
        return failures

    try:
        # ── Primary generation call ────────────────────────────────
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": base_prompt}
            ],
            temperature=0.25,
            max_tokens=2500,
        )
        content = response.choices[0].message.content.strip()
        parsed = parse_and_clean(content)

        if not isinstance(parsed.get("variations"), list) or len(parsed["variations"]) == 0:
            raise ValueError("No variations in model response")

        # ── Fix-retry: structural failures ────────────────────────
        failures = get_failures(parsed)
        if failures:
            failure_detail = "\n".join([
                f"Variation {f['id']} ({f['type']}):\n"
                f"  Template preview: {f['template_preview']}...\n"
                f"  Errors: {'; '.join(f['errors'])}"
                for f in failures
            ])

            neutral_topic = extracted.get('neutral_topic', 'derive from input context') if extracted else 'derive from input context'
            image_contents = extracted.get('image_contents', 'derive from input context') if extracted else 'derive from input context'

            fix_prompt = f"""The following variations failed structural validation. Rewrite ALL 5 from scratch.

FAILURES:
{failure_detail}

ROOT CAUSES AND HOW TO FIX THEM:

1. "Missing details block" error:
   Every variation needs lines in this exact format:
     Sale Name: {{{{2}}}}
     Reference: {{{{4}}}}
   Placeholder {{{{N}}}} must come DIRECTLY after the colon and a space.
   WRONG: "Label: Name - {{{{2}}}}"  |  WRONG: "Name: SmartGRID {{{{2}}}}"
   RIGHT: "Sale Name: {{{{2}}}}"     |  RIGHT: "Reference: {{{{4}}}}"

2. "Generic shell" error:
   You wrote filler like "the product" or "the update".
   Use these specific values from the extracted intent:
     Neutral Topic for {{{{2}}}}: {neutral_topic}
     Image Contents for {{{{3}}}}: {image_contents}

3. "Template ends with details block" error:
   End every template with a plain action sentence:
   "If you have any questions, please contact us at {{{{5}}}}."

4. "Not sequential" error:
   Use {{{{1}}}}, {{{{2}}}}, {{{{3}}}} in strict order — no gaps.

5. Remove ALL exclamation marks — none allowed in Utility templates.

6. Raw promotional content error:
   Do not put "45% off" or any discount amount in the body text.
   The placeholder {{{{2}}}} represents the sale name as a variable.
   Example correct label: "Sale Name: {{{{2}}}}"

Original input context: "{user_input[:300]}"

Rewrite all 5 correctly. Output ONLY valid JSON."""

            response2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.25,
                max_tokens=2500,
            )
            content2 = response2.choices[0].message.content.strip()
            parsed2 = parse_and_clean(content2)

            if isinstance(parsed2.get("variations"), list) and len(parsed2["variations"]) >= 5:
                parsed = parsed2
                print("[FIX-RETRY] Applied structural fix")

        # ── Fix-retry: fewer than 5 variations ────────────────────
        if len(parsed.get("variations", [])) < 5:
            fix_prompt2 = f"""You returned only {len(parsed['variations'])} variations.
I need EXACTLY 5: Minimal, Specific, Action-oriented, Confirmatory, Informational.
Each must have a details block with lines in format: SomeLabel: {{{{N}}}}
Output ONLY valid JSON with the same structure."""

            response3 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_prompt},
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": fix_prompt2}
                ],
                temperature=0.25,
                max_tokens=2500,
            )
            content3 = response3.choices[0].message.content.strip()
            parsed3 = parse_and_clean(content3)

            if isinstance(parsed3.get("variations"), list) and len(parsed3["variations"]) >= 5:
                parsed = parsed3
                print("[FIX-RETRY] Applied count fix")

        # ── Sanitize ───────────────────────────────────────────────
        parsed = sanitize_utility(parsed)

        # ── Re-number IDs cleanly ──────────────────────────────────
        for i, var in enumerate(parsed["variations"]):
            var["id"] = i + 1

        # ── Attach per-variation validation results ────────────────
        for var in parsed["variations"]:
            var["validation"] = validate_structure(var.get("template", ""))

        return parsed

    except json.JSONDecodeError as je:
        print(f"[JSON ERROR] {je}")
        raise RuntimeError("Model returned invalid JSON — try rephrasing your input")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    print("[DEBUG] /api/generate called")
    try:
        data = request.get_json(silent=True)
        if not data or "input" not in data:
            return jsonify({"error": "Missing 'input' field"}), 400

        user_input = data["input"].strip()
        if not user_input:
            return jsonify({"error": "Input cannot be empty"}), 400

        print(f"[DEBUG] Input: {user_input[:150]}{'...' if len(user_input) > 150 else ''}")
        print(f"[DEBUG] Intent: {detect_intent(user_input)}")

        result = generate_variations(user_input)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": str(e) or "Failed to generate templates",
            "classification": "Utility",
            "confidence": 0,
            "warning": "Generation failed — try rephrasing or check server logs",
            "variations": []
        }), 500


@app.route("/api/validate", methods=["POST"])
def validate():
    """
    Lightweight endpoint — validates a single template string.
    Used by frontend inline edit + re-validate flow.
    No Groq calls — purely rule-based.
    """
    try:
        data = request.get_json(silent=True)
        if not data or "template" not in data:
            return jsonify({"error": "Missing 'template' field"}), 400

        template = data["template"].strip()
        if not template:
            return jsonify({"error": "Template cannot be empty"}), 400

        return jsonify(validate_structure(template))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """
    Logs Meta approval/rejection feedback.
    If approved, auto-ingests the template into ChromaDB utility_coll
    so it improves future generation quality.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        user_input       = data.get("user_input", "")
        variation        = data.get("variation", "")
        approved         = 1 if data.get("approved") else 0
        rejection_reason = data.get("rejection_reason", "")

        # ── Write to SQLite ───────────────────────────────────────
        conn = sqlite3.connect("feedback.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO feedback (user_input, variation, approved, rejection_reason) "
            "VALUES (?, ?, ?, ?)",
            (user_input, variation, approved, rejection_reason)
        )
        conn.commit()
        row_id = c.lastrowid
        conn.close()
        print(f"[FEEDBACK] Logged row {row_id} — approved={approved}")

        # ── Auto-ingest approved templates into ChromaDB ──────────
        ingested = False
        if approved and variation and utility_coll:
            try:
                existing = utility_coll.get(where={"source": "feedback"})
                next_id = len(existing["ids"]) + 1
                doc_id = f"feedback_{row_id}_{next_id}"
                utility_coll.add(
                    documents=[variation],
                    ids=[doc_id],
                    metadatas=[{
                        "source": "feedback",
                        "user_input": user_input[:200],
                        "feedback_id": row_id
                    }]
                )
                ingested = True
                print(f"[FEEDBACK] Ingested into ChromaDB as {doc_id}")
            except Exception as ce:
                print(f"[FEEDBACK] ChromaDB ingest failed: {ce}")

        return jsonify({"success": True, "id": row_id, "ingested": ingested})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting TSC Template Studio at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)