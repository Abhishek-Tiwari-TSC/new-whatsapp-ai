from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from groq import Groq
import json
import os
from dotenv import load_dotenv
import traceback
import re
import secrets

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

VALID_EMAIL    = os.getenv("APP_EMAIL",    "abhishek.tiwari@thesleepcompany.in")
VALID_PASSWORD = os.getenv("APP_PASSWORD", "Abhishek@123")

# ── ChromaDB ──────────────────────────────────────────────────────────────
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
    print("Failed to load ChromaDB: " + str(e))
    utility_coll   = None
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
        print("Groq initialization failed: " + str(e))
        groq_client = None


# ══════════════════════════════════════════════════════════════════════════
# BANNED WORDS
# ══════════════════════════════════════════════════════════════════════════
BANNED_WORDS_PATTERNS = [
    r'\bdiscount\b', r'\bdiscounts\b', r'\boffer\b', r'\boffers\b',
    r'\bsale\b', r'\bsales\b', r'\bpromo\b', r'\bpromotion\b', r'\bpromotional\b',
    r'\bcoupon\b', r'\bvoucher\b', r'\bcashback\b', r'\brebate\b',
    r'\b\d+\s*%\s*off\b', r'\bflat\s+\d+', r'\bfree\s+gift\b',
    r'\bbest\s+deal\b', r'\bbest\s+price\b', r'\bspecial\s+price\b',
    r'\bno[- ]cost\s+emi\b', r'\bzero[- ]cost\s+emi\b',
    r'\bloyalty\b', r'\breward\b', r'\brewards\b',
    r'\bpoints\b', r'\bperks?\b', r'\bbonus\b', r'\bincentive\b',
    r'\bhurry\b', r'\blimited\s+time\b', r'\blast\s+chance\b',
    r"\bdon't\s+miss\b", r'\bgrab\s+now\b', r'\bact\s+now\b',
    r'\bexpires?\s+soon\b', r'\bwhile\s+stocks?\s+last\b',
    r'\bshop\s+now\b', r'\bshop\s+before\b', r'\bbuy\s+now\b',
    r'\bbook\s+now\b',
    r'\brepublic\s+day\b', r'\bdiwali\b', r'\bfestive\b', r'\bseason\s+sale\b',
    r'\bflash\s+sale\b', r'\bmega\s+sale\b', r'\bbig\s+sale\b',
    r'\bholi\b', r'\bsalary\s+day\b', r'\beid\b', r'\bchristmas\s+sale\b',
    r'\bnew\s+year\s+sale\b', r'\bblack\s+friday\b', r'\bcyber\s+monday\b',
    r'\bexclusive\b', r'\bspecial\s+offer\b', r'\bunbeatable\b',
    r'\bincredible\s+deal\b', r'\bamazing\s+deal\b', r'\bbest\s+ever\b',
    r"\bdon't\s+wait\b", r'\btoday\s+only\b', r'\bonly\s+today\b',
    r'\bclick\s+here\b', r'\bvisit\s+now\b', r'\border\s+now\b',
    r'\bget\s+yours\b', r'\bclaim\b(?!\s+your\s+order)',
    r'\bupgrade\s+now\b',
]

MARKETING_SIGNALS = [
    r'\d+\s*%\s*off', r'\bdiscount\b', r'\bsale\b', r'\boffer\b', r'\bpromo\b',
    r'\brepublic\s+day\b', r'\bfestive\b', r'\bdiwali\b', r'\bflash\s+sale\b',
    r'\blimited\s+time\b', r'\bhurry\b', r'\blast\s+chance\b', r'\bexclusive\b',
    r'\bdeal\b', r'\bcoupon\b', r'\bvoucher\b', r'\bcashback\b', r'\bfree\s+gift\b',
    r'\bloyalty\b', r'\breward\b', r'\bholi\b', r'\bsalary\s+day\b',
    r'\bno.cost\s+emi\b', r'\bshop\s+now\b', r'\bshop\s+before\b',
    r'\bbuy\s+now\b', r'\bbook\s+now\b', r'\bget\s+\d+', r'\bsave\s+\d+',
    r'\b\d+\s*%\b',
]

UTILITY_ANCHORS = [
    "order", "deliver", "dispatch", "shipped", "track",
    "return", "refund", "exchange", "replace",
    "visit", "store", "walked in", "came in",
    "appointment", "booking", "schedule", "slot",
    "trial", "100 night", "sleep trial",
    "custom", "size", "measurement",
    "review", "feedback", "rating",
    "purchase", "bought", "received",
    "invoice", "payment", "warranty", "installation",
]


def has_marketing_content(text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in MARKETING_SIGNALS)


def is_pure_promotion(user_input: str) -> bool:
    lower      = user_input.lower()
    has_promo  = has_marketing_content(lower)
    has_anchor = any(anchor in lower for anchor in UTILITY_ANCHORS)
    return has_promo and not has_anchor


def _sentence_contains_banned(sentence: str) -> bool:
    s = sentence.lower()
    return any(re.search(p, s) for p in BANNED_WORDS_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — standard utility (non-marketing)
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_UTILITY = """You are a Meta WhatsApp Business API template compliance specialist for The Sleep Company (a premium mattress brand in India).

Your ONLY job: produce Meta-approved UTILITY templates.

━━━ PLACEHOLDER RULES ━━━
- Number sequentially: {{1}}, {{2}}, {{3}} etc.
- NEVER reuse a number for a different value
- NEVER use named placeholders like {{CustomerName}}
- Use the MINIMUM placeholders needed — only truly dynamic values

━━━ FORBIDDEN GENERIC FIELDS — never include these lines ━━━
- "Customer Name: {{N}}"
- "Reference Number: {{N}}" / "Reference: {{N}}"
- "Label: {{N}}"
- "Description: {{N}}"
- "Update: {{N}}"
- "Name: {{N}}"
Every field must be specific and contextually meaningful to the use case.

━━━ BANNED WORDS — must NEVER appear in template text ━━━
discount, offer, sale, promo, coupon, voucher, cashback, rebate, free gift,
best deal, best price, loyalty, reward, points, bonus, incentive, hurry,
limited time, last chance, don't miss, grab now, expires soon, shop now,
buy now, book now, exclusive, flash sale, festive, republic day, diwali,
holi, eid, % off, no-cost EMI, today only, click here, order now, claim now

━━━ TONE RULES ━━━
- No exclamation marks
- No bullet point lists
- No decorative emojis
- No "don't hesitate", "feel free", "we'd love to"
- Factual, neutral, professional

━━━ VARIATION TYPES — generate EXACTLY 5 ━━━
1. Minimal        — shortest, fewest placeholders
2. Specific       — more context-specific detail fields
3. Action-oriented — clear next step for customer
4. Confirmatory   — asks customer to confirm or acknowledge
5. Informational  — most complete, explains status and next steps

━━━ OUTPUT FORMAT — valid JSON only, no markdown fences ━━━
{
  "input_classification": "Utility",
  "output_classification": "Utility",
  "promotional_content_detected": false,
  "extracted_utility_context": "one sentence",
  "warning": null,
  "variations": [
    {
      "id": 1,
      "type": "Minimal",
      "header": "NONE",
      "template": "...",
      "placeholder_map": {"{{1}}": "what this represents"},
      "why": "1-2 sentences"
    }
  ]
}"""


# ══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — image header (marketing/sale queries)
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_IMAGE = """You are a Meta WhatsApp Business API template compliance specialist for The Sleep Company (a premium mattress brand in India).

The user has provided a promotional message/template that contains banned words (sale, discount, % off, etc.).
Your job is to REWRITE it as a Meta-approved Utility IMAGE HEADER template.

━━━ HOW IMAGE HEADER TEMPLATES WORK ━━━
- header: IMAGE — the promotional banner/creative is sent as the image
- The body text is what you write — it must be utility-compliant
- The image does the promotional heavy lifting; the body just provides context

━━━ YOUR CORE TASK: CONTEXT-PRESERVING REWRITE ━━━
READ the user's input carefully. Understand what it is saying — the theme, emotion, and intent.
Then REWRITE it keeping the same flow and feel, but:
1. Replace every banned phrase with a {{variable}} placeholder
2. Keep all neutral, non-promotional language as-is
3. Preserve the paragraph structure and emotional tone where possible

EXAMPLE INPUT:
  "Big comfort upgrade pending? Make sure your home is ready.
   Recline, relax, and enjoy every moment, just the way everyday comfort should feel.
   Shop at the Comfort Carnival Sale with up to 40% OFF.
   Also, enjoy extra savings with bank offers & No-Cost EMI."

EXAMPLE CORRECT REWRITE (Minimal variation):
  "Big comfort upgrade pending? Make sure your home is ready.

   Recline, relax, and enjoy every moment, just the way everyday comfort should feel.

   {{1}} is now live with up to {{2}} savings.
   Also, enjoy extra savings with {{3}}."

  placeholder_map: {
    "{{1}}": "Comfort Carnival Sale",
    "{{2}}": "40% OFF",
    "{{3}}": "bank offers & No-Cost EMI"
  }

EXAMPLE CORRECT REWRITE (Specific variation — adds validity):
  "Big comfort upgrade pending? Make sure your home is ready.

   Recline, relax, and enjoy every moment, just the way everyday comfort should feel.

   {{1}} is now live. Savings of up to {{2}} available.
   {{3}} also applicable. Valid until {{4}}."

  placeholder_map: {
    "{{1}}": "Comfort Carnival Sale",
    "{{2}}": "40% OFF",
    "{{3}}": "Bank offers & No-Cost EMI",
    "{{4}}": "validity date e.g. 30 April 2025"
  }

━━━ WHAT COUNTS AS A BANNED PHRASE (must become {{variable}}) ━━━
sale, discount, % off, X% off, offer, promo, coupon, cashback, rebate,
free gift, no-cost EMI, zero-cost EMI, bank offer, loyalty, reward, points,
hurry, limited time, last chance, shop now, buy now, exclusive, flash sale,
festive, republic day, diwali, holi, eid, today only, order now, claim now

━━━ WHAT IS SAFE TO KEEP AS LITERAL TEXT ━━━
Comfort, upgrade, home, ready, recline, relax, enjoy, every moment, everyday,
feel, live, available, applicable, valid, details, image, update, contact us,
reply, reach out, assist, information, pending, share, notification — and any
other neutral descriptive language that carries no promotional implication.

━━━ RULES ━━━
- NO "Dear Customer" openers
- No exclamation marks
- No bullet point lists
- No emojis
- Preserve paragraph breaks from the original where they make sense
- Each variation must be meaningfully different (different length, structure, or detail level)
- NEVER invent context not present in the input
- placeholder_map values = the actual send-time values, NOT descriptions like "sale event name"

━━━ VARIATION TYPES — generate EXACTLY 5 ━━━
1. Minimal        — keeps neutral phrases, swaps only banned words, shortest form
2. Specific       — adds a validity date or product category placeholder
3. Action-oriented — ends with a clear store visit / contact instruction
4. Confirmatory   — ends by asking customer to reply to confirm interest
5. Informational  — most complete rewrite, all detail preserved, all banned words masked

━━━ OUTPUT FORMAT — valid JSON only, no markdown fences ━━━
{
  "input_classification": "Marketing",
  "output_classification": "Utility",
  "promotional_content_detected": true,
  "extracted_utility_context": "one sentence describing what the campaign is about",
  "warning": null,
  "variations": [
    {
      "id": 1,
      "type": "Minimal",
      "header": "IMAGE",
      "template": "rewritten body text preserving input context, with {{1}} {{2}} etc. for banned phrases",
      "placeholder_map": {
        "{{1}}": "actual value to insert at send time e.g. Comfort Carnival Sale",
        "{{2}}": "actual value to insert at send time e.g. 40% OFF"
      },
      "why": "1-2 sentences explaining why this passes Meta utility review"
    }
  ]
}"""


# ══════════════════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════════════════
def detect_intent(user_input: str) -> str:
    lower = user_input.lower()
    if any(k in lower for k in ["review", "feedback", "rate", "rating", "screenshot", "google review"]):
        return "review_request"
    if any(k in lower for k in ["return", "refund", "exchange", "replace"]):
        return "return_refund"
    if any(k in lower for k in ["trial", "100 night", "sleep trial"]):
        return "trial_update"
    if any(k in lower for k in ["appointment", "booking", "schedule", "slot"]):
        return "appointment"
    if any(k in lower for k in ["custom", "size", "measurement", "dimension"]):
        return "custom_order"
    if any(k in lower for k in ["visit", "store", "walked in", "came in", "in-store"]):
        return "store_visit_followup"
    if any(k in lower for k in ["order", "dispatch", "shipped", "delivery", "track"]):
        return "order_update"
    if has_marketing_content(lower):
        return "notification_update"
    return "customer_update"


# ══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER RENUMBERING
# ══════════════════════════════════════════════════════════════════════════
def renumber_placeholders(template: str) -> str:
    lines        = template.split("\n")
    lline        = re.compile(r"^([^:{{]+):\s*\{{(\d+)\}}\s*$")
    ph           = re.compile(r"\{{(\d+)\}}")
    occurrences  = []

    for li, line in enumerate(lines):
        m = lline.match(line.strip())
        if m:
            label, orig = m.group(1).strip(), m.group(2)
            occurrences.append((li, orig, True, label))
        else:
            for match in ph.finditer(line):
                occurrences.append((li, match.group(1), False, None))

    counter          = [1]
    key_to_new       = {}
    orig_first_label = {}

    def next_num():
        n = counter[0]; counter[0] += 1; return n

    occurrence_new = []
    for li, orig, is_label, label in occurrences:
        if is_label:
            key = ("label", label)
            if key not in key_to_new:
                key_to_new[key] = next_num()
            occurrence_new.append(key_to_new[key])
            if orig not in orig_first_label:
                orig_first_label[orig] = label
        else:
            if orig in orig_first_label:
                key = ("label", orig_first_label[orig])
                occurrence_new.append(key_to_new[key])
            else:
                key = ("inline", orig)
                if key not in key_to_new:
                    key_to_new[key] = next_num()
                occurrence_new.append(key_to_new[key])

    occ_iter     = iter(zip(occurrences, occurrence_new))
    result_lines = []

    for li, line in enumerate(lines):
        m = lline.match(line.strip())
        if m:
            (_, orig, _, _), new_num = next(occ_iter)
            result_lines.append(re.sub(
                r"\{{" + re.escape(orig) + r"\}}",
                "{{" + str(new_num) + "}}", line, count=1))
        else:
            def replacer(match, _iter=occ_iter):
                (_, orig, _, _), new_num = next(_iter)
                return "{{" + str(new_num) + "}}"
            result_lines.append(ph.sub(replacer, line))

    return "\n".join(result_lines)


# ══════════════════════════════════════════════════════════════════════════
# FORBIDDEN FIELD REMOVAL
# ══════════════════════════════════════════════════════════════════════════
FORBIDDEN_FIELD_PATTERNS = [
    r'^customer\s+name\s*:',
    r'^reference\s+(number|no\.?|#)?\s*:',
    r'^label\s*:',
    r'^description\s*:',
    r'^update\s+type\s*:',
    r'^update\s*:',
    r'^name\s*:',
]

def _strip_forbidden_fields(template: str) -> str:
    lines   = template.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip().lower()
        if any(re.match(p, stripped) for p in FORBIDDEN_FIELD_PATTERNS):
            continue
        cleaned.append(line)
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned))
    return result.strip()


# ══════════════════════════════════════════════════════════════════════════
# STRUCTURE VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def _has_multiline_structure(template: str) -> bool:
    non_empty = [l for l in template.split('\n') if l.strip()]
    return len(non_empty) >= 2  # image templates can be 2 lines


def validate_structure(variations: list) -> list:
    failed = []
    for i, var in enumerate(variations):
        t = var.get("template", "")
        if not _has_multiline_structure(t):
            failed.append(i)
            var["_structure_fail"] = True
    return failed


# ══════════════════════════════════════════════════════════════════════════
# IMAGE TEMPLATE BODY LENGTH ENFORCEMENT
# ══════════════════════════════════════════════════════════════════════════
def _enforce_image_body_length(template: str) -> str:
    """For image templates, keep body to max 4 non-empty lines."""
    lines     = template.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if len(non_empty) <= 4:
        return template
    # Keep first 4 non-empty lines, drop the rest
    kept    = []
    count   = 0
    for line in lines:
        if line.strip():
            if count < 4:
                kept.append(line)
                count += 1
            # else skip
        else:
            if kept:  # keep blank lines between kept content only
                kept.append(line)
    return '\n'.join(kept).strip()


# ══════════════════════════════════════════════════════════════════════════
# SANITISATION
# ══════════════════════════════════════════════════════════════════════════
def sanitize_utility(data: dict, is_image_mode: bool = False) -> dict:
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U00002600-\U000027BF"
        "\u2714\u2705\u274C\u2713\u2022"
        "]+", flags=re.UNICODE
    )

    for var in data.get("variations", []):
        template = var.get("template", "")

        # Strip forbidden generic field lines
        template = _strip_forbidden_fields(template)

        lines         = template.split('\n')
        cleaned_lines = []

        for line in lines:
            sentences       = re.split(r'(?<=[.?])\s+', line)
            clean_sentences = [s for s in sentences if not _sentence_contains_banned(s)]
            cleaned_line    = ' '.join(clean_sentences).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        cleaned = '\n'.join(cleaned_lines)
        cleaned = emoji_pattern.sub("", cleaned).strip()
        cleaned = re.sub(r'^\s*[✔✓•\-\*]\s*.+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        # For image mode just ensure header is set — don't truncate the rewritten body
        if is_image_mode:
            var["header"] = "IMAGE"   # ensure header is always set

        var["template"] = renumber_placeholders(cleaned)

    data["output_classification"] = "Utility"
    data["classification"]        = "Utility"
    data["warning"]               = None
    return data


# ══════════════════════════════════════════════════════════════════════════
# JSON HELPER
# ══════════════════════════════════════════════════════════════════════════
def _fix_json_newlines(s: str) -> str:
    result      = []
    in_string   = False
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


# ══════════════════════════════════════════════════════════════════════════
# CAMPAIGN ESSENCE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════
def _extract_campaign_essence(user_input: str) -> str:
    prompt = (
        'Extract the core campaign/offer topic from the input below.\n'
        'Return ONLY a short phrase (3-10 words) describing what the campaign is about.\n'
        'Examples:\n'
        '  "Give me a 45% off sale template" -> "45% off sale on The Sleep Company products"\n'
        '  "Republic Day 30% discount on SmartGRID" -> "Republic Day discount on SmartGRID mattress"\n\n'
        'Input: "' + user_input + '"\n\n'
        'Reply with ONLY the short phrase. No explanation, no JSON.'
    )
    try:
        resp    = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60,
        )
        essence = resp.choices[0].message.content.strip().strip('"').strip("'")
        return essence or user_input
    except Exception:
        return user_input


# ══════════════════════════════════════════════════════════════════════════
# MORE VARIATIONS HELPER
# ══════════════════════════════════════════════════════════════════════════
def _request_more_variations(parsed: dict, content: str, base_prompt: str,
                              system_prompt: str, campaign_essence) -> dict:
    current_count = len(parsed['variations'])
    parts = [
        "You returned only " + str(current_count) + " variations. I need EXACTLY 5.",
        "Types required: Minimal, Specific, Action-oriented, Confirmatory, Informational.",
        "All must be pure Utility compliant.",
        "NEVER include generic lines like 'Customer Name: {{N}}', 'Reference Number: {{N}}', 'Label: {{N}}'.",
    ]
    if campaign_essence:
        parts.append("Campaign essence for variable masking: " + campaign_essence)
    parts.append("Each variation must be MEANINGFULLY DIFFERENT from the others.")
    parts.append("Output ONLY valid JSON with the same structure.")

    fix_prompt = "\n".join(parts)

    resp2 = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": base_prompt},
            {"role": "assistant", "content": content},
            {"role": "user",      "content": fix_prompt}
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    c2 = re.sub(
        r'^```json\s*|\s*```$',
        '',
        resp2.choices[0].message.content.strip(),
        flags=re.MULTILINE
    ).strip()

    try:
        p2 = json.loads(_fix_json_newlines(c2))
        if isinstance(p2.get("variations"), list) and len(p2["variations"]) >= 5:
            for var in p2["variations"]:
                if "template" in var:
                    var["template"] = var["template"].replace('\\n', '\n')
            return p2
    except Exception:
        pass
    return parsed


# ══════════════════════════════════════════════════════════════════════════
# STRUCTURE FIX HELPER
# ══════════════════════════════════════════════════════════════════════════
def _fix_structure(parsed: dict, content: str, base_prompt: str,
                   failed_indices: list, system_prompt: str, is_image_mode: bool) -> dict:
    failed_types     = [parsed["variations"][i].get("type", "unknown") for i in failed_indices]
    failed_types_str = ", ".join(failed_types)
    print("[DEBUG] Fixing structure for: " + failed_types_str)

    if is_image_mode:
        parts = [
            "The following variation types are too short or empty: " + failed_types_str,
            "",
            "For IMAGE templates, body must be 2-4 lines. Example:",
            '  "We have shared an update for {{1}}.\nValid until: {{2}}\n\nReply for assistance."',
            "",
            "NEVER use verbose paragraphs starting with 'Dear Customer, Your product...'",
            "Regenerate ONLY these failed types. Output ONLY valid JSON.",
        ]
    else:
        parts = [
            "The following variation types are too short (under 2 lines): " + failed_types_str,
            "",
            "Requirements:",
            "- Minimum 2 non-empty lines",
            "- Follow the approved examples style",
            "- NEVER use: 'Customer Name: {{N}}', 'Reference: {{N}}', 'Label: {{N}}'",
            "",
            "Regenerate ONLY these failed types. Output ONLY valid JSON.",
        ]

    fix_prompt = "\n".join(parts)

    resp_fix = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": base_prompt},
            {"role": "assistant", "content": content},
            {"role": "user",      "content": fix_prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    fix_raw = re.sub(
        r'^```json\s*|\s*```$',
        '',
        resp_fix.choices[0].message.content.strip(),
        flags=re.MULTILINE
    ).strip()
    fix_raw = _fix_json_newlines(fix_raw)

    try:
        fix_parsed    = json.loads(fix_raw)
        fix_vars      = fix_parsed.get("variations", [])
        fixed_by_type = {v.get("type", "").lower(): v for v in fix_vars}
        for idx in failed_indices:
            var_type = parsed["variations"][idx].get("type", "").lower()
            if var_type in fixed_by_type:
                fv = fixed_by_type[var_type]
                fv["template"] = fv["template"].replace("\\n", "\n")
                parsed["variations"][idx] = fv
                print("  [Fixed] " + var_type)
    except Exception as fix_err:
        print("[WARNING] Structure fix parse failed: " + str(fix_err))

    return parsed


# ══════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════
def generate_variations(user_input: str) -> dict:
    if not groq_client:
        raise ValueError("LLM client not initialized — check GROQ_API_KEY in .env")
    if not utility_coll:
        raise ValueError("ChromaDB utility collection not loaded")

    detected_intent    = detect_intent(user_input)
    marketing_detected = has_marketing_content(user_input)
    pure_promo         = is_pure_promotion(user_input)

    print("[DEBUG] Detected intent: " + detected_intent)
    print("[DEBUG] Marketing: " + str(marketing_detected) + ", Pure promo: " + str(pure_promo))

    # ── Determine mode ─────────────────────────────────────────────────
    is_image_mode = pure_promo  # pure promotional = image header mode

    if pure_promo:
        input_classification = "Marketing"
        campaign_essence     = _extract_campaign_essence(user_input)
        system_prompt        = SYSTEM_PROMPT_IMAGE
        print("[DEBUG] Mode: IMAGE HEADER | Campaign essence: " + campaign_essence)
    elif marketing_detected:
        input_classification = "Mixed"
        campaign_essence     = None
        system_prompt        = SYSTEM_PROMPT_UTILITY
        print("[DEBUG] Mode: MIXED UTILITY")
    else:
        input_classification = "Utility"
        campaign_essence     = None
        system_prompt        = SYSTEM_PROMPT_UTILITY
        print("[DEBUG] Mode: STANDARD UTILITY")

    # ── RAG retrieval ──────────────────────────────────────────────────
    rag_query         = campaign_essence if campaign_essence else user_input
    util_res          = utility_coll.query(query_texts=[rag_query], n_results=8)
    approved_examples = util_res['documents'][0] if util_res.get('documents') else []

    print(f"[DEBUG] RAG returned {len(approved_examples)} approved examples")

    sep            = "\n---\n"
    approved_block = sep.join(approved_examples) if approved_examples else "None available"

    # ── Build prompt ───────────────────────────────────────────────────
    if is_image_mode:
        base_prompt = (
            "INPUT MESSAGE TO REWRITE:\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + user_input + "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "TASK: Rewrite the above message as EXACTLY 5 IMAGE HEADER Utility template variations.\n\n"
            "CORE INSTRUCTION:\n"
            "- Read the input message carefully — understand its theme, structure, and intent\n"
            "- Keep all neutral language exactly as-is (comfort, upgrade, home, relax, enjoy, feel, etc.)\n"
            "- Replace ONLY the banned phrases with {{variables}}: sale names, % off values, EMI terms, bank offers, event names\n"
            "- The rewritten body should still FEEL like the original message, just with banned phrases as {{variables}}\n"
            "- DO NOT replace neutral emotional language with {{variables}}\n"
            "- DO NOT write generic 'We have shared an update' templates — that ignores the input context\n"
            "- placeholder_map values = the ACTUAL send-time values (e.g. 'Comfort Carnival Sale', '40% OFF'), NOT descriptions\n\n"
            "BANNED PHRASES that must become {{variables}}:\n"
            "sale names, % off, discount amounts, 'No-Cost EMI', 'bank offers', event names (Diwali, Holi, etc.)\n\n"
            "SAFE TO KEEP AS LITERAL TEXT:\n"
            "comfort, upgrade, home, ready, recline, relax, enjoy, every moment, everyday, feel,\n"
            "pending, available, applicable, valid, details, contact us, reply, information\n\n"
            "- header: \"IMAGE\" for all variations\n"
            "- input_classification: \"Marketing\"\n"
            "- output_classification: \"Utility\"\n"
            "- promotional_content_detected: true\n"
            "- warning: null\n"
            "- Output ONLY valid JSON — no markdown, no explanation"
        )
    else:
        base_prompt = (
            "APPROVED UTILITY TEMPLATES FROM OUR DATABASE:\n"
            "Study these carefully. Mirror their structure and field names.\n"
            "Do NOT copy verbatim — adapt to the current use case.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + approved_block +
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "USER REQUEST: \"" + user_input + "\"\n"
            "Detected intent: " + detected_intent + "\n\n"
            "TASK: Generate EXACTLY 5 Utility template variations.\n"
            "- input_classification: \"" + input_classification + "\"\n"
            "- output_classification: \"Utility\"\n"
            "- promotional_content_detected: " + str(marketing_detected).lower() + "\n"
            "- warning: null\n"
            "- STRICTLY follow the structure seen in the approved examples above\n"
            "- NEVER include: 'Customer Name: {{N}}', 'Reference Number: {{N}}', 'Label: {{N}}'\n"
            "- Use MINIMUM placeholders — only truly dynamic values\n"
            "- Each variation must be meaningfully different\n"
            "- Zero banned words in template text\n"
            "- Output ONLY valid JSON — no markdown, no explanation"
        )

    # ── Call LLM ──────────────────────────────────────────────────────
    content = ""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": base_prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        content = _fix_json_newlines(content)
        parsed  = json.loads(content)

        for var in parsed.get("variations", []):
            if "template" in var:
                var["template"] = var["template"].replace('\\n', '\n')

        if not isinstance(parsed.get("variations"), list) or len(parsed["variations"]) == 0:
            raise ValueError("No variations in model response")

        # Ensure 5 variations
        if len(parsed["variations"]) < 5:
            parsed = _request_more_variations(
                parsed, content, base_prompt, system_prompt, campaign_essence
            )

        # Structure validation
        failed_indices = validate_structure(parsed["variations"])
        if failed_indices:
            print("[WARNING] " + str(len(failed_indices)) + " variation(s) failed structure check")
            parsed = _fix_structure(
                parsed, content, base_prompt, failed_indices, system_prompt, is_image_mode
            )

        # Normalise metadata
        parsed["output_classification"] = "Utility"
        parsed["classification"]        = "Utility"
        parsed["input_classification"]  = input_classification
        parsed["warning"]               = None

        # Final sanitisation
        parsed = sanitize_utility(parsed, is_image_mode=is_image_mode)

        for var in parsed.get("variations", []):
            var.pop("_structure_fail", None)
            # Ensure header field exists on all variations
            if "header" not in var:
                var["header"] = "IMAGE" if is_image_mode else "NONE"

        for i, var in enumerate(parsed["variations"]):
            var["id"] = i + 1

        return parsed

    except json.JSONDecodeError as je:
        print("[JSON ERROR] Raw content:\n" + content[:800] + "\n" + str(je))
        raise RuntimeError("Model returned invalid JSON — try rephrasing your input")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Generation failed: " + str(e))


# ══════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET"])
def login_page():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if email == VALID_EMAIL.lower() and password == VALID_PASSWORD:
        session.permanent     = False
        session['logged_in']  = True
        session['user_email'] = email
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid email or password."}), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
@login_required
def generate():
    print("[DEBUG] /api/generate called")
    try:
        data = request.get_json(silent=True)
        if not data or "input" not in data:
            return jsonify({"error": "Missing 'input' field"}), 400

        user_input = data["input"].strip()
        if not user_input:
            return jsonify({"error": "Input cannot be empty"}), 400
        if len(user_input) > 2000:
            return jsonify({"error": "Input too long — keep it under 2000 characters"}), 400

        print("[DEBUG] User input: " + user_input[:150])
        result = generate_variations(user_input)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error":          str(e) or "Failed to generate templates",
            "classification": "Error",
            "warning":        None,
            "variations":     []
        }), 500


if __name__ == "__main__":
    print("Starting TSC Template Studio at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)