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

# ── Secret key ────────────────────────────────────────────────────────────
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# ── Credentials ───────────────────────────────────────────────────────────
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
    print(f"Failed to load ChromaDB: {e}")
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
        print(f"Groq initialization failed: {e}")
        groq_client = None

# ══════════════════════════════════════════════════════════════════════════
# BANNED WORDS — words that must NEVER appear in template TEXT
# Note: these can appear as variable VALUES (sent at message time), not in the template body
# ══════════════════════════════════════════════════════════════════════════
BANNED_WORDS_PATTERNS = [
    # Discount / price
    r'\bdiscount\b', r'\bdiscounts\b', r'\boffer\b', r'\boffers\b',
    r'\bsale\b', r'\bsales\b', r'\bpromo\b', r'\bpromotion\b', r'\bpromotional\b',
    r'\bcoupon\b', r'\bvoucher\b', r'\bcashback\b', r'\brebate\b',
    r'\b\d+\s*%\s*off\b', r'\bflat\s+\d+', r'\bfree\s+gift\b',
    r'\bbest\s+deal\b', r'\bbest\s+price\b', r'\bspecial\s+price\b',
    r'\bno[- ]cost\s+emi\b', r'\bzero[- ]cost\s+emi\b',
    # Loyalty / rewards
    r'\bloyalty\b', r'\breward\b', r'\brewards\b',
    r'\bpoints\b', r'\bperks?\b', r'\bbonus\b', r'\bincentive\b',
    # Urgency
    r'\bhurry\b', r'\blimited\s+time\b', r'\blast\s+chance\b',
    r'\bdon\'t\s+miss\b', r'\bgrab\s+now\b', r'\bact\s+now\b',
    r'\bexpires?\s+soon\b', r'\bwhile\s+stocks?\s+last\b',
    r'\bshop\s+now\b', r'\bshop\s+before\b', r'\bbuy\s+now\b',
    r'\bbook\s+now\b',
    # Festivals / events
    r'\brepublic\s+day\b', r'\bdiwali\b', r'\bfestive\b', r'\bseason\s+sale\b',
    r'\bflash\s+sale\b', r'\bmega\s+sale\b', r'\bbig\s+sale\b',
    r'\bholi\b', r'\bsalary\s+day\b', r'\beid\b', r'\bchristmas\s+sale\b',
    r'\bnew\s+year\s+sale\b', r'\bblack\s+friday\b', r'\bcyber\s+monday\b',
    # Superlatives / hype
    r'\bexclusive\b', r'\bspecial\s+offer\b', r'\bunbeatable\b',
    r'\bincredible\s+deal\b', r'\bamazing\s+deal\b', r'\bbest\s+ever\b',
    r'\bdon\'t\s+wait\b', r'\btoday\s+only\b', r'\bonly\s+today\b',
    # Marketing CTAs
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
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a Meta WhatsApp Business API template compliance specialist for The Sleep Company (a premium mattress brand in India).

Your ONLY job is to produce Meta-approved UTILITY templates — regardless of what the user gives you.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE CORE STRATEGY — VARIABLE MASKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Meta reviews the TEMPLATE TEXT, not the VALUES that go into placeholders at send time.

THEREFORE:
- Any promotional concept (sale, discount, offer, event name, % off) must become a {{variable}}
- The template body uses only neutral, factual, utility-compliant language
- The actual promotional value is inserted at send time — Meta never sees it

EXAMPLE:
Input: "45% off sale template"
WRONG approach: Reject it or hallucinate a store visit template
CORRECT approach:
  Template: "Hi {{1}},\n\nWe have shared an update regarding your {{2}}.\n\nPlease find the details below:\nCustomer Name: {{1}}\nUpdate: {{2}}\nValid Until: {{3}}\n\nReply to this message if you need any assistance."

At send time: {{2}} = "45% off sale on SmartGRID mattresses", {{3}} = "30 April 2026"
Meta only sees: "We have shared an update regarding your {{2}}." — perfectly utility compliant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BANNED WORDS — NEVER appear in template TEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These words must NOT appear anywhere in the template body. If the context requires them, replace with a {{variable}}:

discount, discounts, offer, offers, sale, sales, promo, promotion, promotional,
coupon, voucher, cashback, rebate, free gift, best deal, best price, special price,
loyalty, reward, rewards, points, perks, bonus, incentive,
hurry, limited time, last chance, don't miss, grab now, act now, expires soon,
while stocks last, shop now, shop before, buy now, book now, order now,
exclusive, special offer, unbeatable, flash sale, mega sale, big sale, festive,
republic day, diwali, holi, eid, salary day, christmas sale, new year sale, black friday,
% off, flat X%, X% discount, no-cost EMI, zero cost EMI, incredible deal,
amazing deal, best ever, don't wait, today only, claim now, upgrade now,
click here, visit now, get yours

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMPLATE CLASSIFICATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For PURE MARKETING inputs (sale announcement, discount, offer — no transaction):
  → Use IMAGE NOTIFICATION style (utility notification that an image/update has been shared)
  → The promotional content becomes {{variable}} values

For MIXED inputs (has both transaction + promotion):
  → Focus on the transactional part; put promotional content in a variable if needed

For UTILITY inputs (pure transactional):
  → Standard 4-part utility structure

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APPROVED TEMPLATE STRUCTURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRUCTURE A — IMAGE / UPDATE NOTIFICATION (for marketing inputs)
Used when user wants to share a sale, offer, or campaign image with customers.

Hi {{1}},

We have shared an image corresponding to your {{2}} update.

Customer Name: {{1}}
Update Type: {{2}}
Reference: {{3}}

Reply to this message for any assistance.

---

STRUCTURE B — CUSTOMER UPDATE NOTIFICATION (for marketing inputs)
Used when user wants to inform customers about something happening at TSC.

Dear {{1}},

We would like to inform you about an update from The Sleep Company.

Customer Name: {{1}}
Update: {{2}}
Date: {{3}}

For any queries, please contact us at {{4}}.

---

STRUCTURE C — STANDARD UTILITY (for transactional inputs)
Used for orders, returns, deliveries, bookings, etc.

PART 1 — CONTEXT STATEMENT (1–2 sentences, factual)
PART 2 — STATUS OR EXPLANATION (1–2 sentences, optional)
PART 3 — KEY DETAILS BLOCK (label: {{N}} per line, minimum 2 fields)
PART 4 — ACTION OR CLOSING (1 clear instruction)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLACEHOLDER RULES — CRITICAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Number placeholders sequentially in order of first appearance: {{1}}, {{2}}, {{3}} …
- Every UNIQUE dynamic value gets its own unique number
- NEVER reuse a number for a different value
- If {{1}} = Customer Name in the greeting, it must still mean Customer Name in the details block
- NEVER use {{CustomerName}} style — always {{1}}, {{2}} etc.
- Images are sent as media headers — do NOT add an image placeholder variable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
META UTILITY TONE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- No exclamation marks
- No bullet point lists inside templates
- No emojis used decoratively
- No phrases: "don't hesitate", "feel free", "we'd love to", "hope you enjoyed", "we're here for you"
- Tone: factual, neutral, professional, helpful

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIATION TYPES — generate EXACTLY 5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All 5 must use different phrasings and varying levels of detail. They must NOT all be the same.

1. Minimal     — shortest, fewest placeholders, simplest language
2. Specific    — more detail fields, reference number, date
3. Action-oriented — clear CTA for customer to take next step
4. Confirmatory   — asks customer to confirm receipt or acknowledge
5. Informational  — most complete, explains what the update is about in neutral terms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — valid JSON only, no markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "input_classification": "Utility" | "Marketing" | "Mixed",
  "output_classification": "Utility",
  "promotional_content_detected": true | false,
  "extracted_utility_context": "one sentence describing the utility context",
  "warning": null | "message shown to user if promo content was detected",
  "variations": [
    {
      "id": 1,
      "type": "Minimal|Specific|Action-oriented|Confirmatory|Informational",
      "template": "full template text with {{1}} {{2}} etc.",
      "placeholder_map": {"{{1}}": "Customer Name", "{{2}}": "Campaign/Update Description", "{{3}}": "Date"},
      "why": "1–2 sentences explaining why this passes Meta utility review"
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
    # Pure marketing / campaign → image notification
    if has_marketing_content(lower):
        return "image_notification"
    return "customer_update"


# ══════════════════════════════════════════════════════════════════════════
# PLACEHOLDER RENUMBERING
# ══════════════════════════════════════════════════════════════════════════
def renumber_placeholders(template: str) -> str:
    """
    Ensures {{N}} placeholders are sequentially numbered from 1 in order of first appearance.
    Preserves label-value associations (Label: {{N}} blocks).
    """
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
# STRUCTURE VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def _has_details_block(template: str) -> bool:
    field_pattern = re.compile(r'.+:\s*\{\{\d+\}\}')
    field_lines   = [l for l in template.split('\n') if field_pattern.search(l.strip())]
    return len(field_lines) >= 1   # relaxed to 1 for minimal image notification templates

def _has_multiline_structure(template: str) -> bool:
    non_empty = [l for l in template.split('\n') if l.strip()]
    return len(non_empty) >= 3    # at least 3 non-empty lines

def validate_structure(variations: list) -> list:
    failed = []
    for i, var in enumerate(variations):
        t = var.get("template", "")
        if not _has_details_block(t) or not _has_multiline_structure(t):
            failed.append(i)
            var["_structure_fail"] = True
    return failed


# ══════════════════════════════════════════════════════════════════════════
# POST-GENERATION SANITISATION
# ══════════════════════════════════════════════════════════════════════════
def sanitize_utility(data: dict) -> dict:
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
        original = template
        lines    = template.split('\n')
        cleaned_lines = []

        for line in lines:
            # Split by sentence, drop sentences with banned words
            sentences      = re.split(r'(?<=[.?])\s+', line)
            clean_sentences = [s for s in sentences if not _sentence_contains_banned(s)]
            cleaned_line   = ' '.join(clean_sentences).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

        cleaned = '\n'.join(cleaned_lines)
        cleaned = emoji_pattern.sub("", cleaned).strip()
        cleaned = re.sub(r'^\s*[✔✓•\-\*]\s*.+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        if cleaned != original:
            var["template"] = cleaned
        else:
            var["template"] = cleaned

        # Always renumber
        var["template"] = renumber_placeholders(var["template"])

    data["output_classification"] = "Utility"
    data["classification"]        = "Utility"
    return data


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

    print(f"[DEBUG] Detected intent: {detected_intent}")
    print(f"[DEBUG] Marketing content: {marketing_detected}, Pure promo: {pure_promo}")

    # ── Determine generation mode ──────────────────────────────────────
    if pure_promo:
        input_classification = "Marketing"
        generation_mode      = "image_notification"
        # Extract the campaign essence (what is being announced)
        campaign_essence     = _extract_campaign_essence(user_input)
        print(f"[DEBUG] Campaign essence: {campaign_essence}")
    elif marketing_detected:
        input_classification = "Mixed"
        generation_mode      = "mixed"
        campaign_essence     = None
    else:
        input_classification = "Utility"
        generation_mode      = "utility"
        campaign_essence     = None

    # ── RAG retrieval ──────────────────────────────────────────────────
    # Use campaign essence as query for better RAG results on marketing inputs
    rag_query = campaign_essence if campaign_essence else user_input
    util_res  = utility_coll.query(query_texts=[rag_query], n_results=6)
    approved_examples = util_res['documents'][0] if util_res.get('documents') else []
    sep              = "\n---\n"
    approved_block   = sep.join(approved_examples) if approved_examples else "None available"

    # ── Build generation prompt ────────────────────────────────────────
    if generation_mode == "image_notification":
        mode_instructions = f"""
GENERATION MODE: IMAGE / UPDATE NOTIFICATION (Variable Masking)

The user wants to send a promotional message. Your task is to generate utility-compliant
templates where ALL promotional content is replaced by {{{{variables}}}}.

Campaign essence extracted: "{campaign_essence}"

KEY RULES FOR THIS MODE:
1. The word "sale", "offer", "discount", "% off" etc. must NOT appear in the template text
2. Instead, reference it as {{{{2}}}} (e.g. "your {{{{2}}}} update" where {{{{2}}}} = "45% off sale")
3. Use IMAGE NOTIFICATION or UPDATE NOTIFICATION structure
4. Keep neutral language: "update", "information", "details", "notification"
5. Always include: Customer Name {{{{1}}}}, the masked promo content {{{{2}}}}, at minimum
6. Add a placeholder_map in your JSON showing what each variable represents at send time

CORRECT EXAMPLE:
Template: "Hi {{{{1}}}},\\n\\nWe have shared an image corresponding to your {{{{2}}}} update.\\n\\nCustomer Name: {{{{1}}}}\\nUpdate: {{{{2}}}}\\nValid Until: {{{{3}}}}\\n\\nReply to this message for any assistance."
placeholder_map: {{"{{{{1}}}}": "Customer Name", "{{{{2}}}}": "45% off sale on SmartGRID", "{{{{3}}}}": "30 April 2026"}}

Generate 5 DISTINCT variations — different phrasings, different placeholder counts, different structures.
DO NOT generate 5 identical or near-identical templates."""

    elif generation_mode == "mixed":
        mode_instructions = f"""
GENERATION MODE: MIXED (Transactional + Promotional)

The input contains both a transaction AND promotional elements.
Focus on the TRANSACTIONAL part. Any promotional details become {{{{variables}}}}.

User input: "{user_input}"
Detected intent: {detected_intent}

Put the transactional context in neutral utility language.
If promotional content is needed, it goes into a variable placeholder."""

    else:
        mode_instructions = f"""
GENERATION MODE: STANDARD UTILITY

The input is a pure utility / transactional context.
Detected intent: {detected_intent}

Use the standard 4-part utility structure.
User input: "{user_input}" """

    base_prompt = f"""APPROVED UTILITY TEMPLATES FROM DATABASE (study structure, do NOT copy verbatim):
{approved_block}

INPUT FROM USER: "{user_input}"
{mode_instructions}

TASK: Generate EXACTLY 5 Utility variations.
- input_classification: "{input_classification}"
- output_classification: "Utility"
- promotional_content_detected: {str(marketing_detected).lower()}
- Each variation MUST be meaningfully different from the others
- ZERO banned words in any template text
- Output ONLY valid JSON — no markdown, no explanation"""

    # ── Call LLM ──────────────────────────────────────────────────────
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": base_prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
        )

        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        content = _fix_json_newlines(content)
        parsed  = json.loads(content)

        # Restore literal newlines in templates
        for var in parsed.get("variations", []):
            if "template" in var:
                var["template"] = var["template"].replace('\\n', '\n')

        if not isinstance(parsed.get("variations"), list) or len(parsed["variations"]) == 0:
            raise ValueError("No variations in model response")

        # ── Ensure variation count ────────────────────────────────────
        if len(parsed["variations"]) < 5:
            parsed = _request_more_variations(parsed, content, base_prompt, generation_mode, campaign_essence)

        # ── Structure validation + auto-fix ──────────────────────────
        failed_indices = validate_structure(parsed["variations"])
        if failed_indices:
            print(f"[WARNING] {len(failed_indices)} variation(s) failed structure check")
            parsed = _fix_structure(parsed, content, base_prompt, failed_indices, generation_mode)

        # ── Normalise metadata ────────────────────────────────────────
        parsed["output_classification"] = "Utility"
        parsed["classification"]        = "Utility"
        parsed["input_classification"]  = input_classification

        if marketing_detected:
            parsed["warning"] = (
                "Promotional content detected. All sale/offer/discount references have been "
                "replaced with {{variables}} — your template text is Meta utility-compliant. "
                "Supply the actual promotional values at send time."
            )
        elif parsed.get("warning") in ["null", "none", "None", "", None]:
            parsed["warning"] = None

        # ── Final sanitisation ────────────────────────────────────────
        parsed = sanitize_utility(parsed)

        for var in parsed.get("variations", []):
            var.pop("_structure_fail", None)

        for i, var in enumerate(parsed["variations"]):
            var["id"] = i + 1

        return parsed

    except json.JSONDecodeError as je:
        print(f"[JSON ERROR] Raw content:\n{content[:800]}\n{je}")
        raise RuntimeError("Model returned invalid JSON — try rephrasing your input")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Generation failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _fix_json_newlines(s: str) -> str:
    result      = []
    in_string   = False
    escape_next = False
    for ch in s:
        if escape_next:
            result.append(ch); escape_next = False
        elif ch == '\\':
            result.append(ch); escape_next = True
        elif ch == '"':
            in_string = not in_string; result.append(ch)
        elif ch == '\n' and in_string:
            result.append('\\n')
        elif ch == '\r' and in_string:
            result.append('\\r')
        elif ch == '\t' and in_string:
            result.append('\\t')
        else:
            result.append(ch)
    return ''.join(result)


def _extract_campaign_essence(user_input: str) -> str:
    """
    Pull out what the campaign is actually about in a neutral phrase.
    e.g. "Give me 45% off sale template" → "45% off sale"
         "Republic Day discount 30% off on mattresses" → "Republic Day offer on mattresses"
    Uses a lightweight LLM call.
    """
    prompt = f"""Extract the core campaign/offer topic from the input below.
Return ONLY a short phrase (3–10 words) describing what the campaign is about, without any promotional framing.
Examples:
  "Give me a 45% off sale template" → "45% off sale on The Sleep Company products"
  "Republic Day 30% discount on SmartGRID" → "Republic Day discount on SmartGRID mattress"
  "Holi offer template for our customers" → "Holi season offer for customers"
  "Send loyalty reward message" → "loyalty reward for customers"

Input: "{user_input}"

Reply with ONLY the short phrase. No explanation, no JSON."""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=60,
        )
        essence = resp.choices[0].message.content.strip().strip('"').strip("'")
        return essence or user_input
    except Exception:
        return user_input


def _request_more_variations(parsed: dict, content: str, base_prompt: str,
                              generation_mode: str, campaign_essence) -> dict:
    fix_prompt = f"""You returned only {len(parsed['variations'])} variations. I need EXACTLY 5.
Types required: Minimal, Specific, Action-oriented, Confirmatory, Informational.
All must be pure Utility with zero banned words.
{"Use image/update notification structure with variable masking for promotional content." if generation_mode == "image_notification" else ""}
{"Campaign essence: " + campaign_essence if campaign_essence else ""}
Each variation must be MEANINGFULLY DIFFERENT from the others.
Output ONLY valid JSON with the same structure."""

    resp2 = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": base_prompt},
            {"role": "assistant", "content": content},
            {"role": "user",      "content": fix_prompt}
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    c2 = re.sub(r'^```json\s*|\s*```$', '',
                response2.choices[0].message.content.strip(),
                flags=re.MULTILINE).strip()
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


def _fix_structure(parsed: dict, content: str, base_prompt: str,
                   failed_indices: list, generation_mode: str) -> dict:
    failed_types = [parsed["variations"][i].get("type", "unknown") for i in failed_indices]
    print(f"[DEBUG] Fixing structure for: {failed_types}")

    fix_prompt = f"""The following variation types have incorrect structure: {", ".join(failed_types)}

Each template MUST:
- Have at minimum 3 non-empty lines
- Include at least one "Label: {{{{N}}}}" field line
- Have distinct paragraphs (separated by blank lines)

{"For IMAGE NOTIFICATION mode: use 'Hi {{{{1}}}},\\n\\nWe have shared [neutral description] regarding your {{{{2}}}} update.\\n\\nCustomer Name: {{{{1}}}}\\n[Detail]: {{{{2}}}}\\n\\nReply for assistance.' pattern" if generation_mode == "image_notification" else ""}

Regenerate ONLY these failed types. Output ONLY valid JSON with the same root structure.
Failed types: {", ".join(failed_types)}"""

    resp_fix = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": base_prompt},
            {"role": "assistant", "content": content},
            {"role": "user",      "content": fix_prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    fix_raw = re.sub(r'^```json\s*|\s*```$', '',
                     resp_fix.choices[0].message.content.strip(),
                     flags=re.MULTILINE).strip()
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
                print(f"  [Fixed] {var_type}")
    except Exception as fix_err:
        print(f"[WARNING] Structure fix parse failed: {fix_err} — keeping originals")

    return parsed


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
        session.permanent    = False
        session['logged_in'] = True
        session['user_email']= email
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

        print(f"[DEBUG] User input: {user_input[:150]}{'...' if len(user_input) > 150 else ''}")
        print(f"[DEBUG] Marketing: {has_marketing_content(user_input)}, Pure promo: {is_pure_promotion(user_input)}")

        result = generate_variations(user_input)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error":          str(e) or "Failed to generate templates",
            "classification": "Error",
            "warning":        "Generation failed — try rephrasing or check server logs",
            "variations":     []
        }), 500


if __name__ == "__main__":
    print("Starting TSC Template Studio at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)