# build_library.py
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import csv

print("🚀 Building WhatsApp Utility Template Library (v3 - precise detection & fixed)")

# ── CONFIG ────────────────────────────────────────────────────────────────
CSV_FILE = "data/approved_templates_fixed.csv"          # Use your cleaned file
DB_PATH = "./whatsapp_template_db"
MIN_CONTENT_LENGTH = 15                                 # skip junk

# ── Embedding model ───────────────────────────────────────────────────────
embedder = SentenceTransformer('all-mpnet-base-v2')

# ── ChromaDB ──────────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)

utility_coll = client.get_or_create_collection(
    name="utility_templates",
    embedding_function=SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2')
)

marketing_coll = client.get_or_create_collection(
    name="marketing_templates",
    embedding_function=SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2')
)

# ── Load CSV ──────────────────────────────────────────────────────────────
print(f"Reading: {CSV_FILE}")

df = pd.read_csv(
    CSV_FILE,
    dtype=str,
    quoting=csv.QUOTE_ALL,
    on_bad_lines='warn',
    encoding='utf-8-sig',
    engine='python'
)

print(f"Rows loaded: {len(df)}")
print("Raw columns:", list(df.columns))

# ── Precise column detection ──────────────────────────────────────────────
content_col = None
cat_col = None

for col in df.columns:
    col_lower = str(col).strip().lower()
    if any(k in col_lower for k in ['content', 'template', 'body', 'message']):
        if not content_col:  # take first match
            content_col = col
    if any(k in col_lower for k in ['category', 'type', 'status']):
        if not cat_col:
            cat_col = col

# Fallback: if detection failed, assume first = content, second = category
if content_col is None and len(df.columns) >= 2:
    content_col = df.columns[0]
    print("Fallback: Using first column as content:", content_col)

if cat_col is None and len(df.columns) >= 2:
    cat_col = df.columns[1]
    print("Fallback: Using second column as category:", cat_col)

if content_col is None:
    print("ERROR: No content column found. Columns:", list(df.columns))
    exit(1)

print(f"Using content column: {content_col}")
if cat_col:
    print(f"Using category column: {cat_col}")
else:
    print("No category column → all treated as Utility")

# ── Rename for internal use ───────────────────────────────────────────────
df = df.rename(columns={
    content_col: "__content"
})

if cat_col:
    df = df.rename(columns={cat_col: "__category"})
    df["__category"] = df["__category"].fillna("Utility").astype(str).str.strip().str.lower()
else:
    df["__category"] = "utility"

# ── Clean content ─────────────────────────────────────────────────────────
df["__content"] = (
    df["__content"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.replace(r'\n\s*\n', '\n', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
)

# ── Filter ────────────────────────────────────────────────────────────────
df = df[df["__content"].str.len() >= MIN_CONTENT_LENGTH].copy()

print(f"Rows after filtering: {len(df)}")

# ── Index ─────────────────────────────────────────────────────────────────
count_u = count_m = 0

for idx, row in df.iterrows():
    content = row["__content"]
    cat = row.get("__category", "utility")

    is_utility = 'utility' in cat or len(cat.strip()) == 0

    coll = utility_coll if is_utility else marketing_coll

    try:
        coll.add(
            documents=[content],
            metadatas=[{"category": "Utility" if is_utility else "Marketing"}],
            ids=[f"row_{idx}"]
        )
        if is_utility:
            count_u += 1
        else:
            count_m += 1
    except Exception as e:
        print(f"Skip row {idx}: {e}")

print("\n" + "═" * 50)
print("INDEXING DONE!")
print(f"Utility   : {count_u:,}")
print(f"Marketing : {count_m:,}")
print(f"Total     : {count_u + count_m:,}")
print("═" * 50)

if count_u == 0:
    print("WARNING: No Utility templates! Check categories in CSV.")
print("\nRun: python app.py")