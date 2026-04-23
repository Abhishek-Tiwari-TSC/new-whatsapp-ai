# fix_and_clean_csv.py
import pandas as pd
import csv
import os

INPUT = "data/approved_templates.csv"
OUTPUT = "data/approved_templates_fixed.csv"

print("=== CSV Diagnostic & Fix Script ===")

if not os.path.exists(INPUT):
    print(f"Error: File not found → {INPUT}")
    exit(1)

# Step 1: Read with maximum tolerance
try:
    df = pd.read_csv(
        INPUT,
        dtype=str,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        escapechar='\\',
        on_bad_lines='skip',          # skip bad rows
        encoding='utf-8-sig',
        engine='python'               # more tolerant parser
    )
except Exception as e:
    print(f"Read error: {e}")
    exit(1)

print(f"Rows initially loaded: {len(df)}")
print("Detected columns:", list(df.columns))

# Step 2: Show first few raw rows for debugging
print("\nFirst 5 rows (raw):")
print(df.head(5).to_string())

# Step 3: Force column names (assume first = content, second = category)
if len(df.columns) >= 2:
    df = df.iloc[:, :2].copy()
    df.columns = ["Template Content", "Template Category"]
else:
    print("Error: File has fewer than 2 columns")
    exit(1)

print("\nAfter forcing columns:")
print(df.head(5).to_string())

# Step 4: Basic cleaning
df["Template Content"] = (
    df["Template Content"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)          # normalize spaces
    .str.replace(r'\n\s*\n', '\n', regex=True)     # remove empty lines
)

df["Template Category"] = (
    df["Template Category"]
    .fillna("Utility")
    .astype(str)
    .str.strip()
    .str.title()
    .str.replace(r'^Utility$', 'Utility', regex=True)
    .str.replace(r'^Marketing$', 'Marketing', regex=True)
)

# Step 5: Remove truly empty/short rows (lower threshold to 5 chars)
df = df[df["Template Content"].str.len() >= 5].copy()

print(f"\nRows after cleaning & filtering: {len(df)}")
print(f"Utility count: {len(df[df['Template Category'].str.contains('Utility', case=False)])}")
print(f"Marketing count: {len(df[df['Template Category'].str.contains('Marketing', case=False)])}")

# Step 6: Save fixed version
df.to_csv(
    OUTPUT,
    index=False,
    quoting=csv.QUOTE_ALL,
    quotechar='"',
    encoding='utf-8'
)

print(f"\nFixed file saved: {OUTPUT}")
print("You can now use this in build_library.py (change INPUT to 'approved_templates_fixed.csv')")