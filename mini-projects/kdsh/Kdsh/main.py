# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

train = pd.read_csv("train.csv")

print(train.shape)
train.head(1)


# %%
import re

def split_claims(text):
    parts = re.split(r"[.;]\s+| and | but | however ", text, flags=re.IGNORECASE)
    claims = []

    for p in parts:
        p = p.strip()
        if len(p) > 20:   # ignore very small fragments
            claims.append(p)

    return claims



# %%
# Mapping between book names in CSV and actual files
BOOK_PATHS = {
    "The Count of Monte Cristo": "books/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "books/In search of the castaways.txt"
}

def load_book(book_name):
    """
    Given a book name from the CSV, load and return the full book text.
    """
    path = BOOK_PATHS[book_name]
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



# %%
def chunk_book(book_text):
 
    raw_paragraphs = book_text.split("\n\n")

    chunks = []
    idx = 0

    for p in raw_paragraphs:
        p = p.strip()
        if len(p) < 100:  
            continue

        chunks.append({
            "idx": idx,
            "text": p
        })
        idx += 1

    return chunks



# %%
def extract_keywords(claim):
    """
    Take a claim sentence and extract important words.
    We ignore short words to avoid noise.
    """
    words = claim.lower().split()
    keywords = [w for w in words if len(w) >= 5]
    return keywords



# %%
def retrieve_chunks(chunks, character, claim):
    """
    Return book chunks that mention the character
    or contain important keywords from the claim.
    """
    keywords = extract_keywords(claim)
    character = character.lower()

    relevant = []

    for c in chunks:
        text = c["text"].lower()

        if character in text:
            relevant.append(c)
        else:
            for kw in keywords:
                if kw in text:
                    relevant.append(c)
                    break

    return relevant



# %%
ABSOLUTE_WORDS = [
    "never",
    "always",
    "only",
    "since childhood",
    "from birth",
    "throughout his life",
    "throughout her life"
]

EVENT_KEYWORDS = [
    "arrested", "re-arrested", "imprisoned", "released",
    "escaped", "escape", "died", "death", "killed",
    "sentenced", "confined"
]

TIME_MARKERS = [
    "again", "this time", "for life"
]

def is_absolute_claim(claim):
    claim = claim.lower()
    for w in ABSOLUTE_WORDS:
        if w in claim:
            return True
    return False

def is_event_claim(claim):
    c = claim.lower()
    has_event = any(k in c for k in EVENT_KEYWORDS)
    has_time = any(t in c for t in TIME_MARKERS) or any(ch.isdigit() for ch in c)
    return has_event and has_time



# %%
def has_hard_contradiction(claim, relevant_chunks):
    """
    Return True if we see a clear contradiction.
    """
    if not is_absolute_claim(claim):
        return False

    for c in relevant_chunks:
        text = c["text"].lower()

        # very simple signals of change or later action
        if "later" in text or "years later" in text or "afterward" in text:
            return True

        if "returned" in text and "never returned" in claim.lower():
            return True

    return False



# %%
def has_event_contradiction(claim, relevant_chunks):
    """
    Return True if the book strongly suggests
    an incompatible event history.
    """
    if not is_event_claim(claim):
        return False

    c = claim.lower()

    for ch in relevant_chunks:
        text = ch["text"].lower()

        # If claim implies a new arrest, but book says already imprisoned
        if "re-arrest" in c or "again" in c:
            if "already imprisoned" in text or "had been imprisoned" in text:
                return True
            if "never released" in text:
                return True

        # If claim says 'for life', but book mentions death shortly after
        if "for life" in c:
            if "died" in text or "death" in text:
                return True

    return False



# %%
def predict_backstory(row):
    claims = split_claims(row["content"])
    book_text = load_book(row["book_name"])
    chunks = chunk_book(book_text)

    for claim in claims:
        relevant_chunks = retrieve_chunks(
            chunks=chunks,
            character=row["char"],
            claim=claim
        )

        # Rule 1: Absolute contradiction
        if has_hard_contradiction(claim, relevant_chunks):
            return 0

        # Rule 2: Event-existence contradiction (NEW)
        if has_event_contradiction(claim, relevant_chunks):
            return 0

    return 1



# %%
def sanitize_claims(claims):
    clean = []

    for c in claims:
        c = c.strip()

        # Reject very short fragments
        if len(c) < 25:
            continue

        # Reject speculation / soft language
        if any(w in c.lower() for w in [
            "might", "could", "possibly", "perhaps", "likely"
        ]):
            continue

        # Reject explanations
        if any(w in c.lower() for w in [
            "because", "therefore", "suggests that"
        ]):
            continue

        clean.append(c)

    return clean


# %%
def split_claims_safe(backstory):
    # Try Gemini first
    # claims = gemini_split_claims(backstory)

    # if claims:
    #     claims = sanitize_claims(claims)

    #     # Safety check: must not reduce info too much
    #     if len(claims) >= 1:
    #         return claims

    # Fallback to regex splitter
    return split_claims(backstory)


# %%
row = train.iloc[0]
claims = split_claims(row["content"])
claim = claims[0]

relevant_chunks = retrieve_chunks(
    chunks=chunk_book(load_book(row["book_name"])),
    character=row["char"],
    claim=claim
)

print("CLAIM:")
print(claim)

print("\nIS ABSOLUTE?:", is_absolute_claim(claim))

print("\nHARD CONTRADICTION?:",
      has_hard_contradiction(claim, relevant_chunks))


# %%
correct = 0
total = len(train)

for _, row in train.iterrows():
    pred = predict_backstory(row)
    gold = 1 if row["label"] == "consistent" else 0

    if pred == gold:
        correct += 1

accuracy = correct / total
print("TOTAL:", total)
print("CORRECT:", correct)
print("ACCURACY:", accuracy)


# %%
correct = 0
total = len(train)

for _, row in train.iterrows():
    pred = predict_backstory(row)
    gold = 1 if row["label"] == "consistent" else 0

    if pred == gold:
        correct += 1

accuracy = correct / total
print("TOTAL:", total)
print("CORRECT:", correct)
print("ACCURACY:", accuracy)

