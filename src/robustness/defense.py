import re
import unicodedata


def normalize_text(text: str) -> str:
    # Unicode normalize
    text = unicodedata.normalize('NFKC', text)
    # Lowercase
    text = text.lower()
    # Replace common obfuscations
    text = text.replace('0', 'o').replace('1', 'l').replace('3', 'e').replace('5', 's').replace('7', 't')
    # Remove repeated punctuation/emoji-like symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
