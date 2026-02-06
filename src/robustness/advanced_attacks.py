#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Adversarial Attacks for Security Research.

Implements realistic evasion techniques used by spammers/phishers:
- Homoglyph attacks (Unicode lookalikes)
- Zero-width character injection
- URL obfuscation techniques
- Number/currency symbol variants
- Invisible character attacks
- Mixed-script attacks

Based on real-world spam/phishing evasion patterns.
"""
import random
import re
from typing import Dict, List

# ============================================================================
# HOMOGLYPH MAPPINGS (Unicode lookalikes)
# ============================================================================
HOMOGLYPHS: Dict[str, List[str]] = {
    'a': ['а', 'ａ', 'ɑ', 'α', '@'],  # Cyrillic а, fullwidth, Latin alpha
    'b': ['Ь', 'ｂ', 'ß'],
    'c': ['с', 'ｃ', 'ç', '¢'],  # Cyrillic с
    'd': ['ԁ', 'ｄ', 'đ'],
    'e': ['е', 'ｅ', 'ε', '3', 'є'],  # Cyrillic е
    'g': ['ɡ', 'ｇ', '9'],
    'h': ['һ', 'ｈ', 'н'],  # Cyrillic һ
    'i': ['і', 'ｉ', '1', 'l', '|', 'ı'],  # Cyrillic і
    'j': ['ј', 'ｊ'],  # Cyrillic ј
    'k': ['κ', 'ｋ', 'к'],
    'l': ['ⅼ', 'ｌ', '1', 'I', '|'],  # Roman numeral ⅼ
    'm': ['м', 'ｍ', 'rn'],
    'n': ['п', 'ｎ', 'ո'],
    'o': ['о', 'ｏ', '0', 'ο', 'օ'],  # Cyrillic о, Greek ο
    'p': ['р', 'ｐ', 'ρ'],  # Cyrillic р, Greek ρ
    'q': ['ԛ', 'ｑ'],
    'r': ['г', 'ｒ'],
    's': ['ѕ', 'ｓ', '$', '5'],  # Cyrillic ѕ
    't': ['τ', 'ｔ', '7', 'т'],
    'u': ['υ', 'ｕ', 'μ'],
    'v': ['ν', 'ｖ', 'ѵ'],
    'w': ['ω', 'ｗ', 'ш'],
    'x': ['х', 'ｘ', '×'],  # Cyrillic х
    'y': ['у', 'ｙ', 'γ'],  # Cyrillic у
    'z': ['ᴢ', 'ｚ', 'ζ'],
    'A': ['А', 'Ａ', 'Α'],  # Cyrillic А, Greek Α
    'B': ['В', 'Ｂ', 'Β'],  # Cyrillic В, Greek Β
    'C': ['С', 'Ｃ', 'Ϲ'],  # Cyrillic С
    'E': ['Е', 'Ｅ', 'Ε'],  # Cyrillic Е, Greek Ε
    'H': ['Н', 'Ｈ', 'Η'],  # Cyrillic Н, Greek Η
    'I': ['Ι', 'Ｉ', '1', 'l'],  # Greek Ι
    'K': ['К', 'Ｋ', 'Κ'],  # Cyrillic К, Greek Κ
    'M': ['М', 'Ｍ', 'Μ'],  # Cyrillic М, Greek Μ
    'N': ['Ν', 'Ｎ'],  # Greek Ν
    'O': ['О', 'Ｏ', '0', 'Ο'],  # Cyrillic О, Greek Ο
    'P': ['Р', 'Ｐ', 'Ρ'],  # Cyrillic Р, Greek Ρ
    'S': ['Ѕ', 'Ｓ', '$'],  # Cyrillic Ѕ
    'T': ['Т', 'Ｔ', 'Τ'],  # Cyrillic Т, Greek Τ
    'X': ['Х', 'Ｘ', 'Χ'],  # Cyrillic Х, Greek Χ
    'Y': ['Υ', 'Ｙ'],  # Greek Υ
    'Z': ['Ζ', 'Ｚ'],  # Greek Ζ
}

# Zero-width and invisible characters
ZERO_WIDTH_CHARS = [
    '\u200b',  # Zero Width Space
    '\u200c',  # Zero Width Non-Joiner
    '\u200d',  # Zero Width Joiner
    '\u2060',  # Word Joiner
    '\ufeff',  # Zero Width No-Break Space (BOM)
]

INVISIBLE_CHARS = [
    '\u00ad',  # Soft Hyphen
    '\u034f',  # Combining Grapheme Joiner
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
]

# URL obfuscation patterns
URL_TRICKS = {
    'dot_variants': [' . ', '[.]', '(dot)', '．', '。'],
    'slash_variants': ['/', '⁄', '∕', '／'],
    'http_variants': ['hxxp', 'h**p', 'http[s]', 'ht tp'],
}

# Currency symbol variants
CURRENCY_VARIANTS = {
    '$': ['＄', '💲', 's', 'S', '﹩'],
    '£': ['￡', 'L', '£'],
    '€': ['Є', 'E', '€'],
    '¥': ['￥', 'Y', '¥'],
}


def homoglyph_attack(text: str, probability: float = 0.15, seed: int = None) -> str:
    """
    Replace characters with visually similar Unicode homoglyphs.
    
    Args:
        text: Input text
        probability: Probability of replacing each character
        seed: Random seed for reproducibility
    
    Returns:
        Text with homoglyph substitutions
    """
    rng = random.Random(seed)
    result = []
    
    for char in text:
        if char.lower() in HOMOGLYPHS and rng.random() < probability:
            # Use lowercase mapping for both cases
            variants = HOMOGLYPHS.get(char) or HOMOGLYPHS.get(char.lower(), [char])
            result.append(rng.choice(variants))
        else:
            result.append(char)
    
    return ''.join(result)


def zero_width_injection(text: str, frequency: float = 0.1, seed: int = None) -> str:
    """
    Insert zero-width characters to break tokenization.
    
    Args:
        text: Input text
        frequency: Average frequency of insertions (per character)
        seed: Random seed
    
    Returns:
        Text with zero-width characters injected
    """
    rng = random.Random(seed)
    result = []
    
    for char in text:
        result.append(char)
        if rng.random() < frequency:
            result.append(rng.choice(ZERO_WIDTH_CHARS))
    
    return ''.join(result)


def url_obfuscation(text: str, seed: int = None) -> str:
    """
    Obfuscate URLs using common evasion techniques.
    
    Args:
        text: Input text
        seed: Random seed
    
    Returns:
        Text with obfuscated URLs
    """
    rng = random.Random(seed)
    
    # Replace dots in URLs
    def obfuscate_url(match):
        url = match.group(0)
        # Replace dots
        dot_rep = rng.choice(URL_TRICKS['dot_variants'])
        url = url.replace('.', dot_rep)
        # Optionally obfuscate http
        if rng.random() < 0.5:
            url = re.sub(r'^https?', rng.choice(URL_TRICKS['http_variants']), url)
        return url
    
    # Match URLs
    text = re.sub(r'https?://[^\s]+', obfuscate_url, text)
    text = re.sub(r'www\.[^\s]+', obfuscate_url, text)
    
    return text


def currency_obfuscation(text: str, seed: int = None) -> str:
    """
    Replace currency symbols with variants.
    
    Args:
        text: Input text
        seed: Random seed
    
    Returns:
        Text with currency symbol variants
    """
    rng = random.Random(seed)
    
    for original, variants in CURRENCY_VARIANTS.items():
        if original in text:
            text = text.replace(original, rng.choice(variants))
    
    return text


def number_obfuscation(text: str, seed: int = None) -> str:
    """
    Obfuscate numbers using various techniques.
    
    Args:
        text: Input text
        seed: Random seed
    
    Returns:
        Text with obfuscated numbers
    """
    rng = random.Random(seed)
    
    # Digit variants
    DIGIT_VARIANTS = {
        '0': ['o', 'O', '⓪', '𝟘'],
        '1': ['l', 'I', '①', '𝟙'],
        '2': ['②', '𝟚', 'z'],
        '3': ['③', '𝟛', 'E'],
        '4': ['④', '𝟜', 'A'],
        '5': ['⑤', '𝟝', 'S'],
        '6': ['⑥', '𝟞', 'b'],
        '7': ['⑦', '𝟟', 'T'],
        '8': ['⑧', '𝟠', 'B'],
        '9': ['⑨', '𝟡', 'g'],
    }
    
    result = []
    for char in text:
        if char in DIGIT_VARIANTS and rng.random() < 0.3:
            result.append(rng.choice(DIGIT_VARIANTS[char]))
        else:
            result.append(char)
    
    return ''.join(result)


def mixed_script_attack(text: str, probability: float = 0.2, seed: int = None) -> str:
    """
    Combine multiple attack types for realistic evasion.
    
    Args:
        text: Input text
        probability: Overall attack intensity
        seed: Random seed
    
    Returns:
        Text with mixed script attacks
    """
    rng = random.Random(seed)
    
    # Apply attacks in sequence with varying probabilities
    if rng.random() < probability:
        text = homoglyph_attack(text, probability=0.1, seed=seed)
    
    if rng.random() < probability * 0.5:
        text = zero_width_injection(text, frequency=0.05, seed=seed)
    
    text = url_obfuscation(text, seed=seed)
    text = currency_obfuscation(text, seed=seed)
    
    if rng.random() < probability * 0.3:
        text = number_obfuscation(text, seed=seed)
    
    return text


def invisible_payload(text: str, payload: str = "spam", seed: int = None) -> str:
    """
    Inject invisible payload using combining characters.
    
    This technique hides malicious content in invisible Unicode.
    
    Args:
        text: Input text
        payload: Hidden payload to inject
        seed: Random seed
    
    Returns:
        Text with hidden payload
    """
    rng = random.Random(seed)
    
    # Encode payload in invisible characters (simplified)
    invisible_payload = ''.join([
        INVISIBLE_CHARS[ord(c) % len(INVISIBLE_CHARS)] 
        for c in payload
    ])
    
    # Insert at random position
    pos = rng.randint(0, len(text))
    return text[:pos] + invisible_payload + text[pos:]


# ============================================================================
# Attack Suite for Robustness Evaluation
# ============================================================================

def get_attack_suite() -> Dict[str, callable]:
    """Return all available attacks."""
    return {
        "homoglyph": lambda t: homoglyph_attack(t, probability=0.15),
        "zero_width": lambda t: zero_width_injection(t, frequency=0.1),
        "url_obfuscate": url_obfuscation,
        "currency_obfuscate": currency_obfuscation,
        "number_obfuscate": number_obfuscation,
        "mixed_script": lambda t: mixed_script_attack(t, probability=0.3),
        "invisible_payload": invisible_payload,
    }


def apply_realistic_attack(text: str, attack_type: str = "mixed_script", seed: int = None) -> str:
    """
    Apply a realistic attack to text.
    
    Args:
        text: Input text
        attack_type: Type of attack to apply
        seed: Random seed
    
    Returns:
        Attacked text
    """
    attacks = get_attack_suite()
    
    if attack_type in attacks:
        return attacks[attack_type](text)
    elif attack_type == "all":
        # Apply all attacks
        for attack_fn in attacks.values():
            text = attack_fn(text)
        return text
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


# ============================================================================
# Defense: Text Normalization
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text to remove evasion attempts.
    
    This is an inference-time defense that reverses common attacks.
    
    Args:
        text: Input text (possibly attacked)
    
    Returns:
        Normalized text
    """
    import unicodedata
    
    # Remove zero-width and invisible characters
    for char in ZERO_WIDTH_CHARS + INVISIBLE_CHARS:
        text = text.replace(char, '')
    
    # Normalize Unicode (NFKC converts fullwidth to ASCII)
    text = unicodedata.normalize('NFKC', text)
    
    # Reverse common homoglyphs
    reverse_homoglyphs = {}
    for ascii_char, variants in HOMOGLYPHS.items():
        for variant in variants:
            if variant not in 'a-zA-Z0-9':  # Don't reverse ASCII
                reverse_homoglyphs[variant] = ascii_char
    
    result = []
    for char in text:
        result.append(reverse_homoglyphs.get(char, char))
    text = ''.join(result)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


if __name__ == "__main__":
    # Demo
    test_texts = [
        "URGENT: You won $1000! Click http://win.example.com now!",
        "Free iPhone at www.apple-deal.com - Limited offer!",
        "Call now: 1-800-555-0123 to claim your £500 prize",
    ]
    
    print("=" * 60)
    print("Advanced Attack Demonstrations")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Homoglyph: {homoglyph_attack(text, seed=42)}")
        print(f"Zero-width: {zero_width_injection(text, seed=42)}")
        print(f"Mixed: {mixed_script_attack(text, seed=42)}")
        print(f"Normalized: {normalize_text(mixed_script_attack(text, seed=42))}")
