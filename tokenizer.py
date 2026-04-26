"""quantum_tokenizer.py — Quantum-aware NLP tokenizer for OpenQASM 2.0 programs."""

import re

MAX_LEN = 192

def tokenize_qasm(code: str, max_len: int = MAX_LEN) -> list:
    """
    Quantum-aware NLP tokenizer for OpenQASM 2.0 programs.

    Normalization steps applied (in order):
      1. Lowercase
      2. Remove inline comments
      3. Normalize include directives → INC_QELIB
      4. Space around punctuation
      5. Arrow token  →  ARROW
      6. Pi constant  →  PI
      7. Float angles →  FVAL
      8. Integer indices → IVAL

    Parameters
    ----------
    code    : str  — raw OpenQASM source
    max_len : int  — maximum token sequence length (default 192)

    Returns
    -------
    list of str tokens, truncated to max_len
    """
    code = code.lower()
    code = re.sub(r'//[^\n]*', '', code)
    code = re.sub(r'include\s+"[^"]+"', 'INC_QELIB', code)
    code = re.sub(r'([\[\](),;{}])', r' \1 ', code)
    code = code.replace('->', ' ARROW ')
    code = re.sub(r'\bpi\b', 'PI', code)
    code = re.sub(r'-?\d+\.\d+[eE]?[-+]?\d*', 'FVAL', code)
    code = re.sub(r'(?<![a-z])-?\d+', 'IVAL', code)
    tokens = [t for t in re.split(r'\s+', code.strip()) if t]
    return tokens[:max_len]
