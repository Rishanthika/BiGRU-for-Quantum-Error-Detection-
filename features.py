"""features.py — 20-dimensional hand-crafted error-signal features for quantum programs."""

import math
from collections import Counter

VALID_GATE_SET = {'h','x','y','z','s','t','sdg','tdg','rx','ry','rz',
                  'cx','cy','cz','ch','swap','ccx','cswap','cu1','crx',
                  'measure','barrier','reset'}
BAD_GATE_SET   = {'hh','xx','qx','gx','vx','ux','zz','pp','nn','mm','qq','ww'}
TWO_Q_SET      = {'cx','cy','cz','ch','swap','ccx','cswap','cu1','crx'}


def extract_signal_features(tokens: list) -> list:
    """
    Extract 20 highly-discriminative error-signal features from a token sequence.

    Features
    --------
    0  has_bad_gate     — bad gate token present (syntax marker)
    1  n_bad_gate       — normalised count of bad gate tokens
    2  no_measure       — missing measure token
    3  early_meas       — measure appears before any valid gate
    4  oor_qubit        — out-of-range qubit index pattern
    5  big_angle        — very large angle value (999 or 1e10)
    6  no_header        — missing OPENQASM header
    7  no_include       — missing include directive
    8  c_as_q           — classical register used as quantum target
    9  gate_ratio        — ratio of valid gate tokens
    10 seq_len_norm      — normalised sequence length
    11 dup_adj           — normalised adjacent duplicate token count
    12 n_fval            — FVAL token density
    13 n_meas_tok        — measure token density
    14 two_q_ratio       — 2-qubit gate ratio
    15 no_qreg           — missing qreg declaration
    16 no_creg           — missing creg declaration
    17 has_barrier       — barrier token present
    18 high_dup          — >5% adjacent duplicates (semantic duplicate flag)
    19 entropy           — normalised token entropy
    """
    tok_set = set(tokens)
    code    = ' '.join(tokens)
    n_toks  = max(len(tokens), 1)

    has_bad_gate   = float(bool(tok_set & BAD_GATE_SET))
    n_bad_gate     = sum(1 for t in tokens if t in BAD_GATE_SET) / n_toks
    no_measure     = float('measure' not in tok_set)

    first_gate = next((i for i, t in enumerate(tokens) if t in VALID_GATE_SET), 999)
    first_meas = next((i for i, t in enumerate(tokens) if t == 'measure'), 999)
    early_meas = float(first_meas < first_gate)

    oor_pattern  = float(any(
        t == 'ival' and i > 0 and tokens[i-1] == '[' and i+1 < len(tokens) and tokens[i+1] == ']'
        for i, t in enumerate(tokens)))
    big_angle    = float('999' in code or '1e10' in code)
    no_header    = float('openqasm' not in tok_set)
    no_include   = float('inc_qelib' not in tok_set)
    c_as_q       = float(any(tokens[i] == 'c' and i+1 < len(tokens) and tokens[i+1] == '['
                             for i in range(len(tokens)-1)))
    gate_ratio   = sum(1 for t in tokens if t in VALID_GATE_SET) / n_toks
    seq_len_norm = n_toks / 192.0
    dup_adj      = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1]) / n_toks
    n_fval       = tokens.count('fval') / n_toks
    n_meas_tok   = tokens.count('measure') / n_toks
    two_q_ratio  = sum(1 for t in tokens if t in TWO_Q_SET) / n_toks
    no_qreg      = float('qreg' not in tok_set)
    no_creg      = float('creg' not in tok_set)
    has_barrier  = float('barrier' in tok_set)
    high_dup     = float(dup_adj > 0.05)

    freq    = Counter(tokens)
    probs   = [v / n_toks for v in freq.values()]
    entropy = -sum(p * math.log2(p + 1e-9) for p in probs) / max(math.log2(n_toks + 2), 1)

    return [has_bad_gate, n_bad_gate, no_measure, early_meas,
            oor_pattern, big_angle, no_header, no_include,
            c_as_q, gate_ratio, seq_len_norm, dup_adj,
            n_fval, n_meas_tok, two_q_ratio, no_qreg, no_creg,
            has_barrier, high_dup, entropy]
