"""Deterministic MC parsing with explicit-answer boundaries and Markdown support."""
import re

PARSER_POLICY = 'deterministic-no-random-fallback-v2'


def explicit_answer(raw, choices, allow_answer_marker=False):
    """Return a decisive parse (including refusal), or None for legacy fallback.

    A bare letter must end or be followed by punctuation; this prevents `is`,
    `Cannot`, `Approximately`, and `I cannot ...` from becoming option letters.
    We do not promote speculative `perhaps the answer is ...` to a final answer.
    """
    clean = re.sub(r'[*`_]', '', raw)
    final = r'final\s+(?:answer|decision)\b(?:\s+is\b)?\s*[:：]?'
    marker = rf'\b(?:{final}|answer\s*[:：])' if allow_answer_marker else rf'\b{final}'
    valid = ''.join(re.escape(c) for c in choices)
    refusal = re.compile(
        r"(?:cannot\b|can\s+not\b|can't\b|unable\b|undetermined\b|unknown\b|"
        r"insufficient\b|none\s+of\s+the\s+above\b|"
        r"(?:I\s+)?(?:cannot\b|can't\b|do\s+not\s+know\b|don't\s+know\b))", re.I)
    answers = []
    last = None
    for match in re.finditer(marker, clean, re.I):
        tail = clean[match.end():].lstrip()
        if refusal.match(tail):
            # If the refusal text is itself an exact provided option, resolve it.
            line = tail.splitlines()[0].strip().rstrip('.').casefold()
            matching = [c for c, text in choices.items()
                        if text and text.strip().rstrip('.').casefold() == line]
            last = (matching[0], 'explicit_answer') if len(matching) == 1 else (None, 'unparsed_refusal')
            continue
        token = re.match(
            rf'(?:\(\s*([{valid}])\s*\)|([{valid}])(?=[ \t]*(?:$|[\n\r.。,;:!？?—–-])))',
            tail, re.I)
        if token:
            answer = (token.group(1) or token.group(2)).upper()
            answers.append(answer)
            mode = 'explicit_final' if match.group().lower().startswith('final') else 'explicit_answer'
            last = answer, mode
    if last is not None:
        answer, mode = last
        return answer, {'mode': mode, 'candidates': answers}
    return None


def fallback_parse(raw, choices):
    """Existing deterministic MMMU fallback, without the defective final regex."""
    valid = ''.join(re.escape(c) for c in choices)
    compact = raw.strip().strip('*`_ ')
    exact = re.fullmatch(rf'\(?([{valid}])\)?[.。]?', compact, flags=re.I)
    if exact:
        answer = exact.group(1).upper()
        return answer, {'mode': 'exact_letter', 'candidates': [answer]}
    response = raw
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '
    candidates = [c for c in choices if f'({c})' in response]
    mode = 'bracket'
    if not candidates:
        candidates = [c for c in choices if f' {c} ' in response]
        mode = 'letter'
    if not candidates and len(response.split()) > 5:
        candidates = [c for c, answer in choices.items() if answer and answer.lower() in response.lower()]
        mode = 'option_text'
    if not candidates:
        return None, {'mode': 'unparsed', 'candidates': []}
    def position(c):
        if mode == 'bracket':
            return response.rfind(f'({c})')
        if mode == 'letter':
            return response.rfind(f' {c} ')
        return response.lower().rfind(choices[c].lower())
    return max(candidates, key=position), {'mode': mode, 'candidates': candidates}


def parse_mc(raw, choices, allow_answer_marker=False):
    return explicit_answer(raw, choices, allow_answer_marker) or fallback_parse(raw, choices)


def parse_pro(raw, choices, finish_reason=None):
    answer, info = parse_mc(raw, choices, allow_answer_marker=True)
    if finish_reason == 'length' and info['mode'] not in ('explicit_answer', 'explicit_final', 'exact_letter'):
        return None, {'mode': 'truncated_unparsed', 'candidates': info['candidates']}
    return answer, info
