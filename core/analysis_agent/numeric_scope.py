"""Shared numeric interval grammar and source-text coverage audit.

SQL never supplies missing user constraints. A recognized interval must be
bound to a schema column, or remain an explicit unresolved obligation.
"""
import re

NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
VERSION = 3


def intervals(text, prefix=''):
    connector = r'\s*(?:(?:값|히스토그램|histogram|분포)(?:에서|은|는|이|가)?\s*)?(?:(?:is|=|:)\s*)?'
    lead = prefix + connector if prefix else r'(?<![\w.])'
    patterns = (
        (lead + '(' + NUMBER + r')\s*(?:[~∼〜–—-]|\bto\b|부터|에서)\s*('
         + NUMBER + r')(?![\d.])', 'ge', 'le'),
        (lead + r'between\s+(' + NUMBER + r')\s+and\s*(' + NUMBER + r')(?![\d.])', 'ge', 'le'),
        (lead + '(' + NUMBER + r')\s*(이상|초과)\s*(' + NUMBER + r')\s*(이하|미만)', None, None),
    )
    # Dates, quoted literals and opaque asset IDs are not filter intervals.
    excluded = [m.span() for m in re.finditer(
        r''''(?:[^']|'')*'|"(?:[^"]|"")*"|(?<!\d)\d{4}-\d{2}(?:-\d{2})?(?!\d)|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b''', text, re.I)]
    # Parenthesized schema-domain annotations in a grouping noun, such as
    # `label(column, 1~31일)별`, describe an axis, not a population filter.
    excluded += [m.span() for m in re.finditer(
        r'\([A-Za-z_][A-Za-z0-9_]*\s*,[^()\n]{1,48}\)\s*별', text)]
    seen = set()
    for pattern, low_op, high_op in patterns:
        for match in re.finditer(pattern, text, re.I):
            if any(a <= match.start() < b or a < match.end() <= b for a, b in excluded):
                continue
            span = match.span()
            if span in seen:
                continue
            seen.add(span)
            if low_op is None:
                low, high = match[1], match[3]
                lo, hi = ('ge' if match[2] == '이상' else 'gt'), ('le' if match[4] == '이하' else 'lt')
            else:
                low, high, lo, hi = match[1], match[2], low_op, high_op
            def number(value):
                value = value.replace(',', '')
                return float(value) if '.' in value else int(value)
            yield {'span':span, 'lower':number(low), 'upper':number(high),
                   'lower_op':lo, 'upper_op':hi}


def uncovered_intervals(text, bound_spans):
    """Find recognized numeric constraints silently skipped by column binding."""
    return [item for item in intervals(text)
            if not any(a <= item['span'][0] and item['span'][1] <= b for a, b in bound_spans)]
