"""Separate observed qualified table identities from natural-language columns."""
import re


def mask_source_mentions(text, sources):
    """Blank known qualified identities, preserving offsets and quoted values.

    A catalog or schema component may also be a real column name. Its presence
    inside the table address is not a request to analyse that column. Separate
    occurrences and predicate literals retain their original meaning.
    """
    from core.analysis_catalog import _source_key
    protected = [match.span() for match in re.finditer(
        r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", text)]
    result = list(text)
    for source in sorted(set(sources), key=len, reverse=True):
        parts = _source_key(source).split('.')
        if not 2 <= len(parts) <= 3 or any(not re.fullmatch(r'[\w-]+', p) for p in parts):
            continue
        forms = [r'\s*\.\s*'.join(r'`?' + re.escape(p) + r'`?' for p in parts),
                 '`' + re.escape('.'.join(parts)) + '`']
        pattern = r'(?<![A-Za-z0-9_])(?:' + '|'.join(forms) + r')(?![A-Za-z0-9_])'
        for match in re.finditer(pattern, text, re.I):
            if any(match.start() < end and match.end() > start for start, end in protected):
                continue
            result[match.start():match.end()] = ' ' * (match.end() - match.start())
    return ''.join(result)
