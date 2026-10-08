"""Recognize a distribution objective, independently of schema-list words."""
import re


def requested(text):
    text = str(text or '')
    return bool(re.search(r'분포|(?<![A-Za-z0-9_])distribution(?![A-Za-z0-9_])', text, re.I)
        and re.search(r'보여|그려|그리|시각화|그래프|차트|(?<![A-Za-z0-9_])(?:show|plot|draw|display|visuali[sz]e)(?![A-Za-z0-9_])', text, re.I)
        and not re.search(r'설명만|표로|표만|explain\s+only|(?:as\s+(?:a\s+)?|in\s+(?:a\s+)?)table|table\s+only', text, re.I))
