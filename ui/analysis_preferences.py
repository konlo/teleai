"""Persist explicit model choices per conversation, without storing credentials."""
from contextlib import closing
from pathlib import Path
import sqlite3

PROVIDERS = {'ollama': '로컬 Ollama', 'databricks': 'Databricks 모델 · 토큰 사용량 과금'}


def _connect(root):
    db = sqlite3.connect(Path(root) / 'conversations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS conversation_preferences '
               '(conversation_id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)')
    return db


def saved_provider(root, conversation):
    with closing(_connect(root)) as db, db:
        row = db.execute('SELECT model_provider FROM conversation_preferences WHERE conversation_id=?',
                         (conversation,)).fetchone()
    return row[0] if row and row[0] in PROVIDERS else 'ollama'


def save_provider(root, conversation, provider):
    if not conversation or provider not in PROVIDERS:
        raise ValueError('Unknown conversation or analysis model provider')
    with closing(_connect(root)) as db, db:
        db.execute('INSERT INTO conversation_preferences VALUES (?,?) '
                   'ON CONFLICT(conversation_id) DO UPDATE SET model_provider=excluded.model_provider',
                   (conversation, provider))


def render_provider_selector(root, conversation):
    import streamlit as st
    key = 'v1_model_provider'
    if (st.session_state.get('v1_provider_conversation') != conversation
            or st.session_state.get(key) not in PROVIDERS):
        st.session_state[key] = saved_provider(root, conversation)
        st.session_state['v1_provider_conversation'] = conversation

    def changed():
        # Only an explicit widget change writes the preference; another open tab
        # rendering old state must not overwrite a more recent saved choice.
        save_provider(root, conversation, st.session_state[key])

    provider = st.selectbox('분석 모델', list(PROVIDERS), format_func=PROVIDERS.__getitem__,
                           key=key, on_change=changed)
    if provider not in PROVIDERS:
        raise ValueError('Unknown analysis model provider')
    return provider
