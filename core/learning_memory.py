import sqlite3
import datetime
import json
import os
from typing import List, Dict, Any, Optional

DB_PATH = "teleai_learning.db"

def _get_connection():
    """Establish and return a connection to the SQLite learning database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the learning database schema if it does not exist."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                original_query TEXT,
                classified_intent TEXT,
                generated_sql TEXT,
                generated_python TEXT,
                user_rating INTEGER DEFAULT 0,
                execution_success BOOLEAN
            )
            """
        )
        conn.commit()

def record_interaction(
    original_query: str,
    classified_intent: str,
    generated_sql: Optional[str] = None,
    generated_python: Optional[str] = None,
    execution_success: bool = True,
    user_rating: int = 0
) -> int:
    """Record an interaction or update it. Returns the row ID."""
    with _get_connection() as conn:
        timestamp = datetime.datetime.utcnow().isoformat()
        cursor = conn.execute(
            """
            INSERT INTO interactions 
            (timestamp, original_query, classified_intent, generated_sql, generated_python, execution_success, user_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, original_query, classified_intent, generated_sql, generated_python, execution_success, user_rating)
        )
        conn.commit()
        return cursor.lastrowid

def update_rating_by_query(original_query: str, rating: int):
    """Update the rating for the most recent matching query."""
    with _get_connection() as conn:
        # Find the max ID for this query to update the most recent one
        conn.execute(
            """
            UPDATE interactions 
            SET user_rating = ? 
            WHERE id = (SELECT MAX(id) FROM interactions WHERE original_query = ?)
            """,
            (rating, original_query)
        )
        conn.commit()

def get_successful_examples(intent: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Retrieve highly rated past interactions to use as few-shot examples."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT original_query, generated_sql, generated_python 
            FROM interactions 
            WHERE classified_intent = ? AND user_rating > 0 
            ORDER BY timestamp DESC 
            LIMIT ?
            """,
            (intent, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

# Initialize the db on module import
init_db()
