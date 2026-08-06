from .connection import get_db_connection

def add_message(session_id, sender, message):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO chat_history (session_id, sender, message) VALUES (?, ?, ?);",
            (session_id, sender, message)
        )
        conn.commit()
    finally:
        conn.close()

def get_history(session_id, limit=50):
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT sender, message, timestamp FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?;",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        rows.reverse()
        return [(row["sender"], row["message"]) for row in rows]
    finally:
        conn.close()

def clear_history(session_id):
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?;", (session_id,))
        conn.commit()
    finally:
        conn.close()
