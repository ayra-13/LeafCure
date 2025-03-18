import sqlite3

def init_db():
    """Initialize SQLite database and create 'chats' table if it does not exist."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_input TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT (datetime('now', 'localtime'))
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database
init_db()
