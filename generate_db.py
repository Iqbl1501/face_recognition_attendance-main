import os
import sqlite3


if __name__ == '__main__':
    files_to_delete = ["database.db", "database.db-journal",
                       "model.keras", "ResultsMap.pkl"]

    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)

    con = sqlite3.connect(os.environ["DATABASE_PATH"])
    cur = con.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS users(
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
                )
                """)

    cur.execute("""CREATE TABLE IF NOT EXISTS dataset(
                    dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                """)

    cur.execute("""CREATE TABLE IF NOT EXISTS attendance(
                    user_id INTEGER,                   
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, timestamp),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                )
                """)
