import sqlite3
import pandas as pd

db_path = r"C:\Users\DELL\Documents\AI\IVF-AI-Sperm-Selection\sperm_analysis.db"
data_path = r"C:\Users\DELL\Documents\AI\IVF-AI-Sperm-Selection\data\sperm_features_extended.csv"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Connected to database")

df = pd.read_csv(data_path)

df.to_sql("sperm_data", conn, if_exists="replace", index=False)

# Create prediction history table
cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    head_area REAL,
    head_perimeter REAL,
    tail_length REAL,
    motility_score REAL,
    prediction TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

print("Tables ready")

sample = pd.read_sql("SELECT * FROM sperm_data LIMIT 5", conn)

print("\nSample Data:")
print(sample)

conn.close()

print("Database setup complete")