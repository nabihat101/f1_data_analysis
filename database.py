import sqlite3
import pandas as pd

CSV_FILE = "f1_race_data.csv"
DB_FILE = "f1_data.db"

df = pd.read_csv(CSV_FILE)

connection = sqlite3.connect(DB_FILE)

df.to_sql("race_results", connection, if_exists="replace", index=False)

connection.close()

print("Database created successfully.")
print(f"Rows added: {len(df)}")