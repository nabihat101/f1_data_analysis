import sqlite3
import pandas as pd

DB_FILE = "f1_data.db"


def get_connection():
    return sqlite3.connect(DB_FILE)


def get_race_data():
    connection = get_connection()
    df = pd.read_sql_query("SELECT * FROM race_results", connection)
    connection.close()
    return df


if __name__ == "__main__":
    connection = get_connection()

    query = """
    SELECT driver, AVG(finish_pos) AS avg_finish
    FROM race_results
    GROUP BY driver
    ORDER BY avg_finish
    """

    df = pd.read_sql_query(query, connection)
    connection.close()

    print(df)
    print(f"Loaded {len(df)} rows from SQL database.")
    print(df.head())