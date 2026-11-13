"""Create the database and tables: python -m database.init_db"""

import re
import sys
from pathlib import Path

from config import Config
from database.db import connect


def main():
    if not Config.DB_USER:
        sys.exit(
            "DB_USER is empty. Copy .env.example to .env and fill in your MySQL credentials."
        )

    if not re.fullmatch(r"[A-Za-z0-9_]+", Config.DB_NAME):
        sys.exit("DB_NAME may only contain letters, digits and underscores.")

    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")

    # Remove SQL comments, including inline comments
    sql = re.sub(r"--.*", "", sql)

    # Split SQL statements by semicolon
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    conn = connect(Config, with_database=False)

    try:
        cur = conn.cursor()

        # Create database
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{Config.DB_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )

        # Select database
        cur.execute(f"USE `{Config.DB_NAME}`")

        # Create tables
        for statement in statements:
            if statement.upper().startswith(("CREATE DATABASE", "USE ")):
                continue

            cur.execute(statement)

        conn.commit()

        print(
            f"Database '{Config.DB_NAME}' is ready "
            "(tables: vehicles, predictions)."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()