"""
Standalone sanity-check script for the QMS complaints database.

What it does, step by step:
  1. Loads DB connection details from the .env file in the project root.
  2. Connects to Postgres.
  3. Inserts one test complaint row.
  4. Reads that same row back and prints it.
  5. Closes the connection.

This is NOT the application — just a manual way to confirm the schema
and connection actually work before building anything on top of them.
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Step 1: load DB_* values from .env into environment variables.
load_dotenv()

DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"],
}


def main():
    # Step 2: connect.
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Step 3: insert one test complaint row.
                cur.execute(
                    """
                    INSERT INTO complaints (
                        status, complaint_reference, product_type,
                        complaint_source, customer_name,
                        product_name, product_strength_grade, batch_lot_number,
                        affected_quantity, manufacturing_date, expiry_date,
                        originating_site_block, impacted_npm,
                        complaint_category, complaint_description,
                        severity_suggested, suggested_next_action, initial_risk_assessment
                    )
                    VALUES (
                        %(status)s, %(complaint_reference)s, %(product_type)s,
                        %(complaint_source)s, %(customer_name)s,
                        %(product_name)s, %(product_strength_grade)s, %(batch_lot_number)s,
                        %(affected_quantity)s, %(manufacturing_date)s, %(expiry_date)s,
                        %(originating_site_block)s, %(impacted_npm)s,
                        %(complaint_category)s, %(complaint_description)s,
                        %(severity_suggested)s, %(suggested_next_action)s, %(initial_risk_assessment)s
                    )
                    RETURNING id;
                    """,
                    {
                        "status": "draft",
                        "complaint_reference": "CC-2026-00001",
                        "product_type": "API",
                        "complaint_source": "Customer email",
                        "customer_name": "Test Pharma Distributors Ltd.",
                        "product_name": "Paracetamol API",
                        "product_strength_grade": "USP Grade",
                        "batch_lot_number": "BATCH-2026-0042",
                        "affected_quantity": "25 kg",
                        "manufacturing_date": "2026-01-10",
                        "expiry_date": "2028-01-10",
                        "originating_site_block": "Site A - Block 2",
                        "impacted_npm": "None reported",
                        "complaint_category": "Discoloration",
                        "complaint_description": "Customer reports slight yellowing of powder in 3 of 20 drums received.",
                        "severity_suggested": "Major",
                        "suggested_next_action": "Initiate batch investigation and retain sample testing.",
                        "initial_risk_assessment": "Moderate risk pending lab confirmation of discoloration cause.",
                    },
                )
                new_id = cur.fetchone()["id"]
                print(f"Inserted test complaint with id = {new_id}")

                # Step 4: read that same row back.
                cur.execute("SELECT * FROM complaints WHERE id = %s;", (new_id,))
                row = cur.fetchone()

        print("\nRow read back from the database:")
        for key, value in row.items():
            print(f"  {key}: {value}")
    finally:
        # Step 5: always close the connection.
        conn.close()


if __name__ == "__main__":
    main()
