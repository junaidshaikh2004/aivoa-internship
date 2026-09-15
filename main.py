"""
FastAPI backend wiring together:
  - the Postgres `complaints` table (db/schema.sql, step 1)
  - the LangGraph AI pipeline (complaint_ai_pipeline.py, step 2)

Four endpoints, no more:
  POST  /complaints/extract-text     raw text -> AI extraction -> draft row
  POST  /complaints/extract-pdf      PDF file  -> AI extraction -> draft row
  PATCH /complaints/{id}/correct     correction message -> updates only the changed columns
  POST  /complaints/{id}/commit      marks a draft row as logged (final)

No React yet - this is tested with curl. No ORM - plain psycopg2, same style
as scripts/test_db.py from step 1.
"""

import io
import json
import os
from datetime import datetime

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from complaint_ai_pipeline import extract_text_from_pdf, run_correction, run_pipeline

load_dotenv()

app = FastAPI()

# Allow the future frontend (Vite or Create React App - not picked yet) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB connection dependency: opens one connection per request, closes it
# after the request is done (success or error). See explanation in chat.
# ---------------------------------------------------------------------------
def get_db():
    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
    )
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str


class CorrectionInput(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# DB query logic: shared by both "extract" endpoints, and by the
# correction/commit endpoints. This is the "talk to Postgres" part.
# ---------------------------------------------------------------------------
COMPLAINT_FIELDS = [
    "product_type", "complaint_source", "customer_name", "product_name",
    "product_strength_grade", "batch_lot_number", "affected_quantity",
    "manufacturing_date", "expiry_date", "originating_site_block",
    "impacted_npm", "complaint_category", "complaint_description",
    "severity_suggested", "suggested_next_action", "initial_risk_assessment",
]


def parse_partial_date(value):
    """The AI pipeline outputs dates as free text: a full 'YYYY-MM-DD', a
    partial 'YYYY-MM' when only month/year is known, or 'Not specified'.
    The DB column is a real DATE, so anything we can't confidently parse
    into one is stored as NULL rather than rejected by Postgres.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y-%m").date().isoformat()
    except ValueError:
        return None


def generate_complaint_reference(cur) -> str:
    year = datetime.now().year
    cur.execute(
        "SELECT COUNT(*) AS count FROM complaints WHERE EXTRACT(YEAR FROM created_at) = %s",
        (year,),
    )
    sequence = cur.fetchone()["count"] + 1
    return f"CC-{year}-{sequence:05d}"


def insert_draft_complaint(cur, extracted: dict) -> dict:
    values = {field: extracted.get(field) for field in COMPLAINT_FIELDS}
    values["manufacturing_date"] = parse_partial_date(values["manufacturing_date"])
    values["expiry_date"] = parse_partial_date(values["expiry_date"])
    values["status"] = "draft"
    values["complaint_reference"] = generate_complaint_reference(cur)

    columns = ["status", "complaint_reference"] + COMPLAINT_FIELDS
    placeholders = ", ".join(f"%({col})s" for col in columns)
    cur.execute(
        f"INSERT INTO complaints ({', '.join(columns)}) VALUES ({placeholders}) RETURNING *",
        values,
    )
    return cur.fetchone()


def to_json_safe(row: dict) -> dict:
    """DB rows contain date/datetime objects that plain json.dumps() (used
    inside complaint_ai_pipeline.py) can't serialize. Round-tripping through
    json with default=str turns them into plain strings first."""
    return json.loads(json.dumps(row, default=str))


# ---------------------------------------------------------------------------
# Endpoints (route/request handling - each one is just: get input, call the
# AI pipeline, call the DB helpers, return the result)
# ---------------------------------------------------------------------------
@app.post("/complaints/extract-text")
def extract_text(payload: TextInput, conn=Depends(get_db)):
    extracted = run_pipeline(payload.text)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        row = insert_draft_complaint(cur, extracted)
        conn.commit()
    return row


@app.post("/complaints/extract-pdf")
async def extract_pdf(file: UploadFile = File(...), conn=Depends(get_db)):
    pdf_bytes = await file.read()
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))  # pypdf accepts a stream, not just a path
    extracted = run_pipeline(text)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        row = insert_draft_complaint(cur, extracted)
        conn.commit()
    return row


@app.patch("/complaints/{complaint_id}/correct")
def correct_complaint(complaint_id: int, payload: CorrectionInput, conn=Depends(get_db)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM complaints WHERE id = %s", (complaint_id,))
        current_row = cur.fetchone()
        if current_row is None:
            raise HTTPException(status_code=404, detail="Complaint not found")

        changed_fields = run_correction(to_json_safe(current_row), payload.message)

        if changed_fields:
            if "manufacturing_date" in changed_fields:
                changed_fields["manufacturing_date"] = parse_partial_date(changed_fields["manufacturing_date"])
            if "expiry_date" in changed_fields:
                changed_fields["expiry_date"] = parse_partial_date(changed_fields["expiry_date"])

            set_clause = ", ".join(f"{col} = %({col})s" for col in changed_fields)
            cur.execute(
                f"UPDATE complaints SET {set_clause} WHERE id = %(id)s RETURNING *",
                {**changed_fields, "id": complaint_id},
            )
            updated_row = cur.fetchone()
            conn.commit()
        else:
            updated_row = current_row

    return {"changed_fields": changed_fields, "complaint": updated_row}


@app.post("/complaints/{complaint_id}/commit")
def commit_complaint(complaint_id: int, conn=Depends(get_db)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "UPDATE complaints SET status = 'logged' WHERE id = %s RETURNING *",
            (complaint_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Complaint not found")
        conn.commit()
    return row
