-- Schema for the QMS complaints table.
-- Run once against a fresh database to create the table.

CREATE TABLE IF NOT EXISTS complaints (
    id BIGSERIAL PRIMARY KEY,

    -- Workflow state
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'logged')),
    complaint_reference TEXT UNIQUE,

    -- Drives which field labels the frontend shows later
    product_type TEXT NOT NULL CHECK (product_type IN ('API', 'FDF')),

    -- Origin & Customer Details
    complaint_source TEXT,
    customer_name TEXT,

    -- Product & Batch Identification
    product_name TEXT,
    product_strength_grade TEXT,
    batch_lot_number TEXT,
    affected_quantity TEXT,
    manufacturing_date DATE,
    expiry_date DATE,

    -- Facility & Material Impact
    originating_site_block TEXT,
    impacted_npm TEXT,

    -- Defect Analysis
    complaint_category TEXT,
    complaint_description TEXT,

    -- AI Risk Assessment
    severity_suggested TEXT,
    suggested_next_action TEXT,
    initial_risk_assessment TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Keep updated_at accurate on every UPDATE.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_complaints_updated_at ON complaints;
CREATE TRIGGER trg_complaints_updated_at
    BEFORE UPDATE ON complaints
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();
