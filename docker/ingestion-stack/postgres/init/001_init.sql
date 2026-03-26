CREATE TABLE IF NOT EXISTS structured_tables (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT,
    source_file TEXT,
    table_name TEXT,
    row_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_structured_tables_source_id
    ON structured_tables (source_id);

CREATE INDEX IF NOT EXISTS idx_structured_tables_row_data_gin
    ON structured_tables
    USING GIN (row_data);
