"""
LangChain + Snowflake (Cortex) Ready-to-run Notebook
- Purpose: NL -> Snowflake SQL using Snowflake Cortex (Cortex REST API / Cortex Analyst) as the generator
- Features included:
  * Auto-fetch schema (limited subset) from Snowflake
  * Prompt templates using fetched schema (few-shot placeholders)
  * SQL validator (forbidden patterns, allowed tables, auto-LIMIT)
  * Execution using a read-only Snowflake role (example role-switch snippet)
  * LangChain-style orchestration scaffolding (custom tool calling Cortex instead of OpenAI)
  * RBAC and governance notes

USAGE NOTES
1) Populate the SNOWFLAKE_ environment variables below or replace in-line (not recommended for prod).
2) The Cortex REST API usage is illustrated via the Snowflake REST endpoints. You will need Cortex enabled on your Snowflake account and an API key / credentials as described in Snowflake docs.
3) This notebook intentionally fetches schema dynamically and builds prompts - you *don't* need to paste schema here.

WARNING: This notebook is an example for interactive exploration. For production harden secrets, use ephemeral credentials, Vault, and least-privilege roles.

"""

# -------------------------------
# Imports
# -------------------------------
import os
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import requests
import pandas as pd
from snowflake import connector
from snowflake.connector import DictCursor

# -------------------------------
# Configuration (ENV preferred)
# -------------------------------
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")         # e.g. 'xyz-abc-12345.us-east-1'
SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD = os.getenv("SNOW_PWD")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE", "ANALYST_RO")   # application read-only role
SNOW_DATABASE = os.getenv("SNOW_DATABASE")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA")

# Cortex REST API config (requires Snowflake Cortex to be enabled)
CORTEX_REST_URL = os.getenv("CORTEX_REST_URL")  # e.g. https://<account>.snowflakecomputing.com/api/cortex/v1/infer
CORTEX_API_KEY = os.getenv("CORTEX_API_KEY")
CORTEX_MODEL = os.getenv("CORTEX_MODEL", "cortex-analyst-v1")

# Safety config
ALLOWED_TABLES = os.getenv("ALLOWED_TABLES")  # comma-separated table names, optional
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
INTERACTIVE_LIMIT = 100

# -------------------------------
# Helpers: Snowflake connection & schema fetch
# -------------------------------

def get_snowflake_connection(role: Optional[str]=None):
    """Return a Snowflake connection using env creds. Optionally set role."""
    conn = connector.connect(
        user=SNOW_USER,
        password=SNOW_PWD,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WAREHOUSE,
        database=SNOW_DATABASE,
        schema=SNOW_SCHEMA,
        role=role or SNOW_ROLE,
        client_session_keep_alive=False,
    )
    return conn


def fetch_tables_and_sample(limit_tables: int = 10, rows_per_table: int = 3) -> Dict[str, Any]:
    """Fetch a compact schema summary and sample rows to ground the model prompts.
    Returns a dict: { 'table_name': { 'columns':[ (name,type) ], 'sample': [row,...] } }
    """
    conn = get_snowflake_connection()
    cur = conn.cursor(DictCursor)
    try:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s LIMIT %s", (SNOW_SCHEMA, limit_tables))
        tables = [r['TABLE_NAME'] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

    results = {}
    conn = get_snowflake_connection()
    cur = conn.cursor(DictCursor)
    try:
        for t in tables:
            # fetch column names/types
            cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s", (SNOW_SCHEMA, t))
            cols = [(r['COLUMN_NAME'], r['DATA_TYPE']) for r in cur.fetchall()]
            # sample rows
            cur.execute(f"SELECT * FROM {SNOW_DATABASE}.{SNOW_SCHEMA}.\"{t}\" LIMIT {rows_per_table}")
            sample = [dict(row) for row in cur.fetchall()]
            results[t] = {"columns": cols, "sample": sample}
    finally:
        cur.close()
        conn.close()
    return results

# -------------------------------
# Prompt builder (few-shot + schema grounding)
# -------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "nl": "Monthly revenue for category 'soap' for the last 12 months",
        "sql": "SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) AS revenue FROM sales.orders WHERE product_category = 'soap' AND order_date >= DATEADD(month, -12, CURRENT_DATE()) GROUP BY 1 ORDER BY 1;"
    },
    {
        "nl": "Top 5 customers by lifetime spend",
        "sql": "SELECT customer_id, SUM(amount) AS lifetime_spend FROM sales.orders GROUP BY 1 ORDER BY lifetime_spend DESC LIMIT 5;"
    }
]

PROMPT_TEMPLATE = """
You are Snowflake Cortex Analyst. Given the following schema summary and sample rows (a compact subset), translate the user's natural-language request into a single READ-ONLY SQL query that runs on Snowflake.

Schema summary:
{schema_blurb}

Sample rows (first rows from tables):
{sample_blurb}

Constraints:
- Return a single SELECT statement only. Do not include DML (UPDATE/DELETE/INSERT), DDL, or procedural code.
- Use table names exactly as provided.
- For interactive responses include "LIMIT {interactive_limit}" if the user did not explicitly ask for an unbounded result.
- If the user asks for aggregates, use appropriate GROUP BY/time functions.
- Avoid proprietary functions not supported by Snowflake standard SQL.
- If the question requires joining multiple tables, join using plausible keys (hint: use columns named *_id where appropriate), but keep joins minimal.

Examples:
{few_shot}

User question: "{user_question}"

Return ONLY the SQL query as the response (no explanation). Make it syntactically valid for Snowflake.
"""


def build_prompt(schema_summary: Dict[str, Any], user_question: str, interactive_limit: int = INTERACTIVE_LIMIT) -> str:
    # Build schema blurb
    parts = []
    sample_parts = []
    for t, meta in schema_summary.items():
        cols = ", ".join([f"{c}:{tpe}" for c, tpe in meta['columns'][:12]])
        parts.append(f"- {t}: {cols}")
        sample_df = pd.DataFrame(meta['sample'])
        if not sample_df.empty:
            sample_parts.append(f"Table {t} sample:\n" + sample_df.head(3).to_csv(index=False))

    schema_blurb = "\n".join(parts)
    sample_blurb = "\n\n".join(sample_parts)
    few_shot = "\n".join([f"NL: {ex['nl']}\nSQL: {ex['sql']}" for ex in FEW_SHOT_EXAMPLES])

    prompt = PROMPT_TEMPLATE.format(schema_blurb=schema_blurb, sample_blurb=sample_blurb, few_shot=few_shot, user_question=user_question, interactive_limit=interactive_limit)
    return prompt

# -------------------------------
# Validator (safety + RBAC enforcement)
# -------------------------------

FORBIDDEN_PATTERNS = [r"\bUPDATE\b", r"\bDELETE\b", r"\bINSERT\b", r"\bCREATE\b", r"\bALTER\b", r"\bDROP\b", r";\s*\bCOPY\b"]

ALLOWED_TABLES_SET = set([t.strip().upper() for t in (ALLOWED_TABLES or "").split(",") if t.strip()]) if ALLOWED_TABLES else None


def validate_sql(sql: str, max_rows: int = MAX_ROWS, interactive_limit: int = INTERACTIVE_LIMIT) -> Tuple[bool, str]:
    """Validate generated SQL string. Returns (is_valid, sanitized_sql or reason)"""
    # Basic forbid checks
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, sql, flags=re.IGNORECASE):
            return False, f"Forbidden statement detected: {pat}"

    # Ensure only a single statement
    if sql.strip().count(";") > 1:
        return False, "Multiple SQL statements detected. Only one SELECT allowed."

    # Enforce SELECT only
    if not re.search(r"\bSELECT\b", sql, flags=re.IGNORECASE):
        return False, "No SELECT found; only SELECT queries are allowed."

    # Optional allowed table check
    if ALLOWED_TABLES_SET:
        found_tables = set([t.strip().upper() for t in re.findall(r"FROM\s+([\w\.\"]+)|JOIN\s+([\w\.\"]+)", sql, flags=re.IGNORECASE)])
        # Note: above regex returns tuples; flatten
        flat = set()
        for a,b in found_tables:
            if a: flat.add(a.upper())
            if b: flat.add(b.upper())
        # crude match against allowed
        for ft in flat:
            if '"' in ft:
                ft_clean = ft.replace('"','')
            else:
                ft_clean = ft.split('.')[-1]
            if ft_clean not in ALLOWED_TABLES_SET:
                return False, f"Table {ft_clean} not allowed for this app."

    # Enforce LIMIT for interactive queries if missing
    if re.search(r"\bLIMIT\b", sql, flags=re.IGNORECASE) is None:
        # append a LIMIT to the query safely
        sanitized = sql.strip().rstrip(';') + f"\nLIMIT {interactive_limit};"
        return True, sanitized

    # For safety, ensure max rows does not exceed configured MAX_ROWS
    m = re.search(r"\bLIMIT\s+(\d+)", sql, flags=re.IGNORECASE)
    if m:
        lim = int(m.group(1))
        if lim > max_rows:
            sanitized = re.sub(r"(LIMIT\s+)\d+", f"\1{max_rows}", sql, flags=re.IGNORECASE)
            return True, sanitized

    return True, sql

# -------------------------------
# Cortex REST integration (text-to-SQL)
# -------------------------------

def call_cortex_rest(prompt: str, model: Optional[str] = None, max_tokens: int = 1024) -> str:
    """Call Snowflake Cortex REST API to get model completion. This assumes you have a valid CORTEX_REST_URL and CORTEX_API_KEY.
    The exact REST payload may vary by Snowflake release; adapt to your account's API contract (see docs).
    Returns the raw text completion.
    """
    if not CORTEX_REST_URL or not CORTEX_API_KEY:
        raise ValueError("CORTEX_REST_URL and CORTEX_API_KEY must be set in ENV to call Cortex REST API.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CORTEX_API_KEY}"
    }
    payload = {
        "model": model or CORTEX_MODEL,
        "input": prompt,
        "max_tokens": max_tokens,
        # additional params like temperature could be supported
    }
    resp = requests.post(CORTEX_REST_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    # The response shape may vary - attempt common paths
    if 'output' in body:
        return body['output']
    if 'choices' in body and isinstance(body['choices'], list):
        return body['choices'][0].get('text') or body['choices'][0].get('message', {}).get('content', '')
    # fallback
    return json.dumps(body)

# -------------------------------
# Execute SQL safely
# -------------------------------

def execute_sql_and_fetch(sql: str, role: Optional[str] = None, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    conn = get_snowflake_connection(role=role)
    cur = conn.cursor()
    try:
        cur.execute(sql)
        # fetch into DataFrame
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(max_rows)
        df = pd.DataFrame(rows, columns=cols)
        return df
    finally:
        cur.close()
        conn.close()

# -------------------------------
# Orchestration: NL -> SQL -> Validate -> Execute -> Summarize
# -------------------------------

def nl_to_insight(user_question: str, schema_summary: Dict[str, Any], use_cortex: bool = True) -> Dict[str, Any]:
    # 1) Build prompt
    prompt = build_prompt(schema_summary, user_question, interactive_limit=INTERACTIVE_LIMIT)

    # 2) Call Cortex to produce SQL
    if use_cortex:
        raw_out = call_cortex_rest(prompt)
    else:
        raise NotImplementedError("Only Cortex path implemented in this notebook per user request.")

    # 3) Clean output (strip surrounding text)
    sql_candidate = raw_out.strip().strip('"')
    # sometimes models return fences; remove
    sql_candidate = re.sub(r"^```sql|```$", "", sql_candidate, flags=re.IGNORECASE).strip()

    # 4) Validate
    ok, validated_or_reason = validate_sql(sql_candidate)
    if not ok:
        return {"ok": False, "error": validated_or_reason, "raw_sql": sql_candidate}
    sql_to_run = validated_or_reason

    # 5) Execute (read-only role)
    try:
        df = execute_sql_and_fetch(sql_to_run, role=SNOW_ROLE)
    except Exception as e:
        return {"ok": False, "error": f"Execution error: {e}", "sql": sql_to_run}

    # 6) Summarize: short automatic summary
    summary = summarize_dataframe(df)
    return {"ok": True, "sql": sql_to_run, "data": df, "summary": summary}

# -------------------------------
# Simple DataFrame summarizer
# -------------------------------

def summarize_dataframe(df: pd.DataFrame, max_preview: int = 5) -> str:
    if df.empty:
        return "Query returned 0 rows."
    parts = []
    parts.append(f"Returned {len(df)} rows (showing up to {max_preview}):")
    parts.append(df.head(max_preview).to_markdown(index=False))
    # add basic numeric summaries
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        parts.append("\nNumeric summaries:")
        parts.append(df[numeric_cols].describe().to_markdown())
    return "\n\n".join(parts)

# -------------------------------
# Example run (MVP)
# -------------------------------
if __name__ == '__main__':
    print("Fetching schema summary (limit 8 tables)...")
    schema = fetch_tables_and_sample(limit_tables=8, rows_per_table=3)
    print("Schema fetched. Example tables:", list(schema.keys())[:8])

    user_q = "Show me monthly revenue for product category 'soap' in the last 12 months"
    print("Building NL->SQL flow for:", user_q)

    result = nl_to_insight(user_q, schema)
    if not result['ok']:
        print("Error:", result.get('error'))
    else:
        print("Generated SQL:\n", result['sql'])
        print("Summary:\n", result['summary'])


# -------------------------------
# RBAC & Production Notes (README-style) - Keep in notebook for ops
# -------------------------------
"""
RBAC & Governance notes (short):

1) Use least-privilege roles: create a Snowflake role for the conversational application with only the necessary SELECT privileges on required schemas/tables. Do not use ACCOUNTADMIN.

2) Use ROW ACCESS POLICIES to enforce row-level security where necessary (PII, masked data). Cortex integrates with Snowflake RBAC; generated SQL will execute under the role supplied by the connection.

3) Ephemeral credentials: for multi-tenant setups use key-pair or OAuth and generate short-lived Snowflake sessions per user, mapping app-level identity to Snowflake roles.

4) Monitoring: log every NL input, generated SQL, execution plan (EXPLAIN), and returned row count. Use Snowflake's QUERY_HISTORY view to reconcile costs.

5) Cost controls: enforce warehouse size, query timeouts, and MAX_ROWS. Consider using a separate small interactive warehouse for conversational paths.

6) Human-in-the-loop: for queries that exceed complexity thresholds (multiple large joins, window functions, lack of LIMIT, high estimated bytes scanned), present the SQL to a data steward for approval before execution.

"""
