```python
# LangChain + Snowflake Cortex Conversational Platform (NL → SQL)
# Ready-to-run notebook with ChatSnowflakeCortex integration + Schema-aware SQL Generation + Result Summarizer Agent

import os
from langchain_community.chatmodels import ChatSnowflakeCortex
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import snowflake.connector
import pandas as pd

# -----------------------------
# Snowflake Connection Settings
# -----------------------------
sf_connection = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA"),
    role=os.getenv("SNOWFLAKE_ROLE") # ensure read-only role for safety
)

cursor = sf_connection.cursor()

# -----------------------------
# Fetch Schema Structure
# -----------------------------
def get_schema_structure():
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
        ORDER BY TABLE_NAME, ORDINAL_POSITION;
    """)
    schema_info = cursor.fetchall()
    schema_dict = {}
    for table, column, dtype in schema_info:
        if table not in schema_dict:
            schema_dict[table] = []
        schema_dict[table].append(f"{column} ({dtype})")
    return schema_dict

schema_structure = get_schema_structure()
schema_text = "\n".join([
    f"Table: {table}\n  Columns: {', '.join(columns)}"
    for table, columns in schema_structure.items()
])

# -----------------------------
# LangChain Chat Model - Snowflake Cortex
# -----------------------------
llm = ChatSnowflakeCortex(
    model="cortex-analyst",   # Snowflake Cortex Analyst for NL→SQL
    temperature=0.0,
    snowflake_connection=sf_connection
)

# -----------------------------
# Prompt Templates for NL → SQL
# -----------------------------
examples = [
    {"input": "Show me top 5 customers by revenue.", "sql": "SELECT CUSTOMER_ID, SUM(REVENUE) AS TOTAL_REVENUE FROM SALES GROUP BY CUSTOMER_ID ORDER BY TOTAL_REVENUE DESC LIMIT 5;"},
    {"input": "What is the average order value this month?", "sql": "SELECT AVG(ORDER_VALUE) FROM ORDERS WHERE ORDER_DATE >= DATE_TRUNC('month', CURRENT_DATE);"},
]

example_template = "User: {input}\nSQL: {sql}"
example_prompt = PromptTemplate(
    input_variables=["input", "sql"],
    template=example_template
)

schema_aware_suffix = (
    "The database schema is as follows:\n{schema}\n"
    "Generate a syntactically correct SQL query for the following user request."
    "User: {query}\nSQL:"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix=schema_aware_suffix,
    input_variables=["query", "schema"]
)

sql_chain = LLMChain(llm=llm, prompt=few_shot_prompt, output_key="sql_query")

# -----------------------------
# SQL Validator
# -----------------------------
ALLOWED_COMMANDS = ["SELECT"]
FORBIDDEN_KEYWORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]

def validate_sql(sql: str):
    sql_upper = sql.upper()
    if not any(sql_upper.startswith(cmd) for cmd in ALLOWED_COMMANDS):
        raise ValueError("Only SELECT queries are allowed.")
    for kw in FORBIDDEN_KEYWORDS:
        if kw in sql_upper:
            raise ValueError(f"Forbidden keyword detected: {kw}")
    if "LIMIT" not in sql_upper:
        sql += " LIMIT 50"  # enforce safe limit
    return sql

# -----------------------------
# Conversational Memory
# -----------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Result Summarizer Agent
# -----------------------------
summarizer_prompt = PromptTemplate(
    input_variables=["dataframe", "user_query"],
    template=(
        "You are a data analyst. The user asked: {user_query}.\n"
        "Here is the SQL result dataframe:\n{dataframe}\n"
        "Summarize the insights in plain English."
    )
)

summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt, output_key="summary")

# -----------------------------
# Query Execution + Summarization
# -----------------------------
def run_query(user_query: str):
    # Step 1: Generate SQL from NL using schema context
    sql_query = sql_chain.run({"query": user_query, "schema": schema_text})
    print("Generated SQL:\n", sql_query)

    # Step 2: Validate SQL
    validated_sql = validate_sql(sql_query)

    # Step 3: Execute SQL
    cursor.execute(validated_sql)
    df = cursor.fetch_pandas_all()

    # Step 4: Summarize Result
    summary = summarizer_chain.run({"dataframe": df.to_string(), "user_query": user_query})

    return df, summary

# -----------------------------
# Example Run
# -----------------------------
user_input = "List top 3 products by sales quantity this year."
df, summary = run_query(user_input)

print("\nRaw DataFrame Result:\n", df)
print("\nInsight Summary:\n", summary)

# -----------------------------
# RBAC Notes
# -----------------------------
# - Assign a READ-ONLY role for this connection (avoid DML privileges).
# - Use Snowflake RBAC to limit schema/table access.
# - Enable query history logging for audit.
# - Apply row-level / column-level security if sensitive data exists.
```
