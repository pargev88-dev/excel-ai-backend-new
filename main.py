from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import os
import uuid

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STATIC_FOLDER"] = STATIC_FOLDER

# ----------------------------
# OpenAI client (env-based key)
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SMART_MODEL = "gpt-4o-mini"


# ----------------------------
# Helpers
# ----------------------------
def dataframe_preview_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Return a small HTML table for preview."""
    try:
        preview = df.head(max_rows)
        return preview.to_html(
            border=1,
            index=False,
            justify="center",
            classes="preview-table",
        )
    except Exception:
        return "<p>Could not render preview table.</p>"


def dataframe_summary_html(df: pd.DataFrame) -> str:
    """Quick numeric + categorical summary in HTML."""
    try:
        parts = []
        parts.append(f"<p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        parts.append("<ul>")
        parts.append(f"<li><strong>Numeric columns:</strong> {', '.join(numeric_cols) or 'None'}</li>")
        parts.append(f"<li><strong>Text/category columns:</strong> {', '.join(cat_cols) or 'None'}</li>")
        parts.append(f"<li><strong>Date/time columns:</strong> {', '.join(date_cols) or 'None'}</li>")
        parts.append("</ul>")

        # Basic numeric stats
        if numeric_cols:
            desc = df[numeric_cols].describe().round(2)
            parts.append("<p><strong>Numeric summary (describe):</strong></p>")
            parts.append(desc.to_html(border=1, classes="summary-table"))

        return "".join(parts)
    except Exception:
        return "<p>No summary available.</p>"


def build_generic_chart(df: pd.DataFrame) -> str | None:
    """
    Build a simple generic chart:
    - If there is a date column + numeric → line chart
    - Else if category + numeric → bar chart
    - Else if numeric only → histogram of first numeric
    Returns relative URL like '/static/chart_xxx.png'
    """
    try:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if not numeric_cols:
            return None

        plt.clf()
        sns.set_style("whitegrid")

        # Case 1: Date + numeric → line
        if date_cols:
            x_col = date_cols[0]
            y_col = numeric_cols[0]
            tmp = df[[x_col, y_col]].dropna().sort_values(x_col).head(500)
            if tmp.empty:
                return None
            plt.figure(figsize=(6, 3))
            plt.plot(tmp[x_col], tmp[y_col], marker="o")
            plt.title(f"{y_col} over {x_col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # Case 2: Category + numeric → bar of top 15
        elif cat_cols:
            cat = cat_cols[0]
            y_col = numeric_cols[0]
            tmp = df[[cat, y_col]].dropna()
            grouped = tmp.groupby(cat)[y_col].sum().sort_values(ascending=False).head(15)
            if grouped.empty:
                return None
            plt.figure(figsize=(6, 3))
            grouped.plot(kind="bar")
            plt.ylabel(y_col)
            plt.title(f"{y_col} by {cat}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        # Case 3: Only numeric → histogram
        else:
            y_col = numeric_cols[0]
            tmp = df[y_col].dropna().head(1000)
            if tmp.empty:
                return None
            plt.figure(figsize=(6, 3))
            plt.hist(tmp, bins=20)
            plt.xlabel(y_col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {y_col}")
            plt.tight_layout()

        filename = f"chart_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(STATIC_FOLDER, filename)
        plt.savefig(chart_path)
        plt.close()

        return f"/static/{filename}"
    except Exception:
        return None


def build_smart_analysis(df: pd.DataFrame, user_prompt: str) -> str:
    """Call OpenAI with dataset context + user prompt."""
    # Metadata for the prompt
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    meta = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_cols": numeric_cols,
        "category_cols": cat_cols,
        "date_cols": date_cols,
    }

    # Small sample of the data for context (safe token size)
    sample = df.head(30)
    sample_csv = sample.to_csv(index=False)

    system_prompt = """
You are an expert Excel and data analyst embedded in a web app called 'Excel AI Assistant'.
Your job is to read a small dataset and the user's request, then respond with a clear,
business-friendly analysis.

Always:
- Start with a 1–2 sentence plain-language summary of what you see in the data.
- Then give 3–6 bullet points of key insights (totals, averages, trends, anomalies).
- If the user is vague (e.g., 'sum', 'summary'), propose helpful options and explain what
  you can calculate for them.
- Suggest at least one good chart type and one useful pivot table layout using real column names.

DO NOT output code. Focus on explanation and specific numbers you can read from the data.
"""

    user_message = f"""
User prompt:
{user_prompt}

Dataset metadata (JSON):
{meta}

First rows of the dataset (CSV):
{sample_csv}
"""

    try:
        resp = client.chat.completions.create(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
            max_tokens=700,
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"AI analysis failed: {e}"


# ----------------------------
# Routes
# ----------------------------
@app.route("/process", methods=["POST"])
def process():
    prompt = (request.form.get("prompt") or "").strip()
    uploaded = request.files.get("file")

    if not prompt and not uploaded:
        return jsonify({"result": "Please provide a prompt or upload a file."}), 400

    df = None
    preview_html = ""
    summary_html = ""
    chart_url = None

    # 1) Load Excel if provided
    if uploaded and uploaded.filename:
        try:
            filename = f"{uuid.uuid4().hex}_{uploaded.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded.save(file_path)

            # Try reading with openpyxl
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            return jsonify(
                {"result": f"Failed to read Excel file: {e}", "preview_html": ""}
            ), 400

    # If no file, still allow AI to answer prompt
    if df is None:
        ai_text = build_smart_analysis(pd.DataFrame(), prompt or "General question")
        return jsonify(
            {
                "result": ai_text,
                "preview_html": "",
                "summary_html": "",
                "chart_url": None,
            }
        )

    # 2) Build preview + summary + chart
    preview_html = dataframe_preview_html(df)
    summary_html = dataframe_summary_html(df)
    chart_url = build_generic_chart(df)

    # 3) Smart AI analysis
    ai_text = build_smart_analysis(df, prompt or "Provide a useful summary of this dataset.")

    return jsonify(
        {
            "result": ai_text,
            "preview_html": preview_html,
            "summary_html": summary_html,
            "chart_url": chart_url,
        }
    )


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
