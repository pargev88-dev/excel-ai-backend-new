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

# Folders for uploads and static charts
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# ----------------------------
# OpenAI client (insert your key)
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ← add your key here


# ----------------------------
# Main route: /process
# ----------------------------
@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("file")
    prompt = request.form.get("prompt")

    if not prompt:
        return jsonify({"result": "Missing prompt."}), 400

    df = None

    # Handle file if provided
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.xlsx")
        file.save(filename)

        try:
            df = pd.read_excel(filename)
            df = df.dropna(how='all').dropna(axis=1, how='all')
        except Exception as e:
            return jsonify({"result": f"Failed to read Excel: {str(e)}"}), 400

    # If there’s a dataframe, build preview + summary + chart
    if df is not None:
        preview_html = df.head(15).to_html(index=False)
        try:
            summary_html = df.describe(include='all').to_html(classes="summary-table")
        except Exception:
            summary_html = "<p>No numeric or categorical columns to summarize.</p>"
        chart_url = generate_chart(df)
        ai_response = ask_ai(prompt, df)
    else:
        # 🧠 Prompt-only mode (no file uploaded)
        preview_html = ""
        summary_html = ""
        chart_url = ""
        ai_response = ask_ai(prompt, pd.DataFrame())

    return jsonify({
        "result": ai_response,
        "preview_html": preview_html,
        "summary_html": summary_html,
        "chart_url": chart_url
    })



# ----------------------------
# AI logic
# ----------------------------
def ask_ai(prompt, df):
    try:
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Limit rows for performance
        if len(df) > 50:
            display_df = df.head(50)
            note = f"[Only first 50 rows shown out of {len(df)} rows]\\n\\n"
        else:
            display_df = df
            note = ""

        data_preview = display_df.to_string(index=False)
        full_prompt = f"{note}You are a data analyst. Here is the Excel data:\\n\\n{data_preview}\\n\\nThe user asked:\\n{prompt}"

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI Error: {str(e)}"


# ----------------------------
# Chart generator
# ----------------------------
def generate_chart(df):
    try:
        plt.clf()
        sns.set(style="whitegrid")

        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] >= 2:
            x, y = numeric.columns[:2]
            sns.scatterplot(data=df, x=x, y=y)
            plt.title(f"{x} vs {y}")
        elif numeric.shape[1] == 1:
            col = numeric.columns[0]
            df[col].plot(kind="hist", bins=20)
            plt.title(f"Histogram of {col}")
        else:
            return None

        chart_path = os.path.join(app.config['STATIC_FOLDER'], "chart.png")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        return "/static/chart.png"

    except Exception:
        return None


# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
