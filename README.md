# Excel AI Backend (Flask)

This is the backend API for the Excel AI Assistant. It accepts an uploaded `.xlsx`
file and a natural-language prompt, processes the data with pandas/Matplotlib,
and uses the OpenAI API to generate an AI response plus optional chart/image output.

## Running locally

```bash
pip install -r requirements.txt

# set your OpenAI key in the environment
export OPENAI_API_KEY="sk-..."
python main.py
```

The server will start on `http://127.0.0.1:3000`.

## Environment variables

- `OPENAI_API_KEY` – your OpenAI API key (required)

## Deploying to Railway

1. Push this folder to a GitHub repository.
2. Create a new Railway service from the repo.
3. In Railway → Variables, add `OPENAI_API_KEY`.
4. Set the start command to:

   ```bash
   python main.py
   ```

   (or `gunicorn 'main:app'` if you prefer a WSGI server).

5. Deploy and use the Railway URL in your frontend instead of the Replit URL.
