# Invoice Extractor API

A small FastAPI service that turns an invoice — image or PDF — into structured,
typed JSON, using Google Gemini's multimodal model.

**Live demo:** [invoice-extract-frontend-zeta.vercel.app/](https://invoice-extract-frontend-zeta.vercel.app/)· 
**Frontend repo:** [github.com/hkorzh/invoice-extract-frontend](https://github.com/hkorzh/invoice-extract-frontend)

## What it does

Upload an invoice file and get back a predictable JSON object:

```json
{
  "vendor": "Acme Supplies Ltd",
  "date": "2026-03-14",
  "line_items": [
    { 
        "description": "Widget A", "quantity": 10, "unit_price": 4.50, "amount": 45.00, "tax_percent": 20 }
  ],
  "subtotal": 45.00,
  "total": 45.00,
  "total_tax": 49.00,
}
```

The model is constrained to a **fixed schema**, so every response has the same shape —
fields land in the same place every time, ready to pipe into a database, an accounting
system, or a spreadsheet. That predictability is the whole point: it's the difference
between an extraction tool and pasting a document into a chatbot.

## Stack

- **FastAPI** + **Uvicorn** — the API
- **Pydantic** — the schema, passed directly to the model as its response schema
- **google-genai** — Gemini client (`gemini-2.5-flash`, multimodal: reads images and PDFs)

## How it works

The uploaded file is sent to Gemini alongside the Pydantic `Invoice` schema as a
structured-output constraint. Gemini returns JSON that already matches that schema,
which FastAPI validates and serves. Fields that aren't present in the document come
back as `null` rather than invented — the model is explicitly instructed not to guess.

## Run locally

```bash
git clone [this-repo]
cd [repo-folder]
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GOO_KEY=your_key     # free key at aistudio.google.com
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/docs> and try `POST /extract`.

## Endpoints

| Method | Path            | Description                                                          |
| ------ | --------------- | -------------------------------------------------------------------- |
| `GET`  | `/`             | Health check                                                         |
| `POST` | `/extract`      | Multipart file upload (PNG, JPEG, WebP, PDF) → NDJSON stream of `extracting` / `retrying` / `finished` / `error` updates; `finished` carries the `Invoice` |
| `POST` | `/extract/live` | Same upload → NDJSON stream that also reports Gemini's progress: `waiting`, `thinking` (with a thought summary), `writing`, then `retrying` / `finished` / `error` |
| `POST` | `/extract/batch` | Multipart upload of several `files` → one NDJSON stream; files are uploaded to the Gemini Files API and extracted concurrently. Every update carries `file` (index) and `filename`; statuses are `queued`, `uploading`, `uploaded`, then the `/extract/live` ones. Ends with `done` and `finished` / `failed` counts |

## Configuration

| Variable         | Required | Description                                                        |
| ---------------- | -------- | ------------------------------------------------------------------ |
| `GOO_KEY` | yes      | Google AI Studio key. Kept server-side; never exposed to the client. |
| `GEMINI_MAX_RETRIES` | no | Attempts per Gemini call before giving up (default 3). |
| `GEMINI_MAX_CONCURRENCY` | no | Batch files processed at once, shared across requests (default 3). |
| `MAX_BATCH_FILES` | no | Most files accepted by `/extract/batch` (default 5). |

## Scope

Deliberately focused: one document type, one fixed schema, one model. The goal is a
clean, reliable demonstration of LLM-into-app integration — not a full product.

## Deployment

Runs on Render's free tier. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```