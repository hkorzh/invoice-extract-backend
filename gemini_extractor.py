"""
Google Gemini integration — all the model-specific logic lives here.

The API key lives on the backend, never in the browser. main.py talks to
this module through extract_invoice_stream(); it never touches the genai
client directly, so swapping models or providers is a one-file change.
"""

import os
from typing import Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from schemas import Invoice


# --- Gemini client. The key lives HERE, on the backend, never in the browser. ---

# Reads a local .env in development; in production the real env vars win.
load_dotenv()

API_KEY = os.getenv("GOO_KEY")
if not API_KEY:
    raise RuntimeError("Set GOO_KEY — get a free key at aistudio.google.com")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"  # free tier, multimodal: reads invoice images + PDFs

# Gemini occasionally errors on one call and succeeds on the very next one;
# retrying a fixed number of times before giving up smooths over that flakiness.
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))


def _call_gemini(file_bytes: bytes, mime_type: str) -> Invoice:
    """Single attempt at the Gemini call. Raises RuntimeError on failure."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                "Extract all the fields that can be related to invoice. If a field is missing from the "
                "document, leave it null. Do not invent values.",
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
            ),
        )
    except errors.APIError as e:
        # APIError stringifies to the full JSON payload; .message is the clean part.
        raise RuntimeError(f"Gemini call failed: {e.message}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

    # response.parsed is already a typed Invoice instance.
    invoice = response.parsed
    if invoice is None:
        raise RuntimeError("Model returned no parseable data.")
    return invoice


def extract_invoice_stream(file_bytes: bytes, mime_type: str) -> Iterator[dict]:
    """Drive the Gemini call with retries, yielding progress updates.

    Each yielded dict has a "status" of "extracting", "retrying", "finished",
    or "error", and a "payload" — the extracted Invoice on success, or
    {"error": message} once retries are exhausted.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        yield {"status": "extracting", "payload": None}
        try:
            invoice = _call_gemini(file_bytes, mime_type)
        except RuntimeError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                yield {"status": "retrying", "payload": None}
            continue
        yield {"status": "finished", "payload": invoice.model_dump()}
        return

    yield {"status": "error", "payload": {"error": last_error}}
