"""
Google Gemini integration — all the model-specific logic lives here.

The API key lives on the backend, never in the browser. main.py talks to
this module through extract_invoice_stream() and extract_invoice_live(); it
never touches the genai client directly, so swapping models or providers is
a one-file change.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

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


PROMPT = (
    "Extract all the fields that can be related to invoice. If a field is missing from the "
    "document, leave it null. Do not invent values."
)


def _call_gemini(file_bytes: bytes, mime_type: str) -> Invoice:
    """Single attempt at the Gemini call. Raises RuntimeError on failure."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), PROMPT],
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


class GeminiStage(str, Enum):
    WAITING = "waiting"    # request sent, nothing back yet
    THINKING = "thinking"  # model is reasoning; `thought` holds the latest summary
    WRITING = "writing"    # model is emitting the JSON answer
    DONE = "done"          # answer complete; `invoice` is filled


@dataclass
class GeminiProgress:
    stage: GeminiStage
    thought: Optional[str] = None
    invoice: Optional[Invoice] = None


def _iterate_gemini(file_bytes: bytes, mime_type: str) -> Iterator[GeminiProgress]:
    """Single streamed attempt at the Gemini call. Raises RuntimeError on failure.

    THINKING is yielded once per thought summary, WRITING once when the answer
    starts, and DONE last, carrying the parsed Invoice.
    """
    yield GeminiProgress(GeminiStage.WAITING)

    answer = ""
    try:
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        for chunk in stream:
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for part in chunk.candidates[0].content.parts or []:
                if not part.text:
                    continue
                if part.thought:
                    yield GeminiProgress(GeminiStage.THINKING, thought=part.text)
                else:
                    if not answer:
                        yield GeminiProgress(GeminiStage.WRITING)
                    answer += part.text
    except errors.APIError as e:
        # APIError stringifies to the full JSON payload; .message is the clean part.
        raise RuntimeError(f"Gemini call failed: {e.message}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e

    # Chunks are fragments of one JSON document, so validate only the whole thing.
    try:
        invoice = Invoice.model_validate_json(answer)
    except ValueError as e:
        raise RuntimeError("Model returned no parseable data.") from e
    yield GeminiProgress(GeminiStage.DONE, invoice=invoice)


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


def extract_invoice_live(file_bytes: bytes, mime_type: str) -> Iterator[dict]:
    """Like extract_invoice_stream(), but streams Gemini's own progress.

    Each yielded dict has a "status" of "waiting", "thinking", "writing",
    "retrying", "finished", or "error", and a "payload" — {"thought": text}
    while thinking, the extracted Invoice on success, or {"error": message}
    once retries are exhausted.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            for progress in _iterate_gemini(file_bytes, mime_type):
                if progress.stage is GeminiStage.DONE:
                    yield {"status": "finished", "payload": progress.invoice.model_dump()}
                    return
                payload = {"thought": progress.thought} if progress.thought else None
                yield {"status": progress.stage.value, "payload": payload}
        except RuntimeError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                yield {"status": "retrying", "payload": None}

    yield {"status": "error", "payload": {"error": last_error}}
