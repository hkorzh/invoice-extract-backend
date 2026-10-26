"""
Google Gemini integration — all the model-specific logic lives here.

The API key lives on the backend, never in the browser. main.py talks to
this module through extract_invoice_stream(), extract_invoice_live() and
extract_invoices_batch(); it never touches the genai client directly, so
swapping models or providers is a one-file change.

Everything goes through the async client (client.aio), so several files can
be processed concurrently without blocking the event loop.
"""

import asyncio
import io
import os
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

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

# How many files of a batch are processed at once. Shared across requests so
# that parallel uploads together stay under the free tier's requests-per-minute.
MAX_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "3"))
_slots = asyncio.Semaphore(MAX_CONCURRENCY)


PROMPT = (
    "Extract all the fields that can be related to invoice. If a field is missing from the "
    "document, leave it null. Do not invent values."
)


def _gemini_error(e: Exception) -> RuntimeError:
    # APIError stringifies to the full JSON payload; .message is the clean part.
    message = e.message if isinstance(e, errors.APIError) else str(e)
    return RuntimeError(f"Gemini call failed: {message}")


def _inline_part(file_bytes: bytes, mime_type: str) -> types.Part:
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)


async def _call_gemini(document: types.Part) -> Invoice:
    """Single attempt at the Gemini call. Raises RuntimeError on failure."""
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=[document, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
            ),
        )
    except Exception as e:
        raise _gemini_error(e) from e

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


async def _iterate_gemini(document: types.Part) -> AsyncIterator[GeminiProgress]:
    """Single streamed attempt at the Gemini call. Raises RuntimeError on failure.

    THINKING is yielded once per thought summary, WRITING once when the answer
    starts, and DONE last, carrying the parsed Invoice.
    """
    yield GeminiProgress(GeminiStage.WAITING)

    answer = ""
    try:
        stream = await client.aio.models.generate_content_stream(
            model=MODEL,
            contents=[document, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Invoice,
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        async for chunk in stream:
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
    except Exception as e:
        raise _gemini_error(e) from e

    # Chunks are fragments of one JSON document, so validate only the whole thing.
    try:
        invoice = Invoice.model_validate_json(answer)
    except ValueError as e:
        raise RuntimeError("Model returned no parseable data.") from e
    yield GeminiProgress(GeminiStage.DONE, invoice=invoice)


async def _extract_live(document: types.Part) -> AsyncIterator[dict]:
    """Drive the streamed Gemini call with retries; see extract_invoice_live()."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async for progress in _iterate_gemini(document):
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


async def extract_invoice_stream(file_bytes: bytes, mime_type: str) -> AsyncIterator[dict]:
    """Drive the Gemini call with retries, yielding progress updates.

    Each yielded dict has a "status" of "extracting", "retrying", "finished",
    or "error", and a "payload" — the extracted Invoice on success, or
    {"error": message} once retries are exhausted.
    """
    document = _inline_part(file_bytes, mime_type)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        yield {"status": "extracting", "payload": None}
        try:
            invoice = await _call_gemini(document)
        except RuntimeError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                yield {"status": "retrying", "payload": None}
            continue
        yield {"status": "finished", "payload": invoice.model_dump()}
        return

    yield {"status": "error", "payload": {"error": last_error}}


async def extract_invoice_live(file_bytes: bytes, mime_type: str) -> AsyncIterator[dict]:
    """Like extract_invoice_stream(), but streams Gemini's own progress.

    Each yielded dict has a "status" of "waiting", "thinking", "writing",
    "retrying", "finished", or "error", and a "payload" — {"thought": text}
    while thinking, the extracted Invoice on success, or {"error": message}
    once retries are exhausted.
    """
    async for update in _extract_live(_inline_part(file_bytes, mime_type)):
        yield update


# --- Batch: several files, each through its own upload -> extract pipeline. ---


@dataclass
class UploadedDocument:
    filename: str
    data: bytes
    mime_type: str


async def _upload(document: UploadedDocument) -> types.File:
    """Single attempt at uploading to the Gemini Files API. Raises RuntimeError on failure."""
    try:
        return await client.aio.files.upload(
            file=io.BytesIO(document.data),
            config=types.UploadFileConfig(
                mime_type=document.mime_type,
                display_name=document.filename,
            ),
        )
    except Exception as e:
        raise _gemini_error(e) from e


async def _upload_with_retries(document: UploadedDocument) -> AsyncIterator[dict]:
    """Yields "uploading" / "retrying", then "uploaded" carrying the File, or "error"."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        yield {"status": "uploading", "payload": None}
        try:
            uploaded = await _upload(document)
        except RuntimeError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                yield {"status": "retrying", "payload": None}
            continue
        yield {"status": "uploaded", "payload": None, "file": uploaded}
        return

    yield {"status": "error", "payload": {"error": last_error}}


async def _process_document(document: UploadedDocument) -> AsyncIterator[dict]:
    """One file's pipeline: upload, then extract. Yields untagged status updates."""
    uploaded = None
    try:
        async for update in _upload_with_retries(document):
            uploaded = update.pop("file", None)
            yield update
        if uploaded is None:
            return

        file_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)
        async for update in _extract_live(file_part):
            yield update

        # Next step (categorizing) will go here, fed by the "finished" invoice.
    finally:
        # Uploaded files would expire after 48h anyway; don't leave them around.
        if uploaded is not None:
            try:
                await client.aio.files.delete(name=uploaded.name)
            except Exception:
                pass


async def _run_document(index: int, document: UploadedDocument, queue: asyncio.Queue) -> bool:
    """Run one file's pipeline under a concurrency slot, tagging every update.

    Returns True when the file ended "finished".
    """
    tag = {"file": index, "filename": document.filename}
    last_status = None
    # Reported before waiting for a slot, so files behind others aren't silent.
    await queue.put({**tag, "status": "queued", "payload": None})
    async with _slots:
        try:
            async for update in _process_document(document):
                last_status = update["status"]
                await queue.put({**tag, **update})
        except Exception as e:
            # Anything unexpected still has to close this file's stream.
            last_status = "error"
            await queue.put({**tag, "status": "error", "payload": {"error": str(e)}})
    return last_status == "finished"


async def extract_invoices_batch(documents: list[UploadedDocument]) -> AsyncIterator[dict]:
    """Process several files concurrently, yielding their updates as they happen.

    Every per-file update carries "file" (its index in the request) and
    "filename", plus the same "status"/"payload" as extract_invoice_live(),
    with "queued", "uploading" and "uploaded" coming before the extraction
    statuses.
    The last update is {"status": "done", "payload": {"finished": n, "failed": m}}.
    """
    queue: asyncio.Queue = asyncio.Queue()
    tasks = [
        asyncio.create_task(_run_document(index, document, queue))
        for index, document in enumerate(documents)
    ]
    all_done = asyncio.gather(*tasks)
    # Wakes the consumer loop below once the last task has queued its final update.
    all_done.add_done_callback(lambda _: queue.put_nowait(None))

    try:
        while (update := await queue.get()) is not None:
            yield update
        results = all_done.result()
    finally:
        # The client went away mid-stream: stop the remaining work.
        for task in tasks:
            task.cancel()

    finished = sum(results)
    yield {"status": "done", "payload": {"finished": finished, "failed": len(results) - finished}}
