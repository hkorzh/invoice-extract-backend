import json
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gemini_extractor import (
    UploadedDocument,
    extract_invoice_live,
    extract_invoice_stream,
    extract_invoices_batch,
)


ACCEPTED = {"image/png", "image/jpeg", "image/webp", "application/pdf"}
MAX_FILES = int(os.getenv("MAX_BATCH_FILES", "5"))


# --- App ---

app = FastAPI(title="Invoice Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


def _check_type(file: UploadFile):
    if file.content_type not in ACCEPTED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {file.content_type}. Send PNG, JPEG, WebP, or PDF.",
        )


def _ndjson(updates):
    async def lines():
        async for update in updates:
            # Payloads may hold pydantic models (the Invoice); dump those as plain JSON.
            yield json.dumps(update, default=lambda model: model.model_dump()) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """Statuses: extracting, retrying, finished, error. Kept for older frontends."""
    _check_type(file)
    file_bytes = await file.read()
    return _ndjson(extract_invoice_stream(file_bytes, file.content_type))


@app.post("/extract/live")
async def extract_live(file: UploadFile = File(...)):
    """Statuses: waiting, thinking (with thought summary), writing, retrying, finished, error."""
    _check_type(file)
    file_bytes = await file.read()
    return _ndjson(extract_invoice_live(file_bytes, file.content_type))


@app.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...)):
    """Files are processed concurrently; every update carries "file" (index) and "filename".

    Per-file statuses: queued, uploading, uploaded, waiting, thinking, writing, retrying,
    finished, error. The stream ends with "done" and finished/failed counts.
    """
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=413, detail=f"Send at most {MAX_FILES} files at once.")
    for file in files:
        _check_type(file)
    documents = [
        UploadedDocument(filename=file.filename or f"file-{i}", data=await file.read(), mime_type=file.content_type)
        for i, file in enumerate(files)
    ]
    return _ndjson(extract_invoices_batch(documents))
