import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gemini_extractor import extract_invoice_live, extract_invoice_stream


ACCEPTED = {"image/png", "image/jpeg", "image/webp", "application/pdf"}


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
    return StreamingResponse(
        (json.dumps(update) + "\n" for update in updates),
        media_type="application/x-ndjson",
    )


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
