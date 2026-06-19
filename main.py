import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gemini_extractor import extract_invoice_stream


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


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if file.content_type not in ACCEPTED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {file.content_type}. Send PNG, JPEG, WebP, or PDF.",
        )

    file_bytes = await file.read()

    def stream():
        for update in extract_invoice_stream(file_bytes, file.content_type):
            yield json.dumps(update) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
