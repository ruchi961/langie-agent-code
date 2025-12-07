"""
ATLAS MCP Server (single file)

- Paid/closed services -> return dummy responses
- Open-source -> implemented (Tesseract OCR, mock ERP, vendor DB)
- Endpoints expected by LangGraph agent

Run:
uvicorn atlas_mcp_server:app --reload --port 8002
"""

import base64
import io
import json
import random
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

# try to import pytesseract; if not available, fallback
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# Reuse atlas_client MockERP if available; otherwise small internal implementation
try:
    from atlas_client import MockERPClient, atlas_fetch
    _HAS_ATLAS_CLIENT = True
except Exception:
    _HAS_ATLAS_CLIENT = False

    class MockERPClient:
        def __init__(self):
            self.pos = {
                "PO-001": {
                    "vendor_name": "ACME Corp",
                    "amount": 550.0,
                    "currency": "USD",
                    "line_items": [
                        {"desc": "Bolts", "qty": 10, "unit_price": 5, "total": 50},
                        {"desc": "Nuts", "qty": 50, "unit_price": 10, "total": 500},
                    ],
                    "status": "OPEN",
                }
            }

            self.grns = {"GRN-001": {"po_id": "PO-001", "received_qty": 60, "received_date": datetime.utcnow().isoformat()}}
            self.history = [
                {"invoice_id": "INV-HIST-001", "amount": 550, "currency": "USD", "date": "2024-02-11"},
                {"invoice_id": "INV-HIST-002", "amount": 200, "currency": "USD", "date": "2024-01-29"},
            ]

        def fetch_pos(self, vendor_name):
            return [{"po_id": k, **v} for k, v in self.pos.items() if v["vendor_name"].lower() == vendor_name.lower()]

        def fetch_grns(self, po_ids):
            return [ {"grn_id": gid, **g} for gid, g in self.grns.items() if g["po_id"] in po_ids ]

        def fetch_history(self, vendor_name):
            return self.history

    async def atlas_fetch(provider: str, vendor_name: str):
        # simple wrapper using MockERPClient
        erp = MockERPClient()
        pos = erp.fetch_pos(vendor_name)
        grns = erp.fetch_grns([p["po_id"] for p in pos])
        history = erp.fetch_history(vendor_name)
        return {"provider_used": provider, "matched_pos": pos, "matched_grns": grns, "history": history, "retrieved_at": datetime.utcnow().isoformat()}

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="ATLAS MCP Server (mock/open-source mix)")

# -------------------------
# Models
# -------------------------
class OCRExtractRequest(BaseModel):
    # Accept either list of base64 strings or filenames
    attachments: Optional[List[str]] = None  # each item: base64 data OR filepath (for local testing)
    provider: Optional[str] = "tesseract"  # provider hint: tesseract/aws_textract/google_vision


class ParseRequest(BaseModel):
    text: str


class EnrichRequest(BaseModel):
    vendor_name: str
    provider: Optional[str] = "vendor_db"  # clearbit / people_data_labs / vendor_db


class FetchPoRequest(BaseModel):
    provider: Optional[str] = "mock_erp"
    vendor_name: str


class FetchGrnRequest(BaseModel):
    provider: Optional[str] = "mock_erp"
    po_ids: List[str]


class FetchHistoryRequest(BaseModel):
    provider: Optional[str] = "mock_erp"
    vendor_name: str


class HitlDecisionRequest(BaseModel):
    checkpoint_state: Dict[str, Any]
    decision: str  # ACCEPT / REJECT


class PostToERPRequest(BaseModel):
    amount: float
    vendor_name: str
    details: Optional[Dict[str, Any]] = None


class SchedulePaymentRequest(BaseModel):
    amount: float
    vendor_name: str
    days_delay: Optional[int] = 7

class ComputeFlagsRequest(BaseModel):
    amount: float
    risk_score: float = 0.5   # default
    
class NotifyRequest(BaseModel):
    vendor_name: str
    invoice_id: str
    message: Optional[str] = None
    
class VendorInput(BaseModel):
    vendor_name: str

# -------------------------
# Utilities
# -------------------------
def _now_iso():
    return datetime.utcnow().isoformat()

def _is_paid_provider(name: Optional[str]) -> bool:
    """Return True if the provider is commercial/paid (we will return dummy)."""
    if not name:
        return False
    n = name.lower()
    paid_indicators = ["sap", "netsuite", "clearbit", "people_data_labs", "aws", "google_cloud", "gcp", "azure"]
    return any(p in n for p in paid_indicators)

def _decode_base64_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data))
    except Exception as e:
        raise ValueError("invalid base64 image") from e

def _load_image_from_path(path: str) -> Image.Image:
    try:
        return Image.open(path)
    except FileNotFoundError:
        raise
    except UnidentifiedImageError:
        raise ValueError("file is not a valid image")


# -------------------------
# 1. UNDERSTAND: OCR + Parse
# -------------------------
@app.post("/ocr_extract")
async def ocr_extract(req: OCRExtractRequest):
    """
    If provider is paid (aws/gvision) -> return dummy structure.
    If provider is 'tesseract' or unspecified -> try to run pytesseract (open-source).
    Accept attachments list: base64 strings or file paths (local testing).
    """
    provider = (req.provider or "tesseract").lower()
    attachments = req.attachments or []

    # Paid providers -> dummy
    if _is_paid_provider(provider):
        return {"provider": provider, "ocr_texts": [{"filename": f"dummy-{i}", "text": "[PAID OCR DUMMY]"} for i in range(len(attachments))], "timestamp": _now_iso()}

    # Open-source path: use pytesseract if available, otherwise fallback deterministic
    results = []
    for i, att in enumerate(attachments):
        text = ""
        filename = f"attachment-{i}"
        # decide if att is base64 or path
        if att.startswith("data:") and "base64," in att:
            # data URI
            b64 = att.split("base64,")[1]
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
            except Exception:
                img = None
        else:
            # maybe raw base64 or path
            # try detect base64: if long and has no slashes
            if re.match(r"^[A-Za-z0-9+/=\s]+$", att) and len(att) > 200:
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(att)))
                except Exception:
                    img = None
            else:
                # treat as file path
                try:
                    img = Image.open(att)
                except Exception:
                    img = None

        if img and TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(img)
            except Exception:
                text = "[ocr_error]"
        elif img and not TESSERACT_AVAILABLE:
            # tesseract binary not installed — return deterministic fallback
            text = f"[OCR_FALLBACK_TEXT_FROM_IMAGE_{i}]"
        else:
            # no image — fallback to deterministic text
            text = f"[OCR_PLACEHOLDER_TEXT_{i}]"

        results.append({"filename": filename, "text": text})

    return {"provider": provider, "ocr_texts": results, "timestamp": _now_iso()}


@app.post("/parse_line_items")
async def parse_line_items(req: ParseRequest):
    """
    Deterministic parsing from invoice text.
    Very simple heuristics:
     - lines split by newline or '|' containing description | qty | price
     - fallback: single FULL_INVOICE item
    """
    text = req.text or ""
    items = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            try:
                qty = float(parts[1])
            except Exception:
                qty = 1.0
            try:
                price = float(parts[2])
            except Exception:
                price = 0.0
            items.append({"description": parts[0], "quantity": qty, "unit_price": price, "total": qty * price})
    if not items:
        # try to extract simple patterns like "ItemName qty price" using regex (e.g., "Widget 2 19.99")
        for line in text.splitlines():
            m = re.search(r"([A-Za-z\s\-]+)\s+(\d+)\s+([\d\.]+)", line)
            if m:
                desc = m.group(1).strip()
                qty = float(m.group(2))
                price = float(m.group(3))
                items.append({"description": desc, "quantity": qty, "unit_price": price, "total": qty * price})
    if not items:
        items.append({"description": "FULL_INVOICE", "quantity": 1, "unit_price": 0.0, "total": 0.0})
    return {"parsed_items": items, "detected_currency": "USD", "parsed_at": _now_iso()}


# -------------------------
# 2. PREPARE: normalize, enrich, flags
# -------------------------
@app.post("/normalize_vendor")
async def normalize_vendor(payload: VendorInput):
    # deterministic normalization: remove punctuation, uppercase
    #print(vendor_name)
    v = payload.vendor_name.strip()
    v = "".join(ch for ch in v if ch.isalnum() or ch.isspace())
    return {"normalized_name": v.upper(), "confidence": 0.98}


@app.post("/enrich_vendor")
async def enrich_vendor(req: EnrichRequest):
    """
    If provider is paid -> return dummy response (Clearbit, PDL)
    If provider is 'vendor_db' or 'mock' -> return real/deterministic enrichment
    """
    provider = (req.provider or "vendor_db").lower()
    vendor = req.vendor_name or "UNKNOWN"

    if _is_paid_provider(provider) or provider in ("clearbit", "people_data_labs", "pdl"):
        # Paid service -> dummy
        return {
            "provider": provider,
            "vendor": vendor,
            "tax_id": f"TAX-DUMMY-{random.randint(1000,9999)}",
            "pan": f"PAN-DUMMY-{random.randint(1000,9999)}",
            "credit_score": random.randint(300,800),
            "risk_score": round(random.random(), 2),
            "note": "paid-provider-dummy"
        }

    # Open-source / local vendor DB implementation
    # Example static vendor DB
    vendor_db = {
        "ACME CORPORATION": {"tax_id": "TAX-ACME-001", "pan": "PANACME001", "credit_score": 720, "risk_score": 0.15},
        "GLOBEX": {"tax_id": "TAX-GLOBEX-01", "pan": "PANGLOBEX01", "credit_score": 650, "risk_score": 0.45},
    }
    norm = vendor.strip().upper()
    profile = vendor_db.get(norm)
    if profile:
        return {"provider": provider, "vendor": vendor, **profile, "note": "vendor_db"}
    # fallback deterministic enrichment
    return {"provider": provider, "vendor": vendor, "tax_id": f"TAX-{norm[:6]}", "pan": f"PAN{norm[:5]}", "credit_score": 600, "risk_score": 0.5, "note": "fallback"}

@app.post("/compute_flags")
async def compute_flags(payload: ComputeFlagsRequest):

    amount = payload.amount
    risk_score = payload.risk_score

    flags = {
        "missing_info": [],
        "risk_score": risk_score
    }

    # Add flags
    if risk_score > 0.7:
        flags["missing_info"].append("high_risk")

    if amount is None or amount == 0:
        flags["missing_info"].append("missing_amount")

    if amount and amount > 10000:
        flags["risk_flag"] = "high_value"

    return {
        "flags": flags,
        "computed_at": _now_iso()
    }

# -------------------------
# 3. RETRIEVE (ERP): fetch_po, fetch_grn, fetch_history
# -------------------------
@app.post("/fetch_po")
async def fetch_po(req: FetchPoRequest):
    provider = (req.provider or "mock_erp").lower()
    vendor = req.vendor_name
    # paid providers -> dummy
    if _is_paid_provider(provider) or provider in ("sap_sandbox", "netsuite"):
        return {"provider": provider, "pos": [{"po_id": f"{provider.upper()}-PO-DUMMY", "vendor": vendor, "amount": 99999.0}], "fetched_at": _now_iso()}
    # open-source mock_erp -> use atlas_fetch or MockERPClient
    if _HAS_ATLAS_CLIENT:
        result = await atlas_fetch(provider, vendor)
        return {"provider": provider, "pos": result["matched_pos"], "fetched_at": _now_iso()}
    else:
        erp = MockERPClient()
        return {"provider": provider, "pos": erp.fetch_pos(vendor), "fetched_at": _now_iso()}


@app.post("/fetch_grn")
async def fetch_grn(req: FetchGrnRequest):
    provider = (req.provider or "mock_erp").lower()
    po_ids = req.po_ids or []
    if _is_paid_provider(provider) or provider in ("sap_sandbox", "netsuite"):
        return {"provider": provider, "grns": [{"grn_id": f"{provider.upper()}-GRN-DUMMY", "po_id": po_ids[0] if po_ids else None}], "fetched_at": _now_iso()}
    if _HAS_ATLAS_CLIENT:
        result = await atlas_fetch(provider, "")  # atlas_fetch returns pos/grns based on vendor; best-effort
        return {"provider": provider, "grns": result["matched_grns"], "fetched_at": _now_iso()}
    else:
        erp = MockERPClient()
        return {"provider": provider, "grns": erp.fetch_grns(po_ids), "fetched_at": _now_iso()}


@app.post("/fetch_history")
async def fetch_history(req: FetchHistoryRequest):
    provider = (req.provider or "mock_erp").lower()
    vendor = req.vendor_name
    if _is_paid_provider(provider) or provider in ("sap_sandbox", "netsuite"):
        return {"provider": provider, "history": [{"invoice_id": f"{provider.upper()}-INV-DUMMY", "amount": 0, "date": _now_iso()}], "fetched_at": _now_iso()}
    if _HAS_ATLAS_CLIENT:
        result = await atlas_fetch(provider, vendor)
        return {"provider": provider, "history": result["history"], "fetched_at": _now_iso()}
    else:
        erp = MockERPClient()
        return {"provider": provider, "history": erp.fetch_history(vendor), "fetched_at": _now_iso()}


# -------------------------
# 4. HITL_DECISION
# -------------------------
@app.post("/accept_or_reject_invoice")
async def accept_or_reject_invoice(req: HitlDecisionRequest):
    decision = (req.decision or "REJECT").upper()
    if decision == "ACCEPT":
        return {"status": "ACCEPTED", "resume_token": f"resume-{random.randint(1000,9999)}", "next_stage": "RECONCILE", "accepted_at": _now_iso()}
    else:
        return {"status": "REJECTED", "next_stage": None, "message": "Marked for manual handling", "rejected_at": _now_iso()}


# -------------------------
# 5. APPROVE
# -------------------------
class ApprovalRequest(BaseModel):
    amount: float


@app.post("/apply_invoice_approval_policy")
async def apply_invoice_approval_policy(req: ApprovalRequest):
    if req.amount <= 1000:
        return {
            "approval_status": "AUTO_APPROVED",
            "approved_at": _now_iso()
        }
    else:
        return {
            "approval_status": "ESCALATED",
            "approver": "FINANCE_LEAD",
            "escalated_at": _now_iso()
        }

# -------------------------
# 6. POSTING
# -------------------------
@app.post("/post_to_erp")
async def post_to_erp(req: PostToERPRequest):
    provider = req.details.get("provider") if req.details else "mock_erp"
    if _is_paid_provider(provider) or provider in ("sap_sandbox", "netsuite"):
        return {"posted": True, "erp_txn_id": f"{provider.upper()}-ERP-DUMMY-{random.randint(1000,9999)}", "posted_at": _now_iso()}
    # open-source mock -> return deterministic txn
    erp = MockERPClient()
    txn_id = f"MOCKERP-{random.randint(1000,9999)}"
    return {"posted": True, "erp_txn_id": txn_id, "posted_at": _now_iso()}


@app.post("/schedule_payment")
async def schedule_payment(req: SchedulePaymentRequest):
    provider = "mock_payments"
    if _is_paid_provider(provider):
        return {"scheduled": True, "scheduled_payment_id": f"{provider.upper()}-PAY-DUMMY-{random.randint(1000,9999)}", "scheduled_at": _now_iso()}
    # deterministic
    return {"scheduled": True, "scheduled_payment_id": f"PAY-{random.randint(1000,9999)}", "scheduled_at": _now_iso(), "delay_days": req.days_delay}


# -------------------------
# 7. NOTIFY
# -------------------------
@app.post("/notify_vendor")
async def notify_vendor(req: NotifyRequest):
    # open-source: pretend to send via local SMTP; paid providers would be SendGrid etc (dummy)
    provider = "local_mail"
    return {"status": "SENT", "provider": provider, "vendor": req.vendor_name, "invoice_id": req.invoice_id, "sent_at": _now_iso()}


@app.post("/notify_finance_team")
async def notify_finance_team(req: NotifyRequest):
    # pretend to post to slack/webhook
    return {"status": "SENT", "channel": "finance", "invoice_id": req.invoice_id, "sent_at": _now_iso()}


# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok", "tesseract": TESSERACT_AVAILABLE}


# -------------------------
# Run with uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("atlas_mcp_server:app", host="0.0.0.0", port=8002, reload=True)
