
# common_mcp_server.py
import uuid
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from difflib import SequenceMatcher

# -----------------------
# CONFIG
# -----------------------
DB_PATH = "common_mcp.db"
REVIEW_BASE_URL = "https://review.example.com/review"  # replace with real reviewer service

# -----------------------
# UTIL: DB
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            raw_payload TEXT,
            current_stage TEXT,
            state_json TEXT,
            hitl_review_url TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS human_review_queue (
            id TEXT PRIMARY KEY,
            workflow_id TEXT,
            reason TEXT,
            created_at TEXT,
            review_url TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def db_upsert_workflow(workflow_id: str, payload: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("SELECT id FROM workflows WHERE id = ?", (workflow_id,))
    exists = cur.fetchone()
    if exists:
        cur.execute("""
            UPDATE workflows
            SET updated_at = ?, raw_payload = ?, state_json = ?
            WHERE id = ?
        """, (now, json.dumps(payload.get("raw_payload")), json.dumps(payload.get("state")), workflow_id))
    else:
        cur.execute("""
            INSERT INTO workflows (id, status, created_at, updated_at, raw_payload, current_stage, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (workflow_id, payload.get("status", "running"), now, now, json.dumps(payload.get("raw_payload")), payload.get("current_stage"), json.dumps(payload.get("state"))))
    conn.commit()
    conn.close()

def db_get_workflow(workflow_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, status, created_at, updated_at, raw_payload, current_stage, state_json, hitl_review_url FROM workflows WHERE id = ?", (workflow_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "status": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "raw_payload": json.loads(row[4]) if row[4] else None,
        "current_stage": row[5],
        "state": json.loads(row[6]) if row[6] else None,
        "hitl_review_url": row[7]
    }

def db_mark_hitl(workflow_id: str, review_url: str, reason: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("UPDATE workflows SET status = ?, updated_at = ?, hitl_review_url = ?, current_stage = ? WHERE id = ?", ("paused", now, review_url, "CHECKPOINT_HITL", workflow_id))
    entry_id = str(uuid.uuid4())
    cur.execute("INSERT INTO human_review_queue (id, workflow_id, reason, created_at, review_url) VALUES (?, ?, ?, ?, ?)",
                (entry_id, workflow_id, reason, now, review_url))
    conn.commit()
    conn.close()

def db_update_stage_and_state(workflow_id: str, stage: str, state: Dict[str, Any], status: str = "running"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute("UPDATE workflows SET current_stage = ?, state_json = ?, updated_at = ?, status = ? WHERE id = ?",
                (stage, json.dumps(state), now, status, workflow_id))
    conn.commit()
    conn.close()

# -----------------------
# Pydantic models
# -----------------------
class Attachment(BaseModel):
    filename: str
    content_type: Optional[str]
    content: Optional[str] = None  # base64 or text

class InvoicePayload(BaseModel):
    invoice_id: Optional[str] = None
    vendor: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = "USD"
    raw_text: Optional[str] = None
    attachments: Optional[List[Attachment]] = []

class WorkflowState(BaseModel):
    line_items: Optional[List[Dict[str, Any]]] = []
    normalized_vendor: Optional[str] = None
    flags: Optional[Dict[str, bool]] = {}
    match_score: Optional[float] = None
    matched_po_id: Optional[str] = None
    accounting_entries: Optional[List[Dict[str, Any]]] = []
    notes: Optional[List[str]] = []

class IntakeResponse(BaseModel):
    workflow_id: str
    status: str
    message: str

class StageResponse(BaseModel):
    workflow_id: str
    stage: str
    status: str
    state: Dict[str, Any]

# -----------------------
# Deterministic helpers
# -----------------------
def deterministic_parse_line_items(payload: InvoicePayload) -> List[Dict[str, Any]]:
    """
    Very simple deterministic parser:
    - If raw_text contains lines like: ITEM | qty | price
    - or tries to use attachments filenames to infer items
    This is intentionally minimal and deterministic.
    """
    items = []
    text = (payload.raw_text or "").strip()
    if not text:
        # fallback: use vendor as single line item (toy example)
        items.append({"description": "UNKNOWN ITEM", "quantity": 1, "unit_price": payload.total_amount or 0.0})
        return items

    # naive line-based extraction
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            # treat as description | qty | price
            try:
                qty = float(parts[1])
            except Exception:
                qty = 1.0
            try:
                price = float(parts[2])
            except Exception:
                price = 0.0
            items.append({"description": parts[0], "quantity": qty, "unit_price": price})
    # if no structured lines found, make one summary item
    if not items:
        items.append({"description": "FULL_INVOICE", "quantity": 1, "unit_price": payload.total_amount or 0.0})
    return items

def deterministic_normalize_vendor(vendor_name: Optional[str]) -> str:
    if not vendor_name:
        return "UNKNOWN_VENDOR"
    v = vendor_name.strip().lower()
    # remove punctuation-like characters
    v = "".join(ch for ch in v if ch.isalnum() or ch.isspace())
    # deterministic mapping rules (expandable)
    mappings = {
        "acme corp": "ACME CORPORATION",
        "acme corporation": "ACME CORPORATION",
        "acme": "ACME CORPORATION",
        "globex": "GLOBEX CORPORATION",
    }
    for k, mapped in mappings.items():
        if v.startswith(k):
            return mapped
    return v.upper()

def compute_flags_from_state(state: WorkflowState) -> Dict[str, bool]:
    flags = {}
    total_from_items = sum((li.get("quantity", 1) * li.get("unit_price", 0) for li in (state.line_items or [])))
    # deterministic tolerance
    invoice_total = None
    # try to parse from possible place in state.notes
    # (in real system we'd use payload.total_amount)
    if state.line_items:
        invoice_total = total_from_items

    flags["total_matches_line_items"] = abs((invoice_total or 0) - total_from_items) < 0.01
    flags["has_po_reference"] = any(("po" in (str(x).lower())) for x in (state.notes or []))
    flags["vendor_normalized"] = bool(state.normalized_vendor)
    return flags

def deterministic_match_score(invoice_state: WorkflowState, candidate_po: Dict[str, Any]) -> float:
    """
    Very small deterministic scoring:
    - vendor name similarity
    - total amount closeness
    - presence of PO id
    Returns score in [0,1]
    """
    score = 0.0
    # vendor similarity
    vendor = (invoice_state.normalized_vendor or "").lower()
    po_vendor = (candidate_po.get("vendor") or "").lower()
    if vendor and po_vendor:
        ratio = SequenceMatcher(None, vendor, po_vendor).ratio()
        score += 0.5 * ratio

    # amount closeness
    invoice_total = sum((li.get("quantity", 1) * li.get("unit_price", 0) for li in (invoice_state.line_items or [])))
    po_amount = float(candidate_po.get("amount", 0) or 0)
    if po_amount > 0:
        closeness = max(0.0, 1 - (abs(invoice_total - po_amount) / max(invoice_total, po_amount, 1.0)))
        score += 0.5 * closeness

    # ensure in [0,1]
    return max(0.0, min(1.0, float(score)))

def generate_review_url(workflow_id: str) -> str:
    token = str(uuid.uuid4())
    return f"{REVIEW_BASE_URL}/{workflow_id}?token={token}"

# -----------------------
# FastAPI app + endpoints
# -----------------------
app = FastAPI(title="COMMON MCP Server")

@app.on_event("startup")
def startup():
    init_db()

@app.post("/intake", response_model=IntakeResponse)
def intake_accept_invoice(payload: InvoicePayload):
    """
    INTAKE 📥 – accept_invoice_payload (Deterministic)
    - Validate schema (Pydantic does that)
    - Persist raw invoice
    - Server: COMMON
    """
    workflow_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    initial_state = WorkflowState().dict()
    db_payload = {
        "status": "running",
        "raw_payload": payload.dict(),
        "current_stage": "INTAKE",
        "state": initial_state,
        "created_at": now
    }
    db_upsert_workflow(workflow_id, db_payload)
    return IntakeResponse(workflow_id=workflow_id, status="running", message="Invoice accepted and persisted.")

@app.post("/understand/{workflow_id}", response_model=StageResponse)
def understand_parse_line_items(workflow_id: str):
    """
    UNDERSTAND 🧠 – parse_line_items (Deterministic)
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    raw = wf.get("raw_payload") or {}
    invoice = InvoicePayload(**raw)
    items = deterministic_parse_line_items(invoice)
    print("items are", items)
    state = WorkflowState(**(wf.get("state") or {}))
    state.line_items = items
    # add note about parse
    state.notes = (state.notes or []) + [f"parsed {len(items)} line_items deterministically"]
    db_update_stage_and_state(workflow_id, "UNDERSTAND", state.dict())
    return StageResponse(workflow_id=workflow_id, stage="UNDERSTAND", status="running", state=state.dict())

@app.post("/prepare/{workflow_id}", response_model=StageResponse)
def prepare_normalize_and_flags(workflow_id: str):
    """
    PREPARE 🛠️ – normalize_vendor, compute_flags (Deterministic)
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    raw = wf.get("raw_payload") or {}
    invoice = InvoicePayload(**raw)
    state = WorkflowState(**(wf.get("state") or {}))
    # normalize vendor deterministically
    state.normalized_vendor = deterministic_normalize_vendor(invoice.vendor)
    # compute flags
    flags = compute_flags_from_state(state)
    state.flags = flags
    state.notes = (state.notes or []) + [f"vendor normalized -> {state.normalized_vendor}"]
    db_update_stage_and_state(workflow_id, "PREPARE", state.dict())
    return StageResponse(workflow_id=workflow_id, stage="PREPARE", status="running", state=state.dict())

@app.post("/match_two_way/{workflow_id}", response_model=StageResponse)
def match_two_way_compute_score(workflow_id: str, threshold: float = 0.85):
    """
    MATCH_TWO_WAY ⚖️ – compute_match_score (Deterministic)
    - If score < threshold => checkpoint HITL
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    state = WorkflowState(**(wf.get("state") or {}))
    # Candidate PO(s) - in real system you'd query PO datastore. Here we use a deterministic example.
    candidate_pos = [
        {"po_id": "PO-123", "vendor": state.normalized_vendor, "amount": sum((li.get("quantity",1)*li.get("unit_price",0) for li in (state.line_items or [])))},
        {"po_id": "PO-999", "vendor": "DIFFERENT VENDOR", "amount": 9999.0}
    ]
    best_score = 0.0
    best_po = None
    for po in candidate_pos:
        s = deterministic_match_score(state, po)
        if s > best_score:
            best_score = s
            best_po = po
    state.match_score = best_score
    if best_po:
        state.matched_po_id = best_po.get("po_id")

    # persist
    db_update_stage_and_state(workflow_id, "MATCH_TWO_WAY", state.dict())

    # if under threshold => checkpoint for human review
    if best_score < threshold:
        reason = f"match_score {best_score:.3f} < threshold {threshold}"
        review_url = generate_review_url(workflow_id)
        db_mark_hitl(workflow_id, review_url, reason)
        # also include checkpoint info in returned state
        state.notes = (state.notes or []) + [f"HITL CHECKPOINT created: {reason}"]
        return StageResponse(workflow_id=workflow_id, stage="MATCH_TWO_WAY", status="paused", state=state.dict())

    return StageResponse(workflow_id=workflow_id, stage="MATCH_TWO_WAY", status="running", state=state.dict())

@app.post("/checkpoint_hitl/{workflow_id}", response_model=StageResponse)
def checkpoint_hitl_save_state(workflow_id: str, reviewer_note: Optional[str] = None):
    """
    CHECKPOINT_HITL ⏸️ – save_state_for_human_review (Deterministic)
    Triggered ONLY IF matching fails. This endpoint allows external system or automated trigger
    to create the LangGraph checkpoint and persist workflow for human review. Returns paused state.
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    # Create LangGraph checkpoint (lightweight representation)
    checkpoint = {
        "checkpoint_id": str(uuid.uuid4()),
        "workflow_id": workflow_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": wf.get("state")
    }
    # persist checkpoint into workflow state as note + create human review entry if not exists
    state = wf.get("state") or {}
    state = dict(state)
    notes = state.get("notes") or []
    notes.append(f"LangGraph checkpoint created: {checkpoint['checkpoint_id']}")
    if reviewer_note:
        notes.append(f"Reviewer note: {reviewer_note}")
    state["notes"] = notes
    review_url = wf.get("hitl_review_url") or generate_review_url(workflow_id)
    db_mark_hitl(workflow_id, review_url, "manual checkpoint created")
    db_update_stage_and_state(workflow_id, "CHECKPOINT_HITL", state, status="paused")
    return StageResponse(workflow_id=workflow_id, stage="CHECKPOINT_HITL", status="paused", state=state)

@app.post("/reconcile/{workflow_id}", response_model=StageResponse)
def reconcile_build_accounting_entries(workflow_id: str):
    """
    RECONCILE 📘 – build_accounting_entries (Deterministic)
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    if wf.get("status") == "paused":
        raise HTTPException(status_code=409, detail="workflow paused for human review")
    state = WorkflowState(**(wf.get("state") or {}))

    # deterministic ledger entry builder
    total = sum((li.get("quantity", 1) * li.get("unit_price", 0) for li in (state.line_items or [])))
    entries = [
        {"entry_id": str(uuid.uuid4()), "type": "credit", "account": "Accounts Payable", "amount": total, "currency": "USD"},
        {"entry_id": str(uuid.uuid4()), "type": "debit", "account": "Expense:Procurement", "amount": total, "currency": "USD"},
    ]
    state.accounting_entries = entries
    state.notes = (state.notes or []) + [f"built {len(entries)} accounting entries deterministically"]
    db_update_stage_and_state(workflow_id, "RECONCILE", state.dict())
    return StageResponse(workflow_id=workflow_id, stage="RECONCILE", status="running", state=state.dict())

@app.post("/complete/{workflow_id}", response_model=StageResponse)
def complete_output_final_payload(workflow_id: str):
    """
    COMPLETE ✅ – output_final_payload (Deterministic)
    - Produce final structured payload
    - Output logs (here we embed into state)
    - Mark workflow complete
    """
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    if wf.get("status") == "paused":
        raise HTTPException(status_code=409, detail="workflow paused for human review; cannot complete")
    state = WorkflowState(**(wf.get("state") or {}))
    final_payload = {
        "workflow_id": workflow_id,
        "normalized_vendor": state.normalized_vendor,
        "line_items": state.line_items,
        "flags": state.flags,
        "match_score": state.match_score,
        "matched_po_id": state.matched_po_id,
        "accounting_entries": state.accounting_entries,
        "completed_at": datetime.now(timezone.utc).isoformat()
    }
    # write final payload into workflow state and mark complete
    state.notes = (state.notes or []) + ["workflow completed"]
    db_update_stage_and_state(workflow_id, "COMPLETE", state.dict(), status="complete")

    # also persist final payload into raw_payload -> for audit we add an 'output' field
    wf_from_db = db_get_workflow(workflow_id)
    raw_payload = wf_from_db.get("raw_payload") or {}
    raw_payload["final_output"] = final_payload
    db_upsert_workflow(workflow_id, {
        "status": "complete",
        "raw_payload": raw_payload,
        "current_stage": "COMPLETE",
        "state": state.dict()
    })
    return StageResponse(workflow_id=workflow_id, stage="COMPLETE", status="complete", state=final_payload)

@app.get("/workflow/{workflow_id}")
def get_workflow(workflow_id: str):
    wf = db_get_workflow(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf
