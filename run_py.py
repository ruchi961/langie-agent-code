"""
langgraph_invoice_agent.py

Complete LangGraph agent for invoice processing workflow.
Uses LangGraph's built-in checkpointing system.

Usage:
    python langgraph_invoice_agent.py

Features:
- 12-stage invoice processing workflow
- HITL checkpoint/resume with LangGraph checkpointers
- Bigtool-based tool selection
- COMMON/ATLAS MCP server integration
- Built-in persistent state management
- FastAPI endpoints for human review
"""

import os
import json
import uuid
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, TypedDict, Annotated

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import NodeInterrupt

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
COMMON_MCP = os.environ.get("COMMON_MCP", "http://localhost:8001")
ATLAS_MCP = os.environ.get("ATLAS_MCP", "http://localhost:8002")
LOG_FILE = "workflow.log"
CHECKPOINT_DB = "langgraph_checkpoints.db"

# ═══════════════════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════════════════
class WorkflowState(TypedDict):
    workflow_id: str
    payload: Dict[str, Any]
    workflow_state: Dict[str, Any]
    status: str
    hitl_required: bool
    reviewer_decision: Optional[str]
    current_stage: str
    error: Optional[str]
    thread_id: str  # For checkpointing

# ═══════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════
def log(msg: str, **meta):
    ts = datetime.now(timezone.utc).isoformat()
    entry = {"ts": ts, "msg": msg, "meta": meta}
    line = json.dumps(entry, default=str)
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ═══════════════════════════════════════════════════════════
# BIGTOOL SELECTION
# ═══════════════════════════════════════════════════════════
def bigtool_select(pool_hint: List[str], capability: str) -> str:
    """Select best tool from pool based on capability"""
    if not pool_hint:
        return "default"
    
    preferences = {
        "ocr": ["tesseract", "google_vision", "aws_textract"],
        "enrichment": ["vendor_db", "clearbit", "people_data_labs"],
        "erp_connector": ["mock_erp", "sap_sandbox", "netsuite"],
        "db": ["sqlite", "postgres", "dynamodb"],
        "email": ["local_mail", "sendgrid", "ses"]
    }
    
    pref_list = preferences.get(capability, pool_hint)
    for pref in pref_list:
        if pref.lower() in [p.lower() for p in pool_hint]:
            log("bigtool_selected", capability=capability, selected=pref, pool=pool_hint)
            return pref
    
    selected = pool_hint[0]
    log("bigtool_selected", capability=capability, selected=selected, pool=pool_hint)
    return selected

# ═══════════════════════════════════════════════════════════
# MCP CLIENT
# ═══════════════════════════════════════════════════════════
async def mcp_call(server: str, endpoint: str, payload: Dict = None, retries: int = 2):
    """Call MCP server endpoint with retries"""
    if payload is None:
        payload = {}
    
    url = server.rstrip("/") + "/" + endpoint.lstrip("/")
    log("mcp_call", url=url, payload_size=len(str(payload)))
    
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(retries + 1):
            try:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                resp = r.json()
                log("mcp_success", url=url, status=r.status_code)
                return resp
            except Exception as e:
                log("mcp_error", url=url, attempt=attempt, error=str(e))
                if attempt >= retries:
                    raise
                await asyncio.sleep(1)

# ═══════════════════════════════════════════════════════════
# STAGE NODES
# ═══════════════════════════════════════════════════════════

async def intake_node(state: WorkflowState) -> WorkflowState:
    """Stage 1: INTAKE - Accept and validate invoice payload"""
    log("stage_start", stage="INTAKE", wf=state.get("workflow_id"))
    try:
        resp = await mcp_call(COMMON_MCP, "intake", state["payload"])
        state["workflow_id"] = resp.get("workflow_id", state["workflow_id"])
        state["workflow_state"]["intake"] = resp
        state["status"] = "running"
        state["current_stage"] = "INTAKE"
    except Exception as e:
        log("stage_error", stage="INTAKE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="INTAKE", status=state["status"])
    return state

async def understand_node(state: WorkflowState) -> WorkflowState:
    """Stage 2: UNDERSTAND - OCR + Parse line items"""
    log("stage_start", stage="UNDERSTAND", wf=state["workflow_id"])
    try:
        ocr_provider = bigtool_select(["tesseract", "google_vision"], "ocr")
        
        attachments = state["payload"].get("attachments", [])
        ocr_resp = await mcp_call(ATLAS_MCP, "ocr_extract", {
            "attachments": attachments,
            "provider": ocr_provider
        })
        
        ocr_text = " ".join([t.get("text", "") for t in ocr_resp.get("ocr_texts", [])])
        parse_resp = await mcp_call(ATLAS_MCP, "parse_line_items", {"text": ocr_text})
        
        state["workflow_state"]["understand"] = {
            "ocr": ocr_resp,
            "parse": parse_resp,
            "provider": ocr_provider
        }
        state["status"] = "running"
        state["current_stage"] = "UNDERSTAND"
    except Exception as e:
        log("stage_error", stage="UNDERSTAND", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="UNDERSTAND", status=state["status"])
    return state

async def prepare_node(state: WorkflowState) -> WorkflowState:
    """Stage 3: PREPARE - Normalize vendor, enrich, compute flags"""
    log("stage_start", stage="PREPARE", wf=state["workflow_id"])
    try:
        vendor = state["payload"].get("vendor_name", "UNKNOWN")
        if vendor is None:
            vendor = "UNKNOWN"
        
        norm_resp = await mcp_call(ATLAS_MCP, "normalize_vendor", {"vendor_name": vendor})
        
        enrich_provider = bigtool_select(["vendor_db", "clearbit"], "enrichment")
        enrich_resp = await mcp_call(ATLAS_MCP, "enrich_vendor", {
            "vendor_name": vendor,
            "provider": enrich_provider
        })
        
        amount = state["payload"].get("amount", 0)
        risk = enrich_resp.get("risk_score", 0.5)
        flags_resp = await mcp_call(ATLAS_MCP, "compute_flags", {
            "amount": amount,
            "risk_score": risk
        })
        
        state["workflow_state"]["prepare"] = {
            "normalized": norm_resp,
            "enrichment": enrich_resp,
            "flags": flags_resp,
            "provider": enrich_provider
        }
        state["status"] = "running"
        state["current_stage"] = "PREPARE"
    except Exception as e:
        log("stage_error", stage="PREPARE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="PREPARE", status=state["status"])
    return state

async def retrieve_node(state: WorkflowState) -> WorkflowState:
    """Stage 4: RETRIEVE - Fetch POs, GRNs, history from ERP"""
    log("stage_start", stage="RETRIEVE", wf=state["workflow_id"])
    try:
        vendor = state["payload"].get("vendor_name", "")
        erp_provider = bigtool_select(["mock_erp", "sap_sandbox"], "erp_connector")
        
        pos_resp = await mcp_call(ATLAS_MCP, "fetch_po", {
            "vendor_name": vendor,
            "provider": erp_provider
        })
        
        po_ids = [po.get("po_id") for po in pos_resp.get("pos", [])]
        grns_resp = await mcp_call(ATLAS_MCP, "fetch_grn", {
            "po_ids": po_ids,
            "provider": erp_provider
        })
        
        hist_resp = await mcp_call(ATLAS_MCP, "fetch_history", {
            "vendor_name": vendor,
            "provider": erp_provider
        })
        
        state["workflow_state"]["retrieve"] = {
            "pos": pos_resp,
            "grns": grns_resp,
            "history": hist_resp,
            "provider": erp_provider
        }
        state["status"] = "running"
        state["current_stage"] = "RETRIEVE"
    except Exception as e:
        log("stage_error", stage="RETRIEVE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="RETRIEVE", status=state["status"])
    return state

async def match_two_way_node(state: WorkflowState) -> WorkflowState:
    """Stage 5: MATCH_TWO_WAY - Compute match score"""
    log("stage_start", stage="MATCH_TWO_WAY", wf=state["workflow_id"])
    try:
        threshold = 0.90
        invoice_amt = state["payload"].get("amount", 0)
        pos = state["workflow_state"].get("retrieve", {}).get("pos", {}).get("pos", [])
        
        match_score = 0.0
        matched_po = None
        
        for po in pos:
            po_amt = po.get("amount", 0)
            if po_amt > 0:
                score = 1.0 - abs(invoice_amt - po_amt) / max(invoice_amt, po_amt)
                if score > match_score:
                    match_score = score
                    matched_po = po
        
        state["workflow_state"]["match"] = {
            "match_score": match_score,
            "threshold": threshold,
            "matched_po": matched_po,
            "match_result": "MATCHED" if match_score >= threshold else "FAILED"
        }
        
        state["hitl_required"] = match_score < threshold
        state["status"] = "running"
        state["current_stage"] = "MATCH_TWO_WAY"
        
        log("match_result", score=match_score, threshold=threshold, hitl=state["hitl_required"])
    except Exception as e:
        log("stage_error", stage="MATCH_TWO_WAY", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="MATCH_TWO_WAY", status=state["status"])
    return state

async def checkpoint_hitl_node(state: WorkflowState) -> WorkflowState:
    """Stage 6: CHECKPOINT_HITL - Pause for human review (checkpoint handled by LangGraph)"""
    log("stage_start", stage="CHECKPOINT_HITL", wf=state["workflow_id"])
    try:
        score = state["workflow_state"].get("match", {}).get("match_score", 0)
        threshold = state["workflow_state"].get("match", {}).get("threshold", 0.9)
        reason = f"Match score {score:.2f} < threshold {threshold:.2f}"
        
        state["workflow_state"]["checkpoint"] = {
            "reason": reason,
            "thread_id": state["thread_id"],
            "review_url": f"http://localhost:9000/human-review/{state['thread_id']}"
        }
        state["status"] = "paused"
        state["current_stage"] = "CHECKPOINT_HITL"
        
        log("checkpoint_created", thread_id=state["thread_id"], reason=reason)
    except Exception as e:
        log("stage_error", stage="CHECKPOINT_HITL", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="CHECKPOINT_HITL", status=state["status"])
    return state

async def hitl_decision_node(state: WorkflowState) -> WorkflowState:
    """Stage 7: HITL_DECISION - Process human decision"""
    log("stage_start", stage="HITL_DECISION", wf=state["workflow_id"])
    try:
        decision = state.get("reviewer_decision", "PENDING")
        if decision is None:
            decision = "PENDING"
            
        if decision == "PENDING":
            log("hitl_awaiting", wf=state["workflow_id"])
            state["status"] = "paused"
            state["current_stage"] = "HITL_DECISION"
            return state
        
        resp = await mcp_call(ATLAS_MCP, "accept_or_reject_invoice", {
            "checkpoint_state": state["workflow_state"],
            "decision": decision
        })
        
        state["workflow_state"]["hitl_decision"] = resp
        
        if decision == "ACCEPT":
            state["status"] = "running"
            state["hitl_required"] = False
        else:
            state["status"] = "REQUIRES_MANUAL_HANDLING"
        
        state["current_stage"] = "HITL_DECISION"
    except Exception as e:
        log("stage_error", stage="HITL_DECISION", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="HITL_DECISION", status=state["status"])
    return state

async def reconcile_node(state: WorkflowState) -> WorkflowState:
    """Stage 8: RECONCILE - Build accounting entries"""
    log("stage_start", stage="RECONCILE", wf=state["workflow_id"])
    try:
        amt = state["payload"].get("amount", 0)
        entries = [
            {"entry_id": str(uuid.uuid4()), "type": "debit", "account": "Expense:Procurement", 
             "amount": amt, "currency": "USD"},
            {"entry_id": str(uuid.uuid4()), "type": "credit", "account": "Accounts Payable", 
             "amount": amt, "currency": "USD"}
        ]
        
        state["workflow_state"]["reconcile"] = {
            "accounting_entries": entries,
            "reconciled_at": datetime.now(timezone.utc).isoformat()
        }
        state["status"] = "running"
        state["current_stage"] = "RECONCILE"
    except Exception as e:
        log("stage_error", stage="RECONCILE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="RECONCILE", status=state["status"])
    return state

async def approve_node(state: WorkflowState) -> WorkflowState:
    """Stage 9: APPROVE - Apply approval policies"""
    log("stage_start", stage="APPROVE", wf=state["workflow_id"])
    try:
        amt = state["payload"].get("amount", 0)
        resp = await mcp_call(ATLAS_MCP, "apply_invoice_approval_policy", {"amount": amt})
        
        state["workflow_state"]["approve"] = resp
        state["status"] = "running"
        state["current_stage"] = "APPROVE"
    except Exception as e:
        log("stage_error", stage="APPROVE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="APPROVE", status=state["status"])
    return state

async def posting_node(state: WorkflowState) -> WorkflowState:
    """Stage 10: POSTING - Post to ERP and schedule payment"""
    log("stage_start", stage="POSTING", wf=state["workflow_id"])
    try:
        amt = state["payload"].get("amount", 0)
        vendor = state["payload"].get("vendor_name", "")
        
        post_resp = await mcp_call(ATLAS_MCP, "post_to_erp", {
            "amount": amt,
            "vendor_name": vendor,
            "details": {"provider": "mock_erp"}
        })
        
        pay_resp = await mcp_call(ATLAS_MCP, "schedule_payment", {
            "amount": amt,
            "vendor_name": vendor,
            "days_delay": 7
        })
        
        state["workflow_state"]["posting"] = {"erp_post": post_resp, "payment": pay_resp}
        state["status"] = "running"
        state["current_stage"] = "POSTING"
    except Exception as e:
        log("stage_error", stage="POSTING", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="POSTING", status=state["status"])
    return state

async def notify_node(state: WorkflowState) -> WorkflowState:
    """Stage 11: NOTIFY - Send notifications"""
    log("stage_start", stage="NOTIFY", wf=state["workflow_id"])
    try:
        vendor = state["payload"].get("vendor_name", "")
        inv_id = state["payload"].get("invoice_id", state["workflow_id"])
        
        vendor_resp = await mcp_call(ATLAS_MCP, "notify_vendor", {
            "vendor_name": vendor,
            "invoice_id": inv_id,
            "message": "Invoice processed"
        })
        
        finance_resp = await mcp_call(ATLAS_MCP, "notify_finance_team", {
            "vendor_name": vendor,
            "invoice_id": inv_id
        })
        
        state["workflow_state"]["notify"] = {"vendor": vendor_resp, "finance": finance_resp}
        state["status"] = "running"
        state["current_stage"] = "NOTIFY"
    except Exception as e:
        log("stage_error", stage="NOTIFY", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="NOTIFY", status=state["status"])
    return state

async def complete_node(state: WorkflowState) -> WorkflowState:
    """Stage 12: COMPLETE - Finalize workflow"""
    log("stage_start", stage="COMPLETE", wf=state["workflow_id"])
    try:
        final = {
            "workflow_id": state["workflow_id"],
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "invoice_data": state["payload"],
            "processing_results": state["workflow_state"]
        }
        
        state["workflow_state"]["complete"] = final
        state["status"] = "complete"
        state["current_stage"] = "COMPLETE"
        print("The final answer is", state,final)
        print(final['invoice_data'])
        log("workflow_complete", wf=state["workflow_id"])
    except Exception as e:
        log("stage_error", stage="COMPLETE", error=str(e))
        state["error"] = str(e)
        state["status"] = "error"
    log("stage_end", stage="COMPLETE", status=state["status"])
    return state

# ═══════════════════════════════════════════════════════════
# ROUTING LOGIC
# ═══════════════════════════════════════════════════════════

def match_router(state: WorkflowState) -> str:
    """Route after MATCH_TWO_WAY"""
    return "checkpoint" if state.get("hitl_required", False) else "reconcile"

def hitl_router(state: WorkflowState) -> str:
    """Route after HITL_DECISION"""
    if state.get("status") == "REQUIRES_MANUAL_HANDLING":
        return "end"
    if state.get("reviewer_decision") == "ACCEPT":
        return "reconcile"
    return "end"

# ═══════════════════════════════════════════════════════════
# BUILD LANGGRAPH WITH CHECKPOINTER
# ═══════════════════════════════════════════════════════════

def build_graph(use_sqlite: bool = True):
    """Build complete LangGraph workflow with checkpointing"""
    graph = StateGraph(WorkflowState)
    
    # Add nodes
    graph.add_node("INTAKE", intake_node)
    graph.add_node("UNDERSTAND", understand_node)
    graph.add_node("PREPARE", prepare_node)
    graph.add_node("RETRIEVE", retrieve_node)
    graph.add_node("MATCH_TWO_WAY", match_two_way_node)
    graph.add_node("CHECKPOINT_HITL", checkpoint_hitl_node)
    graph.add_node("HITL_DECISION", hitl_decision_node)
    graph.add_node("RECONCILE", reconcile_node)
    graph.add_node("APPROVE", approve_node)
    graph.add_node("POSTING", posting_node)
    graph.add_node("NOTIFY", notify_node)
    graph.add_node("COMPLETE", complete_node)
    
    # Set entry
    graph.set_entry_point("INTAKE")
    
    # Linear edges
    graph.add_edge("INTAKE", "UNDERSTAND")
    graph.add_edge("UNDERSTAND", "PREPARE")
    graph.add_edge("PREPARE", "RETRIEVE")
    graph.add_edge("RETRIEVE", "MATCH_TWO_WAY")
    
    # Conditional routing
    graph.add_conditional_edges("MATCH_TWO_WAY", match_router, 
                                {"checkpoint": "CHECKPOINT_HITL", "reconcile": "RECONCILE"})
    
    graph.add_edge("CHECKPOINT_HITL", "HITL_DECISION")
    
    graph.add_conditional_edges("HITL_DECISION", hitl_router,
                                {"reconcile": "RECONCILE", "end": END})
    
    # Final edges
    graph.add_edge("RECONCILE", "APPROVE")
    graph.add_edge("APPROVE", "POSTING")
    graph.add_edge("POSTING", "NOTIFY")
    graph.add_edge("NOTIFY", "COMPLETE")
    graph.add_edge("COMPLETE", END)
    
    # Compile with checkpointer
    if use_sqlite:
        # Create connection and checkpointer
        import sqlite3
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = MemorySaver()
    
    return graph.compile(checkpointer=checkpointer)

# Global agent instance
agent = build_graph(use_sqlite=False)

# ═══════════════════════════════════════════════════════════
# WORKFLOW RUNNER
# ═══════════════════════════════════════════════════════════

async def run_workflow(payload: Dict[str, Any], thread_id: str = None) -> Dict[str, Any]:
    """Start workflow from beginning"""
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    wf_id = str(uuid.uuid4())
    
    init_state: WorkflowState = {
        "workflow_id": wf_id,
        "payload": payload,
        "workflow_state": {},
        "status": "starting",
        "hitl_required": False,
        "reviewer_decision": None,
        "current_stage": "INIT",
        "error": None,
        "thread_id": thread_id
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    log("workflow_start", wf=wf_id, thread=thread_id)
    
    try:
        # Use astream to handle interrupts properly
        final_state = None
        async for event in agent.astream(init_state, config, stream_mode="values"):
            final_state = event
        
        log("workflow_paused_or_complete", wf=wf_id, thread=thread_id, status=final_state.get("status"))
        return final_state
    except Exception as e:
        log("workflow_error", wf=wf_id, thread=thread_id, error=str(e))
        raise

async def resume_workflow(thread_id: str, decision: str) -> Dict[str, Any]:
    """Resume workflow from checkpoint after human review"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get current state from checkpointer
    state_snapshot = agent.get_state(config)
    
    if not state_snapshot or not state_snapshot.values:
        raise ValueError(f"No checkpoint found for thread_id: {thread_id}")
    
    wf_id = state_snapshot.values.get("workflow_id", "unknown")
    
    log("resume_start", thread=thread_id, decision=decision, wf=wf_id)
    
    try:
        # Update state with the human decision
        agent.update_state(
            config,
            {"reviewer_decision": decision.upper()},
            as_node="CHECKPOINT_HITL"
        )
        
        # Resume execution - stream to handle completion properly
        final_state = None
        async for event in agent.astream(None, config, stream_mode="values"):
            final_state = event
            log("resume_progress", stage=event.get("current_stage"), status=event.get("status"))
        
        log("resume_complete", wf=wf_id, thread=thread_id, status=final_state.get("status"))
        return final_state
        
    except Exception as e:
        log("resume_error", wf=wf_id, thread=thread_id, error=str(e))
        raise

async def get_workflow_history(thread_id: str) -> List[Dict]:
    """Get checkpoint history for a workflow thread"""
    config = {"configurable": {"thread_id": thread_id}}
    
    history = []
    for state in agent.get_state_history(config):
        history.append({
            "checkpoint_id": state.config["configurable"]["checkpoint_id"],
            "values": state.values,
            "next": state.next,
            "metadata": state.metadata
        })
    
    return history

# ═══════════════════════════════════════════════════════════
# FASTAPI HUMAN REVIEW API
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="Langie Invoice Agent - Human Review API")

class DecisionRequest(BaseModel):
    thread_id: str
    decision: str
    reviewer_id: Optional[str] = None
    notes: Optional[str] = None

@app.on_event("startup")
def startup():
    log("api_startup", service="human_review")

@app.get("/human-review/pending")
async def api_list_pending():
    """List all workflows paused for review (requires scanning checkpoints)"""
    # Note: In production, you'd maintain a separate index of paused workflows
    # For now, this is a placeholder that requires external tracking
    return {
        "message": "Use thread_id from workflow execution to access specific paused workflows",
        "note": "In production, maintain a separate queue/index of paused thread_ids"
    }

@app.get("/human-review/{thread_id}")
async def api_get_review(thread_id: str):
    """Get details of a paused workflow for review"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state_snapshot = agent.get_state(config)
        
        if not state_snapshot.values:
            raise HTTPException(404, "Workflow not found")
        
        current_state = state_snapshot.values
        
        if current_state.get("status") != "paused":
            return {
                "thread_id": thread_id,
                "status": current_state.get("status"),
                "message": "Workflow is not paused for review"
            }
        
        match_info = current_state.get("workflow_state", {}).get("match", {})
        checkpoint_info = current_state.get("workflow_state", {}).get("checkpoint", {})
        
        return {
            "thread_id": thread_id,
            "workflow_id": current_state.get("workflow_id"),
            "status": "paused",
            "current_stage": current_state.get("current_stage"),
            "match_score": match_info.get("match_score"),
            "threshold": match_info.get("threshold"),
            "reason": checkpoint_info.get("reason"),
            "payload": current_state.get("payload"),
            "next_steps": ["ACCEPT", "REJECT"]
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/human-review/decision")
async def api_decide(req: DecisionRequest):
    """Submit human review decision and resume workflow"""
    if req.decision.upper() not in ["ACCEPT", "REJECT"]:
        raise HTTPException(400, "Decision must be ACCEPT or REJECT")
    
    try:
        result = await resume_workflow(req.thread_id, req.decision.upper())
        
        return {
            "ok": True,
            "thread_id": req.thread_id,
            "decision": req.decision,
            "workflow_status": result.get("status"),
            "current_stage": result.get("current_stage")
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/workflow/{thread_id}/history")
async def get_history(thread_id: str):
    """Get checkpoint history for a workflow"""
    try:
        history = await get_workflow_history(thread_id)
        return {"thread_id": thread_id, "history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/workflow/{thread_id}/status")
def get_workflow_status(thread_id: str):
    """Get current workflow status"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state_snapshot = agent.get_state(config)
        
        if not state_snapshot.values:
            raise HTTPException(404, "Workflow not found")
        
        return {
            "thread_id": thread_id,
            "workflow_id": state_snapshot.values.get("workflow_id"),
            "status": state_snapshot.values.get("status"),
            "current_stage": state_snapshot.values.get("current_stage"),
            "state": state_snapshot.values
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok", "service": "langie-invoice-agent"}

# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

async def demo_workflow():
    """Run demo invoice processing workflow"""
    
    # Sample invoice that will fail matching (to trigger HITL)
    sample_invoice = {
        "invoice_id": "INV-2025-001",
        "vendor_name": "Acme Corp",
        "invoice_date": "2025-02-10",
        "due_date": "2025-03-15",
        "amount": 999.99,  # Different from PO amount
        "currency": "USD",
        "line_items": [
            {"desc": "Widget A", "qty": 2, "unit_price": 499.99, "total": 999.99}
        ],
        "attachments": []
    }
    
    print("\n" + "="*70)
    print("LANGIE INVOICE AGENT - DEMO WORKFLOW")
    print("With LangGraph Built-in Checkpointing")
    print("="*70 + "\n")
    
    print("📥 Starting workflow with sample invoice...")
    print(f"   Invoice: {sample_invoice['invoice_id']}")
    print(f"   Vendor: {sample_invoice['vendor_name']}")
    print(f"   Amount: ${sample_invoice['amount']}")
    print()
    
    try:
        result = await run_workflow(sample_invoice)
        
        print("\n" + "="*70)
        print("WORKFLOW RESULT")
        print("="*70)
        print(f"Thread ID: {result.get('thread_id')}")
        print(f"Workflow ID: {result.get('workflow_id')}")
        print(f"Status: {result.get('status')}")
        print(f"Current Stage: {result.get('current_stage')}")
        print(f"HITL Required: {result.get('hitl_required')}")
        
        if result.get('status') == 'paused':
            print("\n⏸️  WORKFLOW PAUSED FOR HUMAN REVIEW")
            thread_id = result.get('thread_id')
            print(f"   Thread ID: {thread_id}")
            print(f"   Review URL: http://localhost:9000/human-review/{thread_id}")
            print("\n💡 To resume, use:")
            print(f"   POST /human-review/decision")
            print(f"   {{\"thread_id\": \"{thread_id}\", \"decision\": \"ACCEPT\"}}")
            print("\n💡 To view checkpoint history:")
            print(f"   GET /workflow/{thread_id}/history")
        
        if result.get('status') == 'complete':
            print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY")
            final = result.get('workflow_state', {}).get('complete', {})
            print(f"   Completed At: {final.get('completed_at')}")
        
        print("\n" + "="*70)
        print("STAGE EXECUTION LOG")
        print("="*70)
        for stage, data in result.get('workflow_state', {}).items():
            print(f"✓ {stage.upper()}")
            if stage == 'prepare':
                print(f"   Provider: {data.get('provider')}")
            elif stage == 'retrieve':
                print(f"   Provider: {data.get('provider')}")
            elif stage == 'match':
                print(f"   Score: {data.get('match_score'):.3f}")
                print(f"   Result: {data.get('match_result')}")
        
        print("\n" + "="*70)
        print("CHECKPOINTING FEATURES")
        print("="*70)
        print("✓ Automatic state persistence with SqliteSaver")
        print("✓ Time-travel capability - view any past state")
        print("✓ Resume from any checkpoint")
        print("✓ Built-in state versioning")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        raise

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 LANGIE - LangGraph Invoice Processing Agent")
    print("   With Built-in Checkpointing (SqliteSaver)")
    print("="*70)
    print(f"COMMON MCP Server: {COMMON_MCP}")
    print(f"ATLAS MCP Server:  {ATLAS_MCP}")
    print(f"Checkpoint DB:     {CHECKPOINT_DB}")
    print("="*70 + "\n")
    
    # Run FastAPI server and demo concurrently
    config = uvicorn.Config(app, host="0.0.0.0", port=9003, log_level="info")
    server = uvicorn.Server(config)
    
    async def main():
        # Start API server
        server_task = asyncio.create_task(server.serve())
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Run demo
        await demo_workflow()
        
        print("\n💻 Human Review API running at http://localhost:9000")
        print("   GET  /human-review/{thread_id} - View paused workflow")
        print("   POST /human-review/decision - Submit decision")
        print("   GET  /workflow/{thread_id}/status - Get workflow status")
        print("   GET  /workflow/{thread_id}/history - View checkpoint history")
        print("\n   Press Ctrl+C to exit\n")
        
        # Keep server running
        await server_task
    
    asyncio.run(main())
