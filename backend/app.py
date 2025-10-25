# backend/app.py
# KofiChat / SmartBizBot - production-minded MVP backend (FastAPI + SQLite + OpenAI)
# Requirements: set OPENAI_API_KEY and ADMIN_KEY environment variables.

import os
import json
import time
import logging
import sqlite3
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# -------------------
# CONFIG - change if needed
# -------------------
DB_PATH = os.getenv("KC_DB_PATH", "kofichat.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "change_this_admin_key")  # set to secure value before deploy
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")  # fallback OK
EMB_DIM = 1536  # dimension of chosen embedding model
KB_DIR = os.getenv("KC_KB_DIR", "tenant_kbs")
TOP_K = int(os.getenv("KC_TOP_K", "4"))
# -------------------

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment before running backend.")
openai.api_key = OPENAI_API_KEY

os.makedirs(KB_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartbizbot")

# FastAPI
app = FastAPI(title="SmartBizBot MVP Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# DB helpers (simple SQLite)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS tenants (
        id TEXT PRIMARY KEY,
        business_name TEXT,
        created_at REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS kube_meta (
        tenant_id TEXT,
        meta_key TEXT,
        meta_val TEXT,
        PRIMARY KEY(tenant_id, meta_key)
    )""")
    conn.commit()
    conn.close()

def add_tenant_db(tenant_id: str, business_name: Optional[str]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO tenants (id, business_name, created_at) VALUES (?,?,?)",
              (tenant_id, business_name or "", time.time()))
    conn.commit()
    conn.close()

def list_tenants_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, business_name, created_at FROM tenants")
    rows = c.fetchall()
    conn.close()
    return [{"tenant_id": r[0], "business_name": r[1], "created_at": r[2]} for r in rows]

# Embedding helpers
def tenant_dir(tenant_id: str):
    d = os.path.join(KB_DIR, tenant_id)
    os.makedirs(d, exist_ok=True)
    return d

def docs_file(tenant_id: str):
    return os.path.join(tenant_dir(tenant_id), "docs.json")

def emb_file(tenant_id: str):
    return os.path.join(tenant_dir(tenant_id), "embeddings.npy")

def load_docs(tenant_id: str):
    f = docs_file(tenant_id)
    if not os.path.exists(f):
        return []
    with open(f, "r", encoding="utf-8") as fh:
        return json.load(fh)

def save_docs(tenant_id: str, docs: List[dict]):
    with open(docs_file(tenant_id), "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)

def load_embs(tenant_id: str):
    p = emb_file(tenant_id)
    if not os.path.exists(p):
        return np.zeros((0, EMB_DIM))
    return np.load(p)

def save_embs(tenant_id: str, arr: np.ndarray):
    np.save(emb_file(tenant_id), arr)

# OpenAI wrappers
def create_embeddings(texts: List[str]) -> List[List[float]]:
    # small wrapper with basic retry
    for _ in range(3):
        try:
            resp = openai.Embeddings.create(model=EMBED_MODEL, input=texts)
            return [d["embedding"] for d in resp["data"]]
        except Exception as e:
            logger.warning("Embedding call failed, retrying: %s", e)
            time.sleep(1)
    raise HTTPException(status_code=500, detail="Embedding API failed")

def chat_with_context(question: str, contexts: List[str]) -> str:
    ctx = "\\n\\n---\\n\\n".join(contexts) if contexts else ""
    user_prompt = f"Context:\\n{ctx}\\n\\nUser question:\\n{question}\\n\\nAnswer using the context when possible. If you are unsure, ask for clarification."
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[
                    {"role":"system","content":"You are a helpful assistant for a small business."},
                    {"role":"user","content":user_prompt}
                ],
                max_tokens=500,
                temperature=0.15
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("Chat API failed, retry: %s", e)
            time.sleep(1)
    raise HTTPException(status_code=500, detail="Chat API failed")

# Simple admin-key dependency
def require_admin(x_admin_key: str = Header(None)):
    if x_admin_key is None or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid ADMIN_KEY")
    return True

# Pydantic
class TenantPayload(BaseModel):
    tenant_id: str
    business_name: Optional[str] = None

class ChatPayload(BaseModel):
    tenant_id: str
    question: str

# Initialize DB
init_db()

# Routes
@app.post("/tenant/create")
def create_tenant(payload: TenantPayload, admin_ok: bool = Depends(require_admin)):
    tid = payload.tenant_id.strip().lower()
    if not tid:
        raise HTTPException(status_code=400, detail="tenant_id required")
    add_tenant_db(tid, payload.business_name)
    # init files
    save_docs(tid, [])
    save_embs(tid, np.zeros((0, EMB_DIM)))
    return {"status":"ok","tenant_id":tid}

@app.post("/tenant/{tenant_id}/upload_csv")
async def upload_csv(tenant_id: str, file: UploadFile = File(...), admin_ok: bool = Depends(require_admin)):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.flush()
    df = pd.read_csv(tmp.name)
    if "question" not in df.columns or "answer" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain question and answer columns")
    docs = load_docs(tenant_id)
    for _, r in df.iterrows():
        docs.append({"question": str(r["question"]), "answer": str(r["answer"])})
    save_docs(tenant_id, docs)
    texts = [d["question"] + "\\n" + d["answer"] for d in docs]
    embs = np.array(create_embeddings(texts))
    save_embs(tenant_id, embs)
    return {"status":"ok","rows": len(docs)}

@app.post("/tenant/{tenant_id}/add_qa")
def add_qa(tenant_id: str, question: str = Form(...), answer: str = Form(...), admin_ok: bool = Depends(require_admin)):
    docs = load_docs(tenant_id)
    docs.append({"question": question, "answer": answer})
    save_docs(tenant_id, docs)
    texts = [d["question"] + "\\n" + d["answer"] for d in docs]
    embs = np.array(create_embeddings(texts))
    save_embs(tenant_id, embs)
    return {"status":"ok"}

@app.post("/chat")
def chat(req: ChatPayload):
    docs = load_docs(req.tenant_id)
    if not docs:
        return {"answer":"No KB found for this tenant."}
    embs = load_embs(req.tenant_id)
    q_emb = np.array(create_embeddings([req.question]))
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = np.argsort(-sims)[:TOP_K]
    contexts = []
    used = []
    for i in idxs:
        if i < len(docs) and sims[int(i)] > 0:
            contexts.append(docs[int(i)]["answer"])
            used.append({"index": int(i), "question": docs[int(i)]["question"], "score": float(sims[int(i)])})
    answer = chat_with_context(req.question, contexts)
    return {"answer": answer, "sources": used}

@app.get("/tenants")
def list_tenants(admin_ok: bool = Depends(require_admin)):
    return {"tenants": list_tenants_db()}

# Basic health
@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
