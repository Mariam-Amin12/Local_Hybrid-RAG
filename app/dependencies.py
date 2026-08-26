from typing import Optional

from chromadb import Key
from fastapi import Depends, Header, HTTPException, Request
from flask import request
from .config import Settings, get_settings

def get_rag(request: Request): # 3alshan a return shared rag object 
    print("[dependency] Providing shared RAG instance", flush=True)
    return request.app.state.rag

def get_executor(request: Request): # bardo a return shared executor object
    print("[dependency] Providing shared executor", flush=True)
    return request.app.state.executor

def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for auth"),
    settings: Settings = Depends(get_settings),

    ):
    if x_api_key is None or x_api_key != settings.API_KEY:
        print("[auth] Rejected request with invalid or missing API key", flush=True)
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    print("[auth] API key accepted", flush=True)
    