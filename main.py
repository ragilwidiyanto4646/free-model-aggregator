#!/usr/bin/env python3
"""Free-model aggregator server (make.fun bounty: free-model-aggregator-01).

One server, two APIs (OpenAI- and Anthropic-compatible), ten simulated
free-tier providers, routing each request to the best free model that can
serve it, with failover when a provider refuses.

Python 3.13, standard library only -- nothing is installed before the
server runs (per the bounty requirements).

Run:
    python3 main.py [--port 8080]
    python3 main.py --list            # print the registered catalog, byte for byte

Environment:
    MAKEFUN_ENV_URL   base URL of the sealed trial env that hosts the
                      10 providers (all model calls go there, and only there)
    PORT              listen port (default 8080)
    MAKEFUN_STATE_PATH  where to write the deterministic state export
                      (default ./state.json)

When MAKEFUN_ENV_URL is unset or its catalog cannot be read, the server
falls back to the built-in simulated catalog below, so it still runs and
can be tested standalone (see tests/fake_env.py for a full fake env, and
run20.py for the 20-request run). A prompt containing
"FORCE-429:<provider-id>" / "FORCE-402:<provider-id>" makes that simulated
provider refuse, so failover can be exercised standalone.
MAKEFUN_NO_AUTO_EXIT=1 disables the exit after the 20th request
(the trial default is to exit once 20 requests are served, or when the
env's /work reports done).

Determinism: no wall-clock time and no unseeded randomness anywhere in
routing, ids, token counts or the state export. Response ids are
sequence-based (chatcmpl-0001, msg-0001, ...). The only sleeps are the
fixed /work poll interval and the 1s delayed graceful exit after the
trial reports done -- neither affects any routing decision or any byte
of the state export.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

VERSION = "1.0.0"

# Largest request body the server reads (10 MiB). A prompt is text; even
# a 1M-token context window is ~4 MB of characters, so this is far above
# any legitimate trial request. Without a cap, an unauthenticated client
# can make the server buffer an arbitrarily large body in memory --
# Content-Length is otherwise trusted blindly in Handler._read_json --
# and a GB-scale body exhausts memory and kills the trial process.
MAX_REQUEST_BYTES = 10 * 1024 * 1024


class _RequestTooLarge(Exception):
    """Content-Length exceeds MAX_REQUEST_BYTES; answered with 413."""

# ---------------------------------------------------------------------------
# Built-in simulated catalog (fallback when no env is reachable).
# Each provider: one wire, its own auth header, its own models.
# quality is the routing score (higher = better). price 0 + free True is
# what makes a model a legal pick; paid models are never legal.
# capabilities: tools / vision / json / long_context.
# ---------------------------------------------------------------------------
BUILTIN_CATALOG = {
    "providers": [
        {
            "id": "groq-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-groq-free",
            "base_path": "/providers/groq-free",
            "models": [
                {"id": "llama-3.3-70b-versatile-free", "quality": 88,
                 "free": True, "price": 0, "context_window": 131072,
                 "capabilities": ["tools", "json", "long_context"]},
                {"id": "llama-3.1-8b-instant-free", "quality": 62,
                 "free": True, "price": 0, "context_window": 131072,
                 "capabilities": ["json", "long_context"]},
            ],
        },
        {
            "id": "cerebras-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-cerebras-free",
            "base_path": "/providers/cerebras-free",
            "models": [
                {"id": "llama-3.3-70b-free", "quality": 86,
                 "free": True, "price": 0, "context_window": 131072,
                 "capabilities": ["tools", "json", "long_context"]},
            ],
        },
        {
            "id": "together-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-together-free",
            "base_path": "/providers/together-free",
            "models": [
                {"id": "Qwen2.5-72B-Instruct-free", "quality": 84,
                 "free": True, "price": 0, "context_window": 32768,
                 "capabilities": ["tools", "json"]},
                {"id": "DeepSeek-R1-Distill-free", "quality": 80,
                 "free": True, "price": 0, "context_window": 32768,
                 "capabilities": ["json"]},
            ],
        },
        {
            "id": "fireworks-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-fireworks-free",
            "base_path": "/providers/fireworks-free",
            "models": [
                {"id": "llama-v3p3-70b-instruct-free", "quality": 83,
                 "free": True, "price": 0, "context_window": 131072,
                 "capabilities": ["tools", "json", "long_context"]},
            ],
        },
        {
            "id": "mistral-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-mistral-free",
            "base_path": "/providers/mistral-free",
            "models": [
                {"id": "mistral-small-3.1-free", "quality": 81,
                 "free": True, "price": 0, "context_window": 128000,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
                {"id": "mistral-large-paid", "quality": 95,
                 "free": False, "price": 2, "context_window": 128000,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
            ],
        },
        {
            "id": "google-gemini-free",
            "wire": "gemini",
            "auth_header": "x-goog-api-key",
            "auth_scheme": "",
            "auth_value": "sim-gemini-free",
            "base_path": "/providers/google-gemini-free",
            "models": [
                {"id": "gemini-2.0-flash-free", "quality": 87,
                 "free": True, "price": 0, "context_window": 1048576,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
                {"id": "gemini-2.5-pro-paid", "quality": 96,
                 "free": False, "price": 3, "context_window": 1048576,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
            ],
        },
        {
            "id": "anthropic-free",
            "wire": "anthropic",
            "auth_header": "x-api-key",
            "auth_scheme": "",
            "auth_value": "sim-anthropic-free",
            "base_path": "/providers/anthropic-free",
            "models": [
                {"id": "claude-haiku-4.5-free", "quality": 90,
                 "free": True, "price": 0, "context_window": 200000,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
            ],
        },
        {
            "id": "deepseek-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-deepseek-free",
            "base_path": "/providers/deepseek-free",
            "models": [
                {"id": "deepseek-chat-free", "quality": 85,
                 "free": True, "price": 0, "context_window": 65536,
                 "capabilities": ["tools", "json"]},
            ],
        },
        {
            "id": "openrouter-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-openrouter-free",
            "base_path": "/providers/openrouter-free",
            "models": [
                {"id": "qwen3-235b-free", "quality": 89,
                 "free": True, "price": 0, "context_window": 131072,
                 "capabilities": ["tools", "json", "long_context"]},
                {"id": "kimi-k2-free", "quality": 82,
                 "free": True, "price": 0, "context_window": 200000,
                 "capabilities": ["tools", "json", "long_context"]},
            ],
        },
        {
            "id": "cohere-free",
            "wire": "openai",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "auth_value": "sim-cohere-free",
            "base_path": "/providers/cohere-free",
            "models": [
                {"id": "command-a-free", "quality": 78,
                 "free": True, "price": 0, "context_window": 256000,
                 "capabilities": ["tools", "vision", "json", "long_context"]},
            ],
        },
    ],
}


def canonical_catalog(raw):
    """Normalize any catalog shape into the canonical document.

    Accepted shapes: {"providers": [...]}, {"catalog": {...}}, a bare list
    of providers, an OpenAI /v1/models list {"data": [...]}, or a bare
    list of models. Missing fields get safe defaults; a model is free
    only when it says so (free/is_free True, or price/cost == 0).
    Result is fully deterministic: providers sorted by id, models by id.
    """
    providers = []
    if isinstance(raw, dict):
        inner = raw.get("catalog", raw)
        if isinstance(inner, dict) and "providers" in inner:
            providers = inner["providers"]
        elif isinstance(inner, dict) and "data" in inner:
            providers = [{"id": "env-default", "wire": "openai",
                          "auth_header": "Authorization",
                          "auth_scheme": "Bearer", "auth_value": "",
                          "base_path": "", "models": inner["data"]}]
        elif isinstance(inner, dict) and "models" in inner:
            providers = [{"id": "env-default", "wire": "openai",
                          "auth_header": "Authorization",
                          "auth_scheme": "Bearer", "auth_value": "",
                          "base_path": "", "models": inner["models"]}]
        elif isinstance(inner, list):
            providers = inner
    elif isinstance(raw, list):
        providers = raw

    out_providers = []
    for p in providers:
        if not isinstance(p, dict):
            continue
        models = p.get("models") or p.get("data") or []
        out_models = []
        for m in models:
            if isinstance(m, str):
                m = {"id": m}
            if not isinstance(m, dict):
                continue
            mid = m.get("id") or m.get("model") or m.get("name", "")
            if not mid:
                continue
            free = m.get("free", m.get("is_free", None))
            price = m.get("price", m.get("cost", m.get("price_per_token", 0)))
            try:
                price = float(price or 0)
            except (TypeError, ValueError):
                price = 0
            if free is None:
                free = (price == 0)
            caps = m.get("capabilities") or m.get("supports") or []
            if isinstance(caps, dict):
                caps = [k for k, v in caps.items() if v]
            ctx = m.get("context_window", m.get("max_context",
                        m.get("context_length", 8192)))
            try:
                ctx = int(ctx)
            except (TypeError, ValueError):
                ctx = 8192
            quality = m.get("quality", m.get("rating", m.get("score", 50)))
            try:
                quality = float(quality)
            except (TypeError, ValueError):
                quality = 50
            out_models.append({
                "id": str(mid), "quality": quality,
                "free": bool(free), "price": price,
                "context_window": ctx,
                "capabilities": sorted(str(c) for c in caps),
            })
        out_models.sort(key=lambda m: m["id"])
        out_providers.append({
            "id": str(p.get("id") or p.get("name") or "provider"),
            "wire": str(p.get("wire") or p.get("api") or "openai").lower(),
            "auth_header": str(p.get("auth_header") or p.get("authHeader")
                               or "Authorization"),
            "auth_scheme": str(p.get("auth_scheme") or p.get("authScheme") or ""),
            "auth_value": str(p.get("auth_value") or p.get("authValue")
                              or p.get("api_key") or p.get("token") or ""),
            "base_path": str(p.get("base_path") or p.get("basePath")
                             or p.get("base_url") or p.get("url") or ""),
            "models": out_models,
        })
    out_providers.sort(key=lambda p: p["id"])
    return {"version": 1, "providers": out_providers}


def catalog_bytes(catalog):
    """The exact bytes --list prints and the server registers."""
    return (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode()


def fetch_json(url, headers=None, timeout=10):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def discover_catalog(env_url):
    """Read the env catalog, trying the plausible endpoints in order.

    Returns (catalog, source) where source names the endpoint that worked,
    or (builtin, "builtin") when nothing answered. Never raises.
    """
    if not env_url:
        return canonical_catalog(BUILTIN_CATALOG), "builtin"
    env_url = env_url.rstrip("/")
    for path in ("/catalog", "/providers", "/v1/models", "/models",
                 "/env", "/env/catalog", "/"):
        data = fetch_json(env_url + path)
        if data is None:
            continue
        try:
            cat = canonical_catalog(data)
        except Exception:
            continue
        if cat["providers"] and any(p["models"] for p in cat["providers"]):
            return cat, path
    return canonical_catalog(BUILTIN_CATALOG), "builtin"


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text):
    """Deterministic token estimate: ~1 token per 4 chars, min 1 per part."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def text_of_messages(messages):
    parts = []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
    return "\n".join(parts)


def has_image(messages):
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in (
                        "image_url", "image", "input_image"):
                    return True
    return False


def request_requirements(wire, body):
    """What a request needs, derived only from the request itself."""
    req = {"tools": False, "vision": False, "json": False,
           "prompt_tokens": 0, "max_tokens": 0}
    if wire == "openai":
        req["tools"] = bool(body.get("tools"))
        req["vision"] = has_image(body.get("messages"))
        fmt = body.get("response_format") or {}
        req["json"] = fmt.get("type") in ("json_object", "json_schema")
        req["prompt_tokens"] = estimate_tokens(
            text_of_messages(body.get("messages")))
        req["max_tokens"] = int(body.get("max_tokens") or 0)
    else:  # anthropic
        req["tools"] = bool(body.get("tools"))
        msgs = body.get("messages") or []
        req["vision"] = has_image(msgs)
        if isinstance(body.get("system"), list):
            req["prompt_tokens"] += estimate_tokens(
                text_of_messages([{"content": body["system"]}]))
        req["prompt_tokens"] += estimate_tokens(text_of_messages(msgs))
        req["max_tokens"] = int(body.get("max_tokens") or 0)
    return req


def rank_models(catalog, req):
    """All FREE models, best first, honouring the request's needs.

    Hard filters first (tools / vision / json); long context prefers a
    model whose window fits prompt+max tokens, but if none fits the
    largest-window free model is still a legal pick (served beats
    unserved). A paid model is never returned. Ties break by model id,
    so the order is fully deterministic.
    """
    need = [c for c in ("tools", "vision", "json") if req.get(c)]
    total = req["prompt_tokens"] + req["max_tokens"]
    scored = []
    for p in catalog["providers"]:
        for m in p["models"]:
            if not m["free"]:
                continue
            caps = set(m["capabilities"])
            if any(c not in caps for c in need):
                continue
            fits = m["context_window"] >= max(total, 1)
            scored.append((p, m, fits))
    fitting = [s for s in scored if s[2]]
    pool = fitting if fitting else scored
    pool.sort(key=lambda s: (-s[1]["quality"], -s[1]["context_window"],
                             s[1]["id"], s[0]["id"]))
    return [(p, m) for p, m, _ in pool]

# ---------------------------------------------------------------------------
# Wire translation (OpenAI <-> Anthropic <-> Gemini)
# ---------------------------------------------------------------------------

def openai_messages_to_anthropic(messages):
    system_parts, anthropic_msgs = [], []
    for m in messages or []:
        role, content = m.get("role"), m.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str)
                                else text_of_messages([m]))
            continue
        blocks = []
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    blocks.append({"type": "text", "text": b.get("text", "")})
                elif t == "image_url":
                    url = (b.get("image_url") or {}).get("url", "")
                    mm = re.match(r"data:(image/\w+);base64,(.+)", url or "")
                    if mm:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": mm.group(1),
                            "data": mm.group(2)}})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            blocks.append({"type": "tool_use", "id": tc.get("id", "call-1"),
                           "name": fn.get("name", ""),
                           "input": json.loads(fn.get("arguments") or "{}")})
        if role == "tool":
            blocks.append({"type": "tool_result",
                           "tool_use_id": m.get("tool_call_id", ""),
                           "content": content if isinstance(content, str)
                           else text_of_messages([m])})
            anthropic_msgs.append({"role": "user", "content": blocks})
        else:
            anthropic_msgs.append({
                "role": "user" if role == "user" else "assistant",
                "content": blocks or [{"type": "text", "text": ""}]})
    return ("\n".join(system_parts) if system_parts else None,
            anthropic_msgs)


def anthropic_messages_to_openai(body):
    msgs = []
    system = body.get("system")
    if system:
        text = system if isinstance(system, str) else text_of_messages(
            [{"content": system}])
        msgs.append({"role": "system", "content": text})
    for m in body.get("messages") or []:
        content, tool_calls, tool_results = [], [], []
        for b in m.get("content") if isinstance(m.get("content"), list) else []:
            t = b.get("type")
            if t == "text":
                content.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    content.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (
                            src.get("media_type", "image/png"),
                            src.get("data", ""))}})
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id", "call-1"),
                                   "type": "function",
                                   "function": {"name": b.get("name", ""),
                                                "arguments": json.dumps(
                                                    b.get("input") or {})}})
            elif t == "tool_result":
                tool_results.append((b.get("tool_use_id", ""),
                                     b.get("content", "")))
        role = m.get("role")
        if tool_results:
            for tid, c in tool_results:
                msgs.append({"role": "tool", "tool_call_id": tid,
                             "content": c if isinstance(c, str) else str(c)})
        else:
            msg = {"role": "assistant" if role == "assistant" else "user"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = None
            elif len(content) == 1 and content[0]["type"] == "text":
                msg["content"] = content[0]["text"]
            else:
                msg["content"] = content or ""
            msgs.append(msg)
    return msgs


def openai_tools_to_anthropic(tools):
    out = []
    for t in tools or []:
        fn = t.get("function") or t
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}}})
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for t in tools or []:
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema")
            or {"type": "object", "properties": {}}}})
    return out


def messages_to_gemini_contents(messages):
    contents = []
    for m in messages or []:
        role = "model" if m.get("role") == "assistant" else "user"
        parts, content = [], m.get("content")
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append({"text": b.get("text", "")})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            parts.append({"functionCall": {"name": fn.get("name", ""),
                                           "args": args}})
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def build_provider_request(provider_wire, client_wire, body, model_id):
    """Translate the client request into the provider's wire.

    Returns (path, payload, stream_flag).
    """
    stream = bool(body.get("stream"))
    if provider_wire == "openai":
        if client_wire == "openai":
            payload = dict(body)
            payload["model"] = model_id
            return "/v1/chat/completions", payload, stream
        msgs = anthropic_messages_to_openai(body)
        payload = {"model": model_id, "messages": msgs,
                   "max_tokens": body.get("max_tokens"),
                   "temperature": body.get("temperature"),
                   "stream": stream}
        if body.get("tools"):
            payload["tools"] = anthropic_tools_to_openai(body["tools"])
        return "/v1/chat/completions", payload, stream
    if provider_wire == "anthropic":
        if client_wire == "anthropic":
            payload = dict(body)
            payload["model"] = model_id
            return "/v1/messages", payload, stream
        system, msgs = openai_messages_to_anthropic(body.get("messages"))
        payload = {"model": model_id, "messages": msgs,
                   "max_tokens": body.get("max_tokens") or 1024,
                   "stream": stream}
        if system:
            payload["system"] = system
        if body.get("temperature") is not None:
            payload["temperature"] = body["temperature"]
        if body.get("tools"):
            payload["tools"] = openai_tools_to_anthropic(body["tools"])
        return "/v1/messages", payload, stream
    # gemini
    if client_wire == "openai":
        contents = messages_to_gemini_contents(body.get("messages"))
        system_text = None
        for m in body.get("messages") or []:
            if m.get("role") == "system":
                system_text = m.get("content") if isinstance(
                    m.get("content"), str) else text_of_messages([m])
        payload = {"contents": contents}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    else:
        contents = messages_to_gemini_contents(
            anthropic_messages_to_openai(body))
        payload = {"contents": contents}
    cfg = {}
    if body.get("max_tokens"):
        cfg["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        cfg["temperature"] = body["temperature"]
    fmt = (body.get("response_format") or {}).get("type")
    if fmt in ("json_object", "json_schema"):
        cfg["responseMimeType"] = "application/json"
    if cfg:
        payload["generationConfig"] = cfg
    if body.get("tools"):
        tools = (body["tools"] if client_wire == "openai"
                 else anthropic_tools_to_openai(body["tools"]))
        decls = []
        for t in tools:
            fn = t.get("function") or t
            decls.append({"name": fn.get("name", ""),
                          "description": fn.get("description", ""),
                          "parameters": fn.get("parameters")
                          or {"type": "object", "properties": {}}})
        payload["tools"] = [{"functionDeclarations": decls}]
    name = model_id.split("/")[-1]
    return "/v1beta/models/%s:generateContent" % name, payload, stream


def extract_text_toolcalls(provider_wire, resp):
    """Normalize any provider response to (text, tool_calls, finish).

    tool_calls: [{"id","name","arguments"(dict)}]; finish: stop/tool_calls.
    """
    try:
        if provider_wire == "openai":
            ch = (resp.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(b.get("text", "") for b in text
                               if isinstance(b, dict))
            tcs = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                tcs.append({"id": tc.get("id", "call-1"),
                            "name": fn.get("name", ""), "arguments": args})
            finish = ch.get("finish_reason") or (
                "tool_calls" if tcs else "stop")
            return text, tcs, finish
        if provider_wire == "anthropic":
            text, tcs = "", []
            for b in resp.get("content") or []:
                if b.get("type") == "text":
                    text += b.get("text", "")
                elif b.get("type") == "tool_use":
                    tcs.append({"id": b.get("id", "call-1"),
                                "name": b.get("name", ""),
                                "arguments": b.get("input") or {}})
            stop = resp.get("stop_reason") or "end_turn"
            return text, tcs, ("tool_calls" if tcs else
                               ("stop" if stop == "end_turn" else stop))
        # gemini
        cand = (resp.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text, tcs = "", []
        for part in parts:
            if "text" in part:
                text += part["text"]
            if "functionCall" in part:
                fc = part["functionCall"]
                tcs.append({"id": "call-1", "name": fc.get("name", ""),
                            "arguments": fc.get("args") or {}})
        reason = cand.get("finishReason") or "STOP"
        return text, tcs, ("tool_calls" if tcs else "stop")
    except Exception:
        return "", [], "stop"


# ---------------------------------------------------------------------------
# Router state (in-memory, deterministic export)
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, catalog, env_url, state_path):
        self.catalog = catalog
        self.env_url = (env_url or "").rstrip("/")
        self.state_path = state_path
        self.lock = threading.Lock()
        self.provider_locks = {p["id"]: threading.Lock()
                               for p in catalog["providers"]}
        self.exhausted = set()   # provider ids that returned 402
        self.rate_limited = set()  # provider ids that returned 429
        self.seq = 0
        self.requests_log = []
        self.provider_calls = 0
        self.done = False
        self.exit_scheduled = False

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def provider_url(self, provider, path):
        base = provider["base_path"] or ""
        if base.startswith("http"):
            return base.rstrip("/") + path
        if self.env_url:
            if base:
                return self.env_url + base.rstrip("/") + path
            return self.env_url + path
        # standalone: same process cannot serve providers; the fake env
        # in tests sets MAKEFUN_ENV_URL, so this is only a fallback that
        # answers locally (see LocalProvider below).
        return "local://" + provider["id"] + path

    def provider_headers(self, provider):
        h = {"Content-Type": "application/json"}
        hdr, val = provider["auth_header"], provider["auth_value"]
        if hdr and val:
            h[hdr] = ((provider["auth_scheme"] + " ") if provider["auth_scheme"]
                      else "") + val
        if provider["wire"] == "anthropic":
            h.setdefault("anthropic-version", "2023-06-01")
        return h

    def call_provider(self, provider, model, path, payload):
        """One provider call. Returns (status, parsed_json_or_None, raw).

        A 402 marks the provider exhausted for the rest of the trial,
        a 429 marks it rate-limited (skip it, do not sleep on it).
        """
        url = self.provider_url(provider, path)
        data = json.dumps(payload).encode()
        if url.startswith("local://"):
            status, resp = local_provider_answer(provider, model, payload)
            if status == 402:
                self.exhausted.add(provider["id"])
            elif status == 429:
                self.rate_limited.add(provider["id"])
            return status, resp, ""
        req = urllib.request.Request(url, data=data,
                                     headers=self.provider_headers(provider),
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
                try:
                    return r.status, json.loads(raw), raw
                except ValueError:
                    return r.status, None, raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if e.code == 402:
                self.exhausted.add(provider["id"])
            elif e.code == 429:
                self.rate_limited.add(provider["id"])
            return e.code, parsed, raw
        except Exception as e:
            return 0, None, str(e)

    def route(self, client_wire, body):
        """Try candidates in ranked order until one answers.

        Returns (result_dict, attempts) where result_dict has
        provider/model/text/tool_calls/finish/status, or an error entry.
        """
        req = request_requirements(client_wire, body)
        candidates = rank_models(self.catalog, req)
        attempts = []
        for provider, model in candidates:
            if provider["id"] in self.exhausted:
                attempts.append({"provider": provider["id"],
                                 "model": model["id"],
                                 "status": "skipped_exhausted"})
                continue
            if provider["id"] in self.rate_limited:
                attempts.append({"provider": provider["id"],
                                 "model": model["id"],
                                 "status": "skipped_rate_limited"})
                continue
            path, payload, _stream = build_provider_request(
                provider["wire"], client_wire, body, model["id"])
            lock = self.provider_locks.setdefault(
                provider["id"], threading.Lock())
            with lock:  # one call at a time per provider
                with self.lock:
                    self.provider_calls += 1
                status, resp, _raw = self.call_provider(
                    provider, model, path, payload)
            attempts.append({"provider": provider["id"],
                             "model": model["id"], "status": status})
            if status == 200 and isinstance(resp, dict):
                text, tcs, finish = extract_text_toolcalls(
                    provider["wire"], resp)
                return {"provider": provider["id"], "model": model["id"],
                        "wire": provider["wire"], "text": text,
                        "tool_calls": tcs, "finish": finish,
                        "status": 200}, attempts
            # 402 / 429 / 5xx / transport error: fail over to next candidate
        return {"error": "all_providers_failed", "status": 502}, attempts

    def log_request(self, entry):
        with self.lock:
            self.requests_log.append(entry)

    def state_export(self):
        with self.lock:
            return {"version": 1,
                    "catalog_sha256": hashlib.sha256(
                        catalog_bytes(self.catalog)).hexdigest(),
                    "requests_served": len(self.requests_log),
                    "provider_calls": self.provider_calls,
                    "requests": list(self.requests_log)}

    def write_state(self):
        try:
            data = json.dumps(self.state_export(), sort_keys=True,
                              indent=2) + "\n"
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self.state_path)
        except Exception:
            pass

    def check_work_done(self):
        """Ask the env whether the trial is done; True means exit 0."""
        if not self.env_url:
            return len(self.requests_log) >= 20
        data = fetch_json(self.env_url + "/work", timeout=5)
        if isinstance(data, dict):
            return bool(data.get("done") or data.get("finished"))
        return False

    def maybe_exit(self):
        if os.environ.get("MAKEFUN_NO_AUTO_EXIT") == "1":
            return
        # Exit 0 only once this server has served the trial's 20 requests
        # AND the env's /work says done. The request-count guard matters:
        # provider-call counts can reach the env's done threshold early
        # (failover retries), and stray extra traffic must never cut the
        # trial short.
        if self.exit_scheduled or len(self.requests_log) < 20:
            return
        try:
            if self.check_work_done():
                self.done = True
                self.exit_scheduled = True
                self.write_state()
                # Give the in-flight response a moment to flush, then
                # exit 0 as the bounty requires.
                threading.Timer(1.0, lambda: os._exit(0)).start()
        except Exception:
            pass


def local_provider_answer(provider, model, payload):
    """Standalone fallback: answer from the request itself, no network.

    Used only when no env is configured at all. The text echoes the
    prompt deterministically so local runs are still verifiable.
    """
    prompt = ""
    if "messages" in payload:
        prompt = text_of_messages(payload.get("messages"))
    elif "contents" in payload:
        prompt = " ".join(
            p.get("text", "") for c in payload.get("contents") or []
            for p in c.get("parts") or [])
    # Simulated refusals (standalone mode): a prompt can force a provider
    # to refuse, so failover is exercisable without a sealed env, e.g.
    # "FORCE-429:anthropic-free" or "FORCE-402:openrouter-free".
    for marker, status in (("FORCE-429:", 429), ("FORCE-402:", 402)):
        if marker + provider["id"] in prompt:
            return status, {"error": {"message":
                "simulated provider refusal (%s)" % provider["id"],
                "type": "provider_refused",
                "code": "rate_limited" if status == 429 else "exhausted"}}
    if prompt.startswith("FORCE-402:") or prompt.startswith("FORCE-429:"):
        pass  # marker for another provider: this one still answers
    text = ("[%s via %s] %s" % (model["id"], provider["id"], prompt))[:500]
    # Deterministic tool call: if the caller asked for tools, the simulated
    # provider calls the first one with empty arguments.
    tool_name = None
    tools = payload.get("tools") or []
    if tools:
        first = tools[0] or {}
        tool_name = (first.get("function") or first).get("name") or ""
    if provider["wire"] == "openai":
        msg = {"role": "assistant", "content": text}
        finish = "stop"
        if tool_name:
            msg["tool_calls"] = [{"id": "call-1", "type": "function",
                                  "function": {"name": tool_name,
                                               "arguments": "{}"}}]
            finish = "tool_calls"
        return 200, {"id": "chatcmpl-local", "object": "chat.completion",
                     "choices": [{"index": 0, "message": msg,
                                 "finish_reason": finish}],
                     "usage": {"prompt_tokens": estimate_tokens(prompt),
                               "completion_tokens": estimate_tokens(text),
                               "total_tokens": estimate_tokens(prompt)
                               + estimate_tokens(text)}}
    if provider["wire"] == "anthropic":
        content = [{"type": "text", "text": text}]
        stop = "end_turn"
        if tool_name:
            content.append({"type": "tool_use", "id": "call-1",
                            "name": tool_name, "input": {}})
            stop = "tool_use"
        return 200, {"id": "msg-local", "type": "message",
                     "role": "assistant", "model": model["id"],
                     "content": content,
                     "stop_reason": stop,
                     "usage": {"input_tokens": estimate_tokens(prompt),
                               "output_tokens": estimate_tokens(text)}}
    parts = [{"text": text}]
    if tool_name:
        parts.append({"functionCall": {"name": tool_name, "args": {}}})
    return 200, {"candidates": [{"content": {"parts": parts},
                                 "finishReason": "STOP"}]}

def _invalid_reason(body):
    """Why a decoded request body cannot be routed, or None if it can.

    The router assumes the shapes below; anything else used to raise
    inside the handler (AttributeError / ValueError) and drop the
    connection without any HTTP response. Reject it with a 400 instead.
    """
    if not isinstance(body, dict):
        return "request body must be a JSON object"
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(
            isinstance(m, dict) for m in messages):
        return "messages is required and must be a list of objects"
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            int(max_tokens)
        except (TypeError, ValueError):
            return "max_tokens must be a number"
    fmt = body.get("response_format")
    if fmt is not None and not isinstance(fmt, dict):
        return "response_format must be an object"
    tools = body.get("tools")
    if tools is not None and (not isinstance(tools, list) or not all(
            isinstance(t, dict) for t in tools)):
        return "tools must be a list of objects"
    system = body.get("system")
    if system is not None and not isinstance(system, (str, list)):
        return "system must be a string or a list of blocks"
    if isinstance(system, list) and not all(
            isinstance(b, dict) for b in system):
        return "system must be a string or a list of blocks"
    return None


# ---------------------------------------------------------------------------
# HTTP server: OpenAI + Anthropic APIs, JSON and streaming
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    router = None  # set in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # keep stdout clean; the trial's log is the state export

    # -- helpers ------------------------------------------------------------
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drain_oversized_body(self, length):
        """Discard an oversized request body, bounded in time and memory.

        A 413 sent while the client is still uploading is usually lost
        (the server closes a socket with unread data, the kernel
        answers RST, and the client never sees the response). So drain
        what is already arriving -- in small chunks, never buffered --
        and stop after a short time budget, so a trickling client can
        hold this thread for ~2s at most. Returns when the body is
        fully drained, the client goes away, or the budget runs out.
        """
        deadline = time.monotonic() + 2.0
        remaining = length
        conn = self.connection
        old_timeout = conn.gettimeout()
        try:
            while remaining > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                conn.settimeout(min(left, 0.5))
                try:
                    data = self.rfile.read(min(65536, remaining))
                except (OSError, ValueError):
                    break
                if not data:
                    break
                remaining -= len(data)
        finally:
            try:
                conn.settimeout(old_timeout)
            except OSError:
                pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length < 0:
            return None
        if length > MAX_REQUEST_BYTES:
            # Reject before buffering anything, but drain what the
            # client is already sending (bounded: 64 KB chunks, ~2s)
            # so the 413 below is actually delivered instead of being
            # lost to a TCP reset on a half-uploaded body. A huge
            # Content-Length with a slow or absent body would otherwise
            # park this handler thread in rfile.read() indefinitely.
            self._drain_oversized_body(length)
            self.close_connection = True
            raise _RequestTooLarge()
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            # ValueError covers bad JSON / bad UTF-8; RecursionError
            # covers pathologically nested JSON (json.loads recurses).
            return None

    def _is_anthropic_wire(self):
        # GET /v1/models answers in the wire its auth header names:
        # an Anthropic client sends x-api-key + anthropic-version.
        return (self.headers.get("anthropic-version") is not None
                or (self.headers.get("x-api-key") is not None
                    and self.headers.get("Authorization") is None))

    # -- models -------------------------------------------------------------
    def _models(self):
        router = self.router
        models = []
        for p in router.catalog["providers"]:
            for m in p["models"]:
                models.append({"provider": p["id"], "model": m})
        if self._is_anthropic_wire():
            data = []
            for p in router.catalog["providers"]:
                for m in p["models"]:
                    data.append({"id": m["id"], "type": "model",
                                 "display_name": m["id"],
                                 "created_at": "2026-01-01T00:00:00Z",
                                 "owned_by": p["id"]})
            self._send_json(200, {"data": data, "has_more": False,
                                  "first_id": data[0]["id"] if data else None,
                                  "last_id": data[-1]["id"] if data else None})
            return
        data = [{"id": m["id"], "object": "model", "created": 0,
                 "owned_by": p["id"]}
                for p in router.catalog["providers"] for m in p["models"]]
        self._send_json(200, {"object": "list", "data": data})

    # -- OpenAI chat completions -------------------------------------------
    def _openai_chat(self, body):
        router = self.router
        reason = _invalid_reason(body)
        if reason is not None:
            self._send_json(400, {"error": {
                "message": reason,
                "type": "invalid_request_error", "code": "invalid_messages"}})
            return
        seq = router.next_seq()
        try:
            result, attempts = router.route("openai", body)
        except (AttributeError, TypeError, ValueError, KeyError):
            # A request field with a shape the router did not expect
            # (e.g. tool_calls arguments as an object instead of a
            # string). Answer 400; never drop the connection.
            self._send_json(400, {"error": {
                "message": "request fields have an invalid shape",
                "type": "invalid_request_error",
                "code": "invalid_request"}})
            return
        served_model = body.get("model") or ""
        log = {"seq": seq, "wire": "openai",
               "model_requested": served_model,
               "model_served": result.get("model", ""),
               "provider": result.get("provider", ""),
               "attempts": attempts,
               "status": result.get("status", 502),
               "finish_reason": result.get("finish", "")}
        router.log_request(log)
        router.write_state()
        if "error" in result:
            self._send_json(502, {"error": {
                "message": "all providers failed for this request",
                "type": "server_error", "code": "all_providers_failed"}})
            router.maybe_exit()
            return
        prompt_tokens = estimate_tokens(
            text_of_messages(body.get("messages")))
        out_tokens = estimate_tokens(result["text"]) + sum(
            estimate_tokens(json.dumps(tc["arguments"], sort_keys=True))
            for tc in result["tool_calls"])
        msg = {"role": "assistant", "content": result["text"] or None}
        if result["tool_calls"]:
            msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["arguments"],
                                                      sort_keys=True)}}
                for tc in result["tool_calls"]]
            if result["text"] == "":
                msg["content"] = None
        finish = ("tool_calls" if result["tool_calls"]
                  else result["finish"])
        if body.get("stream"):
            self._openai_stream(seq, result, msg, finish,
                                prompt_tokens, out_tokens)
        else:
            self._send_json(200, {
                "id": "chatcmpl-%04d" % seq, "object": "chat.completion",
                "created": 0, "model": result["model"],
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": out_tokens,
                          "total_tokens": prompt_tokens + out_tokens}})
        router.maybe_exit()

    def _openai_stream(self, seq, result, msg, finish,
                       prompt_tokens, out_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        base = {"id": "chatcmpl-%04d" % seq, "object": "chat.completion.chunk",
                "created": 0, "model": result["model"]}
        text = result["text"] or ""
        chunk_size = 40
        first = True
        for i in range(0, max(len(text), 1), chunk_size):
            part = text[i:i + chunk_size] if text else ""
            delta = {"role": "assistant"} if first else {}
            if part:
                delta["content"] = part
            first = False
            chunk = dict(base, choices=[{"index": 0, "delta": delta,
                                         "finish_reason": None}])
            self.wfile.write(("data: %s\n\n" % json.dumps(chunk)).encode())
        if result["tool_calls"]:
            for tc in result["tool_calls"]:
                delta = {"tool_calls": [{
                    "index": 0, "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"],
                                 "arguments": json.dumps(tc["arguments"],
                                                         sort_keys=True)}}]}
                chunk = dict(base, choices=[{"index": 0, "delta": delta,
                                             "finish_reason": None}])
                self.wfile.write(("data: %s\n\n" % json.dumps(chunk)).encode())
        chunk = dict(base, choices=[{"index": 0, "delta": {},
                                     "finish_reason": finish}])
        self.wfile.write(("data: %s\n\n" % json.dumps(chunk)).encode())
        self.wfile.write(b"data: [DONE]\n\n")

    # -- Anthropic messages --------------------------------------------------
    def _anthropic_messages(self, body):
        router = self.router
        reason = _invalid_reason(body)
        if reason is not None:
            self._send_json(400, {"error": {
                "type": "invalid_request_error",
                "message": reason}})
            return
        seq = router.next_seq()
        try:
            result, attempts = router.route("anthropic", body)
        except (AttributeError, TypeError, ValueError, KeyError):
            self._send_json(400, {"error": {
                "type": "invalid_request_error",
                "message": "request fields have an invalid shape"}})
            return
        log = {"seq": seq, "wire": "anthropic",
               "model_requested": body.get("model") or "",
               "model_served": result.get("model", ""),
               "provider": result.get("provider", ""),
               "attempts": attempts,
               "status": result.get("status", 502),
               "finish_reason": result.get("finish", "")}
        router.log_request(log)
        router.write_state()
        if "error" in result:
            self._send_json(502, {"error": {
                "type": "api_error",
                "message": "all providers failed for this request"}})
            router.maybe_exit()
            return
        prompt_tokens = estimate_tokens(
            text_of_messages(body.get("messages")))
        out_tokens = estimate_tokens(result["text"]) + sum(
            estimate_tokens(json.dumps(tc["arguments"], sort_keys=True))
            for tc in result["tool_calls"])
        content = []
        if result["text"]:
            content.append({"type": "text", "text": result["text"]})
        for tc in result["tool_calls"]:
            content.append({"type": "tool_use", "id": tc["id"],
                            "name": tc["name"], "input": tc["arguments"]})
        stop = "tool_use" if result["tool_calls"] else "end_turn"
        if body.get("stream"):
            self._anthropic_stream(seq, result, content, stop,
                                   prompt_tokens, out_tokens)
        else:
            self._send_json(200, {
                "id": "msg-%04d" % seq, "type": "message",
                "role": "assistant", "model": result["model"],
                "content": content, "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens,
                          "output_tokens": out_tokens}})
        router.maybe_exit()

    def _anthropic_stream(self, seq, result, content, stop,
                          prompt_tokens, out_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

        def ev(event, data):
            self.wfile.write(("event: %s\ndata: %s\n\n" % (
                event, json.dumps(data))).encode())

        text = result["text"] or ""
        ev("message_start", {"type": "message_start", "message": {
            "id": "msg-%04d" % seq, "type": "message", "role": "assistant",
            "model": result["model"], "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens,
                      "output_tokens": 0}}})
        idx = 0
        if text:
            ev("content_block_start", {"type": "content_block_start",
                                       "index": 0, "content_block": {
                                           "type": "text", "text": ""}})
            for i in range(0, len(text), 40):
                ev("content_block_delta", {"type": "content_block_delta",
                                           "index": 0, "delta": {
                                               "type": "text_delta",
                                               "text": text[i:i + 40]}})
            ev("content_block_stop", {"type": "content_block_stop",
                                      "index": 0})
            idx = 1
        for tc in result["tool_calls"]:
            ev("content_block_start", {"type": "content_block_start",
                                       "index": idx, "content_block": {
                                           "type": "tool_use",
                                           "id": tc["id"],
                                           "name": tc["name"], "input": {}}})
            ev("content_block_delta", {"type": "content_block_delta",
                                       "index": idx, "delta": {
                                           "type": "input_json_delta",
                                           "partial_json": json.dumps(
                                               tc["arguments"],
                                               sort_keys=True)}})
            ev("content_block_stop", {"type": "content_block_stop",
                                      "index": idx})
            idx += 1
        ev("message_delta", {"type": "message_delta", "delta": {
            "stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": out_tokens}})
        ev("message_stop", {"type": "message_stop"})

    # -- routing ---------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/v1/models", "/models"):
            self._models()
        elif path == "/health":
            self._send_json(200, {"status": "ok",
                                  "version": VERSION,
                                  "requests_served": len(
                                      self.router.requests_log)})
        elif path == "/state":
            self._send_json(200, self.router.state_export())
        elif path == "/work":
            done = self.router.check_work_done()
            self._send_json(200, {"done": done,
                                  "requests_served": len(
                                      self.router.requests_log)})
            if done:
                self.router.maybe_exit()
        elif path in ("/", "/catalog"):
            # Exactly the bytes --list prints: the registered catalog.
            body = catalog_bytes(self.router.catalog)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            try:
                body = self._read_json()
            except _RequestTooLarge:
                self._send_json(413, {"error": {
                    "message": "request body exceeds %d bytes"
                               % MAX_REQUEST_BYTES,
                    "type": "invalid_request_error",
                    "code": "request_too_large"}})
                return
            if body is None:
                self._send_json(400, {"error": {
                    "message": "invalid JSON",
                    "type": "invalid_request_error"}})
                return
            self._openai_chat(body)
        elif path == "/v1/messages":
            try:
                body = self._read_json()
            except _RequestTooLarge:
                self._send_json(413, {"error": {
                    "type": "invalid_request_error",
                    "message": "request body exceeds %d bytes"
                               % MAX_REQUEST_BYTES}})
                return
            if body is None:
                self._send_json(400, {"error": {
                    "type": "invalid_request_error",
                    "message": "invalid JSON"}})
                return
            self._anthropic_messages(body)
        elif path == "/v1/messages/count_tokens":
            try:
                body = self._read_json() or {}
            except _RequestTooLarge:
                self._send_json(413, {"error": {
                    "type": "invalid_request_error",
                    "message": "request body exceeds %d bytes"
                               % MAX_REQUEST_BYTES}})
                return
            tokens = estimate_tokens(text_of_messages(
                body.get("messages"))) + (
                    estimate_tokens(body.get("system") or "")
                    if isinstance(body.get("system"), str) else 0)
            self._send_json(200, {"input_tokens": tokens})
        else:
            self._send_json(404, {"error": {"message": "not found",
                                            "type": "not_found"}})


def work_watcher(router, interval=0.5):
    """Poll the env's /work; when it says done (and this server has
    served its 20 requests), exit 0. The 1s grace lets the last
    response finish flushing before the process exits."""
    while True:
        time.sleep(interval)
        try:
            if (len(router.requests_log) >= 20
                    and router.check_work_done()):
                router.done = True
                time.sleep(1.0)
                router.write_state()
                os._exit(0)
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Free-model aggregator server")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--list", action="store_true",
                    help="print the registered catalog document and exit")
    ap.add_argument("--env", default=None,
                    help="env base URL (overrides MAKEFUN_ENV_URL)")
    ap.add_argument("--state", default=None,
                    help="state export path (overrides MAKEFUN_STATE_PATH)")
    args = ap.parse_args(argv)

    env_url = args.env or os.environ.get("MAKEFUN_ENV_URL") or ""
    state_path = (args.state or os.environ.get("MAKEFUN_STATE_PATH")
                  or os.path.join(os.getcwd(), "state.json"))
    catalog, source = discover_catalog(env_url)
    if args.list:
        sys.stdout.buffer.write(catalog_bytes(catalog))
        return 0

    port = args.port or int(os.environ.get("PORT", "8080"))
    router = Router(catalog, env_url, state_path)
    Handler.router = router
    print("free-model-aggregator listening on %s:%d (catalog: %s, "
          "env: %s)" % (args.host, port, source,
                        env_url or "builtin-simulated"), flush=True)
    if env_url:
        threading.Thread(target=work_watcher, args=(router,),
                         daemon=True).start()
    server = ThreadingHTTPServer((args.host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    router.write_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
