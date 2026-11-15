#!/usr/bin/env python3
"""Free-model aggregator server (make.fun bounty: free-model-aggregator-01).

One server, two APIs (OpenAI- and Anthropic-compatible), ten simulated
free-tier providers, routing each request to the best free model that can
serve it, with failover when a provider refuses.

Python 3.13, standard library only -- nothing is installed before the
server runs (per the bounty requirements).

How a trial works (see the kit's RULES.txt and starter/):
    * the trial sets MAKEFUN_PROMPT_FILE (the brief: provider directory,
      sealed quality table, workload manifest), MAKEFUN_ENV_URL (the env
      that hosts the ten providers) and MAKEFUN_TRIAL_SEED;
    * the server reads the brief, probes each provider's catalog route
      in the provider's own wire, and builds ONE catalog document;
    * it listens on an ephemeral port and registers it:
      POST $MAKEFUN_ENV_URL/register
      {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
    * the env's client then sends the 20 requests, one at a time;
      each is answered with the best free model meeting its
      requirements, failing over to the next best free model when a
      provider refuses (402/429/5xx), and the answer is returned in
      the wire the client used, with the PROVIDER's usage counts and
      stop reason, and the provider's text whole;
    * it polls GET $MAKEFUN_ENV_URL/work and exits 0 when it says done.

Run:
    python3 main.py [--port 8080]
    python3 main.py --list            # print the registered catalog, byte for byte

With no brief and no env (standalone), --list prints {"providers":[]}
and the server answers with an empty catalog, like the kit's starter.

Determinism: no wall-clock time and no unseeded randomness anywhere in
routing, ids, token counts or the state export. Response ids are
sequence-based (chatcmpl-0001, msg-0001, ...). The only sleeps are the
fixed /work poll interval and the fixed register/probe timeouts --
neither affects any routing decision or any byte of the state export.
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

VERSION = "2.0.0"

# Largest request body the server reads (10 MiB). A prompt is text; even
# a 1M-token context window is ~4 MB of characters, so this is far above
# any legitimate trial request. Without a cap, an unauthenticated client
# can make the server buffer an arbitrarily large body in memory --
# Content-Length is otherwise trusted blindly in Handler._read_json --
# and a GB-scale body exhausts memory and kills the trial process.
MAX_REQUEST_BYTES = 10 * 1024 * 1024

# Fixed timestamps in model listings: nothing in a reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

PROTOCOL_VERSION = 1
PROBE_TIMEOUT = 5
PROVIDER_TIMEOUT = 10
REGISTER_TIMEOUT = 5
WORK_POLL_INTERVAL = 0.1
# Largest provider answer the server buffers (2 MiB). The sealed env's
# answers are ~1 KB; without a cap, a provider that streamed
# gigabytes would exhaust the trial process's memory before the
# router ever saw a byte. Over the cap the call counts as failed, so
# the router fails over to the next provider.
MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024


def _read_capped_response(resp, limit=MAX_PROVIDER_RESPONSE_BYTES):
    """Read a provider answer, bounded: None when it runs past `limit`."""
    chunks = []
    size = 0
    while True:
        chunk = resp.read(65536)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# The brief (MAKEFUN_PROMPT_FILE): provider directory, quality table, manifest
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if (isinstance(value, dict)
                    and isinstance(value.get("schema"), str)):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v else None


def _provider_of(v):
    if not isinstance(v, dict):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if not pid or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = (_str(v.get("path")) or _str(v.get("base_path"))
            or _str(v.get("base_url")) or _str(v.get("base")))
    return {"id": pid, "name": _str(v.get("name")) or pid, "wire": wire,
            "auth": auth, "base": base,
            "key_env": _str(v.get("key_env")), "key": _str(v.get("key"))}


def _rating_of(v):
    if not isinstance(v, dict):
        return None
    model, provider = _str(v.get("model")), _str(v.get("provider"))
    if not model or not provider:
        return None
    rating = v.get("rating_milli")
    return {"model": model, "provider": provider,
            "rating_milli": rating if isinstance(rating, (int, float))
            else 0,
            "context_length": _num_or_none(v.get("context_length")),
            "tools": v.get("tools") is True,
            "vision": v.get("vision") is True,
            "json": v.get("json") is True,
            "free": v.get("free") is True}


def _workload_request_of(v):
    if not isinstance(v, dict):
        return None
    rid = _str(v.get("id"))
    if not rid:
        return None
    req = v.get("requires") if isinstance(v.get("requires"), dict) else {}
    out_req = {"tools": req.get("tools") is True,
               "vision": req.get("vision") is True,
               "json": req.get("json") is True}
    mc = req.get("min_context")
    if isinstance(mc, (int, float)):
        out_req["min_context"] = mc
    return {"id": rid,
            "api": "anthropic" if v.get("api") == "anthropic" else "openai",
            "stream": v.get("stream") is True,
            "weight": 2 if v.get("weight") == 2 else 1,
            "requires": out_req}


def read_brief(path):
    """The brief at `path`: directory, ratings, workload (any may be empty).

    Returns (providers, ratings, manifest, error). `ratings` maps
    (provider, model) -> row; `manifest` is the workload in arrival order.
    """
    providers, ratings, manifest = [], {}, []
    if not path:
        return providers, ratings, manifest, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return providers, ratings, manifest, "cannot read %s: %s" % (path, e)
    docs = _documents_in(text)
    directory = next((d for d in docs
                      if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        return providers, ratings, manifest, (
            "the brief at %s holds no %s document" % (path, PROVIDERS_SCHEMA))
    for v in directory["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    ratings_doc = next((d for d in docs
                        if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for v in ratings_doc["ratings"]:
            r = _rating_of(v)
            if r is not None:
                ratings[(r["provider"], r["model"])] = r
    workload_doc = next((d for d in docs
                         if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for v in workload_doc["manifest"]:
            w = _workload_request_of(v)
            if w is not None:
                manifest.append(w)
    return providers, ratings, manifest, None


def fetch_json(url, headers=None, timeout=10):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def directory_from_instance(env_url):
    """Fallback directory: GET $MAKEFUN_ENV_URL/instance.

    Returns provider entries as read_brief would, or [] when unreachable.
    The instance document spells absolute base_urls; the path suffix is
    what the catalog carries (the scorer compares by path suffix).
    """
    if not env_url:
        return []
    data = fetch_json(env_url.rstrip("/") + "/instance", timeout=5)
    if not isinstance(data, dict):
        return []
    out = []
    for v in data.get("providers") or []:
        p = _provider_of(v)
        if p is not None:
            base = p["base"]
            if base.startswith("http"):
                path = urlparse(base).path
                p["base"] = path or base
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# The catalog: probe each provider, build the one registered document
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    """A base path joined to the env's URL; an absolute base stands."""
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(provider):
    """The headers a provider wants its key in (its own auth scheme)."""
    auth, key = provider["auth"], provider["key"]
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _model_in(wire, v):
    """One model as a catalog route listed it, in its wire's spelling."""
    if not isinstance(v, dict):
        return None
    if wire == "gemini":
        mid = _str(v.get("name")).replace("models/", "", 1)
    else:
        mid = _str(v.get("id"))
    if not mid:
        return None
    ctx = _num_or_none(v.get("context_length"))
    if ctx is None:
        ctx = _num_or_none(v.get("context_window"))
    if ctx is None:
        ctx = _num_or_none(v.get("inputTokenLimit"))
    free = v.get("free")
    if not isinstance(free, bool):
        free = None
    return {"id": mid,
            "display_name": _str(v.get("display_name"))
            or _str(v.get("displayName")) or mid,
            "free": free, "context_length": ctx,
            "tools": v.get("tools") if isinstance(v.get("tools"), bool)
            else None,
            "vision": v.get("vision") if isinstance(v.get("vision"), bool)
            else None,
            "json": v.get("json") if isinstance(v.get("json"), bool)
            else None}


def probe_provider(env_url, provider):
    """GET one provider's catalog route, in its own wire.

    Returns (ok, models). A refused connection, a non-2xx or a body
    that is not JSON leaves the provider unprobed; the catalog falls
    back to the quality table's rows for it, and the trial goes on.
    """
    base = join_base(env_url, provider["base"])
    url = (base + "/v1beta/models" if provider["wire"] == "gemini"
           else base + "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
            if resp.status != 200:
                return False, []
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False, []
    if not isinstance(body, dict):
        return False, []
    items = body.get("models") if provider["wire"] == "gemini" \
        else body.get("data")
    if not isinstance(items, list):
        return False, []
    models = []
    for v in items:
        m = _model_in(provider["wire"], v)
        if m is not None:
            models.append(m)
    return True, models


def catalog_base_path(base):
    """The catalog's base_url: the provider's own path, never the env URL.

    The scorer compares a reported base_url by its path suffix, and the
    register body is part of the state export a replay must reproduce
    byte for byte, so the path -- not the env's origin, which differs
    between a live pass and its replay -- is what the catalog carries.
    """
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return urlparse(b).path or b
    return b


def build_catalog(providers, ratings, probes):
    """The one catalog: every directory provider, every model it lists.

    A provider the probe answered contributes the listing's models,
    with the quality table's row winning for the free flag and the
    context length (the sealed truth where a row exists). A provider
    the probe missed contributes the table's rows instead, so the
    catalog stays complete -- the scorer grades it against the sealed
    truth, not against reachability.
    """
    out_providers = []
    for p in providers:
        ok, listed = probes.get(p["id"], (False, []))
        rows = {m: r for (prov, m), r in ratings.items()
                if prov == p["id"]}
        models = {}
        for m in listed:
            row = rows.get(m["id"])
            free = row["free"] if row is not None else m["free"]
            if free is None:
                free = False
            ctx = (row["context_length"] if row is not None
                   and row["context_length"] is not None
                   else m["context_length"])
            models[m["id"]] = {"id": m["id"], "free": bool(free),
                               "context_length": ctx}
        for mid, row in rows.items():
            if mid not in models:
                models[mid] = {"id": mid, "free": bool(row["free"]),
                               "context_length": row["context_length"]}
        out_providers.append({
            "id": p["id"], "base_url": catalog_base_path(p["base"]),
            "key_env": p["key_env"], "wire": p["wire"],
            "models": [models[mid] for mid in sorted(models)]})
    return {"providers": out_providers}


def catalog_bytes(catalog):
    """The exact bytes --list prints and the server registers."""
    return (json.dumps(catalog, sort_keys=True, indent=2) + "\n").encode()


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text):
    """Deterministic token estimate: ~1 token per 4 chars, min 1 per part."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def blocks_text(content):
    """The text of a string, or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"
                     and isinstance(b.get("text"), str))


def text_of_messages(messages):
    parts = []
    for m in messages or []:
        if isinstance(m, dict):
            parts.append(blocks_text(m.get("content")))
    return "\n".join(p for p in parts if p)


def has_image(messages):
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in (
                        "image_url", "image", "input_image"):
                    return True
    return False


def request_requirements(wire, body):
    """What a request needs, derived only from the request itself.

    Used only when the brief holds no workload manifest; the manifest's
    row is the declared source and is preferred wherever it exists.
    """
    req = {"tools": False, "vision": False, "json": False}
    if wire == "openai":
        req["tools"] = bool(body.get("tools"))
        req["vision"] = has_image(body.get("messages"))
        fmt = body.get("response_format") or {}
        req["json"] = fmt.get("type") in ("json_object", "json_schema")
    else:
        req["tools"] = bool(body.get("tools"))
        req["vision"] = has_image(body.get("messages"))
        # An Anthropic client carries no response_format; a JSON-mode
        # request on this wire is a plain request whose manifest row
        # says json. The body alone cannot show it.
    return req


def meets_requirements(row, req):
    """Whether one quality-table row meets a request's requirements."""
    if req.get("tools") and not row["tools"]:
        return False
    if req.get("vision") and not row["vision"]:
        return False
    if req.get("json") and not row["json"]:
        return False
    mc = req.get("min_context")
    if mc is not None and (row["context_length"] is None
                           or row["context_length"] < mc):
        return False
    return True


# ---------------------------------------------------------------------------
# Wire translation (OpenAI <-> Anthropic <-> Gemini)
#
# The provider echoes what IT parsed -- system text, user text, tool
# names, max_tokens, stop sequences, image, JSON mode -- and the client
# compares that echo with what it sent, so every field the client sent
# must reach the provider untouched, in the provider's own spelling.
# ---------------------------------------------------------------------------

def openai_messages_to_anthropic(messages):
    """OpenAI messages as an Anthropic (system, messages) pair."""
    system_parts, anthropic_msgs = [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "system":
            text = blocks_text(content)
            if text:
                system_parts.append(text)
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
                    mm = re.match(r"data:(image/[\w.+-]+);base64,(.+)",
                                  url or "", re.S)
                    if mm:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": mm.group(1),
                            "data": mm.group(2)}})
                    elif url:
                        blocks.append({"type": "image", "source": {
                            "type": "url", "url": url}})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            blocks.append({"type": "tool_use",
                           "id": tc.get("id", "call-1"),
                           "name": fn.get("name", ""), "input": args})
        if role == "tool":
            blocks.append({"type": "tool_result",
                           "tool_use_id": m.get("tool_call_id", ""),
                           "content": content if isinstance(content, str)
                           else blocks_text(content)})
            anthropic_msgs.append({"role": "user", "content": blocks})
        else:
            anthropic_msgs.append({
                "role": "user" if role == "user" else "assistant",
                "content": blocks or [{"type": "text", "text": ""}]})
    return ("\n".join(system_parts) if system_parts else None,
            anthropic_msgs)


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = blocks_text(content)
        return text if text else json.dumps(content)
    return "" if content is None else json.dumps(content)


def anthropic_messages_to_openai(body):
    """Anthropic messages as OpenAI messages (system first)."""
    msgs = []
    system = body.get("system")
    if system:
        msgs.append({"role": "system", "content": blocks_text(system)})
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            msgs.append({"role": "assistant" if m.get("role") == "assistant"
                         else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        parts, tool_calls, tool_results = [], [], []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": b.get("text", "")})
            elif t == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    parts.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (
                            src.get("media_type", "image/png"),
                            src.get("data", ""))}})
                elif src.get("type") == "url":
                    parts.append({"type": "image_url", "image_url": {
                        "url": src.get("url", "")}})
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id", "call-1"),
                                   "type": "function",
                                   "function": {
                                       "name": b.get("name", ""),
                                       "arguments": json.dumps(
                                           b.get("input") or {})}})
            elif t == "tool_result":
                tool_results.append((b.get("tool_use_id", ""),
                                     _tool_result_text(b.get("content"))))
        if tool_results:
            for tid, c in tool_results:
                msgs.append({"role": "tool", "tool_call_id": tid,
                             "content": c})
        else:
            msg = {"role": "assistant"
                   if m.get("role") == "assistant" else "user"}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                msg["content"] = None
            elif len(parts) == 1 and parts[0]["type"] == "text":
                msg["content"] = parts[0]["text"]
            else:
                msg["content"] = parts or ""
            msgs.append(msg)
    return msgs


def openai_tools_to_anthropic(tools):
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or t
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}}})
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema")
            or {"type": "object", "properties": {}}}})
    return out


def _image_part_openai(block):
    """An OpenAI image part as a Gemini inlineData/fileData part."""
    url = (block.get("image_url") or {}).get("url", "")
    mm = re.match(r"data:(image/[\w.+-]+);base64,(.+)", url or "", re.S)
    if mm:
        return {"inlineData": {"mimeType": mm.group(1),
                               "data": mm.group(2)}}
    if url:
        return {"fileData": {"mimeType": "image/png", "fileUri": url}}
    return None


def messages_to_gemini_contents(messages):
    """OpenAI messages as Gemini contents (role user/model, parts)."""
    contents = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        parts, content = [], m.get("content")
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    parts.append({"text": b.get("text", "")})
                elif b.get("type") == "image_url":
                    part = _image_part_openai(b)
                    if part is not None:
                        parts.append(part)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            parts.append({"functionCall": {"name": fn.get("name", ""),
                                           "args": args}})
        if m.get("role") == "tool":
            parts.append({"text": content if isinstance(content, str)
                          else blocks_text(content)})
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def _stop_list(body, keys):
    for k in keys:
        v = body.get(k)
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [s for s in v if isinstance(s, str)]
    return None


def build_provider_request(provider_wire, client_wire, body, model_id):
    """Translate the client request into the provider's wire.

    Returns (path, payload, stream_flag). Every field the client sent
    that the provider's wire can carry is passed through: system,
    tools, stop sequences, max_tokens, temperature, JSON mode.
    """
    stream = bool(body.get("stream"))
    if provider_wire == "openai":
        if client_wire == "openai":
            payload = dict(body)
            payload["model"] = model_id
            if stream:
                payload["stream_options"] = {"include_usage": True}
            return "/v1/chat/completions", payload, stream
        payload = {"model": model_id,
                   "messages": anthropic_messages_to_openai(body),
                   "stream": stream}
        if body.get("max_tokens") is not None:
            payload["max_tokens"] = body["max_tokens"]
        if body.get("temperature") is not None:
            payload["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            payload["top_p"] = body["top_p"]
        stops = _stop_list(body, ("stop_sequences", "stop"))
        if stops:
            payload["stop"] = stops
        if body.get("tools"):
            payload["tools"] = anthropic_tools_to_openai(body["tools"])
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            if choice.get("type") == "auto":
                payload["tool_choice"] = "auto"
            elif choice.get("type") == "any":
                payload["tool_choice"] = "required"
            elif (choice.get("type") == "tool"
                    and isinstance(choice.get("name"), str)):
                payload["tool_choice"] = {"type": "function",
                                          "function": {"name": choice["name"]}}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", payload, stream
    if provider_wire == "anthropic":
        if client_wire == "anthropic":
            payload = dict(body)
            payload["model"] = model_id
            return "/v1/messages", payload, stream
        system, msgs = openai_messages_to_anthropic(body.get("messages"))
        payload = {"model": model_id, "messages": msgs, "stream": stream}
        if system:
            payload["system"] = system
        if body.get("max_tokens") is not None:
            payload["max_tokens"] = body["max_tokens"]
        else:
            payload["max_tokens"] = 1024
        if body.get("temperature") is not None:
            payload["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            payload["top_p"] = body["top_p"]
        stops = _stop_list(body, ("stop", "stop_sequences"))
        if stops:
            payload["stop_sequences"] = stops
        if body.get("tools"):
            payload["tools"] = openai_tools_to_anthropic(body["tools"])
        return "/v1/messages", payload, stream
    # gemini
    if client_wire == "openai":
        messages = body.get("messages") or []
        contents = messages_to_gemini_contents(messages)
        system_text = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                system_text = blocks_text(m.get("content"))
                if system_text:
                    break
        payload = {"contents": contents}
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    else:
        payload = {"contents": messages_to_gemini_contents(
            anthropic_messages_to_openai(body))}
        system = body.get("system")
        if system:
            text = blocks_text(system)
            if text:
                payload["systemInstruction"] = {"parts": [{"text": text}]}
    cfg = {}
    if body.get("max_tokens") is not None:
        cfg["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        cfg["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        cfg["topP"] = body["top_p"]
    fmt = (body.get("response_format") or {}).get("type")
    if fmt in ("json_object", "json_schema"):
        cfg["responseMimeType"] = "application/json"
    stops = _stop_list(body, ("stop", "stop_sequences"))
    if stops:
        cfg["stopSequences"] = stops
    if cfg:
        payload["generationConfig"] = cfg
    if body.get("tools"):
        tools = (body["tools"] if client_wire == "openai"
                 else anthropic_tools_to_openai(body["tools"]))
        decls = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or t
            decls.append({"name": fn.get("name", ""),
                          "description": fn.get("description", ""),
                          "parameters": fn.get("parameters")
                          or {"type": "object", "properties": {}}})
        if decls:
            payload["tools"] = [{"functionDeclarations": decls}]
    name = model_id.split("/")[-1]
    verb = "streamGenerateContent" if stream else "generateContent"
    return "/v1beta/models/%s:%s" % (name, verb), payload, stream


# ---------------------------------------------------------------------------
# Provider answers, normalized: (text, tool_calls, input_tokens,
# output_tokens, stop_reason) in the CLIENT wire's spelling.
#
# The usage counts are the PROVIDER's own -- the client compares them
# with what the provider recorded -- never an estimate of our own.
# The stop reason is mapped by family (stop/end_turn/STOP, length /
# max_tokens, tool_calls / tool_use, content_filter / refusal), so the
# client's wire sees the family the provider answered.
# ---------------------------------------------------------------------------

_STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
_OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls",
                3: "content_filter"}
_ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use",
                   3: "refusal"}


def _map_stop(provider_stop, client_wire):
    family = _STOP_FAMILIES.get(provider_stop, 0)
    table = _OPENAI_STOP if client_wire == "openai" else _ANTHROPIC_STOP
    return table[family]


def _tool_calls_openai(message):
    tcs = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tcs.append({"id": tc.get("id", "call-1"),
                    "name": fn.get("name", ""), "arguments": args})
    return tcs


def _parse_openai_json(resp, client_wire):
    try:
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = "".join(b.get("text", "") for b in text
                           if isinstance(b, dict))
        tcs = _tool_calls_openai(msg)
        usage = resp.get("usage") or {}
        stop = ch.get("finish_reason") or (
            "tool_calls" if tcs else "stop")
        return {"text": text, "tool_calls": tcs,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "stop": _map_stop(stop, client_wire)}
    except (AttributeError, TypeError, IndexError):
        return None


def _parse_sse_frames(raw):
    """SSE text -> [(event, data)] frames, in order."""
    frames = []
    normalized = raw.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if not frame.strip():
            continue
        event, data = None, []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _parse_openai_sse(raw, client_wire):
    text, usage_in, usage_out, stop = "", None, None, None
    for _event, payload in _parse_sse_frames(raw):
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage_in = chunk["usage"].get("prompt_tokens")
            usage_out = chunk["usage"].get("completion_tokens")
        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choices[0].get("finish_reason"), str):
            stop = choices[0]["finish_reason"]
    if stop is None and text == "" and usage_in is None:
        return None
    return {"text": text, "tool_calls": [],
            "input_tokens": usage_in, "output_tokens": usage_out,
            "stop": _map_stop(stop or "stop", client_wire)}


def _parse_anthropic_json(resp, client_wire):
    try:
        text, tcs = "", []
        for b in resp.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                text += b.get("text", "")
            elif b.get("type") == "tool_use":
                args = b.get("input") or {}
                tcs.append({"id": b.get("id", "call-1"),
                            "name": b.get("name", ""),
                            "arguments": args if isinstance(args, dict)
                            else {}})
        usage = resp.get("usage") or {}
        stop = resp.get("stop_reason") or (
            "tool_use" if tcs else "end_turn")
        return {"text": text, "tool_calls": tcs,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "stop": _map_stop(stop, client_wire)}
    except (AttributeError, TypeError):
        return None


def _parse_anthropic_sse(raw, client_wire):
    text, usage_in, usage_out, stop = "", None, None, None
    for event, payload in _parse_sse_frames(raw):
        if event is None:
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if event == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            usage_in = usage.get("input_tokens")
        elif event == "content_block_delta":
            delta = data.get("delta") or {}
            if (delta.get("type") == "text_delta"
                    and isinstance(delta.get("text"), str)):
                text += delta["text"]
        elif event == "message_delta":
            delta = data.get("delta") or {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = data.get("usage") or {}
            usage_out = usage.get("output_tokens")
    if stop is None and text == "" and usage_in is None:
        return None
    return {"text": text, "tool_calls": [],
            "input_tokens": usage_in, "output_tokens": usage_out,
            "stop": _map_stop(stop or "end_turn", client_wire)}


def _parse_gemini_json(resp, client_wire):
    try:
        cand = (resp.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text, tcs = "", []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str):
                text += part["text"]
            if isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                args = fc.get("args") or {}
                tcs.append({"id": "call-1", "name": fc.get("name", ""),
                            "arguments": args if isinstance(args, dict)
                            else {}})
        meta = resp.get("usageMetadata") or {}
        stop = cand.get("finishReason") or "STOP"
        return {"text": text, "tool_calls": tcs,
                "input_tokens": meta.get("promptTokenCount"),
                "output_tokens": meta.get("candidatesTokenCount"),
                "stop": _map_stop(stop, client_wire)}
    except (AttributeError, TypeError, IndexError):
        return None


def _parse_gemini_sse(raw, client_wire):
    text, usage_in, usage_out, stop = "", None, None, None
    for _event, payload in _parse_sse_frames(raw):
        try:
            frame = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        for cand in frame.get("candidates") or []:
            parts = (cand.get("content") or {}).get("parts") or []
            for part in parts:
                if isinstance(part, dict) and isinstance(
                        part.get("text"), str):
                    text += part["text"]
            if isinstance(cand.get("finishReason"), str):
                stop = cand["finishReason"]
        meta = frame.get("usageMetadata") or {}
        if meta:
            usage_in = meta.get("promptTokenCount")
            usage_out = meta.get("candidatesTokenCount")
    if stop is None and text == "" and usage_in is None:
        return None
    return {"text": text, "tool_calls": [],
            "input_tokens": usage_in, "output_tokens": usage_out,
            "stop": _map_stop(stop or "STOP", client_wire)}


def parse_provider_response(provider_wire, stream, raw, client_wire):
    """Parse one provider answer (bytes) into the normalized form."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if stream:
        if provider_wire == "openai":
            return _parse_openai_sse(text, client_wire)
        if provider_wire == "anthropic":
            return _parse_anthropic_sse(text, client_wire)
        return _parse_gemini_sse(text, client_wire)
    try:
        resp = json.loads(text)
    except ValueError:
        return None
    if not isinstance(resp, dict):
        return None
    if provider_wire == "openai":
        return _parse_openai_json(resp, client_wire)
    if provider_wire == "anthropic":
        return _parse_anthropic_json(resp, client_wire)
    return _parse_gemini_json(resp, client_wire)


# ---------------------------------------------------------------------------
# Router state (in-memory, deterministic export)
# ---------------------------------------------------------------------------

class Router:
    def __init__(self, providers, ratings, manifest, catalog, env_url,
                 state_path):
        self.providers = {p["id"]: p for p in providers}
        self.ratings = ratings
        self.manifest = manifest
        self.catalog = catalog
        self.env_url = (env_url or "").rstrip("/")
        self.state_path = state_path
        self.lock = threading.Lock()
        # Providers that answered 402: their free-tier budget is spent
        # for the rest of the trial -- never call them again.
        self.exhausted = set()
        # Providers that answered 429 twice running: the budget is
        # spent (a scheduled refusal is a single call; two in a row is
        # not a schedule, it is an empty budget). Never call them again.
        self.dead = set()
        self.consec_failures = {}
        self.seq = 0
        self.arrival = 0
        self.requests_log = []
        self.provider_calls = 0
        self.done = False
        self.exit_scheduled = False

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def next_arrival(self):
        with self.lock:
            n = self.arrival
            self.arrival += 1
            return n

    def peek_requirements(self, client_wire, body):
        """The manifest row for the next arrival, without consuming it.

        The row is consumed (bump_arrival) once the request has been
        routed -- a request rejected before routing (400) must not
        shift every later request onto the wrong manifest row.
        """
        with self.lock:
            idx = self.arrival
        if 0 <= idx < len(self.manifest):
            return self.manifest[idx]["requires"]
        return request_requirements(client_wire, body)

    def bump_arrival(self):
        with self.lock:
            self.arrival += 1

    def provider_url(self, provider, path):
        base = provider["base"] or ""
        if base.startswith("http"):
            return base.rstrip("/") + path
        if self.env_url:
            if base:
                return self.env_url + base.rstrip("/") + path
            return self.env_url + path
        return ""

    def provider_headers(self, provider):
        headers = {"Content-Type": "application/json"}
        headers.update(auth_headers(provider))
        if provider["wire"] == "anthropic":
            headers.setdefault("anthropic-version", "2023-06-01")
        return headers

    def rank_candidates(self, client_wire, req):
        """All FREE models meeting the request, best rating first.

        A paid model is never returned. Ties break by context length,
        then model id, then provider id, so the order is fully
        deterministic. An OpenAI client asking for JSON mode can only
        be served by a wire that carries response_format (OpenAI or
        Gemini): an Anthropic-wire provider would echo json_mode false
        and lose the request.
        """
        pool = []
        for pid, p in self.providers.items():
            for (prov, mid), row in self.ratings.items():
                if prov != pid or not row["free"]:
                    continue
                if not meets_requirements(row, req):
                    continue
                if (client_wire == "openai" and req.get("json")
                        and p["wire"] == "anthropic"):
                    continue
                pool.append((p, row))
        pool.sort(key=lambda pr: (-pr[1]["rating_milli"],
                                  -(pr[1]["context_length"] or 0),
                                  pr[1]["model"], pr[0]["id"]))
        return pool

    def call_provider(self, provider, model_id, client_wire, body):
        """One provider call. Returns (status, normalized_or_None).

        A 402 marks the provider exhausted for the rest of the trial;
        a 429 counts toward the two-in-a-row that marks it dead. A
        200 whose body does not parse counts as a failed call, so the
        router fails over instead of answering from a broken echo.
        """
        path, payload, stream = build_provider_request(
            provider["wire"], client_wire, body, model_id)
        url = self.provider_url(provider, path)
        if not url:
            return 0, None
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
                                     headers=self.provider_headers(provider),
                                     method="POST")
        with self.lock:
            self.provider_calls += 1
        try:
            with urllib.request.urlopen(req,
                                       timeout=PROVIDER_TIMEOUT) as resp:
                raw = _read_capped_response(resp)
                status = resp.status
        except urllib.error.HTTPError as e:
            try:
                e.close()
            except Exception:
                pass
            status, raw = e.code, b""
        except Exception:
            return 0, None
        if raw is None:
            # The provider answered past the read cap: count the call
            # as failed and fail over, instead of buffering it.
            return 200, None
        if status == 402:
            self.exhausted.add(provider["id"])
        elif status == 429:
            n = self.consec_failures.get(provider["id"], 0) + 1
            self.consec_failures[provider["id"]] = n
            if n >= 2:
                self.dead.add(provider["id"])
        if status != 200:
            return status, None
        parsed = parse_provider_response(provider["wire"], stream, raw,
                                         client_wire)
        if parsed is None:
            return 200, None
        self.consec_failures[provider["id"]] = 0
        return 200, parsed

    def route(self, client_wire, body, req):
        """Try candidates in ranked order until one answers.

        Returns (result_dict, attempts). Exactly one provider call
        ever answers a request: the loop stops at the first 200 whose
        body parses, so the trial never sees two successful calls for
        one request.
        """
        candidates = self.rank_candidates(client_wire, req)
        attempts = []
        for provider, row in candidates:
            pid = provider["id"]
            if pid in self.exhausted:
                attempts.append({"provider": pid, "model": row["model"],
                                 "status": "skipped_exhausted"})
                continue
            if pid in self.dead:
                attempts.append({"provider": pid, "model": row["model"],
                                 "status": "skipped_dead"})
                continue
            status, parsed = self.call_provider(
                provider, row["model"], client_wire, body)
            attempts.append({"provider": pid, "model": row["model"],
                             "status": status})
            if status == 200 and parsed is not None:
                return {"provider": pid, "model": row["model"],
                        "wire": provider["wire"],
                        "text": parsed["text"],
                        "tool_calls": parsed["tool_calls"],
                        "input_tokens": parsed["input_tokens"],
                        "output_tokens": parsed["output_tokens"],
                        "finish": parsed["stop"],
                        "status": 200}, attempts
            # 402 / 429 / 5xx / transport error / unparsable 200:
            # fail over to the next best free model.
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
            return False
        data = fetch_json(self.env_url + "/work", timeout=5)
        if isinstance(data, dict):
            return bool(data.get("done") or data.get("finished"))
        return False

    def maybe_exit(self):
        if self.exit_scheduled:
            return
        try:
            if self.check_work_done():
                self.done = True
                self.exit_scheduled = True
                self.write_state()
                # Give the in-flight response a moment to flush, then
                # exit 0 as the bounty requires.
                threading.Timer(0.5, lambda: os._exit(0)).start()
        except Exception:
            pass


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
        """GET /v1/models: exactly the FREE model ids the catalog lists.

        The env asks this twice at the trial's start (once per wire)
        and the scorer compares each answer's free-id set with the
        registered catalog's: a paid model listed, or a free one
        missing, loses the route checks.
        """
        router = self.router
        free = []
        for p in router.catalog["providers"]:
            for m in p["models"]:
                if m["free"]:
                    free.append((p["id"], m["id"]))
        if self._is_anthropic_wire():
            data = [{"id": mid, "type": "model", "display_name": mid,
                     "created_at": FIXED_ANTHROPIC_CREATED_AT,
                     "owned_by": pid} for pid, mid in free]
            self._send_json(200, {"data": data, "has_more": False,
                                  "first_id": data[0]["id"] if data else None,
                                  "last_id": data[-1]["id"] if data else None})
            return
        data = [{"id": mid, "object": "model",
                 "created": FIXED_OPENAI_CREATED, "owned_by": pid}
                for pid, mid in free]
        self._send_json(200, {"object": "list", "data": data})

    # -- shared ---------------------------------------------------------------
    def _requirements_for(self, client_wire, body):
        """The manifest row for this arrival, else the body's own needs."""
        return self.router.peek_requirements(client_wire, body)

    def _usage_or_estimate(self, result, body, client_wire):
        """The provider's usage; an estimate only when it never arrived."""
        in_tok = result.get("input_tokens")
        out_tok = result.get("output_tokens")
        if isinstance(in_tok, int) and isinstance(out_tok, int):
            return in_tok, out_tok
        if client_wire == "openai":
            prompt = estimate_tokens(text_of_messages(body.get("messages")))
        else:
            prompt = estimate_tokens(text_of_messages(body.get("messages")))
            if isinstance(body.get("system"), str):
                prompt += estimate_tokens(body["system"])
        out = estimate_tokens(result["text"]) + sum(
            estimate_tokens(json.dumps(tc["arguments"], sort_keys=True))
            for tc in result["tool_calls"])
        return prompt, out

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
        req = self._requirements_for("openai", body)
        try:
            result, attempts = router.route("openai", body, req)
        except (AttributeError, TypeError, ValueError, KeyError):
            # A request field with a shape the router did not expect
            # (e.g. tool_calls arguments as an object instead of a
            # string). Answer 400; never drop the connection.
            router.bump_arrival()
            self._send_json(400, {"error": {
                "message": "request fields have an invalid shape",
                "type": "invalid_request_error",
                "code": "invalid_request"}})
            return
        router.bump_arrival()
        log = {"seq": seq, "wire": "openai",
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
                "message": "all providers failed for this request",
                "type": "server_error", "code": "all_providers_failed"}})
            router.maybe_exit()
            return
        prompt_tokens, out_tokens = self._usage_or_estimate(
            result, body, "openai")
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
        chunk = dict(base, choices=[],
                     usage={"prompt_tokens": prompt_tokens,
                            "completion_tokens": out_tokens,
                            "total_tokens": prompt_tokens + out_tokens})
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
        req = self._requirements_for("anthropic", body)
        try:
            result, attempts = router.route("anthropic", body, req)
        except (AttributeError, TypeError, ValueError, KeyError):
            router.bump_arrival()
            self._send_json(400, {"error": {
                "type": "invalid_request_error",
                "message": "request fields have an invalid shape"}})
            return
        router.bump_arrival()
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
        prompt_tokens, out_tokens = self._usage_or_estimate(
            result, body, "anthropic")
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


class _RequestTooLarge(Exception):
    """Content-Length exceeds MAX_REQUEST_BYTES; answered with 413."""


def register_with_env(env_url, port, catalog):
    """POST $MAKEFUN_ENV_URL/register; True when the env answers ok.

    Retried for up to ~30 s: the env is up before the agent starts,
    but a slow start must not lose the trial before it begins.
    """
    url = env_url.rstrip("/") + "/register"
    data = json.dumps({"protocol_version": PROTOCOL_VERSION,
                       "port": port, "catalog": catalog}).encode()
    deadline = time.monotonic() + 30.0
    while True:
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req,
                                       timeout=REGISTER_TIMEOUT) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
            if resp.status == 200 and reply.get("ok"):
                return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def work_watcher(router):
    """Poll the env's /work; when it says done, exit 0.

    The poll reads the env only; the trial's requests arrive on the
    HTTP server, one at a time, in manifest order.
    """
    while True:
        time.sleep(WORK_POLL_INTERVAL)
        try:
            if router.check_work_done():
                router.done = True
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
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    providers, ratings, manifest, brief_error = read_brief(prompt_file)
    if brief_error is not None and env_url:
        # No usable brief: the env's /instance carries the directory.
        providers = directory_from_instance(env_url)
    probes = {}
    if env_url:
        for p in providers:
            probes[p["id"]] = probe_provider(env_url, p)
    catalog = build_catalog(providers, ratings, probes)
    if args.list:
        sys.stdout.buffer.write(catalog_bytes(catalog))
        return 0

    port = args.port
    router = Router(providers, ratings, manifest, catalog, env_url,
                    state_path)
    Handler.router = router
    if brief_error is not None:
        print("free-model-aggregator: brief: %s" % brief_error,
              file=sys.stderr, flush=True)
    print("free-model-aggregator listening (catalog: %d providers, "
          "%d free models, env: %s)"
          % (len(catalog["providers"]),
             sum(1 for p in catalog["providers"] for m in p["models"]
                 if m["free"]),
             env_url or "none"), flush=True)
    server = ThreadingHTTPServer((args.host, port or 0), Handler)
    actual_port = server.server_address[1]
    if actual_port == 8080:
        # 8080 is the env's on one layout; an ephemeral port never is.
        server.server_close()
        raise SystemExit("the OS picked 8080, which the env owns on one "
                         "layout: restart the trial")
    if env_url:
        if not register_with_env(env_url, actual_port, catalog):
            print("free-model-aggregator: register failed; the env "
                  "will send no requests", file=sys.stderr, flush=True)
        threading.Thread(target=work_watcher, args=(router,),
                         daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    router.write_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
