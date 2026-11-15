# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ragilwidiyanto4646/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ragilwidiyanto4646/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs, ten simulated free-tier providers: it serves the OpenAI Chat Completions API and the Anthropic Messages API, routes each of the trial's 20 requests to the best free model that can serve it, fails over when a provider refuses, and reports the catalog it read.

## Run it

```sh
python3 main.py --port 8080
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ztnjpsw2&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

## Details

- Bounty: **The best free AI model aggregator** — $500
  (`free-model-aggregator-01`, https://make.fun/bounty/free-model-aggregator-01)
- Deadline: **2026-11-16 02:30 UTC** (Mon 16 Nov 2026, 09:30 WIB)
- Prizes: $100 / $70 / $50 / $20×7 / $14×10 (20 places, $500 total)
- Score (0–100): quality 60%, served 15%, wire fidelity 10%,
  failover 10%, catalog accuracy 5%. Only a free model may serve a
  request; a paid model is never a legal pick.

## How it works

`main.py` — Python 3.13, **standard library only** (nothing is
installed before the server runs, per the bounty rules).

- **The brief.** On start it reads `MAKEFUN_PROMPT_FILE`: the provider
  directory (id, wire, auth scheme, key, base path), the sealed
  quality table (every model's rating in milli-points, context
  length, tools / vision / JSON capability, free flag) and the
  workload manifest (the 20 requests in arrival order, each with its
  api, stream flag, weight and requirements). If the brief cannot be
  read, the directory is fetched from the env's `GET /instance`
  instead. With no brief and no env, `--list` prints
  `{"providers":[]}` and the server answers with an empty catalog,
  like the kit's starter.
- **Discovery.** It probes each provider's catalog route in the
  provider's own wire (`GET <base>/v1/models` for OpenAI/Anthropic,
  `GET <base>/v1beta/models` for Gemini, each with its own auth
  header) and builds ONE catalog document from the probes merged
  with the quality table — the table wins for the free flag and the
  context length, a provider the probe missed contributes the
  table's rows instead, so the catalog stays complete. The catalog's
  `base_url` is the provider's own path (`/p/<id>`), never the env's
  origin, so the register body is byte-identical between a live pass
  and its replay. `--list` prints exactly the bytes the server
  registers.
- **Registration.** It listens on an ephemeral port (never 8080,
  which the env owns on one layout) and registers it:
  `POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port",
  "catalog"}` (retried for up to 30 s). It then polls
  `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 the moment
  the env says done.
- **Routing.** For every request it takes the requirements from the
  workload manifest's row for that arrival (falling back to the
  request body itself when the brief holds no manifest) and ranks
  the free models that meet them by the quality table's
  `rating_milli`, then context length, then model id (deterministic
  tie-break). A paid model is never returned. An OpenAI client
  asking for JSON mode is only routed to a wire that can carry
  `response_format` (OpenAI or Gemini) — an Anthropic-wire provider
  would echo `json_mode` false and lose the request. The requested
  model name is ignored for the choice — the client names `auto`.
- **Failover.** Candidates are tried in ranked order, one call at a
  time, and the loop stops at the first 200 whose body parses, so
  exactly one provider call ever answers a request. A `402` marks
  that provider exhausted for the rest of the trial, a `429` twice
  running marks it dead (a single scheduled refusal is not a dead
  provider), a 5xx / transport error / unparsable 200 just moves to
  the next candidate. Every attempt is recorded in the state export.
- **Wire fidelity.** Requests are translated across wires with every
  field the provider echoes passed through untouched: system prompt,
  user text, tool names, `max_tokens`, stop sequences, images
  (data-URL ↔ base64 ↔ Gemini inlineData), JSON mode
  (`response_format` ↔ `responseMimeType`). Answers are returned in
  the wire the client used, with the **provider's** usage counts and
  stop reason (mapped by family: stop/end_turn/STOP,
  length/max_tokens, tool_calls/tool_use, content_filter/refusal),
  the provider's text whole, and streams rebuilt in the client's
  wire (OpenAI chunks + usage chunk + `[DONE]`; Anthropic
  `message_start` / content blocks / `message_delta` /
  `message_stop`). `GET /v1/models` lists exactly the free model ids
  the catalog registers, in the wire its auth header names.
- **Determinism.** No wall-clock time and no unseeded randomness in
  any routing decision, id, token count or state byte. Response ids
  are sequence-based (`chatcmpl-0001`, `msg-0001`, …). The state
  export (`state.json`) is sorted-key JSON with a catalog SHA-256.
  Two runs of the same practice trial leave a byte-identical state
  export (verified locally), so a final-phase replay matches.
- **Input limits.** A request body larger than 10 MiB is rejected
  with `413` before a byte is buffered (an unauthenticated client
  must not be able to exhaust the server's memory with one request;
  the body already arriving is drained in bounded chunks so the 413
  is actually delivered), a provider answer larger than 2 MiB counts
  as a failed call and fails over instead of being buffered, and a
  malformed request (bad JSON, deeply nested JSON, wrong field
  shapes) gets a `400` instead of a dropped connection.

## Run it

```bash
# Standalone (no brief, no env): empty catalog, like the starter
python3 main.py --port 8080

# Against a trial env (the trial sets these three variables)
MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
MAKEFUN_PROMPT_FILE=/input/prompt.txt \
MAKEFUN_TRIAL_SEED=<seed> python3 main.py

# Print the registered catalog, byte for byte
python3 main.py --list
```

Try it:

```bash
curl localhost:8080/v1/models
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello! Who are you?"}]}'
curl localhost:8080/v1/messages -H 'Content-Type: application/json' \
  -H 'x-api-key: test' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"auto","max_tokens":100,"messages":[{"role":"user","content":"Hi!"}]}'
```

## Tests

Verified locally 2026-11-15 against the kit's own sealed env and
scorer (the env run directly with node, the practice instances'
`input/env`, the state export scored by `program/scorer/score.mjs` —
the kit's `grade-local --driver process` runs node22 bots only, so
this Python entry was graded through the same sealed programs
without Docker):

| Practice trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
|---|---|---|---|---|---|---|---|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

20/20 requests valid on every trial; every failover request (5, 7,
7) still served validly; the server exits 0 on each trial. Quality
99 is the ceiling on these instances: only one model in the sealed
table rates 1475 and its per-trial budget (10–12 calls) cannot cover
the 17 requests whose best rating is 1475, so no entry can score
quality 100 here. A repeat run of trial-0 leaves a byte-identical
state export, so the final-phase replay matches.

Security checks (bounded, local, standalone server): an 11 MiB body
gets `413` and the server keeps answering; malformed and
pathologically nested JSON get `400`; wrong field shapes get `400`;
`GET /v1/models` answers in both wires; a 3 MiB provider answer is
refused by the read cap and fails over.

## Files

| File | What it is |
|---|---|
| `main.py` | The server — the whole entry (single file, stdlib only) |
| `README.md` | This file |
| `LICENSE` | MIT |
| `.makefun/freshness.json` | Drand anchor written by the kit's `init` |
| `.makefun/owner.json` | Payout wallet claim for this entry |

## Submission status

Entry code is done and tested. make.fun account: @hazel-panda-51.
GitHub owner: ragilwidiyanto4646.

No API key is needed for this bounty at all — the providers are
simulated, and the server needs no real key anywhere.
