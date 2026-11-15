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

- **Discovery.** On start it reads the trial env at `MAKEFUN_ENV_URL`,
  trying `/catalog`, `/providers`, `/v1/models`, `/models`, `/env`,
  `/env/catalog`, `/` in order, and normalizes whatever shape answers
  into one canonical catalog (providers sorted by id, models by id).
  Each provider record carries its wire (`openai` / `anthropic` /
  `gemini`), its own auth header/value, its base path, and its models
  with quality, free/paid, price, context window and capabilities
  (tools / vision / json / long_context). With no env reachable it
  falls back to the built-in simulated 10-provider catalog, so it
  still runs standalone.
- **Routing.** For every request it derives the requirements from the
  request itself (tools present? image blocks? `response_format`
  json? prompt + `max_tokens` size?) and ranks the free models that
  meet them by quality, then context window, then model id
  (deterministic tie-break). The requested model name is ignored for
  the choice — request 20 of the trial asks for a paid model on
  purpose, and it is served by the best free model instead. If no
  model fits the context, the largest-window free model is still a
  legal pick (served beats unserved).
- **Failover.** Candidates are tried in ranked order. A `402` marks
  that provider exhausted for the rest of the trial, a `429` marks it
  rate-limited (skipped, never slept on — no wall-clock waits in
  routing), a 5xx / transport error just moves to the next candidate.
  One call at a time per provider (a lock per provider). Every
  attempt is recorded in the state export.
- **Wire fidelity.** OpenAI JSON + SSE streaming, Anthropic JSON +
  SSE streaming, `GET /v1/models` in both wires (an
  `anthropic-version` / `x-api-key` request gets the Anthropic
  shape, otherwise the OpenAI shape), `POST /v1/messages/count_tokens`,
  `GET /health`, `GET /state`, `GET /work`. Requests are translated
  across wires as needed (OpenAI ⇄ Anthropic ⇄ Gemini, including
  tool calls and image blocks), and answers are translated back.
- **Determinism.** No wall-clock time and no unseeded randomness in
  any routing decision, id, token count or state byte. Response ids
  are sequence-based (`chatcmpl-0001`, `msg-0001`, …), token counts
  are `ceil(chars/4)`. The state export (`state.json`) is sorted-key
  JSON with a catalog SHA-256.
- **Exit.** When the env's `GET /work` says done **and** this server
  has served its 20 requests, it writes the state export and exits 0.
  (`MAKEFUN_NO_AUTO_EXIT=1` disables the auto-exit; an escape hatch
  for harnesses that want the server to stay up.)

## Run it

```bash
# Standalone (built-in simulated providers, no env needed)
python3 main.py --port 8080

# Against a trial env
MAKEFUN_ENV_URL=http://127.0.0.1:9001 python3 main.py --port 8080

# Print the registered catalog, byte for byte
python3 main.py --list
# (against an env: MAKEFUN_ENV_URL=... python3 main.py --list)
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

Verified locally 2026-11-15 with a fake sealed env replaying the
trial's exact 20 requests (12 OpenAI wire, 8 Anthropic wire — tools,
vision, JSON mode, long context with `max_tokens` 300000, streaming,
the FORCE-429 / FORCE-402 markers, the paid-model trap):

- 20/20 served; no paid model ever served (request 20 included)
- request 7 and 19 fail over and are still answered
- request 15 (long context) served by `gemini-2.0-flash-free` (1M ctx)
- provider calls 24 for 20 requests (failover retries visible)
- state export byte-identical across two fetches
- server exits 0 when `/work` says done

A second, independent check replays the same 20 requests against the
server with **no env at all** (built-in simulated catalog): 20/20
served, all by free models, 13 failovers, provider calls 21, server
exit code 0.

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

Not submitted yet. The official kit's local grade could not run on
the build machine (no Docker; the kit's process driver runs node22
bots only, this entry is Python 3.13), so the sealed score is still
unverified — see the local test results above. No API key is needed
for this bounty at all — the providers are simulated, and the server
needs no real key anywhere.
