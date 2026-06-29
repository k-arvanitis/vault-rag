# TODO — finish LiteLLM semantic cache + cost logging integration

Status as of 2026-06-06. The proxy is **healthy** and the plumbing is wired, but two
things are NOT yet verified working. Pick this up later.

## What's already done (committed in config, verified running)
- `docker-compose.yaml`: added **redis** (redis-stack, host port **6380**), gave the
  **ollama** container explicit DNS + resilient pull + host port **11435**, passed
  `OLLAMA_API_BASE`, `NVIDIA_NIM_API_KEY`, `LANGFUSE_*`, and `extra_hosts: host.docker.internal`
  into the litellm container.
- `litellm_config.yaml`: added `success_callback: ["langfuse"]`, `cache: true` +
  `cache_params` (redis-semantic, threshold 0.85, embedder `ollama/nomic-embed-text`,
  `redis_url: redis://redis:6379`, ttl 3600).
- `.env`: added `LITELLM_MASTER_KEY=sk-vault-local-dev` (proxy refused auth without it).
- Verified: proxy `/health/liveliness` = 200; a chat call returns a real answer;
  one semantic-cache entry IS written to Redis after a call.

## ⚠️ Blocker 1 — the app does NOT use the proxy yet
`GENERATION_API_BASE` in `.env` points straight at `https://api.groq.com/openai/v1`,
so the app bypasses LiteLLM entirely (no cache, no cost logging hit).
- [ ] To route through the proxy: set `GENERATION_API_BASE=http://localhost:4000/v1`
      and make the app send `LITELLM_MASTER_KEY` as the API key (llm_utils already
      falls back to it). Decide whether you WANT all traffic through the proxy.

## ⚠️ Blocker 2 — semantic cache stores but does NOT hit
Repeated identical prompts still generate fresh answers (different content, though
latency dropped 2.95s → 0.77s). The embedding is stored in Redis but lookups miss.
Likely causes to investigate:
- [ ] Embedding determinism / similarity threshold — try lowering `similarity_threshold`
      to ~0.7 and confirm `ollama/nomic-embed-text` returns stable vectors.
- [ ] Async write race — the cache write happens after the response; confirm it's not
      a timing issue by spacing calls several seconds apart (call 3 still missed, so
      it's probably NOT just timing — dig into the redisvl index / litellm cache config).
- [ ] Check whether litellm is keying on the full request (temperature, etc.) vs just
      the prompt text.

## ⚠️ Blocker 3 — cost logging to Langfuse not yet confirmed
- [ ] Fire a call through the proxy, then check the Langfuse UI (host `:3000`) for a
      trace with model/tokens/USD cost. Container reaches Langfuse via
      `host.docker.internal:3000` (Langfuse binds 0.0.0.0, so reachable).

## Environment gotchas on THIS box (ai.lmids.sse.gr)
- Host Ollama binds `127.0.0.1:11434` only → unreachable from containers. We use the
  **containerized** ollama (with a DNS fix) for the cache embedder instead.
- Container DNS to `registry.ollama.ai` fails on the default resolver → fixed with
  explicit `dns: [8.8.8.8, 1.1.1.1]` on the ollama service.
- NVIDIA NIM embeddings were tried and rejected (`encoding_format` incompatibility) —
  don't go back to that path.
- Only the **root** `docker-compose.yaml` has these changes. `make up`
  (docker/ingestion-stack/) and `make litellm` (bare host binary) do **NOT** support
  the cache — host can't resolve the `redis`/`ollama` service names. Sync or drop those
  paths if you standardize on the proxy.

## When done
- [ ] Update `litellm_config.yaml` header comment to mention cache + logging.
- [ ] Remove this file once integrated.
