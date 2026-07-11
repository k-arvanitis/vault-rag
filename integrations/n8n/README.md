# n8n channel connectors for Vault RAG

Vault RAG's `/query` endpoint is a plain, unauthenticated `POST` (see main
[README's Configuration](../../README.md#configuration)). These are two exported n8n
workflows that show the pattern described there in a real, importable form — a channel
connector that owns the messaging protocol and calls Vault RAG as a backend.

```
Chat surface (WhatsApp / generic webhook)
        │
        ▼
   n8n workflow  →  POST /query  →  Vault RAG
        │                              │
        │        answer + sources ◄────┘
        ▼
  answer == "Unsupported"?
   │                  │
  yes                 no
   │                  │
   ▼                  ▼
Notify on-call    Reply to the user
   (Slack)         with the answer
```

## Files

- **`vault-rag-webhook.json`** — generic pattern: any HTTP-capable chat platform (Telegram,
  Teams, SMS gateway, a custom frontend) posts `{"question": "..."}` to an n8n webhook URL and
  gets back `{"answer": ..., "sources": [...]}` (or an escalation message if Vault RAG
  returned `Unsupported`).
- **`vault-rag-whatsapp.json`** — same pattern wired to n8n's native WhatsApp Trigger/node
  pair, matching the exact flow described in the main README's "WhatsApp (via n8n)" section.

## Import steps

1. In n8n: **Workflows → Import from File**, pick one of the two `.json` files here.
2. Set these environment variables in your n8n instance (Settings → Environment Variables, or
   your deployment's env config):
   - `VAULT_RAG_API_URL` — e.g. `http://localhost:8000` or your deployed Vault RAG URL.
   - `ESCALATION_SLACK_CHANNEL` — the Slack channel ID/name for the on-call escalation message.
   - `WHATSAPP_PHONE_NUMBER_ID` — only needed for the WhatsApp workflow.
3. Re-select credentials on the **Slack** node (and the **WhatsApp Trigger** / **WhatsApp**
   nodes, for that workflow) — imported workflows reference credentials by ID, which won't
   exist in a new n8n instance. n8n will flag these nodes; click each one and pick/create your
   own credential.
4. Activate the workflow. For the generic webhook, the trigger URL is
   `<your-n8n-host>/webhook/vault-rag-query`; `POST` `{"question": "..."}` to it.

## What this does and doesn't include

- No WhatsApp Business account, Slack workspace, or n8n instance is provisioned by this
  repo — you bring your own, per n8n's standard credential setup for those node types.
- The escalation branch only sends a Slack notification; it does not implement a full
  human-handoff conversation (e.g. an agent replying back through the same WhatsApp thread).
  That's a reasonable next step once this pattern is in place, not built here.
- This mirrors the same contract the Slack bot (`slack_app.py`) already uses — a thin client
  of `POST /query` — so the underlying RAG pipeline behaves identically across every channel.
