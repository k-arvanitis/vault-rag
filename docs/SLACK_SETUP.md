## Slack setup

### 1. Create a Slack app

Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**.

### 2. Enable Socket Mode

**Settings → Socket Mode** → Enable. This creates the App-Level Token (`xapp-...`) — copy it to `SLACK_APP_TOKEN`.

### 3. Add Bot Token Scopes

**OAuth & Permissions → Scopes → Bot Token Scopes**, add:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Receive @mention events |
| `chat:write` | Post messages and replies |
| `im:history` | Receive DM messages |
| `im:write` | Reply in DMs |

### 4. Subscribe to events

**Event Subscriptions → Subscribe to bot events**, add:
- `app_mention`
- `message.im`

### 5. Install and configure

**OAuth & Permissions → Install to Workspace** → copy the Bot User OAuth Token to `SLACK_BOT_TOKEN`.

Add both tokens to `.env`:

```
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 6. Start the bot

```bash
make slack            # local
make docker-slack-up  # Docker
```

Invite the bot to a channel: `/invite @vault-rag`
