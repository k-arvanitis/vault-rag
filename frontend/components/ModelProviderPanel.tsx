"use client";

import { useCallback, useEffect, useState } from "react";
import { X, Loader2, KeyRound } from "lucide-react";
import {
  deleteLLMCredentials,
  getLLMCredentials,
  setLLMCredentials,
  type LLMCredentialsStatus,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Props {
  onClose: () => void;
}

const PROVIDER_LABELS: Record<string, string> = {
  groq: "Groq",
  openrouter: "OpenRouter",
  openai: "OpenAI",
};

export default function ModelProviderPanel({ onClose }: Props) {
  const [status, setStatus] = useState<LLMCredentialsStatus | null>(null);
  const [provider, setProvider] = useState("groq");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    getLLMCredentials()
      .then((s) => {
        setStatus(s);
        if (s.provider) setProvider(s.provider);
        setModel(s.model || "");
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load status"));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleSave = async () => {
    setBusy(true);
    setError(null);
    try {
      await setLLMCredentials(provider, apiKey.trim() || null, model.trim() || null);
      setApiKey("");
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save credentials");
    } finally {
      setBusy(false);
    }
  };

  const handleClear = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteLLMCredentials();
      setApiKey("");
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to clear credentials");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full w-full">
      <div className="flex h-full w-full flex-col bg-background">
        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold text-foreground">Model provider (bring your own key)</p>
            <div className="mt-0.5 flex items-center gap-1.5">
              {status ? (
                status.key_set ? (
                  <>
                    <Badge variant="default">{PROVIDER_LABELS[status.provider || ""] ?? status.provider}</Badge>
                    <span className="text-[10px] text-muted-foreground">
                      key ending {status.key_last4} · {status.model || "default model"}
                    </span>
                  </>
                ) : (
                  <Badge variant="outline">Using operator's own keys</Badge>
                )
              ) : (
                <Skeleton className="h-4 w-32" />
              )}
            </div>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close model provider panel">
            <X />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-5">
          <div className="mx-auto max-w-xl space-y-3">
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Card size="sm">
              <CardContent className="space-y-2.5">
                <div className="space-y-1.5">
                  <label className="text-[11px] font-medium text-muted-foreground">Provider</label>
                  <Select value={provider} onValueChange={(v) => v && setProvider(v)}>
                    <SelectTrigger className="w-full text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(status?.providers || Object.keys(PROVIDER_LABELS)).map((p) => (
                        <SelectItem key={p} value={p}>
                          {PROVIDER_LABELS[p] ?? p}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] font-medium text-muted-foreground">API key</label>
                  <Input
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={status?.key_set ? `Leave blank to keep key ending ${status.key_last4}` : "sk-..."}
                    className="text-xs"
                    autoComplete="off"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-[11px] font-medium text-muted-foreground">
                    Model <span className="font-normal">(optional — provider default if blank)</span>
                  </label>
                  <Input
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="e.g. gpt-4o-mini"
                    className="text-xs"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-1">
                  {status?.key_set && (
                    <Button variant="outline" size="sm" onClick={handleClear} disabled={busy}>
                      Clear (use operator's keys)
                    </Button>
                  )}
                  <Button size="sm" onClick={handleSave} disabled={busy || !provider}>
                    {busy ? <Loader2 className="animate-spin" /> : <KeyRound data-icon="inline-start" />}
                    Save
                  </Button>
                </div>

                <p className="text-[10px] text-muted-foreground">
                  Your key is stored on this server only, never shown again after saving, and never
                  leaves this server except to call the provider you chose. Saving rebuilds the answer
                  agent, so it takes effect on your next question.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
