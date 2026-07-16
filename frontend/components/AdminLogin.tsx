"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { adminLogin } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";

export default function AdminLogin() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await adminLogin(password);
      router.push("/");
      router.refresh();
    } catch (err) {
      setError((err as Error).message || "Login failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-background p-4">
      <Card size="sm" className="w-full max-w-sm">
        <CardContent className="space-y-3">
          <div>
            <p className="text-sm font-semibold text-foreground">Admin login</p>
            <p className="text-xs text-muted-foreground">Viewers can already ask questions without logging in.</p>
          </div>
          <Input
            type="password"
            placeholder="Admin password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy && password && submit()}
            autoFocus
          />
          {error && <p className="text-xs text-destructive">{error}</p>}
          <Button size="sm" className="w-full" onClick={submit} disabled={busy || !password}>
            {busy ? "Logging in…" : "Log in"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
