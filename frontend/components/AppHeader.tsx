"use client";

import { useRouter } from "next/navigation";
import { History, ShieldCheck, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import ThemeToggle from "@/components/ThemeToggle";
import { useAdminSession } from "@/lib/useAdminSession";

interface Props {
  offline: boolean;
  onDismissOffline: () => void;
  /** Conversations stays an in-app overlay (not a route) — selecting a saved
   * conversation must return to "/" to load it into the active chat. */
  onShowHistory: () => void;
}

export default function AppHeader(props: Props) {
  const router = useRouter();
  const { is_admin: isAdmin } = useAdminSession();
  return (
    <div className="flex shrink-0 flex-col">
      <header className="flex items-center justify-between gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex items-center gap-2">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-5" />
          <div className="flex items-baseline gap-2.5">
            <span className="text-base font-semibold tracking-tight text-foreground">Vault RAG</span>
            <span className="hidden font-mono text-[10px] uppercase tracking-wide text-muted-foreground/90 sm:inline">
              verified knowledge assistant
            </span>
            {isAdmin && (
              <span className="rounded-md border border-border px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                Admin
              </span>
            )}
          </div>
        </div>
        <nav className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={props.onShowHistory}>
            <History data-icon="inline-start" />
            <span className="hidden md:inline">Conversations</span>
          </Button>
          {isAdmin && (
            <Button variant="ghost" size="sm" onClick={() => router.push("/admin/sources")}>
              <ShieldCheck data-icon="inline-start" />
              <span className="hidden md:inline">Admin workspace</span>
            </Button>
          )}
          <Separator orientation="vertical" className="mx-1 h-5" />
          <ThemeToggle />
        </nav>
      </header>

      {props.offline && (
        <div className="flex items-center justify-between border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
          <p>The document service is temporarily unavailable. Please try again shortly.</p>
          <Button variant="ghost" size="icon-xs" onClick={props.onDismissOffline} aria-label="Dismiss">
            <X />
          </Button>
        </div>
      )}
    </div>
  );
}
