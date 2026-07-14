"use client";

import { useRouter } from "next/navigation";
import { History, Library, MessageSquareWarning, FlaskConical, FolderSync, ShieldCheck, Plug, ChevronDown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import ThemeToggle from "@/components/ThemeToggle";

interface Props {
  offline: boolean;
  onDismissOffline: () => void;
  /** Conversations stays an in-app overlay (not a route) — selecting a saved
   * conversation must return to "/" to load it into the active chat. */
  onShowHistory: () => void;
}

export default function AppHeader(props: Props) {
  const router = useRouter();
  return (
    <div className="flex shrink-0 flex-col">
      <header className="flex items-center justify-between gap-2 border-b border-border bg-card px-4 py-2.5">
        <div className="flex items-center gap-2">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-5" />
          <div className="flex items-baseline gap-2.5">
            <span className="text-base font-semibold tracking-tight text-foreground">Vault RAG</span>
            <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
              document intelligence
            </span>
          </div>
        </div>
        <nav className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => router.push("/sources")}>
            <Library data-icon="inline-start" />
            <span className="hidden md:inline">Sources</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={props.onShowHistory}>
            <History data-icon="inline-start" />
            <span className="hidden md:inline">Conversations</span>
          </Button>

          {/* Workspace administration — not primary actions for a normal user. */}
          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="ghost" size="sm" />}>
              <ShieldCheck data-icon="inline-start" />
              <span className="hidden md:inline">Quality</span>
              <ChevronDown data-icon="inline-end" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuGroup>
                <DropdownMenuItem onClick={() => router.push("/feedback")}>
                  <MessageSquareWarning data-icon="inline-start" />
                  Feedback
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => router.push("/quality/evaluation")}>
                  <FlaskConical data-icon="inline-start" />
                  Quality Evaluation
                </DropdownMenuItem>
              </DropdownMenuGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          <DropdownMenu>
            <DropdownMenuTrigger render={<Button variant="ghost" size="sm" />}>
              <Plug data-icon="inline-start" />
              <span className="hidden md:inline">Integrations</span>
              <ChevronDown data-icon="inline-end" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuGroup>
                <DropdownMenuItem onClick={() => router.push("/connectors/google-drive")}>
                  <FolderSync data-icon="inline-start" />
                  Google Drive
                </DropdownMenuItem>
              </DropdownMenuGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          <Separator orientation="vertical" className="mx-1 h-5" />
          <ThemeToggle />
        </nav>
      </header>

      {props.offline && (
        <div className="flex items-center justify-between border-b border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200">
          <p>
            <strong>Backend offline</strong> — start the Python server (<code className="font-mono">make api</code>).
          </p>
          <Button variant="ghost" size="icon-xs" onClick={props.onDismissOffline} aria-label="Dismiss">
            <X />
          </Button>
        </div>
      )}
    </div>
  );
}
