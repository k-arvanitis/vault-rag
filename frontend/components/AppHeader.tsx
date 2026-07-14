"use client";

import { History, MessageSquareWarning, FlaskConical, FolderSync, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import ThemeToggle from "@/components/ThemeToggle";

interface Props {
  offline: boolean;
  onDismissOffline: () => void;
  onShowHistory: () => void;
  onShowFeedback: () => void;
  onShowEval: () => void;
  onShowDrive: () => void;
}

const NAV_ITEMS = (props: Props) => [
  { label: "History", icon: History, onClick: props.onShowHistory },
  { label: "Feedback", icon: MessageSquareWarning, onClick: props.onShowFeedback },
  { label: "Evaluation", icon: FlaskConical, onClick: props.onShowEval },
  { label: "Drive sync", icon: FolderSync, onClick: props.onShowDrive },
];

export default function AppHeader(props: Props) {
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
          {NAV_ITEMS(props).map(({ label, icon: Icon, onClick }) => (
            <Button key={label} variant="ghost" size="sm" onClick={onClick}>
              <Icon data-icon="inline-start" />
              <span className="hidden md:inline">{label}</span>
            </Button>
          ))}
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
