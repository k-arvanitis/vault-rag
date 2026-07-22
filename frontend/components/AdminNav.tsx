"use client";

import { useRouter, usePathname } from "next/navigation";
import { MessageSquare, Library, FlaskConical, Plug, MessageSquareWarning, LogOut, Gauge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import ThemeToggle from "@/components/ThemeToggle";
import { useAdminSession } from "@/lib/useAdminSession";
import { adminLogout } from "@/lib/api";

const ITEMS = [
  { href: "/admin/sources", label: "Sources", icon: Library },
  { href: "/admin/quality", label: "Quality", icon: FlaskConical },
  { href: "/admin/integrations/google-drive", label: "Integrations", icon: Plug },
  { href: "/admin/feedback", label: "Feedback", icon: MessageSquareWarning },
  { href: "/admin/usage", label: "Usage", icon: Gauge },
] as const;

export default function AdminNav() {
  const router = useRouter();
  const pathname = usePathname();
  const { access_mode: accessMode, refresh } = useAdminSession();

  return (
    <header className="flex shrink-0 items-center justify-between gap-2 border-b border-border bg-card px-4 py-2.5">
      <div className="flex items-baseline gap-2.5">
        <span className="text-base font-semibold tracking-tight text-foreground">Vault RAG</span>
        <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
          admin
        </span>
      </div>
      <nav className="flex items-center gap-1">
        <Button variant="ghost" size="sm" onClick={() => router.push("/")}>
          <MessageSquare data-icon="inline-start" />
          <span className="hidden md:inline">Chat</span>
        </Button>
        {ITEMS.map(({ href, label, icon: Icon }) => (
          <Button
            key={href}
            variant={pathname.startsWith(href) ? "secondary" : "ghost"}
            size="sm"
            onClick={() => router.push(href)}
          >
            <Icon data-icon="inline-start" />
            <span className="hidden md:inline">{label}</span>
          </Button>
        ))}
        <Separator orientation="vertical" className="mx-1 h-5" />
        {accessMode === "admin_viewer" && (
          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              await adminLogout();
              refresh();
              router.push("/");
            }}
          >
            <LogOut data-icon="inline-start" />
            <span className="hidden md:inline">Log out</span>
          </Button>
        )}
        <ThemeToggle />
      </nav>
    </header>
  );
}
