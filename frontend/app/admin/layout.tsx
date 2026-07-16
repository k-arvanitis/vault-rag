"use client";

import { usePathname, useRouter } from "next/navigation";
import { ShieldAlert } from "lucide-react";
import { useAdminSession } from "@/lib/useAdminSession";
import { Button } from "@/components/ui/button";
import AdminNav from "@/components/AdminNav";

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { access_mode: accessMode, is_admin: isAdmin, loaded } = useAdminSession();

  // /admin/login has no nav and is never guarded -- it's how a viewer gets in.
  if (pathname === "/admin/login") return <>{children}</>;

  if (!loaded) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    );
  }

  if (accessMode === "admin_viewer" && !isAdmin) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-3 bg-background text-center">
        <ShieldAlert className="h-10 w-10 text-muted-foreground" />
        <p className="text-sm font-semibold text-foreground">Admin access required</p>
        <p className="max-w-sm text-sm text-muted-foreground">
          This area is restricted to administrators. Log in to continue.
        </p>
        <Button size="sm" onClick={() => router.push("/admin/login")}>
          Admin login
        </Button>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      <AdminNav />
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
