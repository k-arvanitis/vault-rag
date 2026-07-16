# Admin/User Workspace Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the frontend into a User workspace (Chat, Conversations only) and an Admin workspace (Chat, Sources, Quality, Integrations, Feedback) under a new `/admin/*` route group, with a single client-side guard so a viewer can never see the admin page shell — not just a hidden nav item.

**Architecture:** New `/admin` route segment with a shared `layout.tsx` that checks `useAdminSession()` once and either renders a 403 page or the admin nav + page content. Existing admin components (`Sources` table, `EvalPanel`, `FeedbackPanel`, `GoogleDrivePanel`) move under `/admin/*` paths unchanged; their old top-level routes become client-side redirects that forward query strings. `AppHeader.tsx` (used only on `/`) drops the admin-only nav items it used to conditionally render and gains a single "Admin" link.

**Tech Stack:** Next.js 14 App Router, React client components, `useAdminSession()` hook (`frontend/lib/useAdminSession.ts`), Playwright for e2e, Vitest for unit tests.

## Global Constraints

- Package manager: uv only for Python, never pip (not applicable to this frontend-only plan, but do not touch `src/` Python files).
- No new dependencies — everything here is existing Next.js/React/lucide-react/shadcn primitives already in the repo.
- Every new/modified function needs no new backend change — `require_admin` in `api.py` is untouched; this plan is frontend-only.
- Do not duplicate the chat implementation — the Admin workspace's "Chat" nav item links to the existing `/` route, no new chat component or route.
- ruff/pytest are not relevant here (no Python changes); run `npm run test` (Vitest) and `npx tsc --noEmit` as the verification gate per task where noted.

---

## File Structure

```
frontend/app/admin/
  layout.tsx                          NEW — guard + AdminNav shell for all /admin/* children
  sources/page.tsx                    NEW — moved from app/sources/page.tsx, content unchanged
  quality/page.tsx                    NEW — renders EvalPanel (was app/quality/evaluation/page.tsx)
  feedback/page.tsx                   NEW — renders FeedbackPanel (was app/feedback/page.tsx)
  integrations/google-drive/page.tsx  NEW — renders GoogleDrivePanel (was app/connectors/google-drive/page.tsx)
  login/page.tsx                      UNCHANGED — already at this path

frontend/app/sources/page.tsx                 MODIFY — becomes a redirect to /admin/sources
frontend/app/feedback/page.tsx                MODIFY — becomes a redirect to /admin/feedback
frontend/app/connectors/google-drive/page.tsx MODIFY — becomes a redirect to /admin/integrations/google-drive
frontend/app/quality/evaluation/page.tsx      MODIFY — becomes a redirect to /admin/quality

frontend/components/AdminNav.tsx      NEW — persistent nav bar for the Admin workspace
frontend/components/AppHeader.tsx     MODIFY — drop Sources/Quality/Integrations/login-logout, add "Admin" link

frontend/e2e/admin-viewer-access.spec.ts MODIFY — assert guard behavior on /admin/* paths
```

---

### Task 1: Admin route guard + layout

**Files:**
- Create: `frontend/app/admin/layout.tsx`
- Test: `frontend/e2e/admin-viewer-access.spec.ts` (extended in Task 6, not this task — this task ships the guard itself; Task 6 is where it's asserted end-to-end since the guard depends on pages that don't exist until Tasks 3-4)

**Interfaces:**
- Consumes: `useAdminSession()` from `frontend/lib/useAdminSession.ts` — returns `{ access_mode: "open" | "admin_viewer", is_admin: boolean, loaded: boolean, refresh: () => Promise<void> }` (already exists, unchanged).
- Consumes: `AdminNav` component from Task 2 — `import AdminNav from "@/components/AdminNav"`, no props.
- Produces: every file under `frontend/app/admin/**/page.tsx` (except `login/page.tsx`) is guarded automatically by Next.js layout nesting — later tasks don't add their own guard logic.

- [ ] **Step 1: Write the layout with guard + 403 view**

```tsx
// frontend/app/admin/layout.tsx
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
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors referencing `admin/layout.tsx` (errors about `AdminNav` not existing yet are expected until Task 2 — stop here if any *other* error appears, otherwise proceed to Task 2 before re-checking).

---

### Task 2: AdminNav component

**Files:**
- Create: `frontend/components/AdminNav.tsx`

**Interfaces:**
- Consumes: `useAdminSession()` (for `refresh` after logout), `adminLogout()` from `@/lib/api` (existing, used today in `AppHeader.tsx:100-103`).
- Produces: `export default function AdminNav()` — no props, rendered by `app/admin/layout.tsx` (Task 1).

- [ ] **Step 1: Write the component**

```tsx
// frontend/components/AdminNav.tsx
"use client";

import { useRouter, usePathname } from "next/navigation";
import { MessageSquare, Library, FlaskConical, Plug, MessageSquareWarning, LogOut } from "lucide-react";
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
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors in `AdminNav.tsx` or `admin/layout.tsx`.

- [ ] **Step 3: Commit**

```bash
git add frontend/app/admin/layout.tsx frontend/components/AdminNav.tsx
git commit -m "Add admin workspace guard and nav"
```

---

### Task 3: Move Sources under /admin

**Files:**
- Create: `frontend/app/admin/sources/page.tsx`
- Modify: `frontend/app/sources/page.tsx` (replace entire contents with a redirect)

**Interfaces:**
- Consumes: nothing new — this task relocates `SourcesPage` verbatim.
- Produces: `/admin/sources` renders the existing Sources table; `/sources` (any query string) 302s to it client-side.

- [ ] **Step 1: Move the file**

```bash
mkdir -p frontend/app/admin/sources
git mv frontend/app/sources/page.tsx frontend/app/admin/sources/page.tsx
```

- [ ] **Step 2: Recreate the old route as a redirect**

```tsx
// frontend/app/sources/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function SourcesRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/admin/sources${window.location.search}`);
  }, [router]);
  return null;
}
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual verify**

Run: `cd frontend && npm run dev`
Visit `http://localhost:3002/sources?doc=test` — expect an immediate client-side navigation to `/admin/sources?doc=test`, table renders (or "No sources added yet." if empty), AdminNav visible at top with "Sources" highlighted.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/admin/sources/page.tsx frontend/app/sources/page.tsx
git commit -m "Move Sources screen under /admin"
```

---

### Task 4: Move Quality, Feedback, Integrations under /admin

**Files:**
- Create: `frontend/app/admin/quality/page.tsx`
- Create: `frontend/app/admin/feedback/page.tsx`
- Create: `frontend/app/admin/integrations/google-drive/page.tsx`
- Modify: `frontend/app/quality/evaluation/page.tsx` (redirect)
- Modify: `frontend/app/feedback/page.tsx` (redirect)
- Modify: `frontend/app/connectors/google-drive/page.tsx` (redirect)

**Interfaces:**
- Consumes: `EvalPanel`, `FeedbackPanel`, `GoogleDrivePanel` (existing components, `onClose: () => void` prop, unchanged).
- Produces: `/admin/quality`, `/admin/feedback`, `/admin/integrations/google-drive` render the existing panels.

- [ ] **Step 1: Create the three new pages**

```tsx
// frontend/app/admin/quality/page.tsx
"use client";

import { useRouter } from "next/navigation";
import EvalPanel from "@/components/EvalPanel";

export default function AdminQualityPage() {
  const router = useRouter();
  return <EvalPanel onClose={() => router.push("/")} />;
}
```

```tsx
// frontend/app/admin/feedback/page.tsx
"use client";

import { useRouter } from "next/navigation";
import FeedbackPanel from "@/components/FeedbackPanel";

export default function AdminFeedbackPage() {
  const router = useRouter();
  return <FeedbackPanel onClose={() => router.push("/")} />;
}
```

```tsx
// frontend/app/admin/integrations/google-drive/page.tsx
"use client";

import { useRouter } from "next/navigation";
import GoogleDrivePanel from "@/components/GoogleDrivePanel";

export default function AdminGoogleDrivePage() {
  const router = useRouter();
  return <GoogleDrivePanel onClose={() => router.push("/")} />;
}
```

- [ ] **Step 2: Replace the three old routes with redirects**

```tsx
// frontend/app/quality/evaluation/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function QualityEvaluationRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/admin/quality${window.location.search}`);
  }, [router]);
  return null;
}
```

```tsx
// frontend/app/feedback/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function FeedbackRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/admin/feedback${window.location.search}`);
  }, [router]);
  return null;
}
```

```tsx
// frontend/app/connectors/google-drive/page.tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function GoogleDriveRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/admin/integrations/google-drive${window.location.search}`);
  }, [router]);
  return null;
}
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual verify**

With `npm run dev` running, visit `/quality/evaluation`, `/feedback`, `/connectors/google-drive` — each redirects to its `/admin/*` counterpart and renders inside the AdminNav shell.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/admin/quality frontend/app/admin/feedback frontend/app/admin/integrations \
        frontend/app/quality/evaluation/page.tsx frontend/app/feedback/page.tsx \
        frontend/app/connectors/google-drive/page.tsx
git commit -m "Move Quality, Feedback, Integrations screens under /admin"
```

---

### Task 5: Simplify AppHeader to the User workspace nav

**Files:**
- Modify: `frontend/components/AppHeader.tsx`

**Interfaces:**
- Consumes: `useAdminSession()` (already imported) — only `is_admin` is needed now, `access_mode`/`refresh`/`adminLogout` are no longer used here (logout moved to `AdminNav`, Task 2).
- Produces: no prop or export signature change — `AppHeader(props: Props)` keeps the same `{ offline, onDismissOffline, onShowHistory }` props consumed by `frontend/app/page.tsx:110-114`.

- [ ] **Step 1: Rewrite the nav section**

Replace lines 1-128 of `frontend/components/AppHeader.tsx` with:

```tsx
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
            <span className="hidden font-mono text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
              document intelligence
            </span>
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
              <span className="hidden md:inline">Admin</span>
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
```

Note: in `access_mode="open"` (default), `is_admin` is always `true` (see `frontend/lib/useAdminSession.ts:6-8`), so the "Admin" link is visible to everyone today — same as current behavior where Sources/Quality were unconditionally visible. Only `admin_viewer` mode actually hides it.

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors. `Library`, `MessageSquareWarning`, `FlaskConical`, `FolderSync`, `Plug`, `ChevronDown`, `LogIn`, `LogOut`, `DropdownMenu*` imports and `adminLogout` are gone from this file — confirm no other symbol in the file still references them.

- [ ] **Step 3: Commit**

```bash
git add frontend/components/AppHeader.tsx
git commit -m "Simplify AppHeader to the User workspace nav"
```

---

### Task 6: Update e2e coverage for the guard

**Files:**
- Modify: `frontend/e2e/admin-viewer-access.spec.ts`

**Interfaces:**
- Consumes: Playwright `test`/`expect`/`page.route` (existing pattern in the file, mocking `**/admin/session`).
- Produces: no new exports — this is a test file.

- [ ] **Step 1: Replace the test file contents**

```ts
// frontend/e2e/admin-viewer-access.spec.ts
import { test, expect } from "@playwright/test";

// Real backend enforcement of ACCESS_MODE=admin_viewer is covered by
// tests/test_admin_auth.py (14 cases against require_admin() directly) --
// that's what actually stops a viewer from uploading/deleting even if they
// forge a request. These tests prove the frontend half: a viewer session
// never even sees the admin page shell for /admin/*, and an admin session
// can reach every admin screen and switch back to the User workspace
// without logging out. The dev server this suite runs against stays in
// ACCESS_MODE=open (today's default) -- admin_viewer is exercised here by
// mocking GET /admin/session's response, not by restarting the backend.

test("a viewer is blocked from every /admin/* route", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "admin_viewer", is_admin: false }),
    })
  );

  for (const path of ["/admin/sources", "/admin/quality", "/admin/feedback", "/admin/integrations/google-drive"]) {
    await page.goto(path);
    await expect(page.getByText(/Admin access required/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /Admin login/i })).toBeVisible();
  }

  // No "Admin" link in the User workspace nav for a viewer.
  await page.goto("/");
  await expect(page.getByRole("button", { name: /^Admin$/i })).not.toBeVisible();
});

test("an admin session reaches every admin screen and can return to chat", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "admin_viewer", is_admin: true }),
    })
  );

  await page.goto("/");
  await expect(page.getByRole("button", { name: /^Admin$/i })).toBeVisible();
  await page.getByRole("button", { name: /^Admin$/i }).click();
  await expect(page).toHaveURL(/\/admin\/sources$/);

  for (const [label, urlPart] of [
    ["Quality", "/admin/quality"],
    ["Integrations", "/admin/integrations/google-drive"],
    ["Feedback", "/admin/feedback"],
  ] as const) {
    await page.getByRole("button", { name: new RegExp(`^${label}$`, "i") }).click();
    await expect(page).toHaveURL(new RegExp(urlPart.replace(/\//g, "\\/") + "$"));
  }

  // Chat link switches back to the User workspace without logging out.
  await page.getByRole("button", { name: /^Chat$/i }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("button", { name: /^Admin$/i })).toBeVisible();
});

test("old routes redirect to their /admin/* equivalents", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "open", is_admin: true }),
    })
  );

  await page.goto("/sources?doc=test.pdf");
  await expect(page).toHaveURL(/\/admin\/sources\?doc=test\.pdf$/);

  await page.goto("/feedback");
  await expect(page).toHaveURL(/\/admin\/feedback$/);

  await page.goto("/connectors/google-drive");
  await expect(page).toHaveURL(/\/admin\/integrations\/google-drive$/);

  await page.goto("/quality/evaluation");
  await expect(page).toHaveURL(/\/admin\/quality$/);
});
```

- [ ] **Step 2: Run the suite (requires a live dev server per `frontend/e2e/README.md` — this suite is not wired into CI)**

Run:
```bash
cd frontend && npm run dev &
sleep 3
npx playwright test e2e/admin-viewer-access.spec.ts
kill %1
```
Expected: all 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add frontend/e2e/admin-viewer-access.spec.ts
git commit -m "Update e2e coverage for the admin workspace guard"
```

---

### Task 7: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Type-check the whole frontend**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 2: Run unit tests**

Run: `cd frontend && npm run test`
Expected: all existing tests (`MessageList.test.tsx`, `InspectorPanel` test, etc.) still PASS — none of them touch the moved routes, so no changes expected here, only confirming nothing broke.

- [ ] **Step 3: Run lint**

Run: `cd frontend && npm run lint` (if the script exists — check `package.json`; skip if absent, ruff/pytest are the project's configured linters for Python, not this frontend)

- [ ] **Step 4: Manual smoke test in the browser**

With `npm run dev` running: as an "open"-mode (default) session, click through Conversations, Admin → Sources → Quality → Integrations → Feedback → Chat, confirm no console errors and each screen renders its existing content unchanged.

---

## Self-Review Notes

- **Spec coverage:** every route in the spec's file tree is created (Task 3, 4); guard + 403 (Task 1); AppHeader/AdminNav split (Task 2, 5); old-route redirects with query-string forwarding (Task 3, 4); e2e coverage for guard + switching + redirects (Task 6). Backend `require_admin` is explicitly out of scope and untouched.
- **Placeholder scan:** no TBD/TODO; every step has literal code.
- **Type consistency:** `AdminSession` shape (`access_mode`, `is_admin`) used identically across `useAdminSession.ts`, `admin/layout.tsx`, `AdminNav.tsx`, `AppHeader.tsx` — matches the existing `frontend/lib/api.ts:497-500` type, unchanged.
