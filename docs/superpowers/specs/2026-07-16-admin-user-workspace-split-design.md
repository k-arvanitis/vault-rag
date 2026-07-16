# Admin/User workspace split — design

Status: approved (design sections walked through with user, 2026-07-16)

## Problem

Admin-only screens (Sources management, evaluation runs, feedback review,
Google Drive config) currently live at top-level routes (`/sources`,
`/feedback`, `/connectors/google-drive`, `/quality/evaluation`) with
admin-gating only applied inside individual nav menu items
(`AppHeader.tsx`) and per-button inside those pages (`Sidebar.tsx`,
`EvalPanel.tsx`). Direct navigation to those URLs is ungated on the
frontend — confirmed via grep, `FeedbackPanel` and `GoogleDrivePanel` have
zero `isAdmin` checks. A viewer who knows or guesses the URL gets the full
page shell rendered client-side; only the underlying API calls 403
server-side (`api.py` `require_admin`, already correct).

Goal: introduce a real User workspace / Admin workspace split with a single
guard point for all admin screens, without duplicating chat or
reimplementing gating logic that already works.

## Route structure

New `/admin` route group; a shared layout does the guard once for every
child:

```
app/
  page.tsx                              User workspace: Chat (unchanged)
  admin/
    layout.tsx                          client-side guard
    sources/page.tsx                    wraps existing Sources content
    quality/page.tsx                    wraps existing EvalPanel
    feedback/page.tsx                   wraps existing FeedbackPanel
    integrations/google-drive/page.tsx  wraps existing GoogleDrivePanel
    login/page.tsx                      existing AdminLogin (path unchanged)
  sources/page.tsx                      redirect() -> /admin/sources
  feedback/page.tsx                     redirect() -> /admin/feedback
  connectors/google-drive/page.tsx      redirect() -> /admin/integrations/google-drive
  quality/evaluation/page.tsx           redirect() -> /admin/quality
```

No new component logic for Sources/Quality/Feedback/Integrations — the new
`page.tsx` files under `/admin/*` render the same existing components
(`Sidebar`/Sources table, `EvalPanel`, `FeedbackPanel`, `GoogleDrivePanel`).
The old top-level `page.tsx` files become one-line `redirect()` calls,
forwarding the full search string so query-param deep links keep working
(e.g. `/sources?doc=x` -> `/admin/sources?doc=x`).

Citations/Evidence-panel links live inside the chat page (`/`) itself, not
under `/sources`, so they are unaffected by this reorg.

## Guard (`app/admin/layout.tsx`)

Client component. Calls `useAdminSession()`:

- while loading -> spinner, no content flash
- `access_mode === "admin_viewer" && !is_admin` -> render a 403 page in
  place (not a redirect chain), with a link to `/admin/login`
- otherwise -> render `children` inside the Admin nav shell

In `open` mode, `is_admin` defaults to `true`
(`frontend/lib/useAdminSession.ts:9-29`), so the guard is a no-op there —
matches current behavior where "open" mode has no viewer restriction at
all. This also means the guard fails open if the backend is unreachable
(same default). Not a new behavior; noted so it isn't mistaken for a bug
introduced by this change.

This guard is UX only. The actual security boundary is unchanged:
`require_admin` in `api.py` already enforces 401 (open mode, bad/missing
API key) or 403 (admin_viewer mode, no valid session) on every admin
mutation/read route. The frontend guard exists so a viewer never even sees
the admin page shell, not because the API needed new protection.

## AppHeader nav split

`AppHeader.tsx` becomes workspace-aware, keyed off `useAdminSession()` +
`usePathname()`:

- **User workspace** (any path outside `/admin/*`, default landing `/`):
  nav shows only Chat (home) and Conversations (existing `onShowHistory`
  sheet trigger, unchanged). If `is_admin`, an "Admin" link appears,
  pointing at `/admin/sources`.
- **Admin workspace** (`/admin/*`): nav shows Chat (-> `/`), Sources,
  Quality, Integrations, Feedback. The Chat link doubles as the way back to
  the User workspace — switching workspace is just a nav click, no logout,
  no extra state.
- Non-logged-in admin (admin_viewer mode, no session): no "Admin" link in
  the User workspace nav; direct navigation to `/admin/*` hits the guard's
  403 page, which links to `/admin/login`.

Existing per-item gates inside admin pages (`Sidebar.tsx` upload/delete/
reprocess, `EvalPanel.tsx` Run eval button) are left as-is — now redundant
defense in depth inside an already-guarded page, not the primary gate.

## Testing

- Update `frontend/e2e/admin-viewer-access.spec.ts`: replace old-path
  hidden-menu-item assertions with `/admin/*` path assertions — viewer
  hitting `/admin/sources` (etc.) directly sees the 403 page, not the real
  content; admin sees the real content.
- Add coverage for all four `/admin/*` subpaths behind the guard.
- No backend test changes required — `require_admin` already covers the
  real boundary.

## Out of scope

- No change to `require_admin` / backend session logic.
- No change to `Sidebar.tsx` / `EvalPanel.tsx` internal gating.
- No new chat route or chat component — Admin workspace's "Chat" nav item
  links to the existing `/`.
