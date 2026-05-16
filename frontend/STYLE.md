# UI style — Oikia front-desk look

A compact, neutral, content-forward dashboard style. Light-first, with dark mode driven
by a single class on `<html>` and CSS-variable colour tokens (no `dark:` variants needed
in components). Flat surfaces with thin borders — almost no shadows. Tailwind only.

Copy the four pieces in **§6** into a new project to get the same look + dark toggle.

---

## 1. Principles

- **Neutral slate + one accent.** Greys do 95% of the work; a single indigo accent (`brand`)
  for links, the "sent" chat bubble, progress fills, focus rings. Status uses Tailwind's
  `emerald` / `amber` / `red` (positive / caution / error) — only ever as light pastel
  chips (`*-50` bg, `*-700` text) or banners (`*-50` bg, `*-200` border, `*-800` text).
- **Borders, not shadows.** Cards/panels are `rounded-lg border border-ink-200 bg-surface`.
  No `shadow-*` anywhere. Depth is implied by `surface` being a touch brighter (light) or
  lighter (dark) than the `bg-ink-50` canvas.
- **Dense.** Body text is `text-xs`/`text-sm`; chips and meta lines drop to `text-[10px]` /
  `text-[11px]`. Inner padding `p-2.5`/`p-3`; vertical rhythm `space-y-1.5` inside a card,
  `space-y-3` between cards.
- **Content first.** The most important panel (results) is at the top of the sidebar and
  only renders when it has something. Secondary state (a form-like profile) shows only its
  filled fields with a "+ N more" toggle. Diagnostics live in collapsible sections.

## 2. Colour tokens

CSS-variable–backed so one `.dark` class on `<html>` flips everything. `ink` is a neutral
scale that **inverts** in dark mode: `ink-50` = the canvas / darkest-text-on-light, … ,
`ink-900` = primary text. `surface` = the elevated card/panel fill.

| token | role | light | dark |
|---|---|---|---|
| `bg-ink-50` | app canvas, sidebar bg | slate-50 `#f8fafc` | slate-900 `#0f172a` |
| `bg-ink-100` | hover step, track, placeholder bg | slate-100 | slate-800 |
| `bg-surface` | card / panel / header fill | white | slate-800 `#1e293b` |
| `border-ink-200` / `border-ink-100` | card border / divider | slate-200 / -100 | slate-700 / -800 |
| `text-ink-800` | primary text | slate-800 | slate-100 |
| `text-ink-600` / `text-ink-500` | secondary / muted labels | slate-600 / -500 | slate-300 / -400 |
| `text-ink-400` / `text-ink-300` | faint meta / "—" placeholder | slate-400 / -300 | slate-500 / -600 |
| `bg-brand` | accent fill (button, progress, sent bubble) | indigo-600 `#4f46e5` | indigo-600 |
| `text-brand-dark` / `ring-brand-dark` | links, focus/highlight ring | indigo-700 `#4338ca` | indigo-300 `#a5b4fc` |
| `brand.light` | not currently used much | indigo-300 `#a5b4fc` | — |
| `bg-emerald-50 text-emerald-700` | positive chip | — | (pastel; acceptable as-is in dark) |
| `bg-amber-50 text-amber-700` / `border-amber-200 bg-amber-50 text-amber-800` | caution chip / banner | — | — |
| `bg-red-50 text-red-800` / `border-red-200` | error banner | — | — |
| `bg-slate-900 text-slate-50` | code / SQL block | dark | dark (kept dark in both) |

Score badges (Walk/Transit/Bike etc.): `>=70` → emerald-100/700, `40–69` → amber-100/700,
`<40` → ink-100/500.

## 3. Typography

- Font: system stack — `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif`, with `-webkit-font-smoothing: antialiased`. No web fonts.
- Sizes: `text-xl font-bold tracking-tight` app title · `text-sm` chat/body · `text-xs`
  panels and most UI · `text-[11px]` meta/labels/inline buttons · `text-[10px]` chips & badges.
- Panel headers: `text-xs font-semibold uppercase tracking-wide text-ink-500`.
- Numbers/code: `font-mono`.

## 4. Shape, spacing, layout

- Radii: cards/panels `rounded-lg`; mini-cards/buttons `rounded-md`; chips `rounded` (≈4px);
  pills/progress `rounded-full`; chat bubbles `rounded-2xl`.
- Sidebar: fixed `w-[300px] flex-shrink-0`, `hidden lg:block`, `overflow-y-auto`,
  `border-l border-ink-200 bg-ink-50 p-3 space-y-3`. Order = **most important first**
  (results → profile → debug).
- Shell: `flex h-screen flex-col` → header `flex items-center justify-between border-b border-ink-200 bg-surface px-6 py-3` → `flex flex-1 overflow-hidden` { main `flex flex-1 flex-col` ; aside }.

## 5. Component patterns

**Panel** — `rounded-lg border border-ink-200 bg-surface`; header bar
`flex items-center justify-between border-b border-ink-100 px-3 py-2` with an uppercase
title and an optional right widget (count, % bar); body `p-3` (`space-y-1` for rows,
`space-y-3` for sub-cards).

**Card (e.g. a listing)** — like a panel but `overflow-hidden`; optional image banner at
top (`rounded-t-lg`, `object-cover`, `loading="lazy"`, `onError`→placeholder); body `p-2.5 space-y-1.5 text-xs`:
title `font-semibold text-ink-800` → meta line `text-ink-600` with secondary bits in
`text-ink-400` joined by `·` → an icon-stat row `flex flex-wrap gap-x-3 gap-y-0.5 text-ink-500`
(emoji + value) → feature chips → a `line-clamp-2` description + a `text-[10px] text-brand-dark hover:underline`
"Show more ▾"/"Show less ▲" toggle → an action row `flex items-center justify-between pt-0.5`.

**Placeholder image** — grey box `bg-ink-100 text-ink-400`, centred stroke-only 24×24 SVG
icon (`fill="none" stroke="currentColor" strokeWidth="1.5"`) + a small uppercase label.
Use a short box (`h-[72px]`) for placeholders, a tall banner (`h-[150px]`) for real images.

**Chip / badge** — `rounded px-1.5 py-0.5 text-[10px]`, semantic bg+text pair (see §2).
Only render present features; absence of a chip implies absence (never "No pool").

**Collapsible section** — full-width `<button>` header
`flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-ink-700 hover:bg-ink-50`
with `+`/`−` (or `▾`/`▲`) on the right; revealed body `px-3 pb-3 text-xs`.

**Progress bar** — track `h-1.5 w-16 overflow-hidden rounded-full bg-ink-100`, fill
`h-full bg-brand` with `style={{ width: \`${pct}%\` }}`, label `text-xs text-ink-500`.

**Chat bubble** — `whitespace-pre-wrap rounded-2xl px-4 py-2`; user → `bg-brand text-white`,
container `justify-end items-end`; assistant → `border border-ink-200 bg-surface text-ink-800`,
`justify-start items-start`; loading state = a pulsing pill `h-2 w-12 animate-pulse rounded-full bg-ink-200`;
meta line below in `text-[11px] text-ink-400` (· separators).

**Inline mini-card strip** — `flex gap-2 overflow-x-auto pb-1`; each item
`w-56 flex-shrink-0 flex items-center gap-2 rounded-md border border-ink-200 bg-surface p-1.5 text-xs`
with a tiny `h-[44px] w-[60px]` thumbnail, truncated lines (`truncate`), and a
`text-[11px] text-brand-dark hover:underline` "View →" that `scrollIntoView`s + briefly
adds `ring-2 ring-brand-dark` to the target panel.

**Buttons** — secondary: `rounded-md border border-ink-200 bg-surface px-3 py-1 text-xs text-ink-700 hover:bg-ink-100`.
Primary/CTA: `rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-dark`.
Tiny inline ("Show more", "View →"): `text-[10px]/[11px] text-brand-dark hover:underline`.
Tiny boxed ("Book →"): `rounded border border-ink-200 bg-ink-50 px-2 py-1 text-[11px] text-ink-700 hover:bg-ink-100`.

**Status banner** — sticky strip, full width, `border-t border-<c>-200 bg-<c>-50 px-4 py-2 text-sm text-<c>-800`
with a `<strong>` lead. amber = pending / human-in-the-loop, emerald = success, red = error.

**Inputs** — `rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-sm focus:border-brand focus:outline-none disabled:opacity-50`.

**Code / SQL** — `<pre class="overflow-x-auto rounded bg-slate-900 p-2 font-mono text-[10px] text-slate-50">` (literal slate so it stays dark in both themes).

## 6. Dark-mode setup (copy into a new project)

**`tailwind.config.ts`**
```ts
darkMode: "class",
theme: { extend: { colors: {
  ink: Object.fromEntries([50,100,200,300,400,500,600,700,800,900].map(n =>
    [n, `rgb(var(--ink-${n}) / <alpha-value>)`])),
  surface: "rgb(var(--surface) / <alpha-value>)",
  brand: { DEFAULT: "#4f46e5", light: "#a5b4fc", dark: "rgb(var(--brand-strong) / <alpha-value>)" },
}}},
```

**`globals.css`**
```css
:root {
  color-scheme: light;
  --ink-50:248 250 252; --ink-100:241 245 249; --ink-200:226 232 240; --ink-300:203 213 225;
  --ink-400:148 163 184; --ink-500:100 116 139; --ink-600:71 85 105; --ink-700:51 65 85;
  --ink-800:30 41 59; --ink-900:15 23 42; --surface:255 255 255; --brand-strong:67 56 202;
}
.dark {
  color-scheme: dark;
  --ink-50:15 23 42; --ink-100:30 41 59; --ink-200:51 65 85; --ink-300:71 85 105;
  --ink-400:100 116 139; --ink-500:148 163 184; --ink-600:203 213 225; --ink-700:226 232 240;
  --ink-800:241 245 249; --ink-900:248 250 252; --surface:30 41 59; --brand-strong:165 180 252;
}
body { background: rgb(var(--ink-50)); color: rgb(var(--ink-900)); }
```

**`app/layout.tsx`** — no-flash init (runs before paint), plus `suppressHydrationWarning` on `<html>`:
```tsx
const themeInit = `try{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.classList.add('dark')}catch(e){}`;
// in RootLayout:
<html lang="en" suppressHydrationWarning>
  <head><script dangerouslySetInnerHTML={{ __html: themeInit }} /></head>
  <body>{children}</body>
</html>
```

**`components/ThemeToggle.tsx`** — a `"use client"` button that toggles `document.documentElement.classList`
+ persists to `localStorage.theme` (see this repo's file for the exact code).

That's it — every component just uses `bg-ink-50` / `bg-surface` / `text-ink-800` / etc. and
both themes work. The only colours that don't flip via vars are the `emerald/amber/red`
status pastels (fine as-is) and the literal `slate-900` code block (intentionally dark).
