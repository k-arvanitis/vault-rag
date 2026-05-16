"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export default function ThemeToggle() {
  const [mounted, setMounted] = useState(false);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setMounted(true);
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* ignore — private mode etc. */
    }
    setDark(next);
  };

  return (
    <button
      onClick={toggle}
      aria-label="Toggle dark mode"
      title={dark ? "Switch to light theme" : "Switch to dark theme"}
      className="rounded-md border border-ink-200 bg-surface p-1.5 text-ink-600 transition-colors hover:bg-ink-100 hover:text-ink-800"
    >
      {mounted ? (
        dark ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />
      ) : (
        <span className="block h-3.5 w-3.5" />
      )}
    </button>
  );
}
