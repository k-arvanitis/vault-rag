"use client";

import { useState } from "react";
import { ChevronDown, Search, X } from "lucide-react";
import { type Document } from "@/lib/api";
import { resolveDisplayTitle } from "@/lib/product";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface Props {
  documents: Document[];
  /** [] = "all sources"; one or more filenames = scoped to those documents. */
  scopedDocIds: string[];
  onChange: (docIds: string[]) => void;
}

/** "Ask across: All sources" control near the conversation header — scopes
 * retrieval to one or more documents instead of relying only on the user
 * naming a filename in the question. Backend ORs across every selected id
 * (src/retriever.py's _metadata_filter). */
export default function SourceScope({ documents, scopedDocIds, onChange }: Props) {
  const [query, setQuery] = useState("");
  const filtered = documents.filter((d) => {
    const q = query.toLowerCase();
    return (
      (d.filename.split("/").pop() ?? d.filename).toLowerCase().includes(q) ||
      resolveDisplayTitle(d).toLowerCase().includes(q)
    );
  });

  const toggle = (filename: string) => {
    onChange(
      scopedDocIds.includes(filename)
        ? scopedDocIds.filter((d) => d !== filename)
        : [...scopedDocIds, filename]
    );
  };

  const scopedName = (filename: string) => {
    const doc = documents.find((d) => d.filename === filename);
    return doc ? resolveDisplayTitle(doc) : filename.split("/").pop();
  };

  const label =
    scopedDocIds.length === 0
      ? `All ${documents.length} sources`
      : scopedDocIds.length === 1
        ? scopedName(scopedDocIds[0])
        : `${scopedDocIds.length} selected sources`;

  return (
    <div className="flex min-w-0 items-center gap-1">
      <span className="shrink-0 text-xs text-muted-foreground">Ask across:</span>
      <Popover onOpenChange={(open) => !open && setQuery("")}>
        <PopoverTrigger
          render={
            <Button variant="outline" size="sm" className="max-w-[220px] justify-between gap-1.5" />
          }
        >
          <span className="truncate">{label}</span>
          <ChevronDown data-icon="inline-end" />
        </PopoverTrigger>
        <PopoverContent align="start" className="w-64 p-1">
          <div className="relative px-1 pb-1 pt-0.5">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search documents…"
              className="h-7 pl-7 text-xs"
            />
          </div>
          <div className="max-h-72 overflow-y-auto">
            {filtered.length === 0 && (
              <p className="px-2 py-1.5 text-xs text-muted-foreground">No matching documents.</p>
            )}
            {filtered.map((d) => {
              const name = resolveDisplayTitle(d);
              const checked = scopedDocIds.includes(d.filename);
              return (
                <label
                  key={d.filename}
                  className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-xs text-foreground hover:bg-accent"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(d.filename)}
                    className="size-3.5"
                  />
                  <span className="truncate">{name}</span>
                </label>
              );
            })}
          </div>
        </PopoverContent>
      </Popover>
      {scopedDocIds.length > 0 && (
        <>
          {scopedDocIds.length > 1 && (
            <Badge variant="secondary">{scopedDocIds.length}</Badge>
          )}
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Clear source scope"
            onClick={() => onChange([])}
          >
            <X />
          </Button>
        </>
      )}
    </div>
  );
}
