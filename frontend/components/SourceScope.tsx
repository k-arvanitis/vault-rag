"use client";

import { ChevronDown, X } from "lucide-react";
import { type Document } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
  const toggle = (filename: string) => {
    onChange(
      scopedDocIds.includes(filename)
        ? scopedDocIds.filter((d) => d !== filename)
        : [...scopedDocIds, filename]
    );
  };

  const label =
    scopedDocIds.length === 0
      ? "All sources"
      : scopedDocIds.length === 1
        ? scopedDocIds[0].split("/").pop()
        : `${scopedDocIds.length} sources`;

  return (
    <div className="flex min-w-0 items-center gap-1">
      <span className="shrink-0 text-xs text-muted-foreground">Ask across:</span>
      <Popover>
        <PopoverTrigger
          render={
            <Button variant="outline" size="sm" className="max-w-[220px] justify-between gap-1.5" />
          }
        >
          <span className="truncate">{label}</span>
          <ChevronDown data-icon="inline-end" />
        </PopoverTrigger>
        <PopoverContent align="start" className="w-64 p-1">
          <div className="max-h-72 overflow-y-auto">
            {documents.map((d) => {
              const name = d.filename.split("/").pop();
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
