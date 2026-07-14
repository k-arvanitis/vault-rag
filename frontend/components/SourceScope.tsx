"use client";

import { X } from "lucide-react";
import { type Document } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const ALL_SOURCES = "__all__";

interface Props {
  documents: Document[];
  /** null = "all sources"; a filename = scoped to that one document. */
  scopedDocId: string | null;
  onChange: (docId: string | null) => void;
}

/** "Ask across: All sources" control near the conversation header — scopes
 * retrieval to one document instead of relying only on the user naming a
 * filename in the question. Only "all" and "one document" are supported today
 * (see TODO.md: multi-select "selected sources" needs a backend filter
 * extension across retriever.py's Qdrant filter builder, not done yet). */
export default function SourceScope({ documents, scopedDocId, onChange }: Props) {
  const scopedDoc = documents.find((d) => d.filename === scopedDocId);

  return (
    <div className="flex min-w-0 items-center gap-1">
      <span className="shrink-0 text-xs text-muted-foreground">Ask across:</span>
      <Select
        value={scopedDocId ?? ALL_SOURCES}
        onValueChange={(v) => onChange(v === ALL_SOURCES ? null : v)}
      >
        <SelectTrigger size="sm" className="max-w-[220px]">
          <SelectValue className="truncate">
            {scopedDoc ? scopedDoc.filename.split("/").pop() : "All sources"}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value={ALL_SOURCES}>All sources</SelectItem>
            {documents.map((d) => (
              <SelectItem key={d.filename} value={d.filename}>
                {d.filename.split("/").pop()}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
      {scopedDoc && (
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Clear source scope"
          onClick={() => onChange(null)}
        >
          <X />
        </Button>
      )}
    </div>
  );
}
