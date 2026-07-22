"use client";

import { useEffect, useState } from "react";
import { setDocumentTitle } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";

interface Props {
  filename: string;
  /** Current display name (extracted title, demo override, or filename) —
   * prefilled so leaving it unchanged and saving is a no-op rename. */
  currentName: string;
  onRenamed: () => void;
  onToast: (msg: string, variant?: "error") => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Rename a source's display title, or leave it as-is / reset it back to the
 * extracted title or original filename — same fully-controlled Dialog
 * pattern as ReprocessDialog.tsx (callers own the trigger + `open` state).
 * "Reset to default" is always offered, not just when a custom title is
 * active — clearing a non-overridden title is a safe no-op on the backend
 * (title_overrides.clear_title), so there's no need to track override-vs-
 * extracted state just to decide whether to show the button. */
export default function RenameDialog({ filename, currentName, onRenamed, onToast, open, onOpenChange }: Props) {
  const [name, setName] = useState(currentName);
  const basename = filename.split("/").pop() ?? filename;

  useEffect(() => {
    if (open) setName(currentName);
  }, [open, currentName]);

  const save = async () => {
    onOpenChange(false);
    try {
      await setDocumentTitle(filename, name.trim());
      onRenamed();
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Rename failed", "error");
    }
  };

  const reset = async () => {
    onOpenChange(false);
    try {
      await setDocumentTitle(filename, null);
      onRenamed();
    } catch (e) {
      onToast(e instanceof Error ? e.message : "Reset failed", "error");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Rename source</DialogTitle>
          <DialogDescription>
            Change how &ldquo;{basename}&rdquo; appears in the knowledge base. Leave it as-is to keep the
            current name, or reset to go back to the extracted title or original filename.
          </DialogDescription>
        </DialogHeader>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={basename} autoFocus />
        <DialogFooter>
          <Button variant="ghost" onClick={reset}>
            Reset to default
          </Button>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!name.trim()}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
