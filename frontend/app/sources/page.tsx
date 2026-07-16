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
