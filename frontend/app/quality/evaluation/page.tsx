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
