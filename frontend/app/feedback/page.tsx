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
