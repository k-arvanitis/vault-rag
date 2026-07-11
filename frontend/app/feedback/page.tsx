"use client";

import { useRouter } from "next/navigation";
import FeedbackPanel from "@/components/FeedbackPanel";

export default function FeedbackPage() {
  const router = useRouter();
  return <FeedbackPanel onClose={() => router.push("/")} />;
}
