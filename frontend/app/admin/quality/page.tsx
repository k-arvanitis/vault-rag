"use client";

import { useRouter } from "next/navigation";
import EvalPanel from "@/components/EvalPanel";

export default function AdminQualityPage() {
  const router = useRouter();
  return <EvalPanel onClose={() => router.push("/")} />;
}
