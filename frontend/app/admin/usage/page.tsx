"use client";

import { useRouter } from "next/navigation";
import UsagePanel from "@/components/UsagePanel";

export default function AdminUsagePage() {
  const router = useRouter();
  return <UsagePanel onClose={() => router.push("/")} />;
}
