"use client";

import { useRouter } from "next/navigation";
import ModelProviderPanel from "@/components/ModelProviderPanel";

export default function AdminModelPage() {
  const router = useRouter();
  return <ModelProviderPanel onClose={() => router.push("/")} />;
}
