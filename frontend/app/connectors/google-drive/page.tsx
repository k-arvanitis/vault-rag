"use client";

import { useRouter } from "next/navigation";
import GoogleDrivePanel from "@/components/GoogleDrivePanel";

export default function GoogleDrivePage() {
  const router = useRouter();
  return <GoogleDrivePanel onClose={() => router.push("/")} />;
}
