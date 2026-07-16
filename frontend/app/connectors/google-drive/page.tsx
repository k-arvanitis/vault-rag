"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function GoogleDriveRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace(`/admin/integrations/google-drive${window.location.search}`);
  }, [router]);
  return null;
}
