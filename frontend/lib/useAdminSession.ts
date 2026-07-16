"use client";

import { useCallback, useEffect, useState } from "react";
import { getAdminSession, type AdminSession } from "@/lib/api";

/** access_mode="open" (the default) always reports isAdmin=true -- there's no
 * login in that mode, everyone has admin capability, matching today's
 * behavior. Only access_mode="admin_viewer" actually gates anything. */
export function useAdminSession() {
  const [session, setSession] = useState<AdminSession>({ access_mode: "open", is_admin: true });
  const [loaded, setLoaded] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setSession(await getAdminSession());
    } catch {
      // Backend unreachable -- default (open/is_admin=true) keeps the UI usable;
      // real endpoints will fail on their own if the backend really is down.
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { ...session, loaded, refresh };
}
