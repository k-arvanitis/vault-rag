import { test, expect } from "@playwright/test";

// Real backend enforcement of ACCESS_MODE=admin_viewer is covered by
// tests/test_admin_auth.py (14 cases against require_admin() directly) --
// that's what actually stops a viewer from uploading/deleting even if they
// forge a request. This test proves the other half: the UI itself correctly
// hides those actions for a viewer session, so there's no button to click in
// the first place. The dev server this suite runs against stays in
// ACCESS_MODE=open (today's default) -- admin_viewer is exercised here by
// mocking GET /admin/session's response, not by restarting the backend.
test("a viewer session sees no upload or delete controls", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "admin_viewer", is_admin: false }),
    })
  );

  await page.goto("/");

  // The upload dropzone (Sidebar's UploadZone) must not render for a viewer.
  await expect(page.getByText(/Upload documents/i)).not.toBeVisible();

  // If any source exists, its per-row Delete/Reprocess icon buttons must not
  // render either (Sidebar only shows them when isAdmin).
  await expect(page.getByRole("button", { name: "Delete document" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Reprocess document" })).toHaveCount(0);

  // The header shows an "Admin login" entry point instead of admin nav.
  await expect(page.getByRole("button", { name: /Admin login/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Integrations/i })).not.toBeVisible();
});
