// frontend/e2e/admin-viewer-access.spec.ts
import { test, expect } from "@playwright/test";

// Real backend enforcement of ACCESS_MODE=admin_viewer is covered by
// tests/test_admin_auth.py (14 cases against require_admin() directly) --
// that's what actually stops a viewer from uploading/deleting even if they
// forge a request. These tests prove the frontend half: a viewer session
// never even sees the admin page shell for /admin/*, and an admin session
// can reach every admin screen and switch back to the User workspace
// without logging out. The dev server this suite runs against stays in
// ACCESS_MODE=open (today's default) -- admin_viewer is exercised here by
// mocking GET /admin/session's response, not by restarting the backend.

test("a viewer is blocked from every /admin/* route", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "admin_viewer", is_admin: false }),
    })
  );

  for (const path of ["/admin/sources", "/admin/quality", "/admin/feedback", "/admin/integrations/google-drive"]) {
    await page.goto(path);
    await expect(page.getByText(/Admin access required/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /Admin login/i })).toBeVisible();
  }

  // No "Admin" link in the User workspace nav for a viewer.
  await page.goto("/");
  await expect(page.getByRole("button", { name: /^Admin$/i })).not.toBeVisible();
});

test("an admin session reaches every admin screen and can return to chat", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "admin_viewer", is_admin: true }),
    })
  );

  await page.goto("/");
  await expect(page.getByRole("button", { name: /^Admin$/i })).toBeVisible();
  await page.getByRole("button", { name: /^Admin$/i }).click();
  await expect(page).toHaveURL(/\/admin\/sources$/);

  for (const [label, urlPart] of [
    ["Quality", "/admin/quality"],
    ["Integrations", "/admin/integrations/google-drive"],
    ["Feedback", "/admin/feedback"],
  ] as const) {
    await page.getByRole("button", { name: new RegExp(`^${label}$`, "i") }).click();
    await expect(page).toHaveURL(new RegExp(urlPart.replace(/\//g, "\\/") + "$"));
  }

  // Chat link switches back to the User workspace without logging out.
  await page.getByRole("button", { name: /^Chat$/i }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("button", { name: /^Admin$/i })).toBeVisible();
});

test("old routes redirect to their /admin/* equivalents", async ({ page }) => {
  await page.route("**/admin/session", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_mode: "open", is_admin: true }),
    })
  );

  await page.goto("/sources?doc=test.pdf");
  await expect(page).toHaveURL(/\/admin\/sources\?doc=test\.pdf$/);

  await page.goto("/feedback");
  await expect(page).toHaveURL(/\/admin\/feedback$/);

  await page.goto("/connectors/google-drive");
  await expect(page).toHaveURL(/\/admin\/integrations\/google-drive$/);

  await page.goto("/quality/evaluation");
  await expect(page).toHaveURL(/\/admin\/quality$/);
});
