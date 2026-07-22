import { describe, it, expect } from "vitest";
import { toDisplayTitle } from "./product";

describe("toDisplayTitle", () => {
  it("title-cases an all-caps extracted title", () => {
    expect(toDisplayTitle("POLICY FOR THE PROCUREMENT OF GOODS AND SERVICES (PGS)")).toBe(
      "Policy for the Procurement of Goods and Services (PGS)"
    );
  });

  it("leaves a mixed-case title unchanged", () => {
    expect(toDisplayTitle("Services Contract Terms")).toBe("Services Contract Terms");
  });

  it("leaves an all-lowercase title unchanged", () => {
    expect(toDisplayTitle("services contract terms")).toBe("services contract terms");
  });

  it("capitalizes the first word even if it's a minor word", () => {
    expect(toDisplayTitle("THE ANNUAL REPORT")).toBe("The Annual Report");
  });
});
