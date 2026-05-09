import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vault RAG",
  description: "Business document intelligence platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-[#0a0a0a] text-zinc-100 h-screen overflow-hidden">
        {children}
      </body>
    </html>
  );
}
