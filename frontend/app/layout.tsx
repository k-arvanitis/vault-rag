import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vault RAG",
  description: "Self-hosted document intelligence — query mixed-format business documents with cited answers",
};

// Runs before paint so the page doesn't flash the wrong theme.
const themeInit = `try{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.classList.add('dark')}catch(e){}`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="antialiased h-screen overflow-hidden">{children}</body>
    </html>
  );
}
