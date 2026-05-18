/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [{ key: "X-Content-Type-Options", value: "nosniff" }],
      },
    ];
  },
  // Proxy /backend/* to the FastAPI server. The browser only ever talks to the
  // Next.js origin (port 3000); Next forwards to the API server-side, so a
  // remote-dev setup needs only the port-3000 SSH forward.
  async rewrites() {
    const apiTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8001";
    return [{ source: "/backend/:path*", destination: `${apiTarget}/:path*` }];
  },
};

export default nextConfig;
