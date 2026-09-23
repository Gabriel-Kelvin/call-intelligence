import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  agentRules: false,
  async rewrites() {
    const backend = process.env.API_BACKEND_URL || "http://127.0.0.1:8000";
    return {
      beforeFiles: [
        {
          source: "/api/:path*",
          destination: `${backend}/api/:path*`,
        },
      ],
      afterFiles: [],
      fallback: [],
    };
  },
};

export default nextConfig;
