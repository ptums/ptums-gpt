/** @type {import('next').NextConfig} */
const nextConfig = {
  basePath: "/home-gpt",
  env: {
    OLLAMA_API_URL: process.env.OLLAMA_API_URL,
    MODEL: process.env.MODEL,
  },
};

module.exports = nextConfig;
