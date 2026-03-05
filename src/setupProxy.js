const { createProxyMiddleware } = require('http-proxy-middleware');

const BACKEND = 'http://localhost:8000';

module.exports = function (app) {
  // WebSocket proxy — must be registered before the HTTP proxy
  app.use(
    '/api/ws',
    createProxyMiddleware({
      target: BACKEND,
      ws: true,
      changeOrigin: true,
    })
  );

  // HTTP API proxy (replaces the "proxy" field in package.json)
  app.use(
    '/api',
    createProxyMiddleware({
      target: BACKEND,
      changeOrigin: true,
    })
  );
};
