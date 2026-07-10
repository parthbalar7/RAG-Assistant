const { createProxyMiddleware } = require('http-proxy-middleware');

const BACKEND = 'http://localhost:8000';

module.exports = function (app) {
  // One proxy for HTTP + WebSocket. http-proxy-middleware only subscribes to
  // the server's `upgrade` event after the first HTTP request passes through
  // this instance — a ws-only mount at /api/ws never sees an HTTP request, so
  // its upgrade handler never arms and WS connections hang in CONNECTING.
  // Mounting one instance at /api (armed instantly by the /api/stats poll)
  // proxies both, and its path filter leaves CRA's HMR socket at /ws alone.
  app.use(
    '/api',
    createProxyMiddleware({
      target: BACKEND,
      changeOrigin: true,
      ws: true,
    }),
  );
};
