import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, normalize, resolve, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vitest/config';

/**
 * Dev only: `MMD_SAMPLE_DIR=/path/to/models pnpm dev` serves that folder at /samples/,
 * so a model can be opened with `?model=/samples/<folder>/<model>.pmx`.
 */
function serveSamples(): Plugin {
  const root = process.env.MMD_SAMPLE_DIR;
  return {
    name: 'serve-mmd-samples',
    apply: 'serve',
    configureServer(server) {
      if (!root) return;
      const base = resolve(root);
      server.middlewares.use('/samples', (req, res, next) => {
        const rel = decodeURIComponent((req.url ?? '/').split('?')[0]);
        const file = normalize(join(base, rel));
        if (!file.startsWith(base + sep) || !existsSync(file) || !statSync(file).isFile()) return next();
        res.setHeader('Cache-Control', 'no-cache');
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [serveSamples()],
  build: {
    target: 'es2022',
  },
  // The physics and bake workers are ES modules sharing the solver code with the main bundle.
  worker: {
    format: 'es',
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
