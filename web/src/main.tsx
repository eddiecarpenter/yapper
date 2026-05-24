/**
 * SPA bootstrap entry point.
 *
 * Creates the React root and renders <App /> inside <StrictMode>.
 * The script tag in `index.html` references this file via the
 * `/src/main.tsx` Vite virtual path — bundling rewrites it to the
 * hashed asset in `dist/` at build time.
 *
 * Strict-mode is on per Task 1 notes: double-invokes effects in
 * development to surface non-idempotent cleanup, which is exactly
 * the discipline the `useDialogue` hook was authored under.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";

const rootEl = document.getElementById("root");
if (rootEl === null) {
  throw new Error("yapper: #root element missing from index.html");
}

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
