// Runs the AI off the main thread.
import { decide } from './ai.js';

self.onmessage = (e) => {
  const { id, view, level } = e.data;
  let action = null;
  try { action = decide(view, level); } catch (err) { console.error(err); }
  self.postMessage({ id, action });
};
