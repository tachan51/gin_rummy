// Inference for the neural leaf-value model (weights in nn-value.js, trained by tools/train_nn.py).
// Small MLP; the first layer is computed sparsely from the one-hot card inputs.
import { NN_LAYERS, NN_SCALE } from './nn-value.js';

const MARK = new Uint8Array(52);

function addRow(h, w, row, n, f = 1) {
  const o = row * n;
  for (let j = 0; j < n; j++) h[j] += f * w[o + j];
}

/** Build an evaluator from layer weights (nn-value.js format). */
export function makeNN(layers, scale = NN_SCALE) {
  const LAYERS = layers.map((l) => ({ nin: l.in, nout: l.out, w: Float32Array.from(l.w), b: Float32Array.from(l.b) }));
  const ACT = LAYERS.map((l) => new Float32Array(l.nout));
  return (my, opp, dead, top, stock, myEval, myDW, oppDW) => evaluate(LAYERS, ACT, scale, my, opp, dead, top, stock, myEval, myDW, oppDW);
}

/**
 * Expected net points of the hand for the player about to draw.
 * my, opp: 10-card hands; dead: discard-pile cards (duplicates allowed); top: card on top of the pile (-1 if none)
 */
function evaluate(LAYERS, ACT, scale, my, opp, dead, top, stock, myEval, myDW, oppDW) {
  const L0 = LAYERS[0], n = L0.nout, w = L0.w, h = ACT[0];
  h.set(L0.b);
  for (let i = 0; i < my.length; i++) addRow(h, w, my[i], n);
  for (let i = 0; i < opp.length; i++) addRow(h, w, 52 + opp[i], n);
  for (let i = 0; i < dead.length; i++) {
    const c = dead[i];
    if (!MARK[c]) { MARK[c] = 1; addRow(h, w, 104 + c, n); }
  }
  for (let i = 0; i < dead.length; i++) MARK[dead[i]] = 0;
  if (top >= 0) addRow(h, w, 156 + top, n);
  addRow(h, w, 208, n, stock / 31);
  addRow(h, w, 209, n, myEval / 30);
  addRow(h, w, 210, n, myDW / 50);
  addRow(h, w, 211, n, oppDW / 50);
  let prev = h;
  for (let k = 1; k < LAYERS.length; k++) {
    for (let j = 0; j < prev.length; j++) if (prev[j] < 0) prev[j] = 0; // ReLU of the previous layer
    const L = LAYERS[k], out = ACT[k];
    out.set(L.b);
    for (let i = 0; i < L.nin; i++) {
      const a = prev[i];
      if (a !== 0) addRow(out, L.w, i, L.nout, a);
    }
    prev = out;
  }
  return prev[0] * scale;
}

export const nnValue = makeNN(NN_LAYERS);
