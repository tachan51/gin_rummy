// Generate training data for the leaf value model: node tools/gen_value.mjs HANDS SEED ['{params}'] > out.csv
// Each row: state at the start of a player's turn (before drawing):
//   myEval (heuristic eval of own 10 cards), oppDW (opponent's true deadwood), stock, outcome (net points for the player)
import { GinHand, makeRng, deadwood } from '../web/js/engine.js';
import { decide, analyze, evalHand, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 100);
const seed = +(process.argv[3] || 1);
const PARAMS = { ...DEFAULT_PARAMS, timeBudget: 0, ...(process.argv[4] ? JSON.parse(process.argv[4]) : {}) };
const rng = makeRng(seed);
const lines = [];
for (let i = 0; i < N; i++) {
  const h = new GinHand({ dealer: i % 2, rng: makeRng(Math.floor(rng() * 2 ** 32)) });
  const rows = [];
  const prng = makeRng(seed * 1000 + i);
  while (h.phase !== 'over') {
    if (h.phase === 'draw' || h.phase === 'upcard') {
      const v = h.view(h.turn);
      const K = analyze(v);
      rows.push([h.turn, evalHand(v.hand, K, DEFAULT_PARAMS, 2), deadwood(h.hands[1 - h.turn]), v.stockCount]);
    }
    h.apply(decide(h.view(h.turn), 'strong', PARAMS, prng));
  }
  const r = h.result || { winner: -1 };
  for (const [p, e, o, s] of rows) {
    const net = r.winner < 0 ? 0 : r.winner === p ? r.points : -r.points;
    lines.push(`${e.toFixed(2)},${o},${s},${net}`);
  }
}
process.stdout.write(lines.join('\n') + '\n');
