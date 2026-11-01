// Training data for the neural leaf-value model: node tools/gen_nn.mjs HANDS SEED ['{params, "level": "normal"}'] > out.jsonl
// One line per position at the start of a player's turn (before drawing), perfect-information view:
//   [myHand[10], oppHand[10], discardPile[...], stock, myEval, myDW, oppDW, outcome]
// outcome = the player's net points of the hand (0 for a drawn hand).
import { GinHand, makeRng, deadwood } from '../web/js/engine.js';
import { decide, analyze, evalHand, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 100);
const seed = +(process.argv[3] || 1);
const P = { ...DEFAULT_PARAMS, timeBudget: 0, searchMin: 200, drawSearchMin: 200, ...(process.argv[4] ? JSON.parse(process.argv[4]) : {}) };
const LEVEL = P.level || 'strong'; // 'normal' (heuristic only) is ~10x faster
const rng = makeRng(seed);
for (let i = 0; i < N; i++) {
  const h = new GinHand({ dealer: i % 2, rng: makeRng(Math.floor(rng() * 2 ** 32)) });
  const prng = makeRng(seed * 7919 + i);
  const rows = [];
  while (h.phase !== 'over') {
    if (h.phase === 'draw' || h.phase === 'upcard') {
      const p = h.turn;
      const v = h.view(p);
      const K = analyze(v);
      rows.push([p, [h.hands[p].slice(), h.hands[1 - p].slice(), h.discardPile.slice(), h.stock.length,
        +evalHand(v.hand, K, P, 2).toFixed(2), deadwood(h.hands[p]), deadwood(h.hands[1 - p])]]);
    }
    h.apply(decide(h.view(h.turn), LEVEL, P, prng));
  }
  const r = h.result || { winner: -1 };
  const out = [];
  for (const [p, row] of rows) out.push(JSON.stringify(row.concat([r.winner < 0 ? 0 : r.winner === p ? r.points : -r.points])));
  process.stdout.write(out.join('\n') + '\n');
}
