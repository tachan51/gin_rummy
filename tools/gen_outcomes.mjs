// Hand-outcome distribution by seat (for the match equity table):
//   node tools/gen_outcomes.mjs HANDS SEED > out.json
// Output: { dealer: {points: count}, nondealer: {points: count}, draw: count, hands }
import { GinHand, makeRng } from '../web/js/engine.js';
import { decide, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 100);
const seed = +(process.argv[3] || 1);
const P = { ...DEFAULT_PARAMS, timeBudget: 0, searchMin: 200, drawSearchMin: 200 };
const rng = makeRng(seed);
const res = { dealer: {}, nondealer: {}, draw: 0, hands: 0 };
for (let i = 0; i < N; i++) {
  const h = new GinHand({ dealer: i % 2, rng: makeRng(Math.floor(rng() * 2 ** 32)) });
  const prng = makeRng(seed * 7919 + i);
  while (h.phase !== 'over') h.apply(decide(h.view(h.turn), 'strong', P, prng));
  const r = h.result || { winner: -1 };
  res.hands++;
  if (r.winner < 0) { res.draw++; continue; }
  const side = r.winner === h.dealer ? 'dealer' : 'nondealer';
  res[side][r.points] = (res[side][r.points] || 0) + 1;
}
process.stdout.write(JSON.stringify(res) + '\n');
