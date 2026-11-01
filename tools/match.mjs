// Match self-play (first to TARGET points): node tools/match.mjs MATCHES A B SEED ['{A params}'] ['{B params}'] [TARGET]
// Levels: strong | normal | easy | v1. Each match is played twice with swapped seats and the same decks.
// Reports A's match win rate with a 95% confidence interval.
import { makeNN } from '../web/js/nn.js';
import { GinMatch, makeRng } from '../web/js/engine.js';
import { decide as decideNew, DEFAULT_PARAMS } from '../web/js/ai.js';
import { decide as decideV1, DEFAULT_PARAMS as V1_PARAMS } from './baseline/ai_v1.js';

const N = +(process.argv[2] || 50);
const A = process.argv[3] || 'strong';
const B = process.argv[4] || 'normal';
const seed = +(process.argv[5] || 1);
const overA = process.argv[6] ? JSON.parse(process.argv[6]) : {};
const overB = process.argv[7] ? JSON.parse(process.argv[7]) : {};
// {"nnFile": "path/to/nn-value.js"} loads alternative neural-network weights
for (const o of [overA, overB]) {
  if (o.nnFile) { const m = await import(new URL(o.nnFile, 'file://' + process.cwd() + '/').href); o.nnModel = makeNN(m.NN_LAYERS, m.NN_SCALE); delete o.nnFile; }
}
const target = +(process.argv[8] || 100);
// fixed sample counts instead of a time budget, so results do not depend on machine load
const BENCH = { timeBudget: 0, searchMin: 200, drawSearchMin: 200, mctsMin: 3000 };
const players = [
  { level: A, params: { ...DEFAULT_PARAMS, ...BENCH, ...overA } },
  { level: B, params: { ...DEFAULT_PARAMS, ...BENCH, ...overB } },
];
const decide = (view, pl, rng) => (pl.level === 'v1' ? decideV1(view, 'strong', V1_PARAMS, rng) : decideNew(view, pl.level, pl.params, rng));

const master = makeRng(seed);
let winsA = 0, matches = 0, hands = 0, ptsA = 0, ptsB = 0;
for (let i = 0; i < N; i++) {
  const mSeed = Math.floor(master() * 2 ** 32);
  const firstDealer = i % 2;
  for (const swap of [0, 1]) {
    // seat p is played by players[p ^ swap]
    const m = new GinMatch({ target, rng: makeRng(mSeed), firstDealer });
    const prng = [makeRng(mSeed ^ 0x9e3779b9), makeRng(mSeed ^ 0x7f4a7c15)];
    while (!m.final) {
      const h = m.newHand();
      let guard = 0;
      while (h.phase !== 'over' && guard++ < 1000) {
        const seat = h.turn;
        const v = h.view(seat);
        v.match = { myScore: m.scores[seat], oppScore: m.scores[1 - seat], target };
        h.apply(decide(v, players[seat ^ swap], prng[seat ^ swap]));
      }
      const r = h.result || { type: 'draw', winner: -1, points: 0 };
      if (r.winner >= 0) { if ((r.winner ^ swap) === 0) ptsA += r.points; else ptsB += r.points; }
      m.recordResult(r);
      hands++;
    }
    matches++;
    if ((m.final.winner ^ swap) === 0) winsA++;
  }
}
const p = winsA / matches;
const ci = 1.96 * Math.sqrt(p * (1 - p) / matches);
console.log(`${A} vs ${B}: ${matches} matches to ${target}, seed ${seed}`);
console.log(`  A match win rate ${(100 * p).toFixed(1)}% ± ${(100 * ci).toFixed(1)}%  (${winsA}/${matches})`);
console.log(`  hands/match ${(hands / matches).toFixed(1)}, A avg net ${((ptsA - ptsB) / hands).toFixed(2)} pts/hand`);
