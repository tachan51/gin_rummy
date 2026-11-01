// Self-play harness: node tools/selfplay.mjs [hands] [A] [B] [seed] ['{"param":value}' for A]
// A/B: strong | v1 | normal | easy
import { GinHand, makeRng } from '../web/js/engine.js';
import { decide as decideNew, DEFAULT_PARAMS } from '../web/js/ai.js';
import { decide as decideV1, DEFAULT_PARAMS as V1_PARAMS } from './baseline/ai_v1.js';
// level 'v1' = frozen first release of the strong AI (benchmark)
const decide = (view, level, params, rng) => (level === 'v1' ? decideV1(view, 'strong', V1_PARAMS, rng) : decideNew(view, level, params, rng));

const N = +(process.argv[2] || 200);
const A = process.argv[3] || 'strong';
const B = process.argv[4] || 'normal';
const seed = +(process.argv[5] || 1);
const overA = process.argv[6] ? JSON.parse(process.argv[6]) : {};
const overB = process.argv[7] ? JSON.parse(process.argv[7]) : {};
// {"valueFile": "path/to/value.js"} loads alternative leaf-value weights
for (const o of [overA, overB]) {
  if (o.valueFile) { o.valueW = (await import(new URL(o.valueFile, 'file://' + process.cwd() + '/').href)).VALUE_W; delete o.valueFile; }
}
// fixed sample counts instead of a time budget, so results do not depend on machine load
const BENCH = { timeBudget: 0, searchMin: 300, drawSearchMin: 300 };
const PA = { ...DEFAULT_PARAMS, ...BENCH, ...overA };
const PB = { ...DEFAULT_PARAMS, ...BENCH, ...overB };

const rng = makeRng(seed);
const players = [
  { level: A, params: PA, rng: makeRng(seed * 7 + 1) },
  { level: B, params: PB, rng: makeRng(seed * 13 + 5) },
];
let pts = [0, 0], wins = [0, 0], draws = 0;
const types = {};
let maxMs = 0, totMs = 0, nDec = 0;
for (let i = 0; i < N; i++) {
  // play each deal twice with swapped seats to reduce variance
  const dealSeed = Math.floor(rng() * 2 ** 32);
  for (const swap of [0, 1]) {
    const h = new GinHand({ dealer: i % 2, rng: makeRng(dealSeed) });
    const seatOf = (p) => (p ^ swap); // seat p is played by players[p^swap]
    let guard = 0;
    while (h.phase !== 'over' && guard++ < 500) {
      const pl = players[seatOf(h.turn)];
      const t0 = performance.now();
      const a = decide(h.view(h.turn), pl.level, pl.params, pl.rng);
      const dt = performance.now() - t0;
      if (pl.level !== "normal" && pl.level !== "easy") { maxMs = Math.max(maxMs, dt); totMs += dt; nDec++; }
      h.apply(a);
    }
    const r = h.result || { type: 'draw', winner: -1, points: 0 };
    types[r.type] = (types[r.type] || 0) + 1;
    if (r.winner < 0) { draws++; continue; }
    const who = seatOf(r.winner);
    pts[who] += r.points;
    wins[who]++;
  }
}
const games = N * 2;
console.log(`${A} vs ${B}: ${games} hands, seed ${seed}`);
console.log(`  wins ${wins[0]} - ${wins[1]}, draws ${draws}`);
console.log(`  points ${pts[0]} - ${pts[1]}  => A avg net ${((pts[0] - pts[1]) / games).toFixed(2)} pts/hand`);
console.log(`  result types`, types);
if (nDec) console.log(`  strong decision time avg ${(totMs / nDec).toFixed(1)}ms, max ${maxMs.toFixed(0)}ms`);
