// Self-play harness: node tools/selfplay.mjs [hands] [A] [B] [seed] ['{"param":value}' for A]
// A/B: strong | normal | easy
import { GinHand, makeRng } from '../web/js/engine.js';
import { decide, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 200);
const A = process.argv[3] || 'strong';
const B = process.argv[4] || 'normal';
const seed = +(process.argv[5] || 1);
const overA = process.argv[6] ? JSON.parse(process.argv[6]) : {};
const overB = process.argv[7] ? JSON.parse(process.argv[7]) : {};
const PA = { ...DEFAULT_PARAMS, ...overA };
const PB = { ...DEFAULT_PARAMS, ...overB };

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
      if (pl.level === "strong") { maxMs = Math.max(maxMs, dt); totMs += dt; nDec++; }
      h.apply(a);
    }
    const r = h.result;
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
