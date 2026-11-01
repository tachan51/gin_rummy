// How well does the simulated opponent predict real discards?
//   node tools/oppsim_eval.mjs HANDS SEED ['{params for the simulated opponent}']
// Plays strong vs strong; at each discard (without knocking) compares the real discard with the
// simulation policy (oppDiscardIdx) given the true 11-card hand.
import { GinHand, makeRng, discardDeadwoods } from '../web/js/engine.js';
import { decide, oppDiscardIdx, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 50);
const seed = +(process.argv[3] || 1);
const SIM = { ...DEFAULT_PARAMS, ...(process.argv[4] ? JSON.parse(process.argv[4]) : {}) };
const P = { ...DEFAULT_PARAMS, timeBudget: 0, searchMin: 100, drawSearchMin: 100 };
const rng = makeRng(seed);
let hit = 0, tot = 0;
const out = new Int32Array(32);
for (let i = 0; i < N; i++) {
  const h = new GinHand({ dealer: i % 2, rng: makeRng(Math.floor(rng() * 2 ** 32)) });
  const prng = makeRng(seed * 7919 + i);
  while (h.phase !== 'over') {
    const p = h.turn;
    const a = decide(h.view(p), 'strong', P, prng);
    if (h.phase === 'discard' && a.type === 'discard' && !a.knock) {
      const hand = h.hands[p];
      // cards the *other* player visibly picked up and still holds
      const picks = new Set();
      for (const e of h.events) { if (e.p !== 1 - p) continue; if (e.type === 'take') picks.add(e.card); else if (e.type === 'discard') picks.delete(e.card); }
      discardDeadwoods(hand, out);
      const k = oppDiscardIdx(hand, out, h.takenCard, { ...SIM, _myPicks: [...picks] });
      if (hand[k] === a.card) hit++;
      tot++;
    }
    h.apply(a);
  }
}
console.log(JSON.stringify({ discards: tot, predicted: +(hit / tot).toFixed(4), sim: process.argv[4] || 'greedy' }));
