// Match equity: probability of winning the match (first to `target`) from a score,
// computed by dynamic programming over the hand-outcome distribution of strong-vs-strong
// self-play (tools/gen_outcomes.mjs). Like a backgammon match equity table.
//
// E(a, b, d) = P(I win the match | my score a, opponent score b, d = 1 if I deal the next hand)
// The dealer alternates every hand (including drawn hands), as in engine.js GinMatch.

import { HAND_OUTCOMES } from './hand-outcomes.js';

function dist(side) {
  const o = HAND_OUTCOMES[side];
  const out = [];
  for (const k of Object.keys(o)) out.push([+k, o[k] / HAND_OUTCOMES.hands]);
  return out;
}
// from the point of view of a player who deals (d = 1) or does not deal (d = 0) this hand
const D_WIN = dist('dealer'), N_WIN = dist('nondealer');
const P_DRAW = HAND_OUTCOMES.draw / HAND_OUTCOMES.hands;
const WIN = [N_WIN, D_WIN];
const LOSE = [D_WIN, N_WIN];

// points distribution of a won hand (either seat), normalised
const ALL_WIN = (() => {
  const m = new Map();
  let tot = 0;
  for (const [k, p] of D_WIN.concat(N_WIN)) { m.set(k, (m.get(k) || 0) + p); tot += p; }
  return [...m].map(([k, p]) => [k, p / tot]);
})();
const MEAN_WIN = ALL_WIN.reduce((s, [k, p]) => s + k * p, 0);

const cache = new Map();

/** Equity table for a target score. Returns get(a, b, d). */
export function equityTable(target) {
  let t = cache.get(target);
  if (t) return t;
  const T = target;
  const E = new Float64Array(T * T * 2);
  const idx = (a, b, d) => (a * T + b) * 2 + d;
  const val = (a, b, d) => (a >= T ? 1 : b >= T ? 0 : E[idx(a, b, d)]);
  for (let a = T - 1; a >= 0; a--) {
    for (let b = T - 1; b >= 0; b--) {
      // R_d: contribution of decided hands when I deal (d=1) or not (d=0); the next hand flips d
      const R = [0, 0];
      for (let d = 0; d < 2; d++) {
        let r = 0;
        for (const [k, p] of WIN[d]) r += p * val(a + k, b, 1 - d);
        for (const [k, p] of LOSE[d]) r += p * val(a, b + k, 1 - d);
        R[d] = r;
      }
      // X0 = R0 + q X1, X1 = R1 + q X0 (drawn hand: same scores, dealer flips)
      const q = P_DRAW;
      const x0 = (R[0] + q * R[1]) / (1 - q * q);
      const x1 = R[1] + q * x0;
      E[idx(a, b, 0)] = x0;
      E[idx(a, b, 1)] = x1;
    }
  }
  t = { target: T, get: val };
  cache.set(target, t);
  return t;
}

/**
 * Utility of a hand result x (my net points: x > 0 I score x, x < 0 the opponent scores -x, 0 drawn hand),
 * measured as my match-winning probability afterwards. Non-integer x (expected values) is interpolated.
 */
export function matchUtility(myScore, oppScore, target, iDealNow) {
  const tab = equityTable(target);
  const dNext = iDealNow ? 0 : 1;
  const LIM = 160;
  const u = new Float64Array(2 * LIM + 1);
  for (let x = -LIM; x <= LIM; x++) {
    u[x + LIM] = x > 0 ? tab.get(myScore + x, oppScore, dNext) : x < 0 ? tab.get(myScore, oppScore - x, dNext) : tab.get(myScore, oppScore, dNext);
  }
  const util = (x) => {
    if (x <= -LIM) return u[0];
    if (x >= LIM) return u[2 * LIM];
    const f = Math.floor(x);
    const w = x - f;
    return u[f + LIM] * (1 - w) + (w > 0 ? u[f + 1 + LIM] * w : 0);
  };
  // Expected utility of a position whose *expected* hand result is v points.
  // Applying util() to an expected value would distort risk (Jensen), so the position is modelled as
  // a drawn hand with the usual probability, otherwise a win with probability q or a loss, with the
  // usual distribution of winning points; q is chosen so that the expected points equal v.
  let uw = 0, ul = 0;
  for (const [k, p] of ALL_WIN) { uw += p * util(k); ul += p * util(-k); }
  const ud = util(0);
  util.leaf = (v) => {
    let q = (v / ((1 - P_DRAW) * MEAN_WIN) + 1) / 2;
    if (q < 0) q = 0; else if (q > 1) q = 1;
    return P_DRAW * ud + (1 - P_DRAW) * (q * uw + (1 - q) * ul);
  };
  return util;
}
