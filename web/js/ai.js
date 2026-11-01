// Gin Rummy AI.
//
// "strong" level:
//  * card tracking: which cards are dead, which are known to be in the opponent's hand
//    (picked from the discard pile), and soft inference from what the opponent discarded
//    or declined (opponent hand probability per card).
//  * hand evaluation = minimum deadwood - alpha * expected deadwood reduction of the next draw
//    (+ bonuses for draws that reach knock range / gin).
//  * draw decision: exact expectation over the unseen cards for drawing from stock vs taking
//    the upcard (1-ply expectimax using the evaluation above).
//  * discard decision: evaluation + danger penalty (probability the opponent can meld the card).
//  * knock decision: Monte-Carlo samples of the opponent hand (simulated from their draw history,
//    respecting known cards) to estimate the undercut risk and the expected knock gain.

import { VAL, suitOf, rankOf, deadwood, discardDeadwoods, defenderResult, arrange, KNOCK_LIMIT, makeRng } from './engine.js';

export const DEFAULT_PARAMS = {
  alpha: 1.2, // weight of draw potential
  alphaDecayStock: 25, // potential fades when stock is below this many cards
  knockBonus: 4, // bonus (deadwood points) for a draw that enters knock range
  ginBonus: 8, // bonus for a draw that makes gin
  beta: 4, // danger penalty weight
  takeMargin: 0.0, // required advantage to take the upcard
  topK: 4, // discards considered per hypothetical draw
  samples: 60, // knock-decision samples
  undercutMax: 0.18, // max acceptable undercut probability
  knockEvMin: 0, // min expected gain to knock
  lateStock: 8, // with this few stock cards or less: knock more readily
  ginHunt: 0, // if > 0, don't knock with deadwood above this (0 = disabled)
  deepK: 0, // 2-ply lookahead over this many best discards (0 = off)
};

// ---------------------------------------------------------------------------
// Knowledge
// ---------------------------------------------------------------------------
export function analyze(view) {
  const me = view.me, opp = 1 - me;
  const status = new Int8Array(52); // 0 unseen, 1 mine, 2 dead, 3 opp known
  for (const c of view.hand) status[c] = 1;
  for (const c of view.discardPile) status[c] = 2;
  const oppKnown = new Set();
  const oppRejected = []; // discarded or declined by opponent
  const oppPicked = []; // ever picked from discard by opponent
  let oppTurns = 0, myTurns = 0;
  for (const e of view.events) {
    if (e.p === opp) {
      if (e.type === 'take') { oppKnown.add(e.card); oppPicked.push(e.card); oppTurns++; }
      else if (e.type === 'stock') { oppTurns++; if (e.declined != null) oppRejected.push(e.declined); }
      else if (e.type === 'discard') { oppKnown.delete(e.card); oppRejected.push(e.card); }
      else if (e.type === 'pass') oppRejected.push(e.card);
    } else if (e.type === 'take' || e.type === 'stock') myTurns++;
  }
  for (const c of oppKnown) status[c] = 3;

  // opponent interest weights for unseen cards
  const w = new Float64Array(52);
  const unseen = [];
  for (let c = 0; c < 52; c++) if (status[c] === 0) { unseen.push(c); w[c] = 1; }
  const adj = (x, f1, f2, fs) => {
    const s = suitOf(x), r = rankOf(x);
    for (let s2 = 0; s2 < 4; s2++) { const c = s2 * 13 + r; if (c !== x && status[c] === 0) w[c] *= fs; }
    for (const d of [-1, 1]) { const r2 = r + d; if (r2 >= 0 && r2 < 13 && status[s * 13 + r2] === 0) w[s * 13 + r2] *= f1; }
    for (const d of [-2, 2]) { const r2 = r + d; if (r2 >= 0 && r2 < 13 && status[s * 13 + r2] === 0) w[s * 13 + r2] *= f2; }
  };
  for (const x of oppRejected) adj(x, 0.7, 0.88, 0.65);
  for (const x of oppPicked) adj(x, 1.6, 1.25, 1.7);

  const handSize = view.phase === 'discard' && view.turn === opp ? 11 : 10;
  const U = Math.max(0, handSize - oppKnown.size);
  let sw = 0;
  for (const c of unseen) sw += w[c];
  const pOpp = new Float64Array(52);
  for (let c = 0; c < 52; c++) if (status[c] === 3) pOpp[c] = 1;
  if (sw > 0) for (const c of unseen) pOpp[c] = Math.min(0.97, (U * w[c]) / sw);
  const pDraw = new Float64Array(52);
  let sd = 0;
  for (const c of unseen) { pDraw[c] = 1 - pOpp[c]; sd += pDraw[c]; }
  if (sd > 0) for (const c of unseen) pDraw[c] /= sd;

  return { me, opp, status, oppKnown, unseen, pOpp, pDraw, oppTurns, myTurns, U, stockCount: view.stockCount };
}

/** Probability that the opponent can use card d in a meld (set or run). */
export function danger(d, K) {
  const q = (c) => (K.status[c] === 3 ? 1 : K.status[c] === 0 ? K.pOpp[c] : 0);
  const r = rankOf(d), s = suitOf(d);
  const qs = [];
  for (let s2 = 0; s2 < 4; s2++) if (s2 !== s) qs.push(q(s2 * 13 + r));
  const [a, b, c] = qs;
  const pSet = a * b + a * c + b * c - 2 * a * b * c;
  const qr = (rr) => (rr >= 0 && rr < 13 ? q(s * 13 + rr) : 0);
  const m2 = qr(r - 2), m1 = qr(r - 1), p1 = qr(r + 1), p2 = qr(r + 2);
  const pRun = 1 - (1 - m2 * m1) * (1 - m1 * p1) * (1 - p1 * p2);
  return 1 - (1 - pSet) * (1 - pRun);
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
const BUF11 = [new Array(11), new Array(11), new Array(11), new Array(11)];
const OUT = [new Int32Array(32), new Int32Array(32), new Int32Array(32), new Int32Array(32)];

function alphaEff(P, K) {
  const st = K.stockCount;
  const f = Math.min(1, Math.max(0, (st - 2) / P.alphaDecayStock));
  return P.alpha * f;
}

/** Evaluate a 10-card hand (lower is better). */
export function evalHand(hand, K, P, level = 2) {
  const dw = deadwood(hand);
  const a = alphaEff(P, K);
  if (a === 0) return dw;
  const h = BUF11[level];
  const out = OUT[level];
  for (let i = 0; i < 10; i++) h[i] = hand[i];
  let pot = 0;
  for (const c of K.unseen) {
    const pc = K.pDraw[c];
    if (pc <= 0) continue;
    h[10] = c;
    const m = discardDeadwoods(h, out);
    if (m < dw) {
      let g = dw - m;
      if (m === 0) g += P.ginBonus;
      else if (m <= KNOCK_LIMIT && dw > KNOCK_LIMIT) g += P.knockBonus;
      pot += pc * g;
    }
  }
  return dw - a * pot;
}

/**
 * Best discard for an 11-card hand. Returns {card, score, idx}.
 * forbidden: card that may not be discarded.
 */
function bestDiscard(hand11, K, P, forbidden, level, dangerCache) {
  const out = OUT[level + 1];
  discardDeadwoods(hand11, out);
  // candidates: topK by deadwood after discard (with danger tiebreak)
  const idx = [];
  for (let k = 0; k < 11; k++) if (hand11[k] !== forbidden) idx.push(k);
  idx.sort((x, y) => out[x] - out[y] || VAL[hand11[y]] - VAL[hand11[x]]);
  const cand = idx.slice(0, P.topK);
  let best = null;
  const rest = new Array(10);
  for (const k of cand) {
    let j = 0;
    for (let i = 0; i < 11; i++) if (i !== k) rest[j++] = hand11[i];
    const d = hand11[k];
    let dg = dangerCache[d];
    if (dg === undefined) dg = dangerCache[d] = danger(d, K);
    const sc = evalHand(rest, K, P, level) + P.beta * dg;
    if (!best || sc < best.score) best = { card: d, score: sc, idx: k };
  }
  return best;
}

// ---------------------------------------------------------------------------
// Knock decision via opponent-hand sampling
// ---------------------------------------------------------------------------
function sampleOpponentHand(K, rng) {
  const pool = K.unseen.slice();
  const hand = [...K.oppKnown];
  // weighted sample without replacement for the unknown part
  const take = (weighted) => {
    let tot = 0;
    for (const c of pool) tot += weighted ? K.pOpp[c] + 1e-6 : 1;
    let x = rng() * tot;
    for (let i = 0; i < pool.length; i++) {
      x -= weighted ? K.pOpp[pool[i]] + 1e-6 : 1;
      if (x <= 0 || i === pool.length - 1) { const c = pool[i]; pool[i] = pool[pool.length - 1]; pool.pop(); return c; }
    }
  };
  while (hand.length < 10 && pool.length) hand.push(take(true));
  // simulate the opponent improving the hand with extra stock draws
  const extra = Math.min(K.oppTurns, 10);
  const out = new Int32Array(32);
  for (let t = 0; t < extra && pool.length; t++) {
    const c = take(false);
    hand.push(c);
    discardDeadwoods(hand, out);
    let bk = -1, bv = 1e9;
    for (let k = 0; k < hand.length; k++) {
      if (K.oppKnown.has(hand[k])) continue;
      if (out[k] < bv || (out[k] === bv && VAL[hand[k]] > VAL[hand[bk]])) { bv = out[k]; bk = k; }
    }
    pool.push(hand[bk]); // returned card is unknown to us anyway
    hand.splice(bk, 1);
  }
  return hand;
}

export function knockStats(hand10, K, P, rng) {
  const myArr = arrange(hand10);
  const mydw = myArr.dw;
  let under = 0, ev = 0;
  const n = P.samples;
  for (let i = 0; i < n; i++) {
    const oh = sampleOpponentHand(K, rng);
    const def = defenderResult(oh, myArr.melds, true);
    if (def.dw <= mydw) { under++; ev -= 25 + (mydw - def.dw); }
    else ev += def.dw - mydw;
  }
  return { pUndercut: under / n, ev: ev / n, dw: mydw };
}

function shouldKnock(hand10, K, P, rng) {
  const dw = deadwood(hand10);
  if (dw === 0) return true;
  if (dw > KNOCK_LIMIT) return false;
  if (P.ginHunt > 0 && dw > P.ginHunt && K.stockCount > P.lateStock) return false;
  const st = knockStats(hand10, K, P, rng);
  if (K.stockCount <= P.lateStock) return st.ev > -5;
  return st.pUndercut <= P.undercutMax && st.ev >= P.knockEvMin;
}

// ---------------------------------------------------------------------------
// Main decision
// ---------------------------------------------------------------------------
export function decideStrong(view, P = DEFAULT_PARAMS, rng = Math.random) {
  const K = analyze(view);
  const dangerCache = {};
  const hand = view.hand;

  if (view.phase === 'upcard' || view.phase === 'draw') {
    if (view.mustStock) return { type: 'stock' };
    const top = view.discardPile[view.discardPile.length - 1];
    // value of taking the upcard
    const h11 = hand.concat([top]);
    const take = bestDiscard(h11, K, P, top, 0, dangerCache);
    // immediate knock/gin after taking is captured by deadwood; check gin explicitly
    // value of drawing from stock (expectation over unseen cards)
    let stockVal = 0;
    const hh = hand.concat([0]);
    for (const c of K.unseen) {
      const pc = K.pDraw[c];
      if (pc <= 0) continue;
      hh[10] = c;
      const b = bestDiscard(hh, K, P, -1, 0, dangerCache);
      stockVal += pc * b.score;
    }
    const takeIt = take.score < stockVal - P.takeMargin;
    if (view.phase === 'upcard') return { type: takeIt ? 'take' : 'pass' };
    return { type: takeIt ? 'take' : 'stock' };
  }

  if (view.phase === 'discard') {
    if (deadwood(hand) === 0) return { type: 'bigGin' };
    const out = new Int32Array(32);
    discardDeadwoods(hand, out);
    // knock candidate: discard with minimal deadwood (prefer dumping high/dangerous)
    let kIdx = -1;
    for (let k = 0; k < 11; k++) {
      if (hand[k] === view.takenCard) continue;
      if (kIdx < 0 || out[k] < out[kIdx] || (out[k] === out[kIdx] && VAL[hand[k]] > VAL[hand[kIdx]])) kIdx = k;
    }
    if (out[kIdx] <= KNOCK_LIMIT) {
      const rest = hand.filter((_, i) => i !== kIdx);
      if (shouldKnock(rest, K, P, rng)) return { type: 'discard', card: hand[kIdx], knock: true };
    }
    // evaluate all discards at the top level
    const scored = scoreDiscards(hand, K, P, view.takenCard, dangerCache);
    if (!P.deepK || scored.length < 2) return { type: 'discard', card: scored[0].card, knock: false };
    // 2-ply: expectation over our next draw of the best follow-up
    const PK = Object.assign({}, P, { topK: 3 });
    let best = null;
    for (const cand of scored.slice(0, P.deepK)) {
      const hh = hand.filter((x) => x !== cand.card);
      hh.push(0);
      let v = 0;
      for (const c of K.unseen) {
        const pc = K.pDraw[c];
        if (pc <= 0) continue;
        hh[10] = c;
        v += pc * bestDiscard(hh, K, PK, -1, 0, dangerCache).score;
      }
      v += P.beta * dangerCache[cand.card];
      if (!best || v < best.v) best = { v, card: cand.card };
    }
    return { type: 'discard', card: best.card, knock: false };
  }
  throw new Error('no decision in phase ' + view.phase);
}

/** All legal discards scored by the heuristic, best first. */
function scoreDiscards(hand11, K, P, forbidden, dangerCache) {
  const res = [];
  const rest = new Array(10);
  for (let k = 0; k < 11; k++) {
    const d = hand11[k];
    if (d === forbidden) continue;
    let j = 0;
    for (let i = 0; i < 11; i++) if (i !== k) rest[j++] = hand11[i];
    let dg = dangerCache[d];
    if (dg === undefined) dg = dangerCache[d] = danger(d, K);
    res.push({ card: d, idx: k, score: evalHand(rest, K, P, 0) + P.beta * dg });
  }
  res.sort((a, b) => a.score - b.score);
  return res;
}

// ---------------------------------------------------------------------------
// Simpler levels
// ---------------------------------------------------------------------------
export function decideGreedy(view, rng = Math.random, noise = 0) {
  const hand = view.hand;
  if (view.phase === 'upcard' || view.phase === 'draw') {
    if (view.mustStock) return { type: 'stock' };
    const top = view.discardPile[view.discardPile.length - 1];
    const before = deadwood(hand);
    const out = new Int32Array(32);
    const h11 = hand.concat([top]);
    discardDeadwoods(h11, out);
    let m = 1e9;
    for (let k = 0; k < 10; k++) m = Math.min(m, out[k]);
    // take if the upcard joins a meld or lowers deadwood clearly
    const takeIt = m < before - (noise ? 4 : 2) && (noise === 0 || rng() > noise);
    if (view.phase === 'upcard') return { type: takeIt ? 'take' : 'pass' };
    return { type: takeIt ? 'take' : 'stock' };
  }
  if (deadwood(hand) === 0) return { type: 'bigGin' };
  const out = new Int32Array(32);
  discardDeadwoods(hand, out);
  let best = -1;
  for (let k = 0; k < 11; k++) {
    if (hand[k] === view.takenCard) continue;
    if (best < 0 || out[k] < out[best] || (out[k] === out[best] && VAL[hand[k]] > VAL[hand[best]])) best = k;
  }
  if (noise && rng() < noise) {
    const legal = hand.filter((c) => c !== view.takenCard);
    const c = legal[Math.floor(rng() * legal.length)];
    const dw = deadwood(hand.filter((x) => x !== c));
    return { type: 'discard', card: c, knock: dw <= KNOCK_LIMIT };
  }
  return { type: 'discard', card: hand[best], knock: out[best] <= KNOCK_LIMIT };
}

export function decide(view, level = 'strong', params, rng) {
  if (level === 'easy') return decideGreedy(view, rng || Math.random, 0.35);
  if (level === 'normal') return decideGreedy(view, rng || Math.random, 0);
  return decideStrong(view, params || DEFAULT_PARAMS, rng || makeRng((Math.random() * 2 ** 32) >>> 0));
}
