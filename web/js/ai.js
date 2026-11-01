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

import { VALUE_W } from './value.js';
import { VAL, suitOf, rankOf, deadwood, discardDeadwoods, defenderResult, arrange, scoreHand, shuffle, KNOCK_LIMIT, makeRng } from './engine.js';

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
  threatSamples: 40, // opponent-hand samples used to estimate the knock threat
  threatDW: 10, // opponent deadwood counted as "can knock soon"
  threatAlpha: 0, // potential weight is reduced by this * threat
  knockChoice: 1, // choose the knock discard by expected knock result
  search: 1, // use the search for discard/knock decisions
  searchCands: 4, // heuristic discard candidates examined by the search
  searchMin: 60, // minimum sampled worlds
  searchMax: 4000, // maximum sampled worlds
  timeBudget: 170, // ms per search (stops after the minimum once exceeded); a whole AI turn stays around 0.5 s
  drawSearch: 1, // also search the draw decision
  drawSearchMin: 60, // minimum sampled worlds for the draw decision
  innerKnockSamples: 20, // knock samples for hypothetical hands inside the draw search
  takeGainW: 1, // value of taking the opponent's discard next turn
  gamma: 0.0, // extra deadwood weight * threat (defence)
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
  const th = K.threat || 0;
  const base = dw * (1 + P.gamma * th);
  const a = alphaEff(P, K) * Math.max(0, 1 - P.threatAlpha * th);
  if (a === 0) return base;
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
  return base - a * pot;
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

/** Opponent-hand samples (cached per decision). */
function oppSamples(K, n, rng) {
  if (!K._samples) K._samples = [];
  while (K._samples.length < n) K._samples.push(sampleOpponentHand(K, rng));
  return K._samples.slice(0, n);
}

/** Probability that the opponent is close to knocking (and their mean deadwood). */
export function estimateThreat(K, P, rng) {
  const S = oppSamples(K, P.threatSamples, rng);
  let close = 0, sum = 0;
  for (const oh of S) {
    const d = deadwood(oh);
    sum += d;
    if (d <= P.threatDW) close++;
  }
  return { threat: close / S.length, oppDW: sum / S.length };
}

export function knockStats(hand10, K, P, rng) {
  const S = oppSamples(K, P.samples, rng);
  let under = 0, ev = 0, mydw = 0;
  for (const oh of S) {
    const r = scoreHand([hand10, oh], 0, false);
    mydw = r.knockerDW;
    if (r.winner === 0) ev += r.points;
    else { under++; ev -= r.points; }
  }
  return { pUndercut: under / S.length, ev: ev / S.length, dw: mydw };
}

function knockOk(st, K, P) {
  if (st.dw === 0) return true;
  if (K.stockCount <= P.lateStock) return st.ev > -5;
  return st.pUndercut <= P.undercutMax && st.ev >= P.knockEvMin;
}

function shouldKnock(hand10, K, P, rng) {
  const dw = deadwood(hand10);
  if (dw === 0) return true;
  if (dw > KNOCK_LIMIT) return false;
  if (P.ginHunt > 0 && dw > P.ginHunt && K.stockCount > P.lateStock) return false;
  return knockOk(knockStats(hand10, K, P, rng), K, P);
}

// ---------------------------------------------------------------------------
// Main decision
// ---------------------------------------------------------------------------
export function decideStrong(view, P = DEFAULT_PARAMS, rng = Math.random) {
  const K = analyze(view);
  if (P.gamma || P.threatAlpha) Object.assign(K, estimateThreat(K, P, rng));
  const dangerCache = {};
  const hand = view.hand;

  if (view.phase === 'upcard' || view.phase === 'draw') {
    if (view.mustStock) return { type: 'stock' };
    const top = view.discardPile[view.discardPile.length - 1];
    if (P.search && P.drawSearch) {
      const takeIt = searchDraw(view, K, P, rng, dangerCache);
      if (view.phase === 'upcard') return { type: takeIt ? 'take' : 'pass' };
      return { type: takeIt ? 'take' : 'stock' };
    }
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
    if (out[kIdx] === 0) return { type: 'discard', card: hand[kIdx], knock: true };
    if (P.search) return searchDiscard(view, K, P, rng, dangerCache);
    if (out[kIdx] <= KNOCK_LIMIT) {
      if (P.ginHunt > 0 && out[kIdx] > P.ginHunt && K.stockCount > P.lateStock) { /* keep playing */ }
      else if (P.knockChoice) {
        // compare every knockable discard by its expected knock result
        let best = null;
        for (let k = 0; k < 11; k++) {
          if (hand[k] === view.takenCard || out[k] > KNOCK_LIMIT) continue;
          const st = knockStats(hand.filter((_, i) => i !== k), K, P, rng);
          if (!best || st.ev > best.st.ev) best = { k, st };
        }
        if (knockOk(best.st, K, P)) return { type: 'discard', card: hand[best.k], knock: true };
      } else {
        const rest = hand.filter((_, i) => i !== kIdx);
        if (shouldKnock(rest, K, P, rng)) return { type: 'discard', card: hand[kIdx], knock: true };
      }
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
// Search: sample opponent hands + stock orders, simulate the opponent's reply
// turn, and score the resulting position with a value model learned from
// self-play (web/js/value.js). Knock options are scored exactly.
// ---------------------------------------------------------------------------
const COMBOS = (() => {
  const out = [];
  const rec = (start, d, acc) => {
    if (acc.length === d) { out.push(acc.slice()); return; }
    for (let i = start; i < 3; i++) { acc.push(i); rec(i, d, acc); acc.pop(); }
  };
  for (let d = 1; d <= 4; d++) rec(0, d, []);
  return out;
})();
export function leafValue(e, o, s, W = VALUE_W) {
  if (e < -15) e = -15; else if (e > 70) e = 70;
  if (o > 100) o = 100;
  const x = [e / 20, o / 20, s / 10];
  let v = W[0];
  for (let i = 0; i < COMBOS.length; i++) {
    let f = 1;
    const cmb = COMBOS[i];
    for (let j = 0; j < cmb.length; j++) f *= x[cmb[j]];
    v += W[i + 1] * f;
  }
  const n = COMBOS.length + 1;
  if (e <= 10) v += W[n];
  if (o <= 10) v += W[n + 1];
  if (e <= 0) v += W[n + 2];
  return v;
}

const SOUT = new Int32Array(32);
const SH11 = new Array(11);
/**
 * Simulate the opponent's turn (greedy: take the discard if it clearly helps, knock as soon as possible)
 * and return my expected net points. myHand: my 10 cards, myEval/myDW: their heuristic value / deadwood.
 */
function simOppTurn(myHand, myEval, myDW, oppHand, top, stock, P) {
  const oh = oppHand;
  for (let i = 0; i < 10; i++) SH11[i] = oh[i];
  // take decision
  SH11[10] = top;
  const before = deadwood(oh);
  discardDeadwoods(SH11, SOUT);
  let m = 1e9;
  for (let k = 0; k < 10; k++) if (SOUT[k] < m) m = SOUT[k];
  let taken = -1;
  if (m < before - 2 || (m <= KNOCK_LIMIT && before > KNOCK_LIMIT)) taken = top;
  else {
    if (stock.length === 0) return 0;
    SH11[10] = stock[stock.length - 1];
    discardDeadwoods(SH11, SOUT);
  }
  let bk = -1;
  for (let k = 0; k < 11; k++) {
    if (SH11[k] === taken) continue;
    if (bk < 0 || SOUT[k] < SOUT[bk] || (SOUT[k] === SOUT[bk] && VAL[SH11[k]] > VAL[SH11[bk]])) bk = k;
  }
  const oppDW = SOUT[bk];
  const x = SH11[bk];
  const stockLeft = stock.length - (taken < 0 ? 1 : 0);
  if (oppDW <= KNOCK_LIMIT) {
    const rest = [];
    for (let k = 0; k < 11; k++) if (k !== bk) rest.push(SH11[k]);
    const r = scoreHand([myHand, rest], 1, false);
    return r.winner === 0 ? r.points : -r.points;
  }
  if (stockLeft <= 2) return 0;
  // my next turn: I may take the opponent's discard x
  for (let i = 0; i < 10; i++) SH11[i] = myHand[i];
  SH11[10] = x;
  let mm = 1e9;
  discardDeadwoods(SH11, SOUT);
  for (let k = 0; k < 10; k++) if (SOUT[k] < mm) mm = SOUT[k];
  const e = mm < myDW ? Math.min(myEval, myEval - (myDW - mm) * P.takeGainW) : myEval;
  return leafValue(e, oppDW, stockLeft, P.valueW || VALUE_W);
}

function sampleWorld(K, rng) {
  const opp = sampleOpponentHand(K, rng);
  const inOpp = new Set(opp);
  const stock = shuffle(K.unseen.filter((c) => !inOpp.has(c)), rng);
  return { opp, stock };
}

function now() { return typeof performance !== 'undefined' ? performance.now() : Date.now(); }

/** Search-based discard decision. Returns an action. */
function searchDiscard(view, K, P, rng, dangerCache) {
  const hand = view.hand;
  const t0 = now();
  const scored = scoreDiscards(hand, K, P, view.takenCard, dangerCache);
  const cands = scored.slice(0, P.searchCands).map((sc) => {
    const rest = hand.filter((c) => c !== sc.card);
    return { card: sc.card, knock: false, rest, ev: evalHand(rest, K, P, 0), dw: deadwood(rest), sum: 0, n: 0 };
  });
  const out = new Int32Array(32);
  discardDeadwoods(hand, out);
  for (let k = 0; k < 11; k++) {
    if (hand[k] === view.takenCard || out[k] > KNOCK_LIMIT) continue;
    cands.push({ card: hand[k], knock: true, rest: hand.filter((_, i) => i !== k), sum: 0, n: 0 });
  }
  if (cands.length === 1) return { type: 'discard', card: cands[0].card, knock: cands[0].knock };
  const pile = view.discardPile;
  for (let w = 0; w < P.searchMax; w++) {
    if (w >= P.searchMin && now() - t0 > P.timeBudget) break;
    const W = sampleWorld(K, rng);
    for (const cd of cands) {
      let v;
      if (cd.knock) {
        const r = scoreHand([cd.rest, W.opp], 0, false);
        v = r.winner === 0 ? r.points : -r.points;
      } else {
        v = simOppTurn(cd.rest, cd.ev, cd.dw, W.opp, cd.card, W.stock, P);
      }
      cd.sum += v; cd.n++;
    }
  }
  let best = null;
  for (const cd of cands) {
    const m = cd.sum / cd.n + (cd.knock ? 0 : 0);
    if (!best || m > best.m) best = { m, cd };
  }
  return { type: 'discard', card: best.cd.card, knock: best.cd.knock };
}

/**
 * The heuristic policy's choice for an 11-card hand (used inside the draw search).
 * Returns {card, knock, rest, ev, dw}.
 */
function policyChoice(h11, forbidden, K, P, rng, dangerCache) {
  const out = new Int32Array(32);
  discardDeadwoods(h11, out);
  let kIdx = -1;
  for (let k = 0; k < 11; k++) {
    if (h11[k] === forbidden) continue;
    if (kIdx < 0 || out[k] < out[kIdx] || (out[k] === out[kIdx] && VAL[h11[k]] > VAL[h11[kIdx]])) kIdx = k;
  }
  if (out[kIdx] <= KNOCK_LIMIT) {
    const rest = h11.filter((_, i) => i !== kIdx);
    const Pk = Object.assign({}, P, { samples: P.innerKnockSamples });
    if (out[kIdx] === 0 || knockOk(knockStats(rest, K, Pk, rng), K, P)) return { card: h11[kIdx], knock: true, rest };
  }
  const best = scoreDiscards(h11, K, P, forbidden, dangerCache)[0];
  const rest = h11.filter((c) => c !== best.card);
  return { card: best.card, knock: false, rest, ev: evalHand(rest, K, P, 0), dw: deadwood(rest) };
}

function optionValue(opt, W, stock, P) {
  if (opt.knock) {
    const r = scoreHand([opt.rest, W.opp], 0, false);
    return r.winner === 0 ? r.points : -r.points;
  }
  return simOppTurn(opt.rest, opt.ev, opt.dw, W.opp, opt.card, stock, P);
}

/** Search-based draw decision: take the upcard vs draw from the stock. */
function searchDraw(view, K, P, rng, dangerCache) {
  const t0 = now();
  const hand = view.hand;
  const top = view.discardPile[view.discardPile.length - 1];
  const takeOpt = policyChoice(hand.concat([top]), top, K, P, rng, dangerCache);
  const stockOpts = new Map();
  let sTake = 0, sStock = 0, n = 0;
  for (let w = 0; w < P.searchMax; w++) {
    if (w >= P.drawSearchMin && now() - t0 > P.timeBudget) break;
    const W = sampleWorld(K, rng);
    sTake += optionValue(takeOpt, W, W.stock, P);
    const st = W.stock.slice();
    const c = st.pop();
    let opt = stockOpts.get(c);
    if (!opt) { opt = policyChoice(hand.concat([c]), -1, K, P, rng, dangerCache); stockOpts.set(c, opt); }
    sStock += optionValue(opt, W, st, P);
    n++;
  }
  return sTake / n > sStock / n;
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

/** Settings for the "normal" level: the first-release strong AI (heuristic only, no search). */
export const NORMAL_OVERRIDES = { search: 0, knockChoice: 0 };

/**
 * level: 'strong' (search + learned value model), 'normal' (heuristic, the former strongest AI),
 *        'easy' (greedy deadwood minimisation)
 */
export function decide(view, level = 'strong', params, rng) {
  if (level === 'easy') return decideGreedy(view, rng || Math.random, 0);
  const r = rng || makeRng((Math.random() * 2 ** 32) >>> 0);
  if (level === 'normal') return decideStrong(view, { ...(params || DEFAULT_PARAMS), ...NORMAL_OVERRIDES }, r);
  return decideStrong(view, params || DEFAULT_PARAMS, r);
}
