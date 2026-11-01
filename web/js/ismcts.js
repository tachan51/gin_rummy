// Single-observer Information Set MCTS (SO-ISMCTS) for Gin Rummy.
//
// * Each iteration samples a "world" consistent with what the AI knows (opponent hand + stock order,
//   see sampleWorld in ai.js) and walks one shared tree whose nodes are the AI's information sets.
// * Opponent actions that are legal only in some worlds are handled with availability counts.
// * The AI's own stock draws branch by the drawn card (it observes it); the opponent's stock draws
//   do not (hidden from the AI).
// * Candidate moves are limited by heuristics (best few discards + knock), the root uses the full
//   heuristic ranking. Leaves are scored with the learned value model (value.js) after simulating
//   up to the AI's next turn, so no long random playouts are needed.
// Values are the AI's net points for the hand.

import { VAL, deadwood, discardDeadwoods, scoreHand, KNOCK_LIMIT } from './engine.js';
import { danger, evalHand, scoreDiscards, leafValue, simOppTurn, sampleWorld, now } from './ai.js';
import { VALUE_W } from './value.js';

const TAKE = 200, STOCK = 201; // action keys; discard c -> c, knock with c -> 52 + c

class Node {
  constructor() { this.ch = new Map(); }
}
class Edge {
  constructor() { this.n = 0; this.w = 0; this.avail = 0; this.node = null; this.obs = null; }
}

const OUT = new Int32Array(32);
const BUF = new Array(11);

function makeCtx(K, P, rng, dangerCache) {
  return { K, P, rng, dangerCache, evalCache: new Map() };
}

function evalCached(hand, ctx) {
  const key = hand.slice().sort((a, b) => a - b).join(',');
  let v = ctx.evalCache.get(key);
  if (v === undefined) {
    v = { e: evalHand(hand, ctx.K, ctx.P, 2), dw: deadwood(hand) };
    ctx.evalCache.set(key, v);
  }
  return v;
}

function dangerOf(c, ctx) {
  let d = ctx.dangerCache[c];
  if (d === undefined) d = ctx.dangerCache[c] = danger(c, ctx.K);
  return d;
}

/** greedy discard index for an 11-card hand (min deadwood, then highest card) */
function greedyIdx(h, taken) {
  discardDeadwoods(h, OUT);
  let bk = -1;
  for (let k = 0; k < h.length; k++) {
    if (h[k] === taken) continue;
    if (bk < 0 || OUT[k] < OUT[bk] || (OUT[k] === OUT[bk] && VAL[h[k]] > VAL[h[bk]])) bk = k;
  }
  return bk;
}

/** Candidate actions in the current (determinized) state, best first. */
function actions(st, ctx, rootCands) {
  if (st.phase === 'draw') return st.pile.length ? [TAKE, STOCK] : [STOCK];
  if (rootCands) return rootCands;
  const p = st.turn, h = st.h[p], P = ctx.P;
  discardDeadwoods(h, OUT);
  const idx = [];
  for (let k = 0; k < h.length; k++) if (h[k] !== st.taken) idx.push(k);
  const sc = (k) => OUT[k] + (p === 0 ? P.beta * dangerOf(h[k], ctx) : 0) - VAL[h[k]] * 0.01;
  idx.sort((a, b) => sc(a) - sc(b));
  const m = p === 0 ? P.mctsCands : P.mctsOppCands;
  const acts = [];
  for (let i = 0; i < idx.length && i < m; i++) acts.push(h[idx[i]]);
  let kIdx = -1;
  for (const k of idx) if (kIdx < 0 || OUT[k] < OUT[kIdx] || (OUT[k] === OUT[kIdx] && VAL[h[k]] > VAL[h[kIdx]])) kIdx = k;
  if (kIdx >= 0 && OUT[kIdx] <= KNOCK_LIMIT) acts.unshift(52 + h[kIdx]);
  return acts;
}

/** Apply an action; returns the observed card for stock draws (else -1). */
function apply(st, a) {
  const h = st.h[st.turn];
  if (a === TAKE) {
    const c = st.pile.pop();
    h.push(c); st.taken = c; st.phase = 'discard';
    return -1;
  }
  if (a === STOCK) {
    if (!st.stock.length) { st.over = true; st.value = 0; return -1; }
    const c = st.stock.pop();
    h.push(c); st.taken = -1; st.phase = 'discard';
    return c;
  }
  const knock = a >= 52;
  const c = knock ? a - 52 : a;
  h.splice(h.indexOf(c), 1);
  st.pile.push(c);
  if (knock) {
    const r = scoreHand(st.h, st.turn, false);
    st.over = true; st.value = r.winner === 0 ? r.points : -r.points;
  } else if (st.stock.length <= 2) {
    st.over = true; st.value = 0;
  } else {
    st.turn ^= 1; st.phase = 'draw'; st.taken = -1;
  }
  return -1;
}

/** Value of my position at the start of my turn, where I may take the opponent's discard x. */
function myTurnLeaf(h0, oppDW, x, stockLen, ctx) {
  const v = evalCached(h0, ctx);
  let e = v.e;
  if (x >= 0) {
    for (let i = 0; i < 10; i++) BUF[i] = h0[i];
    BUF[10] = x;
    discardDeadwoods(BUF, OUT);
    let mm = 1e9;
    for (let k = 0; k < 10; k++) if (OUT[k] < mm) mm = OUT[k];
    if (mm < v.dw) e = e - (v.dw - mm) * ctx.P.takeGainW;
  }
  return leafValue(e, oppDW, stockLen, ctx.P.valueW || VALUE_W);
}

/** Heuristic evaluation of a non-terminal state. */
function leaf(st, ctx) {
  if (st.over) return st.value;
  const [h0, h1] = st.h;
  const top = st.pile.length ? st.pile[st.pile.length - 1] : -1;
  if (st.phase === 'draw') {
    if (st.turn === 0) return myTurnLeaf(h0, deadwood(h1), top, st.stock.length, ctx);
    const v = evalCached(h0, ctx);
    return simOppTurn(h0, v.e, v.dw, h1, top, st.stock, ctx.P);
  }
  if (st.turn === 0) {
    const bk = greedyIdx(h0, st.taken);
    const dwAfter = OUT[bk];
    const rest = h0.filter((_, i) => i !== bk);
    let best = -1e9;
    if (dwAfter <= KNOCK_LIMIT) {
      const r = scoreHand([rest, h1], 0, false);
      best = r.winner === 0 ? r.points : -r.points;
    }
    if (st.stock.length > 2) {
      const v = evalCached(rest, ctx);
      best = Math.max(best, simOppTurn(rest, v.e, v.dw, h1, h0[bk], st.stock, ctx.P));
    } else best = Math.max(best, 0);
    return best;
  }
  // opponent holds 11 cards
  const bk = greedyIdx(h1, st.taken);
  const oppDW = OUT[bk];
  const x = h1[bk];
  const rest = h1.filter((_, i) => i !== bk);
  if (oppDW <= KNOCK_LIMIT) {
    const r = scoreHand([h0, rest], 1, false);
    return r.winner === 0 ? r.points : -r.points;
  }
  if (st.stock.length <= 2) return 0;
  return myTurnLeaf(h0, oppDW, x, st.stock.length, ctx);
}

/** Greedy opponent policy: take the upcard if it clearly helps, discard to minimise deadwood, knock asap. */
function oppPolicy(st) {
  const h = st.h[1];
  if (st.phase === 'draw') {
    if (!st.pile.length) return STOCK;
    const top = st.pile[st.pile.length - 1];
    const before = deadwood(h);
    for (let i = 0; i < 10; i++) BUF[i] = h[i];
    BUF[10] = top;
    discardDeadwoods(BUF, OUT);
    let m = 1e9;
    for (let k = 0; k < 10; k++) if (OUT[k] < m) m = OUT[k];
    return m < before - 2 || (m <= KNOCK_LIMIT && before > KNOCK_LIMIT) ? TAKE : STOCK;
  }
  const bk = greedyIdx(h, st.taken);
  return OUT[bk] <= KNOCK_LIMIT ? 52 + h[bk] : h[bk];
}

function childNode(e, a, actor, obs) {
  if (a === STOCK && actor === 0) {
    if (!e.obs) e.obs = new Map();
    let n = e.obs.get(obs);
    if (!n) { n = new Node(); e.obs.set(obs, n); }
    return n;
  }
  if (!e.node) e.node = new Node();
  return e.node;
}

function iterate(root, st, ctx, rootCands) {
  const P = ctx.P;
  let node = root;
  const path = [];
  let value;
  for (;;) {
    if (st.over) { value = st.value; break; }
    const acts = actions(st, ctx, node === root ? rootCands : null);
    let unexpanded = -1;
    for (const a of acts) {
      const e = node.ch.get(a);
      if (e) e.avail++;
      else if (unexpanded < 0) unexpanded = a;
    }
    const actor = st.turn;
    if (actor === 1 && P.mctsOppGreedy) {
      // opponent follows a fixed greedy policy (chance node) instead of minimising our value
      const a = oppPolicy(st);
      let e = node.ch.get(a);
      if (!e) { e = new Edge(); node.ch.set(a, e); }
      const obs = apply(st, a);
      path.push(e);
      node = childNode(e, a, actor, obs);
      continue;
    }
    if (unexpanded >= 0) {
      const e = new Edge();
      e.avail = 1;
      node.ch.set(unexpanded, e);
      const obs = apply(st, unexpanded);
      childNode(e, unexpanded, actor, obs);
      path.push(e);
      value = leaf(st, ctx);
      break;
    }
    const sign = actor === 0 ? 1 : -1;
    let best = null, bestA = -1, bs = -1e18;
    for (const a of acts) {
      const e = node.ch.get(a);
      const s = (sign * e.w) / e.n / P.mctsScale + P.uctC * Math.sqrt(Math.log(e.avail) / e.n);
      if (s > bs) { bs = s; best = e; bestA = a; }
    }
    const obs = apply(st, bestA);
    path.push(best);
    node = childNode(best, bestA, actor, obs);
  }
  for (const e of path) { e.n++; e.w += value; }
}

/**
 * ISMCTS decision for the 'draw' or 'discard' phase.
 * rootCands: candidate actions at the root (discard phase), as action keys.
 */
export function ismctsDecide(view, K, P, rng, dangerCache) {
  const t0 = now();
  const ctx = makeCtx(K, P, rng, dangerCache);
  let rootCands = null;
  if (view.phase === 'discard') {
    const hand = view.hand;
    rootCands = scoreDiscards(hand, K, P, view.takenCard, dangerCache).slice(0, P.searchCands).map((s) => s.card);
    discardDeadwoods(hand, OUT);
    const knocks = [];
    for (let k = 0; k < hand.length; k++) if (hand[k] !== view.takenCard && OUT[k] <= KNOCK_LIMIT) knocks.push(52 + hand[k]);
    rootCands = knocks.concat(rootCands);
  }
  const root = new Node();
  let it = 0;
  for (; it < P.mctsMax; it++) {
    if (it >= P.mctsMin && now() - t0 > P.timeBudget) break;
    const W = sampleWorld(K, rng);
    const st = {
      h: [view.hand.slice(), W.opp],
      pile: view.discardPile.slice(),
      stock: W.stock,
      turn: 0,
      phase: view.phase === 'discard' ? 'discard' : 'draw',
      taken: view.phase === 'discard' && view.takenCard != null ? view.takenCard : -1,
      over: false,
      value: 0,
    };
    iterate(root, st, ctx, rootCands);
  }
  let bestA = -1, bestN = -1, bestW = 0;
  for (const [a, e] of root.ch) {
    if (e.n > bestN || (e.n === bestN && e.w > bestW)) { bestN = e.n; bestA = a; bestW = e.w; }
  }
  const stats = { iterations: it, ms: now() - t0 };
  if (bestA === TAKE) return { type: 'take', stats };
  if (bestA === STOCK) return { type: 'stock', stats };
  if (bestA >= 52) return { type: 'discard', card: bestA - 52, knock: true, stats };
  return { type: 'discard', card: bestA, knock: false, stats };
}
