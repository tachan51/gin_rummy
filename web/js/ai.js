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
import { ismctsDecide } from './ismcts.js';
import { matchUtility } from './match-equity.js';
import { nnValue } from './nn.js';
import { OPP_MODEL } from './opp-model.js';
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
  mcts: 0, // 1: use ISMCTS (ismcts.js) for draw / discard decisions (no measurable gain, off)
  mctsMin: 100, // minimum iterations
  mctsMax: 100000, // maximum iterations
  mctsCands: 3, // own discard candidates per tree node (besides knock)
  mctsOppCands: 3, // opponent discard candidates per tree node (besides knock)
  uctC: 0.7, // exploration constant
  mctsScale: 25, // value scale (points) used in UCT
  mctsOppGreedy: 0, // 1: opponent nodes follow a fixed greedy policy instead of UCT
  oppModel: 1, // 1: learned opponent-hand model (opp-model.js) instead of the hand-tuned weights
  improveCap: 10, // max simulated stock draws used to 'improve' sampled opponent hands
  oppSim: 0, // 1: smarter opponent discards in the simulation (see simOppTurn)
  oppSimSlack: 2, // consider discards up to this much above the minimum deadwood
  oppSimConn: 0.5, // penalty per own connected card when discarding a card
  oppSimFeed: 4, // penalty per relation to cards I visibly picked up
  oppFilter: 1, // 1: resample sampled opponent hands by how well they explain the opponent's latest play
  filterN: 4, // candidates per sampled hand
  filterBeta: 0.4, // discard likelihood: exp(-beta * (deadwood after the real discard - best possible))
  filterGamma: 0.3, // declined-card likelihood: exp(-gamma * (gain - 2))
  filterNoKnock: 0.5, // likelihood of not knocking although deadwood <= 10
  nn: 0, // 1: neural leaf value (nn.js) instead of the polynomial model
  matchAwareFrom: 0, // ... only once either player has at least this many points
  matchAware: 0, // 1: use the match score (view.match) to value outcomes by match-winning probability
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
  const oppDiscards = []; // [card, opp turn index]
  const oppDeclines = []; // [card, opp turn index] upcard passes / stock draws while a discard was available
  let oppTurns = 0, myTurns = 0;
  for (const e of view.events) {
    if (e.p === opp) {
      if (e.type === 'take') { oppKnown.add(e.card); oppPicked.push(e.card); oppTurns++; }
      else if (e.type === 'stock') { oppTurns++; if (e.declined != null) { oppRejected.push(e.declined); oppDeclines.push([e.declined, oppTurns]); } }
      else if (e.type === 'discard') { oppKnown.delete(e.card); oppRejected.push(e.card); oppDiscards.push([e.card, oppTurns]); }
      else if (e.type === 'pass') { oppRejected.push(e.card); oppDeclines.push([e.card, oppTurns]); }
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

  const K = { me, opp, status, oppKnown, unseen, pOpp, pDraw, oppTurns, myTurns, U, stockCount: view.stockCount, oppDiscards, oppDeclines };
  return K;
}

// ---------------------------------------------------------------------------
// Learned opponent-hand model (logistic regression, tools/opp_eval.mjs + tools/fit_opp.py)
// ---------------------------------------------------------------------------
export const OPP_FEATURES = 19;
/** Features of unseen card c for the opponent-hand model (index 0 = bias). */
export function oppCardFeatures(K, c, f = new Float64Array(OPP_FEATURES)) {
  f.fill(0);
  const r = rankOf(c), s = suitOf(c);
  const rel = (x) => {
    if (rankOf(x) === r) return x === c ? -1 : 0; // same rank
    if (suitOf(x) !== s) return -1;
    const d = Math.abs(rankOf(x) - r);
    return d === 1 ? 1 : d === 2 ? 2 : -1;
  };
  f[0] = 1;
  for (const x of K.oppKnown) { const k = rel(x); if (k >= 0) f[1 + k] += 1; }
  for (const [x, t] of K.oppDiscards) {
    const k = rel(x);
    if (k < 0) continue;
    f[4 + k] += 1;
    if (K.oppTurns - t <= 1) f[15 + k] += 1; // discarded on the opponent's last two turns
  }
  for (const [x] of K.oppDeclines) { const k = rel(x); if (k >= 0) f[7 + k] += 1; }
  let aliveRank = 0;
  for (let s2 = 0; s2 < 4; s2++) { const x = s2 * 13 + r; if (x !== c && (K.status[x] === 0 || K.status[x] === 3)) aliveRank++; }
  let aliveAdj = 0;
  for (const d of [-1, 1]) { const r2 = r + d; if (r2 >= 0 && r2 < 13) { const x = s * 13 + r2; if (K.status[x] === 0 || K.status[x] === 3) aliveAdj++; } }
  f[10] = aliveRank / 3;
  f[11] = aliveAdj / 2;
  f[12] = VAL[c] / 10;
  f[13] = K.oppTurns / 10;
  f[14] = (VAL[c] / 10) * (K.oppTurns / 10);
  f[18] = (r === 0 || r === 12) ? 1 : 0; // A or K: only one run neighbour
  return f;
}

/** Replace the hand-tuned pOpp by the learned model W, normalised so that the unknown cards sum to U. */
export function applyOppModel(K, W = OPP_MODEL) {
  const n = K.unseen.length;
  if (!n || !K.U) return;
  const z = new Float64Array(n);
  const f = new Float64Array(OPP_FEATURES);
  const base = Math.log(K.U / Math.max(1, n - K.U + 1e-9));
  for (let i = 0; i < n; i++) {
    oppCardFeatures(K, K.unseen[i], f);
    let v = base;
    for (let j = 0; j < OPP_FEATURES; j++) v += W[j] * f[j];
    z[i] = v;
  }
  // shift so that the expected number of unknown opponent cards equals U
  let delta = 0;
  for (let it = 0; it < 30; it++) {
    let sum = 0, der = 0;
    for (let i = 0; i < n; i++) { const p = 1 / (1 + Math.exp(-(z[i] + delta))); sum += p; der += p * (1 - p); }
    const step = (K.U - sum) / Math.max(der, 1e-9);
    delta += step;
    if (Math.abs(step) < 1e-6) break;
  }
  let sd = 0;
  for (let i = 0; i < n; i++) {
    const c = K.unseen[i];
    const p = 1 / (1 + Math.exp(-(z[i] + delta)));
    K.pOpp[c] = Math.min(0.99, p);
    K.pDraw[c] = 1 - K.pOpp[c];
    sd += K.pDraw[c];
  }
  if (sd > 0) for (const c of K.unseen) K.pDraw[c] /= sd;
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
export function sampleOpponentHand(K, rng) {
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
  const extra = Math.min(K.oppTurns, K.improveCap ?? 10);
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
  // match-aware: compare outcomes by match-winning probability instead of hand points
  if (P.matchAware && view.match && P.search && Math.max(view.match.myScore, view.match.oppScore) >= P.matchAwareFrom) {
    const m = view.match;
    P = { ...P, util: matchUtility(m.myScore, m.oppScore, m.target, view.dealer === view.me) };
  }
  if (P.nn) P = { ...P, _pile: view.discardPile };
  const K = analyze(view);
  if (P.oppModel) applyOppModel(K, P.oppModelW || OPP_MODEL);
  if (P.oppSim) {
    // cards I picked from the discard pile and still hold (visible to the opponent)
    const picks = new Set();
    for (const e of view.events) {
      if (e.p !== view.me) continue;
      if (e.type === 'take') picks.add(e.card);
      else if (e.type === 'discard') picks.delete(e.card);
    }
    P = { ...P, _myPicks: [...picks] };
  }
  if (P.improveCap != null) K.improveCap = P.improveCap;
  if (P.oppFilter) K.filter = oppFilterInfo(view, P);
  if (P.gamma || P.threatAlpha) Object.assign(K, estimateThreat(K, P, rng));
  const dangerCache = {};
  const hand = view.hand;

  if (view.phase === 'upcard' || view.phase === 'draw') {
    if (view.mustStock) return { type: 'stock' };
    const top = view.discardPile[view.discardPile.length - 1];
    if (P.mcts && view.phase === 'draw') return ismctsDecide(view, K, P, rng, dangerCache);
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
    if (P.mcts) return ismctsDecide(view, K, P, rng, dangerCache);
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
export function scoreDiscards(hand11, K, P, forbidden, dangerCache) {
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
/** same rank, or same suit within two ranks */
function related(a, b) {
  if (a === b) return false;
  const ra = a % 13, rb = b % 13;
  if (ra === rb) return true;
  return ((a / 13) | 0) === ((b / 13) | 0) && Math.abs(ra - rb) <= 2;
}

/**
 * Opponent's discard in the simulation: index into h (11 cards); out = discardDeadwoods(h).
 * Greedy (minimum deadwood, then highest card), or with P.oppSim a smarter rule.
 */
export function oppDiscardIdx(h, out, taken, P) {
  let bk = -1;
  if (P.oppSim) {
    // among discards close to the minimum deadwood, keep connected cards
    // and avoid feeding cards related to what I visibly picked up
    let mn = 1e9;
    for (let k = 0; k < 11; k++) if (h[k] !== taken && out[k] < mn) mn = out[k];
    let bs = 1e9;
    for (let k = 0; k < 11; k++) {
      const c = h[k];
      if (c === taken || out[k] > mn + P.oppSimSlack) continue;
      let conn = 0;
      for (let j = 0; j < 11; j++) if (j !== k && related(c, h[j])) conn++;
      let feed = 0;
      if (P._myPicks) for (const x of P._myPicks) if (related(c, x)) feed++;
      const sc = out[k] + P.oppSimConn * conn + P.oppSimFeed * feed - VAL[c] * 0.01;
      if (sc < bs) { bs = sc; bk = k; }
    }
    return bk;
  }
  for (let k = 0; k < 11; k++) {
    if (h[k] === taken) continue;
    if (bk < 0 || out[k] < out[bk] || (out[k] === out[bk] && VAL[h[k]] > VAL[h[bk]])) bk = k;
  }
  return bk;
}

export function simOppTurn(myHand, myEval, myDW, oppHand, top, stock, P) {
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
    if (stock.length === 0) return P.util ? P.util(0) : 0;
    SH11[10] = stock[stock.length - 1];
    discardDeadwoods(SH11, SOUT);
  }
  const bk = oppDiscardIdx(SH11, SOUT, taken, P);
  const oppDW = SOUT[bk];
  const x = SH11[bk];
  const stockLeft = stock.length - (taken < 0 ? 1 : 0);
  if (oppDW <= KNOCK_LIMIT) {
    const rest = [];
    for (let k = 0; k < 11; k++) if (k !== bk) rest.push(SH11[k]);
    const r = scoreHand([myHand, rest], 1, false);
    const v = r.winner === 0 ? r.points : -r.points;
    return P.util ? P.util(v) : v;
  }
  if (stockLeft <= 2) return P.util ? P.util(0) : 0;
  if (P.nn) {
    // neural leaf value (nn.js): position at the start of my next turn
    const rest = [];
    for (let k = 0; k < 11; k++) if (k !== bk) rest.push(SH11[k]);
    const dead = P._pile ? P._pile.slice() : [];
    if (taken < 0) dead.push(top);
    dead.push(x);
    const lv = (P.nnModel || nnValue)(myHand, rest, dead, x, stockLeft, myEval, myDW, oppDW);
    return P.util ? P.util.leaf(lv) : lv;
  }
  // my next turn: I may take the opponent's discard x
  for (let i = 0; i < 10; i++) SH11[i] = myHand[i];
  SH11[10] = x;
  let mm = 1e9;
  discardDeadwoods(SH11, SOUT);
  for (let k = 0; k < 10; k++) if (SOUT[k] < mm) mm = SOUT[k];
  const e = mm < myDW ? Math.min(myEval, myEval - (myDW - mm) * P.takeGainW) : myEval;
  const lv = leafValue(e, oppDW, stockLeft, P.valueW || VALUE_W);
  return P.util ? P.util.leaf(lv) : lv;
}

/**
 * Likelihood of the opponent's latest observed behaviour given a candidate hand H (10 cards):
 * the last discard d (H + d before discarding), the last declined discard x, and not knocking.
 */
const FOUT = new Int32Array(32);
const FH = new Array(11);
function behaviourLikelihood(H, F) {
  let L = 1;
  if (F.lastDiscard >= 0) {
    for (let i = 0; i < 10; i++) FH[i] = H[i];
    FH[10] = F.lastDiscard;
    discardDeadwoods(FH, FOUT);
    let mn = 1e9;
    for (let k = 0; k < 11; k++) if (FH[k] !== F.lastTaken && FOUT[k] < mn) mn = FOUT[k];
    L *= Math.exp(-F.beta * (FOUT[10] - mn));
  }
  const dw = deadwood(H);
  if (F.lastDeclined >= 0) {
    for (let i = 0; i < 10; i++) FH[i] = H[i];
    FH[10] = F.lastDeclined;
    discardDeadwoods(FH, FOUT);
    let m = 1e9;
    for (let k = 0; k < 10; k++) if (FOUT[k] < m) m = FOUT[k];
    const gain = dw - m;
    if (gain > 2) L *= Math.exp(-F.gamma * (gain - 2));
  }
  if (dw <= KNOCK_LIMIT) L *= dw === 0 ? F.noKnock * F.noKnock : F.noKnock;
  return L;
}

/** Sample an opponent hand; with K.filter, resample candidates by how well they explain the opponent's play. */
function sampleOppFiltered(K, rng) {
  const F = K.filter;
  if (!F) return sampleOpponentHand(K, rng);
  const cands = [], ws = [];
  let tot = 0;
  for (let i = 0; i < F.n; i++) {
    const h = sampleOpponentHand(K, rng);
    const w = behaviourLikelihood(h, F);
    cands.push(h); ws.push(w); tot += w;
  }
  let x = rng() * tot;
  for (let i = 0; i < cands.length; i++) { x -= ws[i]; if (x <= 0) return cands[i]; }
  return cands[cands.length - 1];
}

/** Information about the opponent's latest turn, used by the sampling filter. */
export function oppFilterInfo(view, P) {
  const opp = 1 - view.me;
  let lastDiscard = -1, lastTaken = -1, lastDeclined = -1;
  // walk back to the opponent's latest turn
  for (let i = view.events.length - 1; i >= 0; i--) {
    const e = view.events[i];
    if (e.p !== opp) { if (lastDiscard >= 0) break; continue; }
    if (e.type === 'discard') { if (lastDiscard >= 0) break; lastDiscard = e.card; }
    else if (e.type === 'take') { lastTaken = e.card; break; }
    else if (e.type === 'stock') { if (e.declined != null) lastDeclined = e.declined; break; }
  }
  return { lastDiscard, lastTaken, lastDeclined, n: P.filterN, beta: P.filterBeta, gamma: P.filterGamma, noKnock: P.filterNoKnock };
}

export function sampleWorld(K, rng) {
  const opp = sampleOppFiltered(K, rng);
  const inOpp = new Set(opp);
  const stock = shuffle(K.unseen.filter((c) => !inOpp.has(c)), rng);
  return { opp, stock };
}

export function now() { return typeof performance !== 'undefined' ? performance.now() : Date.now(); }

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
        if (P.util) v = P.util(v);
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
    const v = r.winner === 0 ? r.points : -r.points;
    return P.util ? P.util(v) : v;
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
export const NORMAL_OVERRIDES = { search: 0, knockChoice: 0, mcts: 0, oppModel: 0, oppFilter: 0 };

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
