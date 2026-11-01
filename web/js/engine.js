// Gin Rummy rules engine: cards, meld search, deadwood, layoffs, scoring, hand state.
// Card id: 0..51, suit = floor(id/13) (0♠ 1♥ 2♦ 3♣), rank = id%13 (0=A .. 12=K)

export const SUIT_SYMBOLS = ['♠', '♥', '♦', '♣'];
export const RANK_LABELS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
export const suitOf = (c) => (c / 13) | 0;
export const rankOf = (c) => c % 13;
export const VAL = new Int8Array(52);
for (let c = 0; c < 52; c++) VAL[c] = Math.min(rankOf(c) + 1, 10);
export const cardName = (c) => SUIT_SYMBOLS[suitOf(c)] + RANK_LABELS[rankOf(c)];
export const isRed = (c) => suitOf(c) === 1 || suitOf(c) === 2;

export const KNOCK_LIMIT = 10;
export const GIN_BONUS = 25;
export const BIG_GIN_BONUS = 31;
export const UNDERCUT_BONUS = 25;
// safety valve: players could pass the same cards back and forth forever
export const MAX_TURNS = 200;

// ---------------------------------------------------------------------------
// Meld generation (hand-local bitmasks, hand size <= 31)
// ---------------------------------------------------------------------------
const POS = new Int8Array(52).fill(-1);
const MAXM = 1024;
const MM = new Int32Array(MAXM); // meld masks
const MV = new Int32Array(MAXM); // meld values
const MT = new Int32Array(MAXM); // meld tag (-1 own meld, >=0 layoff target)
let nM = 0;
const SI = new Int32Array(4);

function genMelds(cards) {
  const n = cards.length;
  for (let i = 0; i < n; i++) POS[cards[i]] = i;
  nM = 0;
  // runs
  for (let s = 0; s < 4; s++) {
    const base = s * 13;
    let r = 0;
    while (r < 13) {
      if (POS[base + r] < 0) { r++; continue; }
      let e = r;
      while (e + 1 < 13 && POS[base + e + 1] >= 0) e++;
      if (e - r >= 2) {
        for (let i = r; i <= e - 2; i++) {
          let m = 0, v = 0;
          for (let j = i; j <= e; j++) {
            m |= 1 << POS[base + j];
            v += VAL[base + j];
            if (j - i >= 2) { MM[nM] = m; MV[nM] = v; MT[nM] = -1; nM++; }
          }
        }
      }
      r = e + 1;
    }
  }
  // sets
  for (let r = 0; r < 13; r++) {
    let k = 0;
    for (let s = 0; s < 4; s++) { const p = POS[s * 13 + r]; if (p >= 0) SI[k++] = p; }
    if (k < 3) continue;
    const v = VAL[r];
    if (k === 3) {
      MM[nM] = (1 << SI[0]) | (1 << SI[1]) | (1 << SI[2]); MV[nM] = 3 * v; MT[nM] = -1; nM++;
    } else {
      const full = (1 << SI[0]) | (1 << SI[1]) | (1 << SI[2]) | (1 << SI[3]);
      MM[nM] = full; MV[nM] = 4 * v; MT[nM] = -1; nM++;
      for (let x = 0; x < 4; x++) { MM[nM] = full & ~(1 << SI[x]); MV[nM] = 3 * v; MT[nM] = -1; nM++; }
    }
  }
}
function clearPos(cards) {
  for (let i = 0; i < cards.length; i++) POS[cards[i]] = -1;
}

let bestV = 0;
function dfsMax(i, used, val) {
  if (val > bestV) bestV = val;
  for (let j = i; j < nM; j++) {
    if ((MM[j] & used) === 0) dfsMax(j + 1, used | MM[j], val + MV[j]);
  }
}

function handTotal(cards) {
  let t = 0;
  for (let i = 0; i < cards.length; i++) t += VAL[cards[i]];
  return t;
}

/** Minimum deadwood of a set of cards. */
export function deadwood(cards) {
  genMelds(cards);
  clearPos(cards);
  bestV = 0;
  dfsMax(0, 0, 0);
  return handTotal(cards) - bestV;
}

// combo recording (all disjoint meld subsets)
const MAXC = 8192;
const CM = new Int32Array(MAXC);
const CV = new Int32Array(MAXC);
let nC = 0;
function dfsRec(i, used, val) {
  if (nC < MAXC) { CM[nC] = used; CV[nC] = val; nC++; }
  for (let j = i; j < nM; j++) {
    if ((MM[j] & used) === 0) dfsRec(j + 1, used | MM[j], val + MV[j]);
  }
}

/**
 * For each index k, deadwood of cards with cards[k] removed. Writes into out, returns min.
 */
export function discardDeadwoods(cards, out) {
  genMelds(cards);
  clearPos(cards);
  nC = 0;
  dfsRec(0, 0, 0);
  const total = handTotal(cards);
  let mn = 1e9;
  for (let k = 0; k < cards.length; k++) {
    const bit = 1 << k;
    let b = 0;
    for (let c = 0; c < nC; c++) if ((CM[c] & bit) === 0 && CV[c] > b) b = CV[c];
    const d = total - VAL[cards[k]] - b;
    out[k] = d;
    if (d < mn) mn = d;
  }
  return mn;
}

// path-tracking DFS for arrangements
function dfsAll(i, used, val, path, results, state) {
  if (val > state.best) { state.best = val; results.length = 0; }
  if (val === state.best) results.push(path.slice());
  for (let j = i; j < nM; j++) {
    if ((MM[j] & used) === 0) {
      path.push(j);
      dfsAll(j + 1, used | MM[j], val + MV[j], path, results, state);
      path.pop();
    }
  }
}

function maskToCards(mask, cards) {
  const out = [];
  for (let i = 0; i < cards.length; i++) if (mask & (1 << i)) out.push(cards[i]);
  return out;
}

/** All optimal meld arrangements: [{melds:[[c..]], deadwood:[c..], dw}] */
export function allArrangements(cards) {
  genMelds(cards);
  clearPos(cards);
  const results = [];
  const state = { best: -1 };
  dfsAll(0, 0, 0, [], results, state);
  const total = handTotal(cards);
  const masks = Array.from(MM.subarray(0, nM));
  return results.map((path) => {
    let used = 0;
    const melds = path.map((j) => { used |= masks[j]; return sortMeld(maskToCards(masks[j], cards)); });
    const dwCards = [];
    for (let i = 0; i < cards.length; i++) if (!(used & (1 << i))) dwCards.push(cards[i]);
    dwCards.sort(byRankSuit);
    return { melds, deadwood: dwCards, dw: total - state.best };
  });
}

/** One optimal arrangement (prefers fewer deadwood cards). */
export function arrange(cards) {
  const all = allArrangements(cards);
  all.sort((a, b) => a.deadwood.length - b.deadwood.length);
  return all[0];
}

export function byRankSuit(a, b) { return rankOf(a) - rankOf(b) || suitOf(a) - suitOf(b); }
export function bySuitRank(a, b) { return suitOf(a) - suitOf(b) || rankOf(a) - rankOf(b); }
export function sortMeld(m) {
  const isSet = m.every((c) => rankOf(c) === rankOf(m[0]));
  return m.slice().sort(isSet ? bySuitRank : (a, b) => rankOf(a) - rankOf(b));
}
export function isSetMeld(m) { return m.every((c) => rankOf(c) === rankOf(m[0])); }

// ---------------------------------------------------------------------------
// Layoffs
// ---------------------------------------------------------------------------
/**
 * Defender's best result against knocker melds (own melds + layoffs).
 * Returns {dw, melds:[[..]], layoffs:[{cards:[..], target:idx}], deadwood:[..]}
 */
export function defenderResult(defCards, knockerMelds, allowLayoff = true) {
  genMelds(defCards);
  if (allowLayoff) {
    for (let t = 0; t < knockerMelds.length; t++) {
      const m = knockerMelds[t];
      if (isSetMeld(m)) {
        if (m.length === 3) {
          const r = rankOf(m[0]);
          for (let s = 0; s < 4; s++) {
            const c = s * 13 + r;
            if (!m.includes(c) && POS[c] >= 0 && nM < MAXM) { MM[nM] = 1 << POS[c]; MV[nM] = VAL[c]; MT[nM] = t; nM++; }
          }
        }
      } else {
        const s = suitOf(m[0]);
        let lo = 13, hi = -1;
        for (const c of m) { lo = Math.min(lo, rankOf(c)); hi = Math.max(hi, rankOf(c)); }
        let mask = 0, v = 0;
        for (let r = lo - 1; r >= 0; r--) {
          const c = s * 13 + r;
          if (POS[c] < 0) break;
          mask |= 1 << POS[c]; v += VAL[c];
          if (nM < MAXM) { MM[nM] = mask; MV[nM] = v; MT[nM] = t; nM++; }
        }
        mask = 0; v = 0;
        for (let r = hi + 1; r < 13; r++) {
          const c = s * 13 + r;
          if (POS[c] < 0) break;
          mask |= 1 << POS[c]; v += VAL[c];
          if (nM < MAXM) { MM[nM] = mask; MV[nM] = v; MT[nM] = t; nM++; }
        }
      }
    }
  }
  clearPos(defCards);
  const results = [];
  const state = { best: -1 };
  dfsAll(0, 0, 0, [], results, state);
  const masks = Array.from(MM.subarray(0, nM));
  const tags = Array.from(MT.subarray(0, nM));
  // prefer more own melds (nicer display), any is optimal
  const path = results[0] || [];
  let used = 0;
  const melds = [], layoffs = [];
  for (const j of path) {
    used |= masks[j];
    const cs = maskToCards(masks[j], defCards);
    if (tags[j] < 0) melds.push(sortMeld(cs));
    else layoffs.push({ cards: cs.sort(byRankSuit), target: tags[j] });
  }
  const dwCards = [];
  for (let i = 0; i < defCards.length; i++) if (!(used & (1 << i))) dwCards.push(defCards[i]);
  dwCards.sort(byRankSuit);
  return { dw: handTotal(defCards) - state.best, melds, layoffs, deadwood: dwCards };
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function shuffle(arr, rng = Math.random) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}

// ---------------------------------------------------------------------------
// Hand (one deal)
// ---------------------------------------------------------------------------
export class GinHand {
  constructor({ dealer = 0, rng = Math.random } = {}) {
    const deck = shuffle(Array.from({ length: 52 }, (_, i) => i), rng);
    this.dealer = dealer;
    this.hands = [[], []];
    for (let i = 0; i < 10; i++) {
      this.hands[1 - dealer].push(deck.pop());
      this.hands[dealer].push(deck.pop());
    }
    this.discardPile = [deck.pop()];
    this.stock = deck;
    this.turn = 1 - dealer;
    this.phase = 'upcard'; // upcard | draw | discard | over
    this.upPasses = 0;
    this.mustStock = false;
    this.takenCard = null;
    this.events = [];
    this.result = null;
    this.turnCount = 0;
  }

  top() { return this.discardPile[this.discardPile.length - 1]; }

  view(p) {
    return {
      me: p,
      hand: this.hands[p].slice(),
      discardPile: this.discardPile.slice(),
      stockCount: this.stock.length,
      events: this.events.slice(),
      phase: this.phase,
      turn: this.turn,
      dealer: this.dealer,
      takenCard: this.takenCard,
      mustStock: this.mustStock,
    };
  }

  passUpcard() {
    if (this.phase !== 'upcard') throw new Error('not upcard phase');
    this.events.push({ p: this.turn, type: 'pass', card: this.top() });
    this.upPasses++;
    if (this.upPasses === 1) {
      this.turn = this.dealer;
    } else {
      this.turn = 1 - this.dealer;
      this.phase = 'draw';
      this.mustStock = true;
    }
  }

  takeDiscard() {
    if (!(this.phase === 'upcard' || this.phase === 'draw') || this.mustStock) throw new Error('cannot take discard');
    const c = this.discardPile.pop();
    this.hands[this.turn].push(c);
    this.takenCard = c;
    this.events.push({ p: this.turn, type: 'take', card: c });
    this.phase = 'discard';
    return c;
  }

  drawStock() {
    if (this.phase !== 'draw') throw new Error('cannot draw stock');
    const c = this.stock.pop();
    this.hands[this.turn].push(c);
    this.events.push({ p: this.turn, type: 'stock', declined: this.discardPile.length ? this.top() : null });
    this.takenCard = null;
    this.mustStock = false;
    this.phase = 'discard';
    return c;
  }

  canDiscard(c) {
    return this.phase === 'discard' && this.hands[this.turn].includes(c) && c !== this.takenCard;
  }

  canKnockWith(c) {
    if (!this.canDiscard(c)) return false;
    return deadwood(this.hands[this.turn].filter((x) => x !== c)) <= KNOCK_LIMIT;
  }

  canBigGin() {
    return this.phase === 'discard' && deadwood(this.hands[this.turn]) === 0;
  }

  discard(c, knock = false) {
    if (!this.canDiscard(c)) throw new Error('illegal discard ' + c);
    const h = this.hands[this.turn];
    if (knock && deadwood(h.filter((x) => x !== c)) > KNOCK_LIMIT) throw new Error('cannot knock');
    h.splice(h.indexOf(c), 1);
    this.discardPile.push(c);
    this.events.push({ p: this.turn, type: 'discard', card: c, knock });
    this.takenCard = null;
    this.turnCount++;
    if (knock) {
      this.result = scoreHand(this.hands, this.turn, false);
      this.phase = 'over';
    } else if (this.stock.length <= 2 || this.turnCount >= MAX_TURNS) {
      this.result = { type: 'draw', winner: -1, points: 0 };
      this.phase = 'over';
    } else {
      this.turn = 1 - this.turn;
      this.phase = 'draw';
    }
  }

  bigGin() {
    if (!this.canBigGin()) throw new Error('cannot big gin');
    this.events.push({ p: this.turn, type: 'biggin' });
    this.result = scoreHand(this.hands, this.turn, true);
    this.phase = 'over';
  }

  apply(action) {
    switch (action.type) {
      case 'pass': return this.passUpcard();
      case 'take': return this.takeDiscard();
      case 'stock': return this.drawStock();
      case 'discard': return this.discard(action.card, !!action.knock);
      case 'bigGin': return this.bigGin();
      default: throw new Error('unknown action ' + action.type);
    }
  }
}

/** Score a knock. Knocker picks the optimal arrangement that leaves defender the most deadwood. */
export function scoreHand(hands, knocker, big) {
  const defender = 1 - knocker;
  const kh = hands[knocker], dh = hands[defender];
  const arrs = allArrangements(kh);
  const kdw = arrs[0].dw;
  if (kdw === 0) {
    const kArr = arrs.slice().sort((a, b) => a.melds.length - b.melds.length)[0];
    const def = defenderResult(dh, kArr.melds, false);
    const bonus = big ? BIG_GIN_BONUS : GIN_BONUS;
    return {
      type: big ? 'biggin' : 'gin', knocker, winner: knocker,
      points: bonus + def.dw, bonus,
      knockerArr: kArr, defenderArr: def, knockerDW: 0, defenderDW: def.dw,
    };
  }
  let best = null;
  for (const a of arrs) {
    const def = defenderResult(dh, a.melds, true);
    if (!best || def.dw > best.def.dw) best = { a, def };
  }
  const ddw = best.def.dw;
  if (ddw <= kdw) {
    return {
      type: 'undercut', knocker, winner: defender,
      points: UNDERCUT_BONUS + (kdw - ddw), bonus: UNDERCUT_BONUS,
      knockerArr: best.a, defenderArr: best.def, knockerDW: kdw, defenderDW: ddw,
    };
  }
  return {
    type: 'knock', knocker, winner: knocker, points: ddw - kdw, bonus: 0,
    knockerArr: best.a, defenderArr: best.def, knockerDW: kdw, defenderDW: ddw,
  };
}

// ---------------------------------------------------------------------------
// Match (game to target)
// ---------------------------------------------------------------------------
export const GAME_BONUS = 100;
export const BOX_BONUS = 25;

export class GinMatch {
  constructor({ target = 100, rng = Math.random, firstDealer = 0 } = {}) {
    this.target = target;
    this.rng = rng;
    this.scores = [0, 0];
    this.handsWon = [0, 0];
    this.history = [];
    this.dealer = firstDealer;
    this.hand = null;
    this.final = null;
  }
  newHand() {
    this.hand = new GinHand({ dealer: this.dealer, rng: this.rng });
    return this.hand;
  }
  recordResult(result) {
    this.history.push(result);
    if (result.winner >= 0) {
      this.scores[result.winner] += result.points;
      this.handsWon[result.winner]++;
    }
    // the dealer alternates each hand
    this.dealer = 1 - this.dealer;
    if (this.scores[0] >= this.target || this.scores[1] >= this.target) {
      const w = this.scores[0] >= this.target ? 0 : 1;
      const l = 1 - w;
      const box = [this.handsWon[0] * BOX_BONUS, this.handsWon[1] * BOX_BONUS];
      const totals = [this.scores[0] + box[0], this.scores[1] + box[1]];
      totals[w] += GAME_BONUS;
      const shutout = this.handsWon[l] === 0;
      if (shutout) totals[w] *= 2;
      this.final = { winner: w, totals, box, shutout, margin: totals[w] - totals[l] };
    }
  }
}
