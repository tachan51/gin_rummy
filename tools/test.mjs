// Basic rules-engine tests: node tools/test.mjs
import assert from 'node:assert/strict';
import { deadwood, defenderResult, scoreHand, GinHand, makeRng } from '../web/js/engine.js';
import { decide, leafValue, DEFAULT_PARAMS } from '../web/js/ai.js';

const S = { s: 0, h: 1, d: 2, c: 3 };
const R = { A: 0, J: 10, Q: 11, K: 12 };
const card = (str) => { const s = S[str.at(-1)]; const r = str.slice(0, -1); return s * 13 + (R[r] ?? +r - 1); };
const cs = (str) => str.split(' ').map(card);

assert.equal(deadwood(cs('As 2s 3s 4s 5h 5d 5c Kh Qh Jh')), 0); // run + set + run
assert.equal(deadwood(cs('As 2s 3s 4s 4h 4d Kh Qh 9c 9d')), 38); // A-2-3 run + 4-4-4 set beats A-4 run
assert.equal(deadwood(cs('Ks Kh Kd Kc Qs Js 10s 2c 3c 4c')), 0); // K set + Q-J-10 run (Ks used in set)
assert.equal(deadwood(cs('7s 7h 7d 8s 9s 2c 2d 5h Jc Qd')), 67 - 24); // 7s-8s-9s run beats 7-7-7 set
assert.equal(deadwood(cs('Qs Ks As 2s 3h 5h 7h 9h Jd Jc')), 10 + 10 + 1 + 2 + 3 + 5 + 7 + 9 + 10 + 10); // no K-A wrap

// layoffs: defender extends a run at both ends and completes a set
const def = defenderResult(cs('3h 7h 9s 9c Kd Qd 2c 4c 6c 8d'), [cs('4h 5h 6h'), cs('9h 9d 9c')].map((m) => m.filter(Boolean)), true);
assert.ok(def.layoffs.length >= 2, 'expected layoffs');

// knock vs undercut scoring
const knocker = cs('As 2s 3s 4h 5h 6h 7d 7c 7s 9d'); // deadwood 9
const defender = cs('Kh Qd Jc 10s 2d 3c 8h 9h 5c 6c'); // lots of deadwood
let r = scoreHand([knocker, defender], 0, false);
assert.equal(r.type, 'knock');
assert.equal(r.winner, 0);
const defender2 = cs('Ah 2h 3h 4d 4c 4s Kd Kc Ks 2c'); // deadwood 2 -> undercut
r = scoreHand([knocker, defender2], 0, false);
assert.equal(r.type, 'undercut');
assert.equal(r.winner, 1);
assert.equal(r.points, 25 + 9 - 2);

// leaf value model: better own hand / worse opponent hand => higher value
assert.ok(leafValue(5, 30, 20) > leafValue(25, 30, 20));
assert.ok(leafValue(15, 40, 20) > leafValue(15, 8, 20));

// AI plays legal moves for whole hands (fixed small search for speed)
const FAST = { ...DEFAULT_PARAMS, timeBudget: 0, searchMin: 30, drawSearchMin: 30 };
for (let seed = 1; seed <= 30; seed++) {
  const h = new GinHand({ dealer: seed % 2, rng: makeRng(seed) });
  let n = 0;
  while (h.phase !== 'over') { h.apply(decide(h.view(h.turn), seed % 3 === 0 ? 'easy' : 'strong', FAST, makeRng(seed + n))); n++; assert.ok(n < 200); }
}
console.log('all tests passed');
