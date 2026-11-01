// Opponent-hand inference: quality metrics and training data.
//   node tools/opp_eval.mjs HANDS SEED ['{params}'] [dump.csv]
// params: AI params (also used by both players), e.g. oppModel: 0|1 (learned opponent model), improveCap.
// The metrics are computed with the same settings.
// Metrics at every decision before drawing:
//   logloss of P(card in opponent hand) over unknown cards (vs the uniform guess),
//   sampler: share of the sampled unknown cards that are really in the opponent hand,
//            mean (sampled deadwood - true deadwood) and mean |difference|.
import fs from 'node:fs';
import { GinHand, makeRng, deadwood } from '../web/js/engine.js';
import { decide, analyze, applyOppModel, sampleWorld, oppFilterInfo, oppCardFeatures, DEFAULT_PARAMS } from '../web/js/ai.js';

const N = +(process.argv[2] || 100);
const seed = +(process.argv[3] || 1);
const opts = process.argv[4] ? JSON.parse(process.argv[4]) : {};
const dump = process.argv[5] ? fs.openSync(process.argv[5], 'w') : null;
const P = { ...DEFAULT_PARAMS, timeBudget: 0, searchMin: 100, drawSearchMin: 100, ...opts };
const rng = makeRng(seed);
let ll = 0, llU = 0, nc = 0, ov = 0, novN = 0, bias = 0, absErr = 0, ns = 0;
let buf = [];
for (let i = 0; i < N; i++) {
  const h = new GinHand({ dealer: i % 2, rng: makeRng(Math.floor(rng() * 2 ** 32)) });
  const prng = makeRng(seed * 7919 + i);
  let step = 0;
  while (h.phase !== 'over') {
    if (h.phase === 'draw' || h.phase === 'upcard') {
      const p = h.turn;
      const v = h.view(p);
      const K = analyze(v);
      if (P.oppModel) applyOppModel(K);
      if (P.improveCap != null) K.improveCap = P.improveCap;
      if (P.oppFilter) K.filter = oppFilterInfo(v, P);
      const truth = new Set(h.hands[1 - p]);
      const n = K.unseen.length;
      const base = K.U / n;
      for (const c of K.unseen) {
        const y = truth.has(c) ? 1 : 0;
        const q = Math.min(1 - 1e-6, Math.max(1e-6, K.pOpp[c]));
        ll -= y ? Math.log(q) : Math.log(1 - q);
        llU -= y ? Math.log(base) : Math.log(1 - base);
        nc++;
        if (dump) {
          const f = oppCardFeatures(K, c);
          buf.push(Math.log(K.U / Math.max(1e-9, n - K.U)).toFixed(4) + ',' + Array.from(f).map((x) => +x.toFixed(3)).join(',') + ',' + y);
        }
      }
      if (step++ % 3 === 0 && K.U > 0) {
        const tdw = deadwood(h.hands[1 - p]);
        for (let k = 0; k < 20; k++) {
          const sh = sampleWorld(K, prng).opp;
          let hit = 0;
          for (const c of sh) if (!K.oppKnown.has(c) && truth.has(c)) hit++;
          ov += hit; novN += K.U;
          const d = deadwood(sh) - tdw;
          bias += d; absErr += Math.abs(d); ns++;
        }
      }
      if (dump && buf.length > 5000) { fs.writeSync(dump, buf.join('\n') + '\n'); buf = []; }
    }
    h.apply(decide(h.view(h.turn), 'strong', P, prng));
  }
}
if (dump && buf.length) fs.writeSync(dump, buf.join('\n') + '\n');
console.log(JSON.stringify({
  hands: N, cards: nc,
  logloss: +(ll / nc).toFixed(4), loglossUniform: +(llU / nc).toFixed(4),
  sampledHitRate: +(ov / novN).toFixed(4), dwBias: +(bias / ns).toFixed(2), dwAbsErr: +(absErr / ns).toFixed(2),
}));
