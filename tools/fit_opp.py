"""Fit the opponent-hand model (logistic regression with an offset).

usage: python3 tools/fit_opp.py DATA_DIR [OUT_JS]
  DATA_DIR: CSV files from tools/opp_eval.mjs (rows: offset, features..., label)
  OUT_JS:   default web/js/opp-model.js

P(unseen card c is in the opponent's hand) = sigmoid(offset + w . features(c)),
offset = logit(U / #unseen). Features: oppCardFeatures() in web/js/ai.js.
"""
import glob, json, sys
import numpy as np

data_dir = sys.argv[1]
out_js = sys.argv[2] if len(sys.argv) > 2 else 'web/js/opp-model.js'
D = np.vstack([np.loadtxt(f, delimiter=',', ndmin=2, dtype=np.float64) for f in sorted(glob.glob(data_dir + '/*.csv'))])
off, X, y = D[:, 0], D[:, 1:-1], D[:, -1]
n, d = X.shape
cut = int(n * 0.9)  # contiguous holdout (rows of one position stay together)


def fit(Xt, yt, ot, l2=1e-3, iters=30):
    w = np.zeros(Xt.shape[1])
    for _ in range(iters):
        z = ot + Xt @ w
        p = 1 / (1 + np.exp(-z))
        g = Xt.T @ (p - yt) / len(yt) + l2 * w
        H = (Xt * (p * (1 - p))[:, None]).T @ Xt / len(yt) + l2 * np.eye(Xt.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def logloss(w, Xt, yt, ot):
    p = np.clip(1 / (1 + np.exp(-(ot + Xt @ w))), 1e-6, 1 - 1e-6)
    return -(yt * np.log(p) + (1 - yt) * np.log(1 - p)).mean()


w = fit(X[:cut], y[:cut], off[:cut])
print('rows', n, 'features', d)
print('holdout logloss: model %.4f, offset only %.4f' % (logloss(w, X[cut:], y[cut:], off[cut:]),
                                                         logloss(np.zeros(d), X[cut:], y[cut:], off[cut:])))
w = fit(X, y, off)
names = ['bias', 'held-pick same rank', 'held-pick adj1', 'held-pick adj2', 'discard same rank', 'discard adj1',
         'discard adj2', 'decline same rank', 'decline adj1', 'decline adj2', 'alive rank-mates', 'alive neighbours',
         'value', 'opp turns', 'value x turns', 'recent discard same rank', 'recent discard adj1',
         'recent discard adj2', 'A or K']
for nm, v in zip(names, w):
    print('  %-26s %+.3f' % (nm, v))
with open(out_js, 'w') as f:
    f.write('// Opponent-hand model (logistic regression, tools/opp_eval.mjs + tools/fit_opp.py), %d card samples.\n'
            '// Weights for oppCardFeatures() in ai.js.\n' % n)
    f.write('export const OPP_MODEL = %s;\n' % json.dumps([round(float(x), 5) for x in w]))
print('wrote', out_js)
