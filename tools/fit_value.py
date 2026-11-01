"""Fit the leaf value model used by the search AI.

usage: python3 tools/fit_value.py DATA_DIR [OUT_JS]
  DATA_DIR: CSV files from tools/gen_value.mjs (rows: myEval, oppDW, stock, outcome)
  OUT_JS:   default web/js/value.js

Model: least squares on degree-4 polynomial features of (e/20, o/20, s/10)
plus indicators [e<=10, o<=10, e<=0]. Must match leafValue() in web/js/ai.js.
"""
import glob, itertools, json, sys
import numpy as np

data_dir = sys.argv[1]
out_js = sys.argv[2] if len(sys.argv) > 2 else 'web/js/value.js'
D = np.vstack([np.loadtxt(f, delimiter=',', ndmin=2) for f in sorted(glob.glob(data_dir + '/*.csv'))])
np.random.seed(0)
np.random.shuffle(D)
e = np.clip(D[:, 0], -15, 70)
o = np.clip(D[:, 1], 0, 100)
s = D[:, 2]
y = D[:, 3]


def feats(e, o, s, deg=4):
    x = [e / 20, o / 20, s / 10]
    F = [np.ones_like(e)]
    for d in range(1, deg + 1):
        for comb in itertools.combinations_with_replacement(range(3), d):
            f = np.ones_like(e)
            for c in comb:
                f = f * x[c]
            F.append(f)
    F += [(e <= 10).astype(float), (o <= 10).astype(float), (e <= 0).astype(float)]
    return np.stack(F, 1)


n = len(y)
cut = int(n * 0.8)
X = feats(e, o, s)
w, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
pred = X[cut:] @ w
print('rows', n, 'holdout R2', 1 - ((pred - y[cut:]) ** 2).mean() / y[cut:].var())
w, *_ = np.linalg.lstsq(X, y, rcond=None)
with open(out_js, 'w') as f:
    f.write(
        "// Leaf value model: expected net points of the hand for the player about to draw,\n"
        "// fitted by least squares on %d self-play positions (tools/gen_value.mjs, tools/fit_value.py).\n"
        "// Features: polynomial (degree 4) of e/20, o/20, s/10 + indicators [e<=10, o<=10, e<=0]\n"
        "//   e = own heuristic evaluation, o = opponent deadwood, s = stock count\n"
        "export const VALUE_W = %s;\n" % (n, json.dumps([round(float(x), 5) for x in w])))
print('wrote', out_js)
