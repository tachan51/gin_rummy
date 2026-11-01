"""Train the neural leaf-value model (numpy only).

usage: python3 tools/train_nn.py DATA_DIR [OUT_JS] [--hidden 64,32] [--epochs 30] [--lr 0.001] [--init model.js]
  DATA_DIR: *.jsonl from tools/gen_nn.mjs
  OUT_JS:   default web/js/nn-value.js

Input (212 values), from the point of view of the player about to draw:
  my hand (52 one-hot) | opponent hand (52) | discard pile (52) | top discard (52)
  | stock/31, myEval/30, myDW/50, oppDW/50
Target: the player's net points of the hand / 25.
Gin Rummy is symmetric under permutations of the suits, so each batch is augmented with a
random suit permutation.
"""
import glob, json, sys, time
import numpy as np

args, opts = [], {}
it = iter(sys.argv[1:])
for a in it:
    if a.startswith('--'):
        opts[a] = next(it)
    else:
        args.append(a)
data_dir = args[0]
out_js = args[1] if len(args) > 1 else 'web/js/nn-value.js'
hidden = [int(x) for x in opts.get('--hidden', '64,32').split(',')]
epochs = int(opts.get('--epochs', '30'))
lr = float(opts.get('--lr', '0.001'))
l2 = float(opts.get('--l2', '1e-5'))
SCALE = 25.0
NIN = 212

# ---------------------------------------------------------------- load
t0 = time.time()
rows = []
for f in sorted(glob.glob(data_dir + '/*.jsonl')):
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
n = len(rows)
my = np.zeros((n, 10), np.int16); opp = np.zeros((n, 10), np.int16)
pile = np.full((n, 32), -1, np.int16); top = np.zeros(n, np.int16)
sc = np.zeros((n, 4), np.float32); y = np.zeros(n, np.float32)
for i, r in enumerate(rows):
    my[i] = r[0]; opp[i] = r[1]
    p = r[2][-32:]
    pile[i, :len(p)] = p
    top[i] = r[2][-1]
    sc[i] = [r[3] / 31, r[4] / 30, r[5] / 50, r[6] / 50]
    y[i] = r[7] / SCALE
del rows
print('rows', n, 'load %.1fs' % (time.time() - t0), flush=True)

rng = np.random.default_rng(0)
# hold out the last 10% of the rows (contiguous, so positions of one hand stay on one side;
# a random row split would leak the hand outcome into the holdout set)
cut = int(n * 0.9)
tr, te = np.arange(cut), np.arange(cut, n)


def suit_map(p):
    """card id mapping for a suit permutation p (len 4)."""
    c = np.arange(52)
    return (p[c // 13] * 13 + c % 13).astype(np.int16)


def build(idx, cmap=None):
    m = len(idx)
    X = np.zeros((m, NIN), np.float32)
    a, b, pl, t = my[idx], opp[idx], pile[idx], top[idx]
    if cmap is not None:
        a, b, t = cmap[a], cmap[b], cmap[t]
        pl = np.where(pl >= 0, cmap[np.maximum(pl, 0)], -1)
    rr = np.arange(m)[:, None]
    X[rr, a] = 1
    X[rr, 52 + b] = 1
    valid = pl >= 0
    X[np.repeat(np.arange(m), pl.shape[1])[valid.ravel()], 104 + pl[valid]] = 1
    X[np.arange(m), 156 + t] = 1
    X[:, 208:212] = sc[idx]
    return X


# ---------------------------------------------------------------- polynomial baseline (same features as value.js)
def poly(idx):
    import itertools
    e = np.clip(sc[idx, 1] * 30, -15, 70); o = np.clip(sc[idx, 3] * 50, 0, 100); s = sc[idx, 0] * 31
    x = [e / 20, o / 20, s / 10]
    F = [np.ones_like(e)]
    for d in range(1, 5):
        for comb in itertools.combinations_with_replacement(range(3), d):
            f = np.ones_like(e)
            for c in comb:
                f = f * x[c]
            F.append(f)
    F += [(e <= 10).astype(np.float32), (o <= 10).astype(np.float32), (e <= 0).astype(np.float32)]
    return np.stack(F, 1)


w_poly, *_ = np.linalg.lstsq(poly(tr), y[tr], rcond=None)
pp = poly(te) @ w_poly
var_te = y[te].var()
print('polynomial baseline holdout R2 %.4f' % (1 - ((pp - y[te]) ** 2).mean() / var_te), flush=True)

# ---------------------------------------------------------------- MLP
sizes = [NIN] + hidden + [1]
W = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32) for i in range(len(sizes) - 1)]
B = [np.zeros(sizes[i + 1], np.float32) for i in range(len(sizes) - 1)]
if '--init' in opts:
    # fine-tune from an existing model (nn-value.js format)
    import re
    L = json.loads(re.search(r'NN_LAYERS = (.*);', open(opts['--init']).read()).group(1))
    W = [np.array(l['w'], np.float32).reshape(l['in'], l['out']) for l in L]
    B = [np.array(l['b'], np.float32) for l in L]
    print('initialised from', opts['--init'], [w.shape for w in W])
mW = [np.zeros_like(w) for w in W]; vW = [np.zeros_like(w) for w in W]
mB = [np.zeros_like(b) for b in B]; vB = [np.zeros_like(b) for b in B]
step = 0


def forward(X):
    acts = [X]
    h = X
    for i in range(len(W)):
        h = h @ W[i] + B[i]
        if i < len(W) - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


Xte = build(te)
best = (1e9, None)
bs = 512
for ep in range(epochs):
    t1 = time.time()
    order = rng.permutation(tr)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        cmap = suit_map(rng.permutation(4))
        X = build(idx, cmap)
        acts = forward(X)
        g = (acts[-1][:, 0] - y[idx])[:, None] * (2 / len(idx))
        step += 1
        for i in reversed(range(len(W))):
            gW = acts[i].T @ g + l2 * W[i]
            gB = g.sum(0)
            if i > 0:
                g = (g @ W[i].T) * (acts[i] > 0)
            for P_, G, M, V in ((W, gW, mW, vW), (B, gB, mB, vB)):
                M[i] = 0.9 * M[i] + 0.1 * G
                V[i] = 0.999 * V[i] + 0.001 * G * G
                mh = M[i] / (1 - 0.9 ** step); vh = V[i] / (1 - 0.999 ** step)
                P_[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
    pred = forward(Xte)[-1][:, 0]
    mse = ((pred - y[te]) ** 2).mean()
    print('epoch %d holdout R2 %.4f (%.0fs)' % (ep + 1, 1 - mse / var_te, time.time() - t1), flush=True)
    if mse < best[0]:
        best = (mse, ([w.copy() for w in W], [b.copy() for b in B]))
    if ep >= 3 and mse > best[0] * 1.01:
        break

Wb, Bb = best[1]
print('best holdout R2 %.4f' % (1 - best[0] / var_te))
rnd = lambda a: [round(float(x), 5) for x in np.asarray(a).ravel()]
layers = [{'in': int(w.shape[0]), 'out': int(w.shape[1]), 'w': rnd(w), 'b': rnd(b)} for w, b in zip(Wb, Bb)]
with open(out_js, 'w') as f:
    f.write('// Neural leaf-value model (tools/gen_nn.mjs, tools/train_nn.py), %d positions, holdout R2 %.4f.\n'
            '// Input 212: my hand | opponent hand | discard pile | top discard (52 one-hot each)\n'
            '//            | stock/31, myEval/30, myDW/50, oppDW/50. Output * %g = expected net points.\n'
            % (n, 1 - best[0] / var_te, SCALE))
    f.write('export const NN_SCALE = %g;\n' % SCALE)
    f.write('export const NN_LAYERS = %s;\n' % json.dumps(layers, separators=(',', ':')))
print('wrote', out_js)
