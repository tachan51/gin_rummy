# 開発・公開ガイド

## ファイル構成

| パス | 内容 |
| --- | --- |
| `web/index.html`, `web/style.css` | 画面 |
| `web/js/engine.js` | ルール（メルド探索、デッドウッド、付け札、得点、ハンド進行、マッチ） |
| `web/js/ai.js` | AI（最強＝探索あり / ふつう＝ヒューリスティックのみ / よわい＝貪欲）。しくみは [algo.md](algo.md) |
| `web/js/ismcts.js` | ISMCTS（オプション。既定では無効） |
| `web/js/match-equity.js`, `web/js/hand-outcomes.js` | マッチエクイティ表と、その元になるハンド結果の分布（オプション） |
| `web/js/opp-model.js` | 相手手札の推定モデル（ロジスティック回帰）の重み |
| `web/js/nn.js`, `web/js/nn-value.js` | ニューラルネットの局面評価の推論と重み（オプション） |
| `web/js/value.js` | 探索で使う局面評価モデルの係数（自動生成） |
| `web/js/ai-worker.js` | AI を Web Worker で動かすための入口 |
| `web/js/ui.js` | 画面の制御 |
| `tools/test.mjs` | ルールエンジンと AI のテスト |
| `tools/selfplay.mjs` | AI 同士の自己対戦。結果は [history.md](history.md) |
| `tools/match.mjs` | 100点先取のマッチ単位の自己対戦（マッチ勝率と95%信頼区間） |
| `tools/gen_outcomes.mjs` | ハンド結果の分布の生成 |
| `tools/gen_nn.mjs`, `tools/train_nn.py` | ニューラルネット用の学習データ生成と学習（numpy） |
| `tools/opp_eval.mjs`, `tools/fit_opp.py` | 相手手札の推定の精度測定・学習データ生成と、推定モデルの当てはめ |
| `tools/oppsim_eval.mjs` | シミュレーション内の相手の捨て札が、実際の捨て札と一致する割合 |
| `tools/sweep.sh` | パラメータ違いの自己対戦を並列実行 |
| `tools/baseline/ai_v1.js` | 初版の最強AI（比較用に凍結。レベル名 `v1`） |
| `tools/gen_value.mjs`, `tools/fit_value.py` | 局面評価モデルの学習データ生成と当てはめ |
| `.github/workflows/pages.yml` | GitHub Pages への自動デプロイ |

ビルド不要の静的サイトで、外部ライブラリも使っていません。

## ローカルで動かす

ES Modules を使うため、`file://` で直接開かず HTTP サーバー経由で開いてください。

```bash
cd web && python3 -m http.server 8000
```

ブラウザで http://localhost:8000 を開きます。

## テストと自己対戦

```bash
node tools/test.mjs
```

```bash
node tools/selfplay.mjs 500 strong normal 1
```

引数は `配札数 Aのレベル Bのレベル seed [Aのパラメータ上書きJSON] [Bのパラメータ上書きJSON]` です。各配札を席を入れ替えて2回対戦します。レベルは `strong`（最強）/ `normal`（ふつう）/ `easy`（よわい）/ `v1`（凍結した初版の最強AI）。

結果がマシンの負荷に左右されないよう、自己対戦では時間制限の代わりに探索する世界の数を300に固定しています（`timeBudget: 0, searchMin: 300, drawSearchMin: 300`）。上書きJSONで変えられます。

```bash
node tools/selfplay.mjs 1000 strong strong 7 '{"beta":6}'
```

```bash
tools/sweep.sh 800 31 v1 '{"searchCands":6}' '{"takeGainW":0.5}'
```

### マッチ勝率で比べる

```bash
node tools/match.mjs 300 strong normal 1
```

引数は `マッチ数 Aのレベル Bのレベル seed [Aの上書きJSON] [Bの上書きJSON] [目標点]` です。各マッチを席を入れ替えて同じ山札の順で2回行い、Aのマッチ勝率を95%信頼区間つきで表示します。マッチの勝ち負けは運の影響が大きいので、数%の差を見分けるには数千マッチ必要です。

## 局面評価モデルを学習し直す

```bash
mkdir -p /tmp/vdata
node tools/gen_value.mjs 2500 1 > /tmp/vdata/d1.csv
python3 tools/fit_value.py /tmp/vdata
```

`gen_value.mjs` は最強AI同士で対戦し、各手番開始時の局面（自分の評価値・相手のデッドウッド・山札枚数）と、そのハンドの最終得点を書き出します。複数の seed で並列に実行してから当てはめると効率的です。`fit_value.py` は `web/js/value.js` を上書きします（numpy が必要です）。

## ニューラルネットの局面評価を学習する

```bash
mkdir -p /tmp/nndata
node tools/gen_nn.mjs 3000 1 > /tmp/nndata/d1.jsonl
python3 tools/train_nn.py /tmp/nndata /tmp/nn-value.js --hidden 64,32 --epochs 15
```

- `gen_nn.mjs` の第3引数に `'{"level":"normal"}'` を渡すと、「ふつう」同士の対局で約10倍速くデータを作れます。
- `--init 既存の重み.js --lr 0.0003` で、既存の重みからの追加学習になります。
- 自己対戦での比較は、`'{"nn":1,"nnFile":"/tmp/nn-value.js"}'` のように重みファイルを指定します。

## 相手手札の推定モデルを学習し直す

```bash
mkdir -p /tmp/oppdata
node tools/opp_eval.mjs 1500 1 '{"oppModel":0}' /tmp/oppdata/f1.csv
python3 tools/fit_opp.py /tmp/oppdata
```

`opp_eval.mjs` は精度の指標も表示します。

- 対数損失（何も推測しない場合との比較）
- サンプリングした相手手札の的中率
- 相手のデッドウッドの見積もり誤差

設定を変えて精度を比べるときにも使えます（例：`'{"oppModel":1,"oppFilter":1}'`）。

## GitHub Pages で公開する

1. 変更を `main` にマージします。
2. リポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にします。
3. `main` への push（または Actions タブから `Deploy to GitHub Pages` を手動実行）で、テスト実行後に `web/` が公開されます。

公開URL: `https://<ユーザー名>.github.io/gin_rummy/`
