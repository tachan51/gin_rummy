# 開発・公開ガイド

## ファイル構成

| パス | 内容 |
| --- | --- |
| `web/index.html`, `web/style.css` | 画面 |
| `web/js/engine.js` | ルール（メルド探索、デッドウッド、付け札、得点、ハンド進行、マッチ） |
| `web/js/ai.js` | AI（最強＝探索あり / ふつう＝ヒューリスティックのみ / よわい＝貪欲）。しくみは [algo.md](algo.md) |
| `web/js/ismcts.js` | ISMCTS（オプション。既定では無効） |
| `web/js/value.js` | 探索で使う局面評価モデルの係数（自動生成） |
| `web/js/ai-worker.js` | AI を Web Worker で動かすための入口 |
| `web/js/ui.js` | 画面の制御 |
| `tools/test.mjs` | ルールエンジンと AI のテスト |
| `tools/selfplay.mjs` | AI 同士の自己対戦。結果は [history.md](history.md) |
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

## 局面評価モデルを学習し直す

```bash
mkdir -p /tmp/vdata
node tools/gen_value.mjs 2500 1 > /tmp/vdata/d1.csv
python3 tools/fit_value.py /tmp/vdata
```

`gen_value.mjs` は最強AI同士で対戦し、各手番開始時の局面（自分の評価値・相手のデッドウッド・山札枚数）と、そのハンドの最終得点を書き出します。複数の seed で並列に実行してから当てはめると効率的です。`fit_value.py` は `web/js/value.js` を上書きします（numpy が必要です）。

## GitHub Pages で公開する

1. 変更を `main` にマージします。
2. リポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にします。
3. `main` への push（または Actions タブから `Deploy to GitHub Pages` を手動実行）で、テスト実行後に `web/` が公開されます。

公開URL: `https://<ユーザー名>.github.io/gin_rummy/`
