# 開発・公開ガイド

## ファイル構成

| パス | 内容 |
| --- | --- |
| `web/index.html`, `web/style.css` | 画面 |
| `web/js/engine.js` | ルール（メルド探索、デッドウッド、付け札、得点、ハンド進行、マッチ） |
| `web/js/ai.js` | AI（最強 / ふつう / よわい）。しくみは [algo.md](algo.md) |
| `web/js/ui.js` | 画面の制御 |
| `tools/test.mjs` | ルールエンジンのテスト |
| `tools/selfplay.mjs` | AI 同士の自己対戦。結果は [history.md](history.md) |
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

引数は `配札数 Aのレベル Bのレベル seed [Aのパラメータ上書きJSON] [Bのパラメータ上書きJSON]` です。各配札を席を入れ替えて2回対戦します。レベルは `strong` / `normal` / `easy`。

```bash
node tools/selfplay.mjs 1000 strong strong 7 '{"beta":6}'
```

## GitHub Pages で公開する

1. 変更を `main` にマージします。
2. リポジトリの **Settings → Pages → Build and deployment → Source** を **GitHub Actions** にします。
3. `main` への push（または Actions タブから `Deploy to GitHub Pages` を手動実行）で、テスト実行後に `web/` が公開されます。

公開URL: `https://<ユーザー名>.github.io/gin_rummy/`
