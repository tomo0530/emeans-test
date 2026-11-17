# `extract-to-toolkit` スキル

> [!NOTE]
> このファイル（`README.md`）は人間（開発者・レビュアー）向けの設計・概要
> ドキュメント。エージェントが実行する手順そのものは `SKILL.md` にある。
> `README.md` は Agent Skills 標準の Anatomy（`SKILL.md` + `references/` +
> `scripts/` + `assets/`）には含まれず、`SKILL.md` からもリンクしていない
> ため、スキル実行時にエージェントのコンテキストへロードされることはない
> （トークンを消費しない）。

---

## 概要 / Overview

案件リポジトリ内の `project_module/` 実装を解析し、`libs/` への共通化が見込めるモジュールを発見・スコアリングして、実際の抽出（ファイル移動・import 書き換え・ワークスペース登録）まで一気通貫で行うスキル。[Agent Skills](https://agentskills.io) オープン標準に準拠しており、Claude Code・Codex CLI・Gemini CLI のいずれからも同じ `SKILL.md` で実行できる。

**単一の案件リポジトリ内で完結して動作する。** 複数リポジトリをまたいだ比較は行わない。

### 切り出し先について（重要）

切り出し先は `libs/<lib_name>/`（案件ローカルの lib）であり、**`libs/abeja-toolkit/` の中には作らない。** `libs/abeja-toolkit/` は別リポジトリ `abeja-inc/abeja-toolkit` のミラーで、`make sync-tools` を実行すると `rm -rf` されて再取得される（＝そこに置いたものは消える）ため。ABEJA 全体の共通ツール群 `abeja-toolkit` へ本当に寄贈したい場合は、あちらのリポジトリへ別途 PR を出す運用とし、このスキルの対象外とする。

---

## 背景・目的 / Background

`dsg-project-template` を使って案件を進めると、`project_module/` の中に本来ドメイン固有でないコード（API クライアント、ユーティリティ、汎用アダプター等）が蓄積されがちである。これらは `libs/` に切り出せば案件内で再利用しやすくなるが、「どのコードが移せるか」の見極めと実作業が面倒で後回しにされやすい。

このスキルは以下の問いを自動的に答え、意思決定と作業の両方を省力化する。

- **どのコードが移せるか？** → レイヤー・結合度・汎用性の 3 軸スコアで定量評価
- **どの層のコードが向いているか？** → ドメインロジックに依存しない層（infrastructure / utils 等）を優先
- **移した後の作業は？** → `libs/` への配置・import 書き換え・ワークスペース登録まで自動化

---

## 使い方 / Usage

案件リポジトリのルートで実行する。引数は不要。

```bash
# 通常の使い方（引数なし）
/extract-to-toolkit

# package root を明示する場合
/extract-to-toolkit project_module/src/my_project/
```

スキルが `project_module/src/<pkg>/` を自動探索し、5 フェーズで処理を進める。

---

## 実装内容 / Implementation details

### ファイル構成

実体は `.agents/skills/` に置く。Codex CLI・Gemini CLI はこのパスを直接
自動検出する。Claude Code は `.agents/skills/` を見ないため、
`.claude/skills/extract-to-toolkit` をそこへの symlink にしている。

```
.agents/skills/extract-to-toolkit/   # 実体
  SKILL.md                        # スキル本体（エージェント向け手順）
  references/
    layer-rubric.md               # スコアリング判断の具体例（判定テーブルではない）
    lib-structure.md              # 新規 lib の標準ディレクトリ構造
  evals/                           # テストケースとダミー project_module フィクスチャ
  README.md                       # このドキュメント（人間向け設計・概要）

.claude/skills/extract-to-toolkit  # symlink -> ../../.agents/skills/extract-to-toolkit
```

---

### 処理フロー

```
Phase 1: Setup
  └─ project_module/src/<pkg>/ を自動探索・検証

Phase 2: Scan
  └─ .py を再帰読み込み
     → rel_path / dir_path / imports / symbols / internal_importers を収集
     → libs/ の既存シンボルも読み込み（重複検知用）

Phase 3: Analyze（スコアリング）
  └─ 3 軸でスコアリング → 合計 ≥ 4 のみ候補として残す

Phase 4: Report
  └─ ランキングテーブルを提示（ドメイン名詞フラグ・既存 lib との重複も表示）
     → ユーザーが抽出対象・ライブラリ名を選択

Phase 5: Extract
  └─ libs/<lib_name>/ 作成（既存 libs と同じ underscore 命名）
     → ファイル移動・import 書き換え・元ファイル削除
     → project_module/pyproject.toml に依存追加
     → ルート pyproject.toml に workspace メンバー追加
     → make fix → make all-lint → make test → 新規 lib 個別の lint / type-check
```

---

### スコアリング設計（Phase 3）

合計スコア = レイヤースコア + 結合スコア + 汎用性スコア（各 0–3、最大 9）

**ディレクトリ名・ファイル名には一切依存しない。** `project_module/` は
テンプレートでは空（`__init__.py` のみ）であり、ディレクトリ構造は各
案件チームが独自に定義する（日本語ディレクトリ名や独自命名を使う
チームもある）。そのため「`utils/`なら高スコア」のような命名規則の
テーブル引きは行わず、**LLMがコードの中身を読んで判断する**設計にした。
各軸は判定ルールのテーブルではなく「何を問うべきか」という自然言語の
問いかけとして`SKILL.md` Phase 3に記述してあり、詳細な判断例は
[`references/layer-rubric.md`](references/layer-rubric.md)を参照。

- **レイヤースコア**: 「案件の業務文脈を取り除いても、全く別の案件に
  そのまま持っていって意味が通じるか」を問う
- **結合スコア**: 「このファイルだけを新しいプロジェクトにコピーして、
  サードパーティ依存だけ入れれば動くか」を問う
- **汎用性スコア**: 「名前・中身から案件固有の業務が透けて見えないか、
  実際に複数箇所から使われているか」を問う

---

### 抽出後の成果物（Phase 5）

**新規作成**
```
libs/<lib_name>/
  src/<lib_name>/__init__.py
  src/<lib_name>/py.typed
  src/<lib_name>/<moved files>
  tests/__init__.py
  pyproject.toml
  README.md
```

ディレクトリ名は既存 libs と同じ underscore（`<lib_name>`）。kebab-case
の`<lib-name>`は`pyproject.toml`の`[project] name`（配布名）にのみ使う。
`libs/abeja-toolkit/` の中ではなく `libs/` 直下に作る（理由は「切り出し先
について」の節を参照）。

**更新**
| ファイル | 変更内容 |
|---|---|
| ルート `pyproject.toml` | `[tool.uv.workspace] members` に追加 |
| `project_module/pyproject.toml` | `[project] dependencies` と `[tool.uv.sources]` に追加 |
| `project_module/` 内の `.py` | 移動したモジュールへの import を `from <lib_name> import X` に書き換え |

---

## 動作確認 / Testing & results

スキルはコード生成ではなくエージェント手順書のため、[`evals/`](./evals/) に
テストケース（`evals.json`）とダミー `project_module`（`fixtures/`）を用意し、
それを使って検証する。

- [x] `evals/fixtures/dummy_project_module/` にダミー実装を用意
      （generic な utils/clients と、ドメイン固有な domain/invoice_parser.py の両方を含む）
- [x] `evals/evals.json` にスコアリング・グルーピング・ドメイン固有ファイルの
      誤検出防止をそれぞれ検証する3ケースを定義
- [ ] 実際に3ケースをスクラッチ環境で実行し、`expectations` を満たすか確認
- [ ] Phase 5 実行後に `make fix` / `make all-lint` / `make test` と、新規 lib 個別の lint / type-check が通るか確認
- [ ] import 書き換えと `pyproject.toml` 更新が正しく行われるか確認

---

## 関連リンク / Links

- [`SKILL.md`](./SKILL.md) — エージェント向けの実行手順（本体）
- [`references/layer-rubric.md`](./references/layer-rubric.md) — スコアリング判断の具体例
- [`references/lib-structure.md`](./references/lib-structure.md) — 新規 lib のファイル構造テンプレート
- [`evals/`](./evals/) — テストケースとダミー `project_module` フィクスチャ
- [`libs/`](../../../libs/) — 抽出先ディレクトリ（`libs/<lib_name>/` を新規作成。`libs/abeja-toolkit/` は同期対象なので使わない）
- [`pyproject.toml`](../../../pyproject.toml) — uv ワークスペース定義

---

## コメント / Comments

- **PR・コミットは対象外**: 抽出後のコミット・PR はユーザーが手動で行う。
- **スコア閾値（≥ 4）は調整可能**: 汎用性の判断基準を厳しくしたい場合は ≥ 5 に変更してよい。
- **ファイル名に案件固有名詞がある場合はフラグを出す**: スコアが高くても `invoice_parser.py` のようなファイルは移しにくいため、レポート時に注記する。
- **既存 lib との重複チェック**: Phase 2 で `libs/`（`abeja-toolkit/` の同期分と案件ローカルの lib の両方）の既存シンボルを読み込み、候補との重複をレポートに表示する。
