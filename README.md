[English](./README_en.md)

# `dsg-project-template`

<!-- #### プロジェクトの正式名称もしくはリポジトリ名に書き換えてください。 -->

## プロジェクトテンプレートについて

<!-- ####  dsg-project-template に関する記載です。プロジェクト開始後削除可。 #### -->

`dsg-project-template` から案件用リポジトリを作成して使う手順は、
[`dsg-project-template を使った DS 検証案件の進め方`](./docs/how-to-use-dsg-project-template.md)
を参照してください。
テンプレートから新しいリポジトリを作成した後の GitHub repository settings 確認は、
[`docs/agents/recommended-repository-settings.md`](./docs/agents/recommended-repository-settings.md)
を参照してください。

`dsg-project-template` への Contribution は歓迎しています。
[Development Guide](https://github.com/abeja-inc/dsg-project-template/wiki/Development-Guide) を参照の上、Issue / Pull Request を起票してください。

なお、本 README.md の以下の内容は、そのままプロジェクトの README.md として利用できるようにしています。
デフォルトでは `dsg-project-template` を想定した内容を記載していますが、必要に応じて編集してください（特に本セクションは削除していただいて問題ありません）。

`<!-- #### SOMETHING #### -->` で記載されているコメントは README.md 作成時のガイドラインです。

↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

## 概要  <!-- Required -->

<!-- ####  
このプロジェクト・リポジトリが解決したい課題と、そのアウトカムを数行で簡潔に記載してください。
また、Notion のプロジェクトページへのリンクがあれば、ここに追加してください。
例: 「〇〇事業の□□レポート自動生成のためのバッチ処理一式を提供します。」
例: 「○○社の××予測案件のための検証用リポジトリ。性能 xx % を達成。」
#### -->

### 利用した技術・手法  <!-- Optional -->

<!-- ####
このプロジェクトで利用している主要（特徴的）な技術スタックや手法を記載してください。
主要な技術スタック等が特にない場合は、この節は削除して構いません。
（例: 言語 / フレームワーク / ライブラリ / ミドルウェア / インフラなど）
例:
- Python 3.11
- PyTorch / TensorFlow / scikit-learn
- FastAPI / uvicorn
- ...
#### -->

## アーキテクチャ <!-- Optional -->

<!-- ####
システム全体像や検証環境の全体像がわかる情報を記載してください。
主要なアーキテクチャ情報が特にない場合は、この節は削除して構いません。

- 可能であれば、図や mermaid による図（外部サービスとの連携や主要コンポーネントの構成）があると望ましいです。
- テキストのみの場合も、外部サービス・バッチ・API などの役割関係を箇条書きで説明してください。
#### -->

## Setup Instructions  <!-- Required -->

<!-- ####
開発・検証環境を一から構築するための手順を記載してください。
「この手順どおりに実行すれば環境構築が完了する」状態を目指してください。
コマンドは、できるだけコピー & ペーストでそのまま実行できる形で記載してください。
#### -->

### Requirements  <!-- Required -->

<!-- ####
このプロジェクトを動かすために必要な前提条件を列挙してください。
以下は `dsg-project-template` の例です。プロジェクト固有の要件に応じて修正してください。
#### -->

- `docker compose`
- `git`
- `make`

<!-- #### 必要に応じて項目を追加してください。 #### -->

### How to Setup  <!-- Required -->

<!-- ####
環境構築の具体的な手順を、順を追って記載してください。
以下は `dsg-project-template` が想定している最もシンプルな手順例です。
プロジェクト固有のセットアップ手順に応じて、適宜加筆・修正してください。

特に、以下の内容がある場合は漏れなく記載してください。
- gcloud SDK / AWS CLI / ABEJA CLI などの設定
- 環境変数ファイル (.env など) の作成
- 初回起動やマイグレーション用のコマンド
- context/ 以下の Dockerfile や docker-compose.yml のカスタマイズ内容
#### -->

1. リポジトリをクローン：`git clone <repository_url>`

2. 作業ディレクトリに移動：`cd <repository_name>`

3. `.env.container` を編集し、必要な build オプションを設定します。

   ```bash
   ${EDITOR:-vi} .env.container
   ```

   build オプションのデフォルト値は項目ごとに異なります。必要に応じて `.env.container` を編集してください。

   - `WITH_NODE`（デフォルト: `false`）: Node.js / npm を有効化
   - `WITH_OPENCV`（デフォルト: `false`）: OpenCV 関連の追加依存（`libgl1`, `libglib2.0-0`）を有効化
   - `WITH_GH`（デフォルト: `false`）: GitHub CLI (`gh`) を有効化
   - `WITH_GITLEAKS`（デフォルト: `true`）: `gitleaks` を有効化
   - `WITH_LASTPASS`（デフォルト: `false`）: `lastpass-cli` を有効化
   - `WITH_VIM`（デフォルト: `false`）: `vim-tiny` を有効化

   独自の `apt` 由来パッケージをコンテナに追加したい場合は、
   `context/Dockerfile` から参照される runtime manifest
   （例: `/opt/runtime-manifests/local.txt`）にランタイム依存を追記してください。
   `runtime-assembly` ステージでは `/opt/runtime-manifests/*.txt` を集約して
   `apt install` する実装になっています。本テンプレートでは optional な `apt`
   パッケージをこの仕組みで一元管理し、`final` ステージは user/environment
   wiring に集中させる設計です。

4. コンテナを作成：`make docker-up-cpu`

   （CUDA を使用する場合は `make docker-up-gpu` を利用してください）

   初回起動時に、`context/entrypoint.sh` が以下を自動実行します。
   - `core.hooksPath` の effective 値が未設定のときだけ、`git config --local core.hooksPath .githooks` により Git hook を有効化
   - `uv sync --all-groups` による依存関係同期

   ※ `core.hooksPath` が local / global / system のいずれかで別値に設定されている場合、自動設定は行いません。
   ※ コンテナを使わずホストだけで作業する場合や、このリポジトリで `.githooks` に切り替えたい場合は、`make setup-hooks` を手動実行してください。

   ※ `context/Dockerfile` では、大きい UID での `lastlog` / `faillog` 異常肥大化を避けるため
   `useradd -l` を使用しています。コンテナ内で ssh サーバ等を動かし、ログイン監視や監査のために
   `lastlog` / `faillog` の初期化が必要な場合は、この `-l` を外してください。

   <details><summary>Python version の指定</summary>

   `context/<cpu or gpu>` 配下の `docker-compose.yml` で Python のバージョンを指定してください。

   </details>

   <details><summary>JupyterLab のポート設定</summary>

   JupyterLab を使用する場合は `ports` のコメントアウトを外し、Makefile 内の `start-jupyter` コマンドで指定しているポート番号を `docker-compose.yml` のポート設定に合わせてください。
   デフォルトではポート 8888 を使用します。DGX などの共用マシン上で起動する場合は、空いているポートを確認し、`docker-compose.yml` の `ports` のホスト側ポートを変更してください。
   （`start-jupyter` コマンド側のポート指定も同様に変更してください）

   </details>

4. コンテナに入ります。

   通常の開発作業はデフォルトユーザーで行います：

   ```bash
   make docker-shell
   ```

   パッケージのインストールなど root 権限が必要な場合はこちらを使います：

   ```bash
   make docker-shell-as-root
   ```

   ただし、root 権限の使用は `apt-get install` など OSレベルのパッケージ追加のみに留め、Pythonの依存関係は通常ユーザーで `uv add / uv sync`を使ってインストールしてください。

5. （必要に応じて、コンテナ内で）Python の依存パッケージを再同期します：`uv sync`

   通常はコンテナ起動時に `uv sync --all-groups` が自動実行されるため、
   この手順は依存関係を手動で再同期したい場合にのみ必要です。
   本テンプレートでは、ホストと仮想環境を共有しないため、
   コンテナ内の仮想環境として `.venv` ではなく `.venv_container`
   を使用します（`UV_PROJECT_ENVIRONMENT` により設定）。
   これにより、DGX 等で Docker ボリュームが非 RAID ディスクを使用してしまう問題も回避します。


   ```bash
   uv sync  # すべてのグループを同期します（基本的にはこのコマンドを実行してください）
   uv sync --dev  # dev グループのパッケージのみ同期します
   uv sync --no-dev  # dev グループ以外のパッケージを同期します
   uv sync --group <group_name1> --no-group <group_name2>  # 特定のグループのみ同期 / 同期しないグループを指定します
   ```

#### (Optional) 分析環境のセットアップ

<details><summary>VSCode</summary>

1. VSCode から Docker コンテナへ接続します。

   VSCode の `Dev Containers` 拡張機能を使用し、立ち上げたコンテナに VSCode をアタッチしてください。

2. VSCode のセットアップ

   **コンテナ内** で以下のコマンドを実行し、VSCode の最低限の拡張機能をインストールします。

   ```bash
   # Stable 版 VSCode を使う場合
   make setup-vscode

   # VSCode Insiders を使う場合
   make setup-vscode-insiders
   ```

</details>

<details><summary>JupyterLab</summary>

1. ローカル環境であれば、以下のコマンドを実行して JupyterLab を起動します。

   ```bash
   make start-jupyter
   ```

2. 起動後、ブラウザで [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab) にアクセスします。

   ポート番号を変更している場合は URL 中の 8888 を適宜変更してください。
   また、DGX などの共有マシンで起動している場合は、下記のように SSH ポートフォワーディングを行ってから上記 URL にアクセスしてください。

   ```bash
   ssh -N -L 8888:localhost:{コンテナと繋がっている DGX 側ポート番号} dgx1a
   ```

   ホスト OS が Windows の場合は `%USERPROFILE%/.wslconfig` を作成し、以下の内容を記載してください（作成後、一度 WSL を shutdown してください）。

   ```
   [wsl2]
   localhostForwarding=true
   ```

3. 利用終了後は JupyterLab を shutdown します。

   JupyterLab の GUI から shutdown しても構いませんし、以下のコマンドでコンテナごと停止しても構いません。

   ```bash
   make docker-stop
   ```

   もしくは、JupyterLab のみ停止したい場合は、以下のコマンドでも問題ありません。

   ```bash
   make stop-jupyter
   ```

</details>

## Datasets Description  <!-- Required if data is used -->

<!-- ####
データを利用するプロジェクトでは本節は必須です。データを扱わない場合は、この節を削除してください。

再現実験に必要なデータの準備方法を記載してください。
最低限、以下の観点を含めてください。
- データの入手元 / 取得方法
- データのバージョンや取得日時などの識別情報
- 配置パス（ローカル / S3 / ABEJA Datalake など）
- 匿名化・マスキング・アクセス権限の扱い
- 疑似データ / サンプルデータで代替可能な場合は、その作り方
#### -->

## 実行方法  <!-- Optional -->

<!-- ####
このプロジェクトで提供している主な処理（バッチ / 学習 / 推論 / API 起動など）ごとに、代表的な実行コマンドを記載してください。
主要な処理が特にない場合は、この節は削除して構いません。

例:
- 前処理: `python -m project_module.xxxx.preprocess`
- 学習: `python -m project_module.xxxx.train`
- 推論: `python -m project_module.xxxx.inference`

例:
- API サーバ起動: `make xxxx`
#### -->

## 再現用シナリオ（スモークテスト）  <!-- Required -->

<!-- ####
PJ 終了後、5〜10 分程度で実施できる「README どおりに動作する」ことを確認するための最小限の手順を記載してください。
本プロジェクトが純粋な検証ではない場合、この節は削除しても構いませんが、基本的には用意しておくことを推奨します。

例:
1. サンプルデータを準備する
2. 前処理 + 学習 + 推論の一連のコマンドを実行する
3. 期待される出力結果（ファイル / ログ / メトリクスなど）の確認方法

例：
1. xxx.csv を hogehoge から取得し、`experimentation/dataset/` に配置する
2. `experimentation/notebooks/main.ipynb` を実行する
3. `experimentation/outputs/` 以下に `output.md` が生成されていることを確認する
#### -->

## 開発

<!-- ####
このプロジェクトの開発に関する情報を記載してください。
以下は `dsg-project-template` の例です。必要に応じてプロジェクト固有の構成に合わせて修正してください。
#### -->

### AI コーディングエージェント向けの指示ファイル

- Claude Code / Codex / Gemini CLI などのエージェント向け共通指示の正本は、ルートの `AGENTS.md` です（`project_module/` 向けの追加指示は `project_module/AGENTS.md`）。
  指示を変更するときは `AGENTS.md` を編集し、同じディレクトリの `CLAUDE.md` は `@AGENTS.md` の1行のまま維持してください。
- Claude Code を `project_module/` などのサブディレクトリで起動すると、ルートの `AGENTS.md` は外部ファイルの import として扱われ、承認ダイアログが表示されます。
  承認しないとルートの共通指示が読み込まれないため、承認してください。
- `claude -p` などの非対話実行では承認ができず、ルートの共通指示が読み込まれません。非対話実行はリポジトリのルートで行ってください。

### ブランチ名

本リポジトリで想定しているブランチ名は、`prefix/短い要約` または `prefix-短い要約` 形式です。
prefix には次のような意味があります。

- `feat/` または `feature/` … 新機能
- `fix/` … バグ修正
- `doc/` … ドキュメント更新
- `change/` / `rename/` … 仕様変更・破壊的変更
- `refactor/` / `style/` / `chore/` … リファクタリング・整形・その他雑多な更新
- `test/` … テストコード更新
- `dep/` / `dependabot/` … 依存関係の更新
- `release/` … リリース用ブランチ（対象: `release/**` のみ）
- `copilot/` … GitHub Copilot による自動対応

上記フォーマットを満たしているかどうかは、[`Branch Name Verifier`](.github/workflows/branch_name_verifier.yml) ワークフローにより、`main`, `develop`, `release/**` ブランチへの PR 起票時に自動チェックされます。

<details><summary>また、上記フォーマットのブランチ名に対しては、[`PR Auto Labeler`](.github/workflows/pr-auto-labeler.yml) により、ラベルが PR に自動付与されます。</summary>

  - `feat/` または `feature/`: `feature`
  - `fix/`: `bug`
  - `doc/`: `documentation`
  - `change/` / `rename/`: `change`
  - `refactor/` / `style/` / `chore/`: `other`
  - `test/`: `enhancement` または `other`
  - `dep/` / `dependabot/`: `dependencies`
  - `copilot/`: `other`

  なお、`.github/workflows/**` を変更した場合はファイルマッチにより `cicd` ラベルが自動付与されます。
  また、`pyproject.toml` / `poetry.lock` / `uv.lock` / `requirements*.txt` を変更する場合は `dependencies` ラベルが自動で付与されます。

</details>

### テストと静的解析の実行方法  <!-- Required -->

<!-- ####
品質を保つために用意しているテストや静的解析ツールの実行コマンドを記載してください。
以下は `dsg-project-template` の例です。プロジェクト固有の構成に合わせて修正してください。
#### -->

- テスト実行（`pytest`）: `make test` または `make test (テスト対象のパス)`

  設定については [pytest.ini](./pytest.ini) を参照してください。

  注意：
  pytest の標準動作では、テストが 1 件も収集されなかった場合の終了コードは 5 となります。
  本プロジェクトでは `./conftest.py` の pytest フックにより、このケースを警告表示のうえ成功（終了コード 0）とみなすように設定しています。挙動を変更したい場合は `./conftest.py` 内のフックを編集してください。 Ref. [conftest.py: sharing fixtures across multiple files](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)

- Lint / フォーマット / 型チェック（`ruff`, `mypy`）: `make all-lint <directory or file>`

  個別コマンドは以下のとおりです。

  - Lint: `make lint <directory or file>`
  - フォーマットチェック: `make format <directory or file>`
  - 型チェック: `make type-check <directory or file>`
  - Lint自動修正: `make lint-fix <directory or file>`
  - フォーマット自動修正: `make format-fix <directory or file>`
  - Lint + フォーマット自動修正: `make fix <directory or file>`

  Lint / フォーマッタの設定については [ruff.toml](./ruff.toml) を、型チェックの設定については [mypy.ini](./mypy.ini) を参照してください。
  デフォルトでは `libs/` は `ruff` で除外され、`tests/` は `mypy` で除外されます。
  また、`tests/` は `ruff` では完全には除外されず、`E402` のみ per-file-ignore が設定されています。
  `libs/` 配下の特定ライブラリを `mypy` の対象から除外したい場合は、`mypy.ini` の `exclude` を更新してください。
  正確な対象・除外条件を変更したい場合は、`ruff.toml` と `mypy.ini` を更新してください。

- シークレットスキャン（`gitleaks`）: `make secret-scan` または `make secret-scan <path>`

  ソースコード内にハードコードされた API キーやトークンなどの機密情報を検知します。
  CI（GitHub Actions）では PR と `main` / `develop` への push 時に自動実行されますが、コミット前のローカルチェックとしても利用できます。

  事前に gitleaks のインストールが必要です（推奨バージョン: 8.30.0 以上）。
  - macOS: `brew install gitleaks`
  - Linux / Windows: [GitHub Releases](https://github.com/gitleaks/gitleaks/releases) からバイナリをダウンロード（詳細は [公式 README](https://github.com/gitleaks/gitleaks#installing) を参照）
  - Docker コンテナ内: デフォルトイメージに同梱されています。古いイメージを使っている場合は再ビルド（`make docker-build-cpu` または `make docker-build-gpu`）

  コンテナを起動すると、`context/entrypoint.sh` が effective な
  `core.hooksPath` 未設定時にだけ `core.hooksPath=.githooks`
  を初回自動設定し、`git push` 前に `.githooks/pre-push` が `gitleaks git`
  を実行します。すでに別の hook パスを使っている場合は自動設定されません。
  コンテナを使わない場合や、このリポジトリで `.githooks` に切り替えたい場合は
  `make setup-hooks` を実行してください。
  ローカル hook は `git push --no-verify` でスキップできますが、CI の
  `secret-scan` は継続して実行されます。
  `pre-push` と同じ条件を手元で再現したい場合は、
  `gitleaks git --config .gitleaks.toml --redact --log-opts='<commit>^!'`
  を使ってください。
  `make secret-scan [path]` は `.gitleaks-local.toml` を使う広域の
  ワーキングツリースキャンであり、`pre-push` と完全一致ではありません。

  設定については [.gitleaks.toml](./.gitleaks.toml)（CI 用）および [.gitleaks-local.toml](./.gitleaks-local.toml)（ローカル用）を参照してください。
  誤検知（false positive）が発生した場合は、以下の順で対応してください。
  1. `gitleaks:allow`（行末コメントで個別許可）
  2. `.gitleaks.toml`（CI・ローカル共通の構造的除外。**ディレクトリパス除外はここに入れないでください**）
  3. `.gitleaks-local.toml`（ローカル `make secret-scan` 専用のディレクトリパス除外）
  4. `.gitleaksignore`（fingerprint ベースの除外、最終手段）

  **推奨**: リポジトリの Settings > Branches > Branch protection rules で `secret-scan / gitleaks` を Required status check に追加すると、シークレットが検知された PR のマージをブロックできます。

### 依存関係の更新

プロジェクトの依存関係を更新するには、以下のコマンドを実行してください。

```bash
uv add <package_name>  # パッケージを追加します
uv add <package_name> --group <group_name>  # 特定のグループにパッケージを追加します
uv add <package_name> --dev  # dev グループにパッケージを追加します
uv add --editable <path_to_package>  # ローカルパッケージを editable モードで追加します
uv remove <package_name>  # パッケージを削除します
```

詳細は [https://docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/) を参照してください。

※ `project_module` の依存関係を更新したい場合は `./project_module` に移動した上で `uv add` などを実行してください。

### 依存関係の脆弱性チェック

**コンテナ内** で以下のコマンドを実行すると、`uv.lock` に記録された依存関係（推移的依存を含む）に既知の脆弱性がないかを確認できます。
`uv.lock` がない場合は、先に依存関係を解決してから確認します。

```bash
uv audit --preview-features audit-command  # すべての依存関係グループを確認します
uv audit --preview-features audit-command --no-dev  # dev グループを除いて確認します
```

- `uv audit` は uv 0.12 時点では experimental な機能のため、`--preview-features audit-command` を付けて実行します（付けない場合も実行できますが、警告が表示されます）
- 脆弱性が見つかった場合は、対象パッケージと修正済みバージョンが表示され、終了コードが 0 以外になります
- 詳細は [https://docs.astral.sh/uv/reference/cli/#uv-audit](https://docs.astral.sh/uv/reference/cli/#uv-audit) を参照してください

### CI/CD

- [branch_name_verifier.yml](.github/workflows/branch_name_verifier.yml): ブランチ名の命名規則をチェックする workflow の定義
- [export_sbom.yaml](.github/workflows/export_sbom.yaml): [sdd-security-actions](https://github.com/abeja-inc/sdd-security-actions) の reusable workflow を利用して SBOM (CycloneDX) を生成・エクスポートする workflow 定義。Python 依存関係（uv workspace 全体）と CPU 開発コンテナイメージの 2 つの SBOM を出力します。`main` ブランチへの push、`v*` タグの push および手動実行時に実行されます。
- [finding_changed_package.yml](.github/workflows/finding_changed_package.yml): 変更のあった Python パッケージを検知する workflow。 [lint.yaml](.github/workflows/lint.yaml) や [test.yaml](.github/workflows/test.yaml) で利用されます。
- [lint.yaml](.github/workflows/lint.yaml): Lint / フォーマッタ自動実行の workflow 定義。PR の起票および `main` ブランチへの push 時に実行されます。
- [pr-auto-labeler.yml](.github/workflows/pr-auto-labeler.yml): ブランチ名に応じて PR にラベルを自動付与する PR Auto Labeler の workflow 定義。PR の起票時に実行されます。
- [release-drafter.yml](.github/workflows/release-drafter.yml): Release Drafter によりリリースノートの下書きを更新する workflow 定義。`main` ブランチへの push 時に実行されます。
- [slack.yaml](.github/workflows/slack.yaml): Issue / PR の発行を Slack に通知する workflow 定義
- [test.yaml](.github/workflows/test.yaml): テスト自動実行の workflow 定義。PR の起票および `main` ブランチへの push 時に実行されます。
- [secret-scan.yaml](.github/workflows/secret-scan.yaml): Gitleaks によるシークレットスキャンの workflow 定義。PR の起票および `main` / `develop` ブランチへの push 時に実行されます。

以下、`dsg-project-template` 管理用 CI/CD ワークフロー:

- [create-release-pr.yml](.github/workflows/create-release-pr.yml): `workflow_dispatch` で手動実行し、`develop` などの指定ブランチから `release/{version}` ブランチを作成して、`main` 向けのリリース PR を自動作成する workflow の定義
  - 概要: `version` / `source_branch` / `target_branch` を受け取り、同名ブランチや既存 open PR があればスキップしつつ、リリース用ブランチと PR を作成します。
  - `dsg-project-template` 以外で使う方法:
    - そのままでも利用できます。
    - ただし `v1.2.3` 形式のバージョンバリデーションは `abeja-inc/dsg-project-template` でのみ実行され、他のリポジトリではスキップされます。
    - 別の運用に合わせたい場合は、`source_branch` / `target_branch` のデフォルト値やバリデーション条件を変更してください。
- [sync-main-to-develop.yml](.github/workflows/sync-main-to-develop.yml): `main` への push をトリガーに、`main` → `develop` の sync PR を自動作成する workflow の定義
  - 概要: `main` への push 後に既存 open PR の有無を確認し、未作成なら `develop` 向けの sync PR を 1 件だけ作成します。
  - `dsg-project-template` 以外で使う方法:
    - デフォルトでは `github.repository == 'abeja-inc/dsg-project-template'` の場合のみ動作します。
    - 他リポジトリでも使う場合は、この条件を削除または自分のリポジトリ名に変更してください。
    - 必要に応じて同期元・同期先ブランチ（既定では `main` → `develop`）を変更してください。

### [Abeja Toolkit](https://github.com/abeja-inc/abeja-toolkit) の更新

<details><summary>abeja-toolkit は、以下のコマンドで最新版に同期したり、新しいツールを追加したりできます。</summary>

```shell
chmod +x ./scripts/sync_tools.sh  # 初回のみ実行してください
./scripts/sync_tools.sh [<tool1> <tool2> ...] [--version <version>] [--debug]
```

または、次のように `make` を利用することもできます。

```shell
make sync-tools [<tool1> <tool2> ...] [VERSION=<version:branch or tag>] [DEBUG=1]
```

</details>

### [`dsg-project-template`](https://github.com/abeja-inc/dsg-project-template) の更新

<details><summary>`dsg-project-template` は、以下のコマンドで最新版に同期できます。</summary>

```shell
chmod +x ./scripts/sync_dsg-project-template.sh  # 初回のみ実行してください
./scripts/sync_dsg-project-template.sh [--version <version:branch or tag>] [--debug]
```

または、次のように `make` を利用することもできます。

```shell
make sync-template [VERSION=<version:branch or tag>] [DEBUG=1]
```

詳細は `./scripts/sync_dsg-project-template.sh --help` を参照してください。

</details>

## ディレクトリ構成  <!-- Required -->

<!-- ####
このリポジトリのディレクトリ構成と、主要ディレクトリ / モジュールの役割を簡潔に記載してください。
以下は `dsg-project-template` の例です。プロジェクト固有の構成に合わせて修正してください。
`tree` コマンドの抜粋などを利用しても構いません。
#### -->

```text
.githooks/
  pre-push                      # Git pre-push hook for Gitleaks
.github/
  pull_request_template.md      # PR テンプレートの設定ファイル
  workflows/
    lint.yaml                   # Lint / フォーマッタ自動実行
    test.yaml                   # テスト自動実行
    secret-scan.yaml            # シークレットスキャン（Gitleaks）
    ...
.vscode/                        # VSCode Dev Container / 推奨設定
  settings.json                 # VSCode ワークスペース設定。formatOnSave はデフォルトで off になっています。
context/                        # 分析・検証用 Dockerfile と CPU / GPU 環境向け docker-compose.yml を格納します。
  Dockerfile                    
  .hadolint.yaml
  cpu/docker-compose.yml        # CPU 用コンテナ設定
  gpu/docker-compose.yml        # GPU 用コンテナ設定
experimentation/
  dataset/                      # データセット配置場所
  latest/                       # 直近実験の設定・結果
  notebooks/                    # 実験用ノートブック
  outputs/                      # 実験結果出力
libs/
  abeja-toolkit/                # ABEJA 社内の共通ツール群
    lastpass_secret_provider/   # LastPass からシークレットを取得するツール
    spreadsheet/                # スプレッドシートのデータを読み書きするツール
project_module/
  tests/                        # 単体テスト等。test_*.py という命名規則を想定しています。
  src/project_module/           # プロジェクト固有ロジックの実装。この配下に開発したツールを格納します。
  pyproject.toml                # project_module の依存定義
  README.md                     # project_module の使い方
scripts/                        # テンプレート・ツール同期などの補助スクリプト群
  sync_dsg-project-template.sh  # dsg-project-template 同期スクリプト
  sync_tools.sh                 # abeja-toolkit 同期スクリプト
.gitignore
Makefile                        # 主要コマンドのエントリポイント
conftest.py                     # pytest 共通設定・フック定義
mypy.ini
pyproject.toml                  # 分析・検証・開発に伴う依存関係の定義
pytest.ini
README.md                       # このリポジトリの README
ruff.toml
.gitleaks.toml                  # Gitleaks 設定（CI 用シークレットスキャン）
.gitleaks-local.toml            # Gitleaks 設定（ローカル用、ディレクトリ除外あり）
uv.lock
```

## Tips

<details><summary>DGX / Linux サーバー上のコンテナから GitHub にアクセスする</summary>

本手順は **DGX / Linux サーバーのみ対象**です。macOS + Docker Desktop などは対象外です。
ssh-agent は有効済みが前提です。詳細は [ssh-agent のすすめ / Recommended ssh-agent](https://www.notion.so/abejainc/ssh-agent-Recommended-ssh-agent-1956fbf10b72806c96c1e98cb613bf3f?source=copy_link) を参照してください。

1. `ssh <dgx-host>` で DGX にログインし、`SSH_AUTH_SOCK` が有効であることを確認します。
   （`~/.ssh/config` で `ForwardAgent yes` が有効であることが前提）

   ```bash
   ssh <dgx-host>
   echo $SSH_AUTH_SOCK
   ls -l $SSH_AUTH_SOCK
   ssh-add -l
   ```

2. `context/cpu/docker-compose.yml` または `context/gpu/docker-compose.yml` の
   ssh-agent フォワーディング設定のコメントアウトを解除します（`volumes` と
   `environment` の両方）。

3. コンテナを起動します。

   ```bash
   make docker-up-cpu  # もしくは make docker-up-gpu
   ```

4. コンテナ内で GitHub へアクセスできることを確認します。

   ```bash
   echo $SSH_AUTH_SOCK
   ssh-add -l
   ssh -T git@github.com
   ```

※ `SSH_AUTH_SOCK` が未設定、またはソケットが存在しない場合、
ssh-agent フォワーディング設定のコメントアウトを解除すると `docker compose up` が
`bind source path does not exist` で失敗します。その場合は ssh-agent フォワーディング設定を
コメントアウトのままにしてください。

</details>

<details><summary>コンテナ内で `~/.ssh` を使う</summary>

`~/.ssh` をコンテナ内で直接使いたい場合は、`context/cpu/docker-compose.yml`
または `context/gpu/docker-compose.yml` の `.ssh` マウント設定のコメントアウトを
解除してください。

DGX / Linux サーバー上では `.ssh` をコンテナ内で直接使うのではなく、
直前の `DGX / Linux サーバー上のコンテナから GitHub にアクセスする` に従って
ssh-agent フォワーディングを使う運用を推奨します。

1. `context/cpu/docker-compose.yml` または `context/gpu/docker-compose.yml` の
   `.ssh` マウント設定のコメントアウトを解除します。

2. コンテナを起動します。

   ```bash
   make docker-up-cpu  # もしくは make docker-up-gpu
   ```

3. コンテナ内で `~/.ssh` が見えていることを確認します。

   ```bash
   ls -la ~/.ssh
   ```

※ `~/.ssh` をコンテナ内で使わない場合は、`.ssh` マウント設定をコメントアウトのままにしてください。

※ `~/.ssh` がホスト側に存在しない状態で `.ssh` マウント設定のコメントアウトを解除すると、
`docker compose up` 等が失敗します。その場合は `~/.ssh` を作成してから再度試してください。

</details>

<details><summary>開発コンテナに GitHub CLI (`gh`) を入れる</summary>

`gh` はデフォルトでは無効です。必要な場合のみ `WITH_GH=true` で build すると、
`https://github.com/cli/cli/releases/` から `gh` の `.deb` を取得して導入します。
バージョンは `context/cpu/docker-compose.yml` または
`context/gpu/docker-compose.yml` の `GH_VERSION` で指定します。

特に **DGX / Linux サーバー上では、直前の
`DGX / Linux サーバー上のコンテナから GitHub にアクセスする`
に従って ssh-agent をコンテナ内で有効化する運用を推奨**します。
`gh` でも GitHub への SSH ベースの Git 操作を行うため、この設定を先に済ませておくと
認証トラブルを減らせます。

1. 必要に応じて `context/cpu/docker-compose.yml` または
   `context/gpu/docker-compose.yml` の `GH_VERSION` を変更します。
2. `GH_VERSION` を変更した場合は、`context/cpu/docker-compose.yml` または
   `context/gpu/docker-compose.yml` の `GH_EXPECTED_SHA256_AMD64` /
   `GH_EXPECTED_SHA256_ARM64` も対応する値へ更新します。
3. `WITH_GH=true` を指定してイメージを再 build します。

   ```bash
   WITH_GH=true make docker-build-cpu  # もしくは WITH_GH=true make docker-build-gpu
   ```

4. コンテナを起動します。

   ```bash
   make docker-up-cpu  # もしくは make docker-up-gpu
   ```

5. コンテナ内で GitHub CLI を認証します。

   ```bash
   gh --version
   gh auth login --git-protocol ssh --skip-ssh-key
   gh auth status
   ```

   `gh` の認証状態は永続化しません。コンテナを作り直した場合は、
   必要に応じて再度 `gh auth login --git-protocol ssh --skip-ssh-key`
   を実行してください。

6. 必要に応じて、GitHub 操作を確認します。

   ```bash
   gh repo clone <owner>/<repo>
   gh pr status
   ```

</details>

<details><summary>lastpass-cli (lpass) を使ってシークレットを取得する</summary>

[lastpass-cli](https://github.com/lastpass/lastpass-cli) を使うと、コンテナ内から LastPass に保存されたパスワードや機密情報を取得できます。

**インストール方法**

`.env.container` の `WITH_LASTPASS` を `true` に変更し、コンテナをビルドし直してください。

```bash
# .env.container
WITH_LASTPASS=true
```

```bash
make docker-build-cpu  # GPU 環境の場合は make docker-build-gpu
```

**ログイン・シークレットの取得**

コンテナ内で以下のコマンドを実行してください。

```bash
# LastPass にログイン（メールアドレスを入力）
lpass login <your-lastpass-email>

# パスワードを取得する例
lpass show --password "<entry-name>"

# フィールドを指定して取得する例
lpass show --field="<field-name>" "<entry-name>"

# ログアウト
lpass logout
```

詳細は [lastpass-cli の公式ドキュメント](https://github.com/lastpass/lastpass-cli) を参照してください。

</details>

<details><summary>Notebook の出力を削除する</summary>

Notebook のセル出力に機密情報が含まれる場合、GitHub に push する前に出力を削除することを推奨します。
以下のコマンドで、指定したディレクトリ配下の notebook 出力を一括削除できます。

```bash
make reset-notebook /path/to/directory/
```

</details>

<details><summary>Issue / PR の発行を Slack に通知する</summary>

1. Slack App にアクセスし、通知用の Bot を作成します。
2. App 管理画面左側のナビゲーションメニューから `Incoming Webhooks` を選択し、Webhook を有効化します。
3. 画面を下にスクロールし、Webhook の作成申請を発行します（組織管理者への申請が必要です）。
4. 申請が承認されたら Webhook を作成します。
5. Webhook の URL を、リポジトリの GitHub Repository Secrets に `SLACK_WEBHOOK_URL` という名前で設定します。

</details>

## トラブルシューティング

- [Case: `make docker-up-cpu` や `make docker-up-gpu` 実行時に `~/.gitconfig` もしくは `~/.ssh` が存在しないためエラーになる](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

- [Case: `sudo make docker-up-cpu` や `sudo make docker-up-gpu` 実行時に `useradd: user 'root' already exists` というメッセージとともにエラーが発生する](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

- [Case: `make docker-up-cpu` や `make docker-up-gpu` を実行時に `invalid project name "_hoge-username-userid". must consist only of lowercase alphanumeric characters, hyphens and underscores as well as start with a letter or number` のようなエラーが生じてコンテナビルドできない](https://github.com/abeja-inc/dsg-project-template/wiki/Troubleshooting)

## 制約事項 / 既知の課題

<!-- ####
このプロジェクトを利用する際に注意すべき前提条件や制約、既知の課題を記載してください。
例:
- 大規模データではメモリ使用量が増大する
- 特定のパラメータ設定では精度が十分でない
#### -->

## ライセンス / 担当者・連絡先

<!-- ####
依存しているライブラリ・モデル等のライセンス情報（社外公開の場合）や、
このプロジェクトの主担当者・レビュー担当者などの連絡先を記載してください。
#### -->

## 関連リンク

<!-- ####
このプロジェクトに関連する Slack / Notion や、その他外部ドキュメント、共通基盤のレポジトリなどへのリンクを記載してください。
#### -->
