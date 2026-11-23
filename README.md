[English](./README_en.md)

# emeans-test

e-means と Lloyd、MiniBatch、Elkan k-means のクラスタ品質と学習時間を比較する実験リポジトリです。主な成果は、品質（MSE）と時間の関係を示す曲線です。固定の品質合格幅や全体の時間上限は設けず、事前に定めた条件・予算点・seedで比較します。

本評価は22条件・6手法・各10 seedの計9,520試行が完了しました。品質と速度の比較、22条件の曲線、少数成分の損失、再現性と計時の変動を[検証結果](docs/report/README.md)にまとめています。実装と各段階の検証は[実装進捗](docs/plan/implementation-progress.md)を参照してください。

## 実験の範囲

- **データ**: VIBE配布のarXiv–NomicとLandmark–Nomicの既存埋め込み、および分布形状や規模を制御する合成データ12条件。原文書・原画像やencoderは取得・実行しません。
- **比較手法**: CPUのFaiss Lloyd、NumPy e-means、scikit-learn MiniBatchKMeans、Elkan KMeansと、GPUのnative FP32 Torch e-means、Torch Lloyd対照。
- **主評価**: 最終中心から独立計算するtest MSEとwarm training time。GPUのTorch e-means/Lloydは共通のV100 native FP32再現性profileで比較しました。上流Torchの標準設定で行ったS0の再現性と感度検証は[本評価プロトコル](docs/report/protocol/README.md)に記録しています。
- **副次時間**: Resident-input pipelineは、共通input/初期中心の準備後、実測adapter呼出しの直前から計時し、centroid保存・fsync直後で終わります。GPU transfer・学習・保存を含み、データload/hash、初期化、評価や台帳I/Oは含みません。詳細は設計文書を参照してください。
- **主条件**: 実データ10セルと合成12セルの計22セル、各10 seed。Elkanを含む全手法を対象とし、資源境界に合わないrunもinfeasibleとして記録します。

設計・評価条件の正本は[設計と計画の索引](docs/plan/README.md)から確認できます。

GPU profileの根拠、診断結果、適用範囲、未完了の資格確認は[GPU再現性profile記録](docs/research/2026-11-18-gpu-reproducibility.md)にまとめています。

## 環境とセットアップ

実行環境は既存のGPU対応Dockerコンテナ、CPython `>=3.14,<3.15`、uvです。現在の環境はPython 3.14.9です。GPU手法はV100のSM70でnative FP32演算を使います。Faiss GPUは使いません。

CPU依存にTorchは含まれず、CPUテストはTorchをimportせずに実行できます。GPU依存はoptional extraです。外から継承した`UV_PROJECT_ENVIRONMENT`が別の仮想環境を指す場合があるため、このリポジトリ内の環境を明示してから同期してください。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync
```

GPUテストやGPU実験を行う環境では、GPU extraを追加します。

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --extra gpu
```

`gpu` extraはTorch `2.14.1+cu126`を追加します。CPUとGPUを同じ台帳で比較する場合は、CPU phaseを含む全実験コマンドに`uv run --extra gpu`を付け、lockとextraを固定します。extraなしの`uv run`はoptional依存を外すことがあり、凍結済みの実行環境と一致しなくなります。CPU専用のセットアップ・テストではGPU extraは不要です。

`make`で品質検査する際も同期によるextraの変更を避けるには、上のGPU同期を済ませてから`UV_NO_SYNC=1 make test`、`UV_NO_SYNC=1 make all-lint`を使います。依存変更後は明示的に`uv sync --extra gpu`をやり直してください。

公開Hugging Faceデータの取得に追加のAPI keyは不要です。`HF_TOKEN`は任意で、設定済みの場合に利用できます。トークンをREADMEや実験成果物へ書き込まないでください。

## テスト

通常のテスト実行では、`pytest.ini`がGPU marker付きテストを除外します。

```bash
uv run pytest project_module/tests/e_means -q
```

CUDA対応GPU上でGPUテストを明示的に実行します。

```bash
uv run --extra gpu pytest -o addopts='' -m gpu project_module/tests/e_means/test_methods_gpu.py -q
```

## CLI、設定、成果物

設定は`configs/experiments/e_means/`にあります。

- `smoke.yaml`: 小さな合成データでCLIとartifactを確認
- `pilot.yaml`: 実データ・合成データで実行条件を確認
- `core.yaml`: 実データ10セルの主評価
- `stress.yaml`: 合成データ12セルの主評価
- `sources.yaml`: VIBE revision、ファイルhash、由来とライセンス

入力データとrun artifactはGit管理外です。慣例上の保存先は`experimentation/dataset/e_means/`と`experimentation/outputs/e_means/`ですが、CLI引数で任意の場所へ変更できます。次の例はsmoke設定を使います。

```bash
uv run --extra gpu python scripts/e_means/prepare_data.py \
  --config configs/experiments/e_means/smoke.yaml \
  --data-root experimentation/dataset/e_means

uv run --extra gpu python scripts/e_means/run.py \
  --config configs/experiments/e_means/smoke.yaml \
  --data-root experimentation/dataset/e_means \
  --output-root experimentation/outputs/e_means/smoke \
  --phase cpu --dry-run

uv run --extra gpu python scripts/e_means/run.py \
  --config configs/experiments/e_means/smoke.yaml \
  --data-root experimentation/dataset/e_means \
  --output-root experimentation/outputs/e_means/smoke \
  --phase cpu --resume

uv run --extra gpu python scripts/e_means/run.py \
  --config configs/experiments/e_means/smoke.yaml \
  --data-root experimentation/dataset/e_means \
  --output-root experimentation/outputs/e_means/smoke \
  --phase gpu --devices 0 --resume

uv run --extra gpu python scripts/e_means/analyze.py \
  --runs-root experimentation/outputs/e_means/smoke \
  --output-root experimentation/outputs/e_means/smoke-analysis \
  --require-complete
```

`prepare_data.py`は設定されたデータを準備し、固定revision・hashを検証します。`run.py`は`--phase cpu|gpu`を選び、`--dry-run`、`--resume`、失敗runの明示的な再試行（`--retry-failed --retry-reason TEXT`）を受け付けます。`analyze.py`は部分結果も出力します。`--require-complete`を付けた場合、予定範囲が未完了なら終了コード2、完了なら0です。

runごとの設定・環境情報、中心、metrics、status等は指定したoutput root配下に保存します。大容量のraw data、中心配列、ログをGitへ追加しないでください。最終的な日本語報告と共有図は本評価後に`docs/report/`へ作成する予定です。現時点ではその報告はありません。

再開には、環境・入力・実行コード・lock・commitの同一性が必要です。実験前に専用worktreeと凍結commitを用意し、phase実行中は同worktreeのcommit、コード/config編集、dependency sync、データ再prepareを行いません。`uv`の自動同期を避けるため、凍結したGPU環境では`UV_NO_SYNC=1`または`uv run --no-sync --extra gpu`を使います。親CLIのCPU affinityはhardware fingerprintに含まれます。resumeする各phaseでtaskset/numactl等の有無とaffinity設定を変えないでください。実験を再開するときは`environment.json`の`code.git_head`で示す凍結commitをcheckoutし、元の成功・失敗artifactへ別条件を上書きしないでください。
