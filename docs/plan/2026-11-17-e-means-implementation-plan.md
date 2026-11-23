# e-means Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 同一データ・初期中心と独立した最終MSEを用い、e-meansと既存k-means手法の品質対時間を再現可能に比較する。

**Architecture:** 環境と公開契約を先に固定し、データ、手法、評価・分析を別所有範囲で並行実装する。runnerが実行・再開・artifactを統合し、pilotで実行適合を確認してから条件を凍結し本評価する。

**Tech Stack:** CPython 3.14、uv、NumPy、faiss-cpu同梱e-means、PyTorch cu126、scikit-learn、h5py、Pydantic、pytest、YAML。

**Spec:** [目的・方針](design/00-overview.md)、[実験条件・評価](design/01-experiment-protocol.md)、[実行基盤・公開契約・成果物](design/02-runtime-and-artifacts.md)。実装者は3文書を先に読む。各型の全フィールドと検証規則は公開契約を正本とする。

## Global Constraints

- 設計・計画のレビュー後、ユーザーからB1–B4の実装・実験・commit・pushまで自律実行の指示を受領済み。以下の全タスクを対象とする。
- Python `>=3.14,<3.15`、既存GPU Docker内のuv環境を使う。ホスト実行や新規コンテナ起動を自動代替にしない。Docker環境をread-onlyで確認できない場合は停止して報告する。
- ユーザーは依存追加等の自律実行を許可済み。`exclude-newer = "1 week"`は維持し、既存のcertifi例外を維持する。研究用の追加例外はユーザー承認済みの固定版faiss-cpu 1.16.0に限る。候補版と実機確認は[環境仕様](design/02-runtime-and-artifacts.md#環境)参照。
- 既存V100 32GB ×8、driver 535、物理CPU 40 cores、RAM 504GiB。実行は資源と計時の仕様に従い、GPU非適合をCPUのみで完了扱いしない。
- 全手法はFP32、同一split・共通初期中心、最終centroidでの独立exact MSEを使う。総予算Tごとに独立完走し、warm/cold/転送/評価を分離する。品質マージンと全体時間上限を固定しない。
- 実用比較と機構比較を分け、CPU/GPU時間比をアルゴリズム単独の効果と呼ばない。条件表、seed、budget、ライセンス、計時、失敗の定義は[実験仕様](design/01-experiment-protocol.md)が正本。
- 上流e-meansを再実装しない。GPUは`QuantizedEmeans`の`quantizer=None`、`assignment_backend="native"`、FP32。薄いTorch Lloydは同じ上流assignmentを共有し独立oracleで検証する。
- CPU/GPUを同じ台帳で実行する間は`gpu` extraとlockを固定する。明示的な`uv sync --extra gpu`後、品質検査は`UV_NO_SYNC=1 make ...`、実験CLIは`uv run --extra gpu ...`で実行する。親CLIのCPU affinityもphase間で変えない。実行中のcommit・source編集・依存同期は行わず、検出時は手法のterminal failedを増やさずphaseを中断する。
- 計時仕様の改訂は`e2e_scope=resident-input-pipeline-v1`で識別する。`2026-11-17-timing-v2`で導入し、現行の`plan_revision=2026-11-18-cuda-deterministic-v1`にも引き継いだ。主warm/MSE/条件/seed/予算は維持し、secondaryはresident inputから中心保存までへ限定する。旧load/全評価込みwallは診断metadataへ分離する。詳細は実験仕様を正本とする。
- 実ベクトルのみを取得しencoderを使わない。HF_TOKEN以外の認証は追加しない。データ本体・centroid・run logはGit管理外。既存dirty変更を保持し、関係外の変更を整形・コミットしない。
- 研究用パッケージは型付き関数・配列契約と純粋なlogic/I/Oの分離に従う。ENGへの引き渡し時にはDS integration contractを適用する。

## Review Focus

1. splitの境界をまたぐ厳密重複: 同じvector groupは一つのsplitにだけ入り再生成しても同じmanifestになる（B2データ）。
2. 非有限値・不正shape・正規化済み配布物のゼロnorm: 入力エラーとして明示し成功artifactへ変換しない。一般のL2入力の原点は有効とする（B2データ・評価、B3）。
3. 空clusterや重複点: Lloydは空中心を維持し、最終中心から独立再割当したMSEを返す（B2手法・評価）。
4. 中断、欠損artifact、異なるhash: 成功を誤認せず履歴を保持し、別条件を上書きしない（B3）。
5. GPU非適合やworker失敗: 他runの結果を保持し、failed/infeasible/unsupportedを区別して台帳と分析に残す（B2手法、B3、B4）。

---

## 実行順序・レビュー

B1完了後、B2の3担当は固定された契約を消費し、記載されたファイルのみを所有して並列作業する。契約変更が必要なら合流点で調整する。B3で統合し全体smoke・関連テストを通した後、実装全体を束ねてレビューする。B4はpilot、条件凍結、本評価、報告の順に進む。細かなタスクごとのレビューやコミットを連続させない。コミットはユーザーが許可したまとまり単位とし、既存dirty変更を含めない。

実行時のtool allowlistで利用可能IDを確認する。割当の目安は、定型文書にLuna `gpt-6-luna`、実装にTerra `gpt-6.1-terra`、統計にSol `gpt-6.2-sol`を各xhigh、難しい設計にAstra `gpt-6.2-astra`のmedium/high/xhighを複雑度に応じて使う。追加の高度な論点はAstra highとFable highへまとめて相談できる。最終レビュー前に実装者自身が仕上げる。

重要なprotocol、metrics、再現性、科学的結論の最終ゲートはOpus xhighとGPT-6.2 Sol xhighの双方。判断と修正案を束ねて両者に提示し、指摘を解決した後に完了とする。両者を利用できない場合は承認済みと扱わず報告する。

### B1: 環境と契約

**Files:**
- Modify: `pyproject.toml`, `project_module/pyproject.toml`, `uv.lock`, `context/Dockerfile`, `context/cpu/docker-compose.yml`, `context/gpu/docker-compose.yml`, `ruff.toml`, `mypy.ini`, `.github/workflows/python_test_workflow_call_dispatch.yaml`, `pytest.ini`, `Makefile`（必要箇所のみ）。
- Create: `project_module/src/project_module/e_means/contracts.py`, `project_module/src/project_module/e_means/__init__.py`, `project_module/tests/e_means/test_contracts.py`。

**Interfaces:** Produces `PreparedDataset`, frozen Pydantic `RunSpec`/`ExperimentConfig`, `TimingBreakdown`, `TrainingResult`, `Evaluation`, `DatasetManifest`, `RunRecord`, `RunMetrics`, JSON値型、`derive_seed`。各フィールド・shape・schema version・検証は[公開契約](design/02-runtime-and-artifacts.md#公開契約)を正本にB1で凍結する。B2/B3はこの型からfixtureを作り、別schemaを作らない。

- [x] 既存GPU環境、Python、driver、wheel可用性、cooldown、現状のテストbaselineを確認して結果を記録する。実行環境が不明、または必須wheelが条件を満たさない場合は無理に変更せず該当枝を止める。
- [x] `test_run_spec_rejects_unknown_fields_and_invalid_budget`（未知fieldと非正budgetを拒否）、`test_timing_rejects_negative_seconds`、`test_run_spec_is_frozen`、`test_config_yaml_json_round_trip`（ネスト型とschema_version保存）、`test_artifact_round_trip_rejects_nonfinite`、`test_pilot_config_rejects_test_evaluation`、`test_named_random_streams_are_reproducible_signed_safe`（常に0～2^31-1、同入力同値）を記述する。
- [x] `make test project_module/tests/e_means/test_contracts.py` を実行し、未実装の契約によるREDを確認する。環境解決失敗はREDの代わりにしない。
- [x] 契約と環境設定を一括実装し、cooldownを守って候補依存を解決・lockする。`make all-lint`はformat→type-check→lintの順に修正する。CPUのみのテスト時にGPUやtorch importを必須化しない。上記コマンドで対象テストが収集され全件PASSすることを確認する。
- [x] GPU統合用`gpu` markerを`pytest.ini`に登録し、default suiteからGPU専用テストを除外する。通常suiteと明示GPU suiteの選択が重ならないことを確認する。

### B2: データ、手法、評価・分析（契約固定後に並列）

#### B2-A: データと初期中心

**Files:** Create `project_module/src/project_module/e_means/data.py`, `initialization.py`, `scripts/e_means/prepare_data.py`, `project_module/tests/e_means/test_data.py`, `test_initialization.py`, `configs/experiments/e_means/{smoke,pilot,core,stress}.yaml`。ライセンス通知と小さいmanifestは仕様どおり追加する。

**Interfaces:** Consumes B1 contracts。Produces `load_dataset(dataset_id: str, root: Path, *, condition_id: str, data_seed: int) -> PreparedDataset`、`make_initial_centers(x: np.ndarray, k: int, seed: int) -> np.ndarray`。準備CLIは `uv run --extra gpu python scripts/e_means/prepare_data.py --config <yaml> --data-root <path>`。condition/data_seed別の配列・group・manifestをprepareが物化し、loaderは生成params/n_train/hashを照合する。

- [x] `test_duplicate_groups_never_cross_splits`（厳密重複groupがsplit間で交差せず同seedで再現）、`test_rejects_invalid_normalized_dataset`（非有限値・正規化済み配布物のゼロnorm・不正dimensionを拒否）、`test_initial_centers_are_distinct_reproducible_train_rows`（同seed同値・入力由来・値としてk個異なる）、`test_too_few_unique_rows_rejected`（候補不足を拒否）を書く。
- [x] `make test project_module/tests/e_means/test_data.py project_module/tests/e_means/test_initialization.py` で未実装によるREDを確認する。
- [x] `test_condition_and_data_seed_select_exact_manifest`（異なるrealizationを混用しない）、`test_synthetic_group_ids_survive_round_trip`を加え、主/pilot別conditionと群IDの保持を確認する。
- [x] 指定revisionとサイズ/SHA256の検証、group split・nested subset・syntheticの独立stream、由来/権利/seed/hash manifest、共通初期化を実装する。CLIは公開vectorのみ取得し、入力不正や配布情報不一致を止める。上記テストで収集件数と全PASSを確認する。

#### B2-B: 手法adapter

**Files:** Create `project_module/src/project_module/e_means/methods/{__init__,faiss_lloyd,numpy_e_means,mini_batch,elkan,torch_e_means,torch_lloyd}.py`, `project_module/tests/e_means/test_methods_cpu.py`, `test_methods_gpu.py`。

**Interfaces:** 各adapterはB1を消費し `train(x: np.ndarray, init: np.ndarray, spec: RunSpec) -> TrainingResult` を提供。各methodのbudget単位と注入条件は[実験仕様](design/01-experiment-protocol.md#手法と公平性)に従う。

- [x] `test_methods_use_supplied_centers_and_report_actual_work`（外部中心・全件条件と実処理量）、`test_minibatch_first_batch_accepts_k_over_4096`（初回batchがk以上）、`test_torch_lloyd_one_iteration_matches_fp64_oracle`、`test_empty_center_is_retained`、`test_lloyd_objective_does_not_increase`、`test_lloyd_iteration_prefix_agrees`、`test_unavailable_native_backend_is_unsupported`を書く。GPU数値の初期許容はrtol=1e-4、atol=1e-6。
- [x] `make test project_module/tests/e_means/test_methods_cpu.py` と `uv run --extra gpu pytest -o addopts='' -m gpu project_module/tests/e_means/test_methods_gpu.py -v` で、対象テストが収集され未実装によるREDを確認する。GPUコマンドはV100環境で実行する。
- [x] adapterを実装し、MiniBatchの明示`partial_fit`回数、各手法の実処理量、復活・収束・backend/dtypeを記録する。Torch Lloydは上流native assignmentを使い、代替kernelへの黙った切替を行わない。上記CPU/GPUコマンドで全PASSし、GPUテストがskipだけでないことを確認する。

#### B2-C: 独立評価・統計

**Files:** Create `project_module/src/project_module/e_means/evaluation.py`, `analysis.py`, `scripts/e_means/analyze.py`, `project_module/tests/e_means/test_evaluation.py`, `test_analysis.py`。

**Interfaces:** Consumes B1 contracts。Produces `evaluate_centers(x: np.ndarray, centroids: np.ndarray, *, chunk_rows: int) -> Evaluation`、`evaluate_groups(x: np.ndarray, centroids: np.ndarray, group_ids: np.ndarray, *, chunk_rows: int) -> Mapping[str, JSONValue]`、`summarize_runs(run_dirs: Sequence[Path], out_dir: Path) -> Path`。分析CLIは `uv run --extra gpu python scripts/e_means/analyze.py --runs-root <path> --output-root <path> [--require-complete]`。strict条件とstatusは仕様どおり、欠落は部分reportに残してexit 2、充足時はexit 0。

- [x] `test_exact_mse_matches_hand_computed_oracle`（tiny例の点平均squared L2とcluster sizesを手計算と照合）、`test_chunking_preserves_mse_and_empty_clusters`、`test_nonfinite_centroids_are_rejected`、`test_paired_summary_preserves_unreached_and_failures`（対応seedと未到達・失敗を区別）、`test_paired_bootstrap_uses_seed_as_unit`（小fixtureの対応差を検証）を書く。
- [x] `make test project_module/tests/e_means/test_evaluation.py project_module/tests/e_means/test_analysis.py` で未実装によるREDを確認する。
- [x] `test_summary_mean_quality_median_time_and_repeat_order`（MSE平均/seed内時間中央値→seed間中央値）、`test_reference_is_same_cell_and_seed`、`test_incomplete_points_are_not_bridged`、`test_empty_group_is_null_not_failed`、`test_origin_is_valid_l2_input`を加える。未到達/不完全/基準未定義を区別し、failed初回をretry成功に置換しないこともfixtureで検証する。
- [x] FP32距離/FP64集計、tiny FP64 oracleとspot check、group別損失、対応bootstrapと事前定義水準の観測点のみの最短時間を実装する。上記テストで収集件数と全PASSを確認する。

### B3: runner・統合・smoke

**Files:** Create `project_module/src/project_module/e_means/runner.py`, `_sentinel.py`, `sentinel_analysis.py`, `scripts/e_means/run.py`, `project_module/tests/e_means/test_runner.py`, `test_smoke.py`, `test_sentinel.py`, `test_sentinel_analysis.py`。必要な限定的な設定・通知のみB2と調整する。

**Interfaces:** Consumes B1/B2。Produces `run_trial(spec: RunSpec, dataset: PreparedDataset, output_root: Path) -> Path`。実行CLIは `uv run --extra gpu python scripts/e_means/run.py --config <yaml> --data-root <path> --output-root <path> --phase {cpu,gpu} [--devices 0,1,...] [--resume] [--dry-run] [--retry-failed --retry-reason TEXT]`。dry-runは型・条件・manifest検証とjob台帳出力のみ。resumeはpending/interruptedだけ、failed再試行は理由と別attemptを残す。分析にはrun群の親directoryを渡す。

- [x] `test_resume_retries_interrupted_attempt_without_deleting_history`、`test_resume_preserves_failed_terminal_status`、`test_timing_repeat_and_measurement_id_never_reuse_success`、`test_pilot_never_evaluates_test`、`test_changed_hash_never_reuses_or_overwrites_success`、`test_missing_artifact_never_marks_success`、`test_worker_failure_is_recorded_without_losing_other_results`、`test_logs_do_not_contain_token`、`test_smoke_uses_same_init_and_independent_final_mse`を書く。
- [x] `make test project_module/tests/e_means/test_runner.py project_module/tests/e_means/test_smoke.py` で未実装によるREDを確認する。
- [x] run keyとattempt履歴、atomic publication、必要artifact検証後のsuccess marker、spawn GPU worker、資源/affinity/phase制御、同期計時、秘密値除去、CLIを実装する。上記テストが収集され全PASSすることを確認する。
- [x] 既存GPUコンテナ内で以下を順に実行し、smokeの全対象method・予算点が対応する初期中心、有限MSE、計時区分、statusと再読込可能なartifactを持つことを確認する。`smoke.yaml`は小さなsynthetic専用とする。

  ```bash
  uv run --extra gpu python scripts/e_means/prepare_data.py --config configs/experiments/e_means/smoke.yaml --data-root experimentation/dataset/e_means
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/smoke.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/smoke --phase cpu --dry-run
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/smoke.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/smoke --phase cpu --resume
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/smoke.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/smoke --phase gpu --devices 0 --resume
  uv run --extra gpu python scripts/e_means/analyze.py --runs-root experimentation/outputs/e_means/smoke --output-root experimentation/outputs/e_means/smoke-analysis --require-complete
  ```

- [x] `make test` で実際の収集件数と全PASSを確認する。GPU専用は上の明示コマンドで別に実行する。既存dirty変更へ作用しない対象範囲で `make fix project_module/src/project_module/e_means project_module/tests/e_means scripts/e_means` を実行し、`make all-lint` がformat→type-check→lint順に全成功することを確認する。既存の失敗は分離して報告する。
- [x] B1–B3を自分で仕上げ、差分/テスト/数値oracle/smoke artifactを束ねてOpus xhigh＋GPT-6.1 Sol xhighへ最終レビュー依頼する。判断・修正も両者と協議し、双方の承認後にB4へ進む。

実施記録: B1–B3の最終同一source検証は通常181 PASS / GPU10件除外、明示V100 suite10 PASS（skipなし）、全体format 81 files / mypy 29 source / lint成功。Opus xhighとGPT-6.1 Sol xhighの最終修正ゲートは承認済み。runner初期3件は未実装REDを記録したが、後から追加した一部は事後の回帰テストであり、全件test-firstを実施したとは主張しない。詳細は[実装ゲート記録](../reviews/2026-11-18-e-means-implementation.md)を参照。

### B4: pilot・凍結・本評価・報告

**Files:** Modify `configs/experiments/e_means/{pilot,core,stress}.yaml`（pilotに基づく凍結時のみ）。Create `docs/report/` の日本語報告・共有図・小さい集計表と再現手順。生データ・run artifactはGit管理外に置く。

**Interfaces:** B2/B3の3 CLIとartifact schemaを消費する。設定・条件表・判断基準は実験仕様が正本。

- [x] `pilot.yaml`の別seed/subsetで以下を実行し、V100 native FP32、全8 GPU sentinel、単独対同時干渉、CPU/GPU分離、memory、タイミング区分を検証する。5%以上の干渉時は正式時間を単独GPUで再計時する。

  ```bash
  uv run --extra gpu python scripts/e_means/prepare_data.py --config configs/experiments/e_means/pilot.yaml --data-root experimentation/dataset/e_means
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/pilot.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/pilot --phase cpu --dry-run
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/pilot.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/pilot --phase cpu --resume
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/pilot.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/pilot --phase gpu --devices 0,1,2,3,4,5,6,7 --resume
  uv run --extra gpu python scripts/e_means/analyze.py --runs-root experimentation/outputs/e_means/pilot --output-root experimentation/outputs/e_means/pilot-analysis --require-complete
  ```

- [x] parallel sentinelは、凍結した8 UUID/group/timeoutと転送同期後のadapter hookで待ち合わせ、実学習時刻の共通overlapを監査する。groupの欠損・abort・再使用を成功へ変換せず、新groupで全参加者を再資格確認する。sentinel pipelineに含まれる待機時間は別記し、5%判定にはwarm trainだけを使う。
- [x] 主testを見る前に条件・予算・seed・resolved params・計時cohort・再現許容・適合性・変更理由・設定hashをまとめて凍結する。pilotのtest評価が拒否されていることを確認する。Opus xhigh＋GPT-6.2 Sol xhighの双方がprotocolを承認してから本評価へ進む。異常/非適合は台帳へ残し、成功扱いしない。
- [x] 以下を各設定について順に実行する。`core`は実データ10セル、`stress`は合成12セルを担う。干渉判定の結果、正式GPU計時は `--devices 0` で実行する。根拠と凍結値は[本評価プロトコル](../report/protocol/README.md)を参照。両方の全適合method・budget・10 seedについて台帳が埋まり、failed/infeasible/unsupportedとunreachedを保存する。操作上の再開は同じコマンドで行う。

  ```bash
  uv run --extra gpu python scripts/e_means/prepare_data.py --config configs/experiments/e_means/core.yaml --data-root experimentation/dataset/e_means
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/core.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/core --phase cpu --dry-run
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/core.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/core --phase cpu --resume
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/core.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/core --phase gpu --devices 0 --resume
  uv run --extra gpu python scripts/e_means/analyze.py --runs-root experimentation/outputs/e_means/core --output-root experimentation/outputs/e_means/core-analysis --require-complete
  uv run --extra gpu python scripts/e_means/prepare_data.py --config configs/experiments/e_means/stress.yaml --data-root experimentation/dataset/e_means
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/stress.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/stress --phase cpu --dry-run
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/stress.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/stress --phase cpu --resume
  uv run --extra gpu python scripts/e_means/run.py --config configs/experiments/e_means/stress.yaml --data-root experimentation/dataset/e_means --output-root experimentation/outputs/e_means/stress --phase gpu --devices 0 --resume
  uv run --extra gpu python scripts/e_means/analyze.py --runs-root experimentation/outputs/e_means/stress --output-root experimentation/outputs/e_means/stress-analysis --require-complete
  ```

- [x] 別seed sentinel再現、全セルの台帳・有限値・split/初期中心/設定/版hash・図と数表を照合する。対応bootstrap CI、観測点の到達時間、warm/E2E、失敗・未到達と限界を日本語報告にまとめる。GPU非適合または必須手法全体の欠落は部分完了と明記する。
- [x] protocol、metrics、再現性、科学的結論と修正案をまとめ、Opus xhighおよびGPT-6.2 Sol xhighの双方で最終確認し、指摘を解消してから成果を引き渡す。

[検証結果と再現手順](../report/README.md) · [本評価前レビュー](../reviews/2026-11-17-e-means-pre-test-review.md)
