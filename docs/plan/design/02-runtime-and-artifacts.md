# 実行基盤・公開契約・成果物

## 環境

既存のGPU付きDockerコンテナ内で、uv管理の実験環境をPython `>=3.14,<3.15`へ揃える。ホストでの実行や新しいDockerの起動を自動的な代替にしない。当ツールruntimeはDocker CLIがなく、想定dev環境との関係はread-onlyで確認してから環境を変更する。テンプレートの初期化監査は前提にしない。

候補はfaiss-cpu1.16.0同梱e-means、torch2.14.1の公式cu126 cp314 wheel、numpy2.5.4、sklearn1.9.1、h5py3.16.0。具体的lockは実装時にresolveして実artifactを確認する。[artifact調査](../../research/2026-11-17-e-means-design/astra-feasibility-and-datasets.md) 最新無指定のtorchはV100対応を失う可能性がある。Faiss GPU、CuTe、Triton kernelは必須とせず、torch依存として入るTritonと実際に使うkernelも区別する。

`exclude-newer="1 week"`を維持する。Faiss wheelは2026-11-10 16:40:58.511478 UTC公開なので、通常のcooldownを満たすのは2026-11-18 01:40:58 JSTより後。release日やソースSHAで迂回しない。上流mainのCPU最適化は別の版差評価として正式buildした別環境に分け、wheelのPython部を上書きしない。

torch専用indexのupload-timeもB1で検査する。今回の[公式cu126 index](https://download.pytorch.org/whl/cu126/torch/)には対象cp314 wheelの `data-upload-time="2026-09-30T20:04:15Z"` を確認した。upload-time欠落でuvが拒否する場合に無断例外を入れず、その環境枝を止めて方針を共有する。

root/moduleのPython制約、CPU/GPU Docker定義、ruff/mypy/CI、uv.lockを実装の環境タスクで一括整合する。既存の依存制約は必要性と該当依存を調べて整理し、セキュリティ制約を一律削除しない。今回これらの設定・依存は変更しない。

## 構成

参考の[State-Aware RAG](https://github.com/tomo0530/state-aware-rag-test)のexperiment分離と、[Dreaming](https://github.com/tomo0530/dreaming-test)の設定・runner・報告の小さい構成を借りる。8分冊のサービス設計は写さない。以下は今後作成する予定の配置。

```text
configs/experiments/e_means/       # smoke / pilot / core / stress YAML
project_module/src/project_module/e_means/
  contracts.py                   # 型、設定validation、artifact schema
  data.py                        # 固定artifact取得、split、manifest
  initialization.py              # 共通初期中心
  methods/                       # 手法ごとの薄いadapter
  evaluation.py                  # 共通exact MSEと数値oracle
  runner.py                      # run状態、worker割当、再開
  analysis.py                    # 集計、曲線、CI、report入力
project_module/tests/e_means/     # tiny unit / optional GPU integration
scripts/e_means/                  # prepare_data / run / analyze の薄いCLI
experimentation/dataset/e_means/  # Git管理外のraw/processed
experimentation/outputs/e_means/  # Git管理外のrun artifact
experimentation/notebooks/       # 必要なときだけ結果の閲覧
docs/report/                     # 日本語の検証報告、共有用の図
```

外部Faissの処理をadapterから呼ぶ。e-means本体を再実装しない。Torch Lloydだけは数値検証可能な対照loopを作り、上流native assignmentを共用する。上流helperが使えなければ別kernelへ黙って替えず機構比較を停止する。汎用registry/DB/サービスは作らず明示したmethod mappingで足りる。

## 公開契約

配列はC-contiguousな2次元float32、有限値、dimension/k/件数を境界で検証する。以下のnp.ndarrayはruntime表記であり、Python注釈は `npt.NDArray[np.float32]` / `npt.NDArray[np.int64]` とする。型の詳細はcontracts.pyへまとめ、runnerとadapterで別のschemaを作らない。研究用の関数境界を採用し、production DS-to-ENG用BaseInvokableを全関数へ被せない。[配下の規則](../../../project_module/AGENTS.md)

| 契約 | フィールド/関数 |
| --- | --- |
| `PreparedDataset` | train/validation/test: float32配列、各 `{train,validation,test}_groups`: int64配列またはNone、n_train: int、split_hash/data_hash: str、manifest_path: Path。groupは点と行対応 |
| `RunSpec` | frozen Pydantic model。run_id,dataset_id,condition_id,method,device,measurement_idはstr、n_train/k/seed/data_seed/threadsはint、timing_repeatは非負int、budgetは正のfloat、paramsはJSON mapping。未知field拒否、method別budget/paramsを検査。deviceはcpu/cuda |
| `TimingBreakdown` | cold_seconds（任意）、transfer_seconds、train_seconds、assign_seconds、evaluate_seconds、save_seconds、e2e_seconds: 非負float。clock種別と計時範囲をmetadataへ保存 |
| `TrainingResult` | centroids: `np.ndarray`、timing: `TimingBreakdown`、stats: `Mapping[str, JSONValue]`。JSONValueはJSONの再帰的値型 |
| `Evaluation` | mse: float、n_points: int、empty_clusters: int、cluster_sizes: 1次元int64配列、group_mse: mapping（任意）。件数0のgroupだけMSE=None/empty_groupとcountを保存。主MSE非有限値を成功へ変換しない |
| `ExperimentConfig` | frozen/extra=forbid。schema_version=1、plan_revision、phase(pilot/main)、conditions、methods、seeds、measurement、resources。下記の境界をB1で凍結 |
| `DatasetManifest` | schema_version=1、dataset_id/condition_id/data_seed、source revision/hash/size、processed data/group hash、split hash/seed/algorithm、shape/count、normalized、generator_params、license/attribution。配列と整合検査 |
| `RunRecord` | schema_version=1、run_key、status、attempt、reason_code、開始/終了時刻、artifact paths。statusはpending/running/success/failed/infeasible/unsupported/interrupted |
| `RunMetrics` | schema_version=1、run_key、config/manifest/init hash、split別Evaluation、TimingBreakdown、stats。pilotのtest評価はnullでreason=not_evaluated。JSON round-trip時もNaN/Inf拒否 |
| data | `load_dataset(dataset_id: str, root: Path, *, condition_id: str, data_seed: int) -> PreparedDataset` |
| random streams | `derive_seed(seed: int, stream_name: str) -> int`。UTF-8の `emeans-v1:<seed>:<stream_name>` のSHA256先頭4byteをbig-endian intへ変換し `& 0x7fffffff` で0～2^31-1へ制限。Faissのsigned seedへ負数を渡さない |
| initialization | `make_initial_centers(x: np.ndarray, k: int, seed: int) -> np.ndarray` |
| 各adapter | `train(x: np.ndarray, init: np.ndarray, spec: RunSpec) -> TrainingResult` |
| evaluation | `evaluate_centers(x: np.ndarray, centroids: np.ndarray, *, chunk_rows: int) -> Evaluation` |
| group evaluation | `evaluate_groups(x: np.ndarray, centroids: np.ndarray, group_ids: np.ndarray, *, chunk_rows: int) -> Mapping[str, JSONValue]`。各groupのcount/mse/reasonを返す |
| runner | `run_trial(spec: RunSpec, dataset: PreparedDataset, output_root: Path) -> Path` |
| analysis | `summarize_runs(run_dirs: Sequence[Path], out_dir: Path) -> Path` |

GPU tensorはadapter内に閉じ、保存中心は共通NumPy型へ戻す。GPU転送とCPU返却時間を別に取り、共通評価への変換を片方だけのtrain時間に加算しない。method固有のsampling、revival、実反復数、実batch行数、backend/dtypeはstatsへ出す。

ExperimentConfigのconditionsはcondition_id/dataset_id/n_train/k/dimension/data_seed対応/generator_params、methodsはmethod/budgets/params、measurementは採用cohort/repeat表/evaluate_test、resourcesはCPU threads/affinity/GPU poolとmemory境界を持つ。ネストした構造も未知field拒否とし、pilotのevaluate_test=trueは拒否する。B1でYAML→型→JSONのround-trip、全schema version、B2/C向けtiny artifact fixtureを一度に固定する。

prepare CLIがcondition/data_seed別の配列・group・manifestを物化する。loaderは既存manifestと生成params/n_train/hashを照合して読み、隠れた乱数や別条件を補わない。group別MSEは上記補助関数へ既知IDを渡す。巨大な(n,k)行列は作らず距離chunkを使い、学習batchとは別に設定する。

CLIは `prepare_data.py --config PATH --data-root PATH`、`run.py --config PATH --data-root PATH --output-root PATH --phase {cpu,gpu} [--devices 0,1,...] [--resume] [--dry-run] [--retry-failed --retry-reason TEXT]`、`analyze.py --runs-root PATH --output-root PATH [--require-complete]` とする。dry-runは設定/manifest検証とjob台帳出力まで。analyzeは部分reportも生成でき、require-completeで予定run欠落/非終端/不正artifact、主datasetまたは必須手法全体の欠落があればexit2。全run終端かつscopeを満たせばexit0。個別のfailed/infeasibleも終端記録として残し、曲線の不完全点を隠さない。

## runnerと再開

run keyは計画版、config hash、dataset/condition/data_seed/split/initial-centers hash、method/upstream版、seed、budget、device条件、measurement_id、timing_repeatから作る。run_idはkeyの短縮表示で衝突時拒否。同一key/hashの成功runだけを再利用する。本計時はmain-parallel/main-isolated、sentinelはmodeとGPU UUIDを含むmeasurement_idで区別し、repeat0–2を別artifactにする。retry attemptは同じ科学的runの履歴で、計時反復とは別。analysisの採用cohortは凍結configから読み、勝つattempt/cohortを選ばない。

途中ファイルはatomic renameし、成功markerは必須artifactと有限値検査後に最後に作る。`--resume`はpending/interruptedのみ再実行する。OOM/凍結memory境界はinfeasible、backend/中心注入不可はunsupported、数値不正やworker crash等はfailedとして終端にする。worker crash後は親がworkerを再生成して他pending jobを続ける。failedの再試行は明示flagと理由を要求し、全attemptを残す。初回失敗は主集計で成功へ置換せず、retry成功は感度表へ分離する。秘密値をログへ出さない。

GPU workerはspawnで作り、torch import前にCUDA_VISIBLE_DEVICESを設定する。1worker1GPU・1trial、親がjob一覧を配る。adapterのoptional importはこの境界で遅延してよい。GPU workerのRSS見積合計が384GiBを超える同時起動は避け、残りをOS/共有cache用に確保する。1GPUのinitial working limitは24GiB。これは時間予算と違うOOM防止の資源境界であり、適合しなければchunkを見直して設定版を更新する。

CPU正式計時phaseとGPU正式計時phaseを分ける。読み取り・評価・描画は計時を妨げない範囲で並列化する。共通inputはread-only memmapで共有し、大量配列をprocess queueで複製しない。artifact pathはrun配下へ制限する。

## artifactと秘密情報

1runはspec.json、environment.json、dataset-manifest参照、centroids.npy、metrics.json、status.json、stdout/stderr（必要分のみ）を持つ。environmentにはrepo/upstream SHA、dirty diff hash、wheel/lock/image識別子、Python/CUDA/driver/GPU UUID、CPU affinity/thread、要求/実backend/dtypeを残す。env全体やtokenの値を出力しない。

HF_TOKENは既存root envから取得する。主datasetはpublic/non-gatedで、追加のAPI認証は不要。運用変数はHF_HOME（repo側cache）、CUDA_VISIBLE_DEVICES（workerが設定）、OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS（CPU/GPU phase設定）を使い、ユーザーへ新しい秘密値の入力を要求しない。

data本体・centroid・run logはGit管理外。commitするのはconfig、data license/revision/hash/splitの小さいmanifest、集計表、共有用図、日本語report、再現手順。run artifactを再現用に保存し、Gitへ全量を投入しない。

## 検証と引き渡し

tiny FP64 oracleで既知MSE、distance/label順序、空cluster、重複、非有限値、保存再読込を確認する。Torch Lloydは同初期値の1反復とCPU参照、Lloyd MSE単調性、空中心維持、N反復prefix整合を確認する。GPU nondeterminismはbit一致を要求せずtinyではrtol=1e-4/atol=1e-6を初期数値許容とし、超過は解析して黙って緩めない。

CPU testがtorch/GPUを必須にせず実行でき、GPU marker付きintegrationがV100 native演算を実行することを受入条件とする。runnerには中断/再開、設定hash相違、欠損artifact、1worker失敗、秘密値非出力のテストを置く。統計は手計算できるpaired fixtureで検証する。意味のあるtestとformat→type→lintを束ねて確認する。

実装担当へ渡す順序、並列所有範囲、モデル選択、レビューgate、具体的な将来コマンドは [実装計画](../2026-11-17-e-means-implementation-plan.md) を正本とする。
