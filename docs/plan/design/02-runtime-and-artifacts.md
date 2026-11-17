# 実行基盤・公開契約・成果物

## 環境

既存のGPU付きDockerコンテナ内で、uv管理の実験環境をPython `>=3.14,<3.15`へ揃える。ホストでの実行や新しいDockerの起動を自動的な代替にしない。当ツールruntimeはDocker CLIがなく、想定dev環境との関係はread-onlyで確認してから環境を変更する。テンプレートの初期化監査は前提にしない。

実行環境はCPython 3.14.9とuv lockで確認した。依存はfaiss-cpu 1.16.0同梱e-means、numpy 2.5.4、scikit-learn 1.9.1、h5py 3.16.0等で固定し、Torch 2.14.1+cu126はoptional `gpu` extraに分離している。対象cp314 wheelはV100/SM70上のnative FP32 GEMMと小規模e-means演算で確認した。これはbackend適合性の確認であり性能benchmark結果ではない。[artifact調査](../../research/2026-11-17-e-means-design/astra-feasibility-and-datasets.md) 最新無指定のtorchはV100対応を失う可能性がある。Faiss GPU、CuTe、Triton kernelは必須とせず、torch依存として入るTritonと実際に使うkernelも区別する。

GPU主比較では、両Torch手法に共通の`native-fp32-deterministic-v1` profileを設定する方針とする。`CUBLAS_WORKSPACE_CONFIG=:4096:8`をworker起動時・Torch/CUDA/cuBLAS初回使用前に設定し、warm-up前に`torch.use_deterministic_algorithms(True, warn_only=False)`を有効化する。CUDA graph policyは`upstream_auto`とし、上流sourceやnative FP32を変更しない。PyTorch flagは特定の演算のdeterministic実装選択・未対応演算の例外化を制御するもので、application全体の再現性を単独で保証しない。[GPU再現性profile記録](../../research/2026-11-18-gpu-reproducibility.md) に判断根拠と資格状況をまとめる。

`exclude-newer="1 week"`を維持する。Faiss wheelは2026-11-10 16:40:58.511478 UTC公開なので、通常のcooldownを満たすのは2026-11-18 01:40:58 JSTより後。ユーザーの追加承認により、固定版faiss-cpu 1.16.0には`exclude-newer-package`の0日例外を設定した。他の研究用依存へこの例外を広げない。上流mainのCPU最適化は別の版差評価として正式buildした別環境に分け、wheelのPython部を上書きしない。

torch専用indexのupload-timeも確認済みである。今回の[公式cu126 index](https://download.pytorch.org/whl/cu126/torch/)には対象cp314 wheelの `data-upload-time="2026-09-30T20:04:15Z"` がある。将来uvが拒否するwheelにupload-time欠落があれば、無断例外を加えず該当環境枝を止めて方針を確認する。

root/moduleのPython制約、CPU/GPU Docker定義、ruff/mypy/CI、uv.lockはB1で整合した。既存の依存制約は必要性と該当依存を確認し、セキュリティ制約を一律削除しない。修正batchの統合検証状況は[実装進捗](../implementation-progress.md)で管理する。

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
| `ResourceConfig` | `gpu_deterministic_algorithms: bool = False`を要求値として持つ。他のCPU/GPU thread・affinity・GPU pool・memory設定とともにfrozen configへ保存する。現在の4実験configはprofile利用を明示してtrueを指定する |
| `TimingBreakdown` | cold_seconds（任意）、transfer_seconds、train_seconds、assign_seconds、evaluate_seconds、save_seconds、e2e_seconds: 非負float。全時間はscope metadataとともに保存し、新pipelineでは`e2e_scope="resident-input-pipeline-v1"`を必須にする。旧scopeまたはscopeなしのartifactは新pipeline集計と互換にしない |
| `TrainingResult` | centroids: `np.ndarray`、timing: `TimingBreakdown`、stats: `Mapping[str, JSONValue]`。JSONValueはJSONの再帰的値型 |
| `Evaluation` | mse: float、n_points: int、empty_clusters: int、cluster_sizes: 1次元int64配列、group_mse: mapping（任意）。件数0のgroupだけMSE=None/empty_groupとcountを保存。主MSE非有限値を成功へ変換しない |
| `ExperimentConfig` | frozen/extra=forbid。schema_version=1、plan_revision、phase(pilot/main)、conditions、methods、seeds、measurement、resources。下記の境界をB1で凍結 |
| `DatasetManifest` | schema_version=1、dataset_id/condition_id/data_seed、source revision/hash/size、processed data/group hash、split hash/seed/algorithm、shape/count、normalized、generator_params、license/attribution。配列と整合検査 |
| `RunRecord` | schema_version=1、run_key、status、attempt、reason_code、開始/終了時刻、artifact paths。statusはpending/running/success/failed/infeasible/unsupported/interrupted |
| `RunMetrics` | schema_version=1、run_key、config/manifest/init hash、split別Evaluation、TimingBreakdown、stats。pilotのtest評価はnullでreason=not_evaluated。JSON round-trip時もNaN/Inf拒否 |
| data | `load_dataset(dataset_id: str, root: Path, *, condition_id: str, data_seed: int) -> PreparedDataset` |
| random streams | `derive_seed(seed: int, stream_name: str) -> int`。UTF-8の `emeans-v1:<seed>:<stream_name>` のSHA256先頭4byteをbig-endian intへ変換し `& 0x7fffffff` で0～2^31-1へ制限。Faissのsigned seedへ負数を渡さない |
| initialization | `make_initial_centers(x: np.ndarray, k: int, seed: int) -> np.ndarray` |
| 各adapter | `train(x: np.ndarray, init: np.ndarray, spec: RunSpec, *, before_train: Callable[[], None] | None = None) -> TrainingResult`。optional hookは後方互換のため既定値Noneとし、RunSpecやparamsへserializeしない |
| evaluation | `evaluate_centers(x: np.ndarray, centroids: np.ndarray, *, chunk_rows: int) -> Evaluation` |
| group evaluation | `evaluate_groups(x: np.ndarray, centroids: np.ndarray, group_ids: np.ndarray, *, chunk_rows: int) -> Mapping[str, JSONValue]`。各groupのcount/mse/reasonを返す |
| runner | `run_trial(spec: RunSpec, dataset: PreparedDataset, output_root: Path) -> Path` |
| analysis | `summarize_runs(run_dirs: Sequence[Path], out_dir: Path) -> Path` |

GPU tensorはadapter内に閉じ、保存中心は共通NumPy型へ戻す。GPU転送とCPU返却時間を別に取り、共通評価への変換を片方だけのtrain時間に加算しない。method固有のsampling、revival、実反復数、実batch行数、backend/dtypeはstatsへ出す。

主時間軸`train_seconds`は従来どおりで、transferとcentroid保存を含めない。副次時間`e2e_seconds`はresident-input pipelineを表す。検証済みのcommon inputと初期中心の用意後、runnerがadapterを呼ぶ直前から、返却centroidの保存・fsync完了直後までの連続wall時間であり、GPU transfer、training、CPUへのcentroid返却、保存を含む。input loading/hash/norm、初期中心生成、git identity、NVML、評価、台帳・metrics等の管理I/O、warm-upは除外する。input load/cache、初期化準備時間とwhole-trial wall時間は別metadataに残す。このscopeは`plan_revision="2026-11-17-timing-v2"`で導入した。再現性profileを含む計画版`2026-11-18-cuda-deterministic-v1`ではscopeを変えずに引き継ぎ、4つのexperiment configで現在のrevisionを記録する。

`assign_seconds`はtrain splitの一度の`evaluate_centers`を測る。exact final labelsとtrain MSEおよびcluster countsが融合した時間であり、純粋なassignment kernel時間ではない。train group、validation/testおよびその他の診断は`evaluate_seconds`へ記録し、assignへ二重計上しない。evaluatorの実thread数とbackendを保存する（現行計画はCPU 16、GPU 1）。centroidを保存・fsyncしてから評価する。

6 adapterとdispatcherはkeyword-only `before_train` hookを受け取る。parallel pilot sentinelでのみ、warm-up後、入力検証とGPU transfer / synchronizationが済んだ後、train timer直前に一度呼び出す。warm-up、通常main、main-isolatedではhookなしで待機もしない。各adapterは実際のtrain timer境界で`time.time_ns()`を読み、`train_started_at_ns`/`train_finished_at_ns`を記録する。duration自体は単調clockで測る。GPU endはCUDA synchronize後、device-to-host返却前である。

ExperimentConfigのconditionsはcondition_id/dataset_id/n_train/k/dimension/data_seed対応/generator_params、methodsはmethod/budgets/params、measurementは採用cohort/repeat表/evaluate_testと任意のsentinel設定、resourcesはCPU threads/affinity/GPU pool・memory境界・`gpu_deterministic_algorithms`を持つ。`gpu_deterministic_algorithms`の既定値はfalseであり、現在の4実験configはtrueを明記する。ネストした構造も未知field拒否とし、pilotのevaluate_test=trueは拒否する。B1でYAML→型→JSONのround-trip、全schema version、B2/C向けtiny artifact fixtureを一度に固定する。

`measurement.sentinel`は`group_id`、`participants`、`members`、`timeout_seconds`を持つ。`timeout_seconds`は既定300秒とする。`members`は重複しない、ソート済みの実GPU UUIDである。parallel sentinelはpilotに限り、参加者は凍結した8 UUIDに一致し、各参加CLIが1 GPUを使う。共有host memory reservation 48 GiBを保持する。既存のshared phase lockとper-GPU exclusive lockは維持する。各groupは一度だけ使う識別子で、再試行には新しいgroupと新しいoutput rootを用意する。通常runにはsentinel設定や待機を適用しない。

prepare CLIがcondition/data_seed別の配列・group・manifestを物化する。loaderは既存manifestと生成params/n_train/hashを照合して読み、隠れた乱数や別条件を補わない。group別MSEは上記補助関数へ既知IDを渡す。巨大な(n,k)行列は作らず距離chunkを使い、学習batchとは別に設定する。

CLIは `prepare_data.py --config PATH --data-root PATH`、`run.py --config PATH --data-root PATH --output-root PATH --phase {cpu,gpu} [--devices 0,1,...] [--resume] [--dry-run] [--retry-failed --retry-reason TEXT]`、`analyze.py --runs-root PATH --output-root PATH [--require-complete]` とする。dry-runは設定/manifest検証とjob台帳出力まで。analyzeは部分reportも生成でき、require-completeで予定run欠落/非終端/不正artifact、主datasetまたは必須手法全体の欠落があればexit2。全run終端かつscopeを満たせばexit0。個別のfailed/infeasibleも終端記録として残し、曲線の不完全点を隠さない。

## runnerと再開

run keyは計画版、config/profile hash、GPU reproducibility profile、dataset/condition/data_seed/split/initial-centers hash、method/upstream版、seed、budget、device条件、measurement_id、timing_repeatから作る。run_idはkeyの短縮表示で衝突時拒否。同一key/hashの成功runだけを再利用する。本計時はmain-parallel/main-isolated、sentinelはmodeとGPU UUIDを含むmeasurement_idで区別し、repeat0–2を別artifactにする。retry attemptは同じ科学的runの履歴で、計時反復とは別。analysisの採用cohortは凍結configから読み、勝つattempt/cohortを選ばない。stock rootに別profileのrunをresumeしてはならない。

途中ファイルはatomic renameし、成功markerは必須artifactと有限値検査後に最後に作る。`--resume`はpending/interruptedのみ再実行する。通常のtrial OOMはinfeasible、backend/中心注入不可はunsupported、数値不正やworker crash等はfailedとして終端にする。Frozen identity、config/run key、入力manifest/hash、既存attempt/artifact、再現性profile requested/actualの不一致は`FrozenIdentityError`、sentinel協調の不成立は`SentinelCoordinationError`として科学的なtrial failureと分け、いずれもphase abortとする。workerは専用abort responseを親へ返し、親は新しいfailed attemptを作らずpending jobの配布を停止し、非0で終了する。既存attemptがあればinterruptedとして保全する。worker側InputCacheの再検証不一致も同じabort経路へ通す。汎用数値/algorithm exceptionや通常worker crashは従来のfailed処理を保つ。通常trialのworker crash後は、sentinel cohort外に限り親がworkerを再生成して他pending jobを続ける。failedの再試行は明示flagと理由を要求し、全attemptを残す。秘密値をログへ出さない。

GPU workerはspawnで作り、torch import前にCUDA_VISIBLE_DEVICESを設定する。1worker1GPU・1trial、親がjob一覧を配る。adapterのoptional importはこの境界で遅延してよい。GPU workerのRSS見積合計が384GiBを超える同時起動は避け、残りをOS/共有cache用に確保する。1GPUのinitial working limitは24GiB。これは時間予算と違うOOM防止の資源境界であり、適合しなければchunkを見直して設定版を更新する。

CUDA peakのrun帰属のため、warm-up終了後かつmeasured H2D transfer前にsynchronize、`empty_cache()`、`reset_peak_memory_stats()`を実行する。runごとの`max_memory_allocated`と`max_memory_reserved`を保存する。NVML値はcontextや外部processを含む補助観測で、run固有allocator peakと区別する。共有worker RSSの合計が運用上限を超えた場合、個別runをinfeasibleにせずphase全体をabortし、in-flight attemptをinterruptedとしてpending jobの配布を止める。個別runに帰属できるGPU OOMはinfeasibleのままとする。

CPU正式計時phaseとGPU正式計時phaseを分ける。読み取り・評価・描画は計時を妨げない範囲で並列化する。共通inputはread-only memmapで共有し、大量配列をprocess queueで複製しない。artifact pathはrun配下へ制限する。

parallel sentinel barrierは`data_root`と同じlocal filesystem上のgroup専用directoryで協調する。各UUIDのclaimはexclusive-createで永続化し、全員の到着、共通trial/input identity、owner lock、abort markerなしを確認してから共通の将来release時刻を発行する。各参加者はそのmonotonic時刻まで待って同期し、busy-spinしない。groupはsingle-use/fail-closedとし、stale claim/arrivalを再利用しない。sentinelの資格には、予定した8 UUIDすべての成功、同一input/init hash、非abort groupを要求する。8本の`train_started_at_ns`/`train_finished_at_ns`から`min(finish)-max(start)>0`を確認し、common overlap、start skew、overlap ratioを保存する。窓がほとんど重ならない結果から干渉なしと結論しない。barrier timeout、worker終了、hash不一致、stale group、共有host memory超過では全cohortをabortする。既存in-flight attemptはinterrupted、残りのjobにはterminal recordを作らない。再測定は全参加者で新しいgroup/output rootを使う。

Opus reviewで指摘された親CLIのCPU affinityはhardware fingerprintに含まれる。再開を含む各phaseでtaskset/numactl等の有無とaffinityを揃える。実行は専用worktreeの凍結commitから始め、phase中は同worktreeのcommit・ソース/config編集・dependency sync・再prepareを行わない。同期が必要な`uv run`やMake targetは、事前にenvironmentを凍結した後`UV_NO_SYNC=1`または`uv run --no-sync --extra gpu`を使う。

## artifactと秘密情報

1runはspec.json、environment.json、dataset-manifest参照、centroids.npy、metrics.json、status.json、stdout/stderr（必要分のみ）を持つ。environmentにはrepo/upstream SHA、dirty diff hash、wheel/lock/image識別子、Python/CUDA/cuBLAS/driver/GPU UUID/SM、CPU affinity/thread、要求/実backend/dtypeを残す。再現性profileについてはprofile名・hash・plan revisionとrequested/actualの値を両方保存する。具体的には`gpu_deterministic_algorithms`、`torch.are_deterministic_algorithms_enabled()`、warn-only状態、要求/実`CUBLAS_WORKSPACE_CONFIG`、`fill_uninitialized_memory`、FP32 matmul設定、CUDA graph policyを記録する。formal adapterはinstance限定の観測wrapperでupstream/effective eligibilityを記録し、upstreamの戻り値を変えず、`finally`で一時属性を削除する。`stats.graph_policy="upstream_auto"`と`stats.graph_eligibility`（同じ観測値を示す`upstream_eligible`/`effective_eligible`）を保存し、formal runの`graph_actual_replays`は未計測を示す`null`とする。今回の95 replayは診断での観測値としてのみ報告する。timing metadataには`e2e_scope="resident-input-pipeline-v1"`、`data_load_seconds`とcache state、初期化準備時間、`whole_trial_wall_seconds`、evaluator backend/thread数、train-windowのns境界、sentinelの`coordination_wait_seconds`と共通overlap auditを保存する。旧またはscopeなしartifactを新pipelineの図表へ混ぜない。env全体やtokenの値を出力しない。

HF_TOKENは既存root envから取得する。主datasetはpublic/non-gatedで、追加のAPI認証は不要。運用変数はHF_HOME（repo側cache）、CUDA_VISIBLE_DEVICES（workerが設定）、OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS（CPU/GPU phase設定）を使い、ユーザーへ新しい秘密値の入力を要求しない。

data本体・centroid・run logはGit管理外。commitするのはconfig、data license/revision/hash/splitの小さいmanifest、集計表、共有用図、日本語report、再現手順。run artifactを再現用に保存し、Gitへ全量を投入しない。

## 検証と引き渡し

tiny FP64 oracleで既知MSE、distance/label順序、空cluster、重複、非有限値、保存再読込を確認する。Torch Lloydは同初期値の1反復とCPU参照、Lloyd MSE単調性、空中心維持、N反復prefix整合を確認する。通常のGPU数値再現許容はatol=1e-6/rtol=1e-4のまま維持する。新profileの8 GPU資格判定では、S0/大形状について各GPUのfresh process 3反復でe-means centroid SHA一致を追加条件とし、Lloydには従来の数値許容を用いる。正式runner fresh-process確認は完了したが、準備済み全22条件coverageのGPU実行、8 GPU再資格、cohort干渉判定は未完了であり、詳細は[GPU再現性profile記録](../../research/2026-11-18-gpu-reproducibility.md)に記す。

CPU testがtorch/GPUを必須にせず実行でき、GPU marker付きintegrationがV100 native演算を実行することを受入条件とする。runnerには中断/再開、設定hash相違、欠損artifact、1worker失敗、秘密値非出力のテストを置く。統計は手計算できるpaired fixtureで検証する。意味のあるtestとformat→type→lintを束ねて確認する。

実装担当へ渡す順序、並列所有範囲、モデル選択、レビューgate、具体的な将来コマンドは [実装計画](../2026-11-17-e-means-implementation-plan.md) を正本とする。
