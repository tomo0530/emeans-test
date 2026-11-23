# 実験条件と評価仕様

上位方針は [00-overview](00-overview.md)。以下は実行前に設定へ落とす設計値であり、測定値ではない。

## 手法と公平性

| ID | 実装と位置づけ | 予算点 |
| --- | --- | --- |
| `faiss_lloyd` | faiss-cpuの全件Lloyd、CPU実用基準 | iteration = 1,2,5,10,25,50,100 |
| `numpy_e_means` | 同じFaiss配布物のNumpyEmeans | nepochs = 1,2,3,5,8,12 |
| `mini_batch` | sklearn MiniBatchKMeans、CPUの既存近似更新 | 全件相当pass = 0.25,0.5,1,2,5,10,25 |
| `elkan` | sklearn KMeans(algorithm=elkan)、exact加速対照 | iteration = 1,2,5,10,25,50,100 |
| `torch_e_means` | QuantizedEmeans、quantizer=None、native FP32、V100 | nepochs = 1,2,3,5,8,12 |
| `torch_lloyd` | 共通native assignment＋全件中心更新の薄いGPU対照 | iteration = 1,2,5,10,25,50,100 |

各点は独立した完走run。e-meansのscheduleは総Tに依存するので、長runの途中を別Tの最終結果にしない。他手法も同じartifact契約に揃える。MiniBatchは `B=min(n,max(4096,k))`、必要batch数を `ceil(pass*n/B)` とし、端数と実処理行数を記録する。初回partial_fitにk行以上を渡す。専用乱数streamから復元抽出したB行と外部中心を使い、`partial_fit` を明示回数呼ぶ。tol/max_no_improvementによる隠れた停止と自動初期化を避ける。

全手法は同じFP32 train、k、共通初期中心行列、1 restartを使う。初期中心はseedごとにtrainの厳密重複を除いたvector候補からk個を一様非復元抽出しhashを保存する。候補がk未満なら入力不適合。同seedだけを共通初期値の代替にはしない。外部中心を注入できないadapterは統制比較に参加させずunsupportedと記録する。k-means++/推奨既定値panelは凍結後の追加実験版だけで行い、主22セルの完了条件には含めない。

予算・点数上限・外部中心・明示した停止条件・device/dtype以外は固定上流版の既定値を使い、全resolved paramsを保存する。NumPy/Torch e-meansで共有するbatch/schedule/revivalのアルゴリズム設定は同値にし、版/経路固有の差は別に記録する。MiniBatchのreassignment_ratio等も記録し、主結果を見て調整しない。設定を変える判断はpilot後の凍結時にまとめる。

## GPU再現性profile

GPUの`torch_e_means`と`torch_lloyd`は、共通の`native-fp32-deterministic-v1` profileを使う方針とする。`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True, warn_only=False)`、CUDA graph policy=`upstream_auto`を適用する。選択理由は既定の同seed再現性要件を満たすためで、観測MSEの値や速度ではない。上流source、FP32精度、batch/schedule/revival、seed、予算点、split、許容値は変えない。

このprofileはPyTorch既定の実行条件そのものではない。GPU主図は「再現性を要求したV100/native FP32 Torch profile」での品質–時間比較として示し、stock defaultの速度・品質へ一般化しない。stockのS0再現許容違反と診断結果は独立findingとして残し、profile/cohort/CIをまたいで混ぜない。診断根拠・限界・資格確認の結果は[GPU再現性profile記録](../../research/2026-11-18-gpu-reproducibility.md)を参照する。

Faiss/e-meansの点数上限を全件条件へ明示設定し、実処理件数を確認する。MiniBatchのbatch抽出は手法自体なので残し、累積sample数と重複を含む処理件数を記録する。Lloyd/Elkanの収束停止は可能な範囲で無効化し、同一割当による早期収束は実反復数を記録する。通常L2を主比較にし、入力がunit normでも中心をunit normに制約しない。spherical k-meansは別の目的である。

## 配布データと取得条件

主HF revisionは `07b387891a221b7b073b83d2f752b76462e5fa03`。以下のsize/hashはHF metadataの公称値であり、取得後に全体SHA256を照合する。[配布元](https://huggingface.co/datasets/vector-index-bench/vibe)、[配布形式](https://arxiv.org/html/2505.17810v3#A1.SS8)

| ID | artifact / corpus | 公称bytes / SHA256 |
| --- | --- | --- |
| `arxiv_nomic` | arxiv-nomic-768-normalized.hdf5 / n=1,344,643,d=768 | 4,135,431,488 / `8be0993b978b0d0ef023d21d878251a5ed09e058adb25994553c08388d37d414` |
| `landmark_nomic` | landmark-nomic-768-normalized.hdf5 / n=760,757,d=768 | 2,341,733,696 / `9e870c21f3a33cea1adb9055ce0e7af0988557b5e4e21485a661f40365be286a` |

VIBE配布embeddingはCC BY4.0、コードはMIT。arXivのdescriptive metadataはabstractを含みCC0である。[arXiv公式条件と脚注](https://info.arxiv.org/help/api/tou.html) Landmark生成はGoogle Landmarksのindex集合を使い、その原画像は公式にCC0/PDと記載される。[公式データ説明](https://github.com/cvdfoundation/google-landmark#index-image-licenses) 生成コードのNomic text/vision v1.5はApache-2.0。[VIBE生成コード](https://github.com/vector-index-bench/vibe/blob/1e7f77666aaf1754d14b8d1072aa8d8be1bbedf9/vibe/datasets.py)

配布ベクトルだけをクラスタ/索引用中心学習のbenchmarkに利用し、元文書・原画像やencoderは取得しない。作者・元データ・モデル・配布元・変更（subset/split）・ライセンスURLをデータmanifestとTHIRD_PARTY_NOTICESへ保存する。取得時のcard/metadataが設計と相違すればそのデータ枝を止め、synthetic等は続ける。元データの許諾が曖昧なAGNewsは主対象にしない。

HDF5 `train` はcorpus、`test` は検索queryであり、MLのtrain/testとは違う。corpus内の厳密重複行をhashでgroup化し、groupを分離した固定seed42のsplitを作る。validation/test各10,000行を目標とし、groupにより行数がずれた実値を保存する。残りをcluster_train poolとし、seed42の固定順列の先頭n行をnested subsetにする。publisher query/GTは主MSE評価に使わない。意味的重複・同一論文系列の完全除去は配布vectorだけでは保証できない。

L2正規化済みFP32を保持し、PCA/白色化/再正規化を加えない。非有限値、想定外dimension、group数不足は黙って修正せずdata errorとする。正規化済み配布データのゼロnormは不適合とするが、一般のL2入力・合成・数値oracleの原点は有効な点として扱う。既存前処理を各手法でやり直さない。

## 条件表と反復

実データは両datasetで次の5セルを行う。`full`はholdout後の全train pool。kは固定値で、論文の特定表の直接再現とは呼ばない。

| 条件 | n | k | 狙い |
| --- | --- | --- | --- |
| R0 | 500,000 | 2,048 | 代表点 |
| Rn-small | 100,000 | 2,048 | 件数差 |
| Rn-full | full | 2,048 | 最大corpus |
| Rk-small | 500,000 | 256 | 小さいk |
| Rk-large | 500,000 | 8,192 | 大きいk |

合成は256成分のGaussian mixtureを基本とし、既知group別損失も評価する。各seedで成分中心 `mu_j~N(0,s^2 I_d)`、noise `N(0,I_d)`、基本s=3、成分は等重み。train/validation/testは同じ生成分布から独立に作る。validation/test各10k。成分数と学習kは別の変数であり、k変更を正解クラス数変更と混同しない。

| 条件 | 基本値からの変更 |
| --- | --- |
| S0 | n=100k,d=128,k=256 |
| Sn / Sd / Sk | それぞれn=1M / d=1024 / k=2048を単独変更 |
| Sratio | n=16,384,k=2,048（n/k=8の境界） |
| Soverlap | s=0.5 |
| Saniso | noiseの対角共分散を0.1～10の等間隔値、trace=dに正規化 |
| Srare | 32成分に合計1%、残り224成分に99%。rare中心のs=10、rare noise std=0.5 |
| Soutlier | 1%をN(0,20^2 I)へ置換、outlier groupを保存 |
| Sduplicate | 生成train候補の10%を同一成分内の既存点へ置換。holdoutは独立生成し厳密重複groupを再確認 |
| Sinteraction-1 / 2 | Sd+Srare / Sk+Soverlap |

主評価は独立seed 0–9の10組。実データはdata_seed=42の固定split、初期中心/手法乱数が組内対応。合成はdata_seed=組seedで、data生成と学習の名前付き乱数streamを分け、組内でdataと初期中心を共有する。10組のCIは合成ではdata生成と学習双方の変動を含む。乱数は共通derive_seedを通し、`data/centers`、`data/noise`、`init`、`method/<id>`を分離する。

pilotは学習seed100–102、専用condition_idで識別する。実データpilotはtrain poolから別の固定順列でsubsetを作り、主trainと重なり得ることを認める。pilotのrunner/analysisはtrainとvalidationだけを評価し、主testの評価・集計を拒否する。合成pilotはdata_seed100–102で独立生成する。sentinelにもpilot用condition_idを付ける。

pilotでは実行適合、メモリ、時間計測の分解を確認する。表の内容を変更する場合は主testを見る前に理由・新しい表・設定hashを凍結する。主結果を見てseedや有利な予算点を追加しない。未到達が多くても主表を残し、追加は探索版として分離する。

## 評価と曲線

主品質は**test MSE**、主x軸はwarm **train_seconds**。最終保存centroidへの独立したexact squared L2最小値の点平均を用い、次元数では割らない。主曲線と主評価は変更しない。副次図は、範囲を版管理した`e2e_seconds`を **Resident-input pipeline (s)** として示す。validationは設定選択用とする。主評価器はFP32の距離計算とFP64集計を使い、微小入力のFP64総当たりと照合する。実装のobj、学習中label、approx assignmentを最終MSEに使わない。大きなnormでの相殺誤差はtiny oracleとFP64 spot checkで検出する。

GPU手法の主曲線には上記の共通再現性profileを必ず表示する。stock診断は別profileの結果として別に報告し、主品質・時間・CIへpoolしない。deterministic profileの測定値からstock defaultの速度や一般的なGPU速度を推定しない。

`e2e_seconds`の新しい範囲は`e2e_scope="resident-input-pipeline-v1"`で識別する。検証済みでresidentな共通inputと初期中心が用意された後、実測adapter呼出しの直前から、返されたcentroidの保存とfsyncが終わるまでの連続wall時間とする。GPU transfer、学習、centroidのCPU返却、保存を含む。初回data hash・検証、input load/cache、初期中心生成、git identity、NVML、評価、run台帳や他の管理I/O、warm-up、最終metrics/status書込は含めない。input load/cacheとwhole-trial wall時間は別metadataとして保存する。旧範囲の`e2e_seconds`にこのscopeを付け替えず、旧artifactは新しいpipeline集計と互換にしない。

主runはtrain/validation/test MSE、成分別MSE（synthetic）、クラスタサイズ、空cluster、revival数、最大/平均サイズ比、非有限値・失敗を記録する。成分ごとの評価件数も保存し、Srare等で件数0ならgroup MSEはnullと理由empty_groupを残す。全体MSEの非有限値やrun失敗とは区別する。実データの意味的クラス精度をMSEだけから主張しない。

各セル・予算点では10seedの**MSE算術平均**と**時間中央値**を点推定にする。品質はrepeat0を使い、3回計時する点はseed内時間中央値を先に取り、次にseed間を集約する。repeat間のMSE差も数値再現検査に残し、最良repeatを選ばない。bootstrapは乱数1729、seed indexの復元抽出10,000回を同セルの全手法・全予算点・両軸へ共用し、percentile95%の点ごとCIを出す。反復計時や個々のdata点を独立seedに数えない。CIは曲線全体や選択した最良点の同時保証ではない。

10/10成功・有限の点のみ主曲線へ集約する。部分成功点はincomplete(n/10)を表示し、成功数・失敗heatmap・全seed生値へ接続する。欠落点を線で橋渡ししない。成功seedだけの平均で手法を順位付けしない。

固定の品質合格幅は置かない。相対曲線では同cell・同seedのFaiss Lloyd@100 test MSEを基準に `delta_s=100*(MSE_method,s/MSE_ref,s-1)` を取り、10seed平均を描く。基準失敗/ゼロMSEによる比の未定義はreference_undefinedとして絶対MSEを残す。平均MSEの比とは混ぜない。

補助水準 `{0,0.1,0.5,1,2,5,10}%` は事前固定する。集約到達時間は、10/10成功の点で平均deltaが水準以下となる**観測済み予算点の時間中央値の最小値**。CI上限も水準以下かは補助列だけに示す。seed別にも到達率と観測最短時間を出す。未達かつ全予算成功ならunreached、失敗等が残り未達ならincomplete、基準不成立ならreference_undefinedを区別し、到達seedだけの時間中央値で優劣を断定しない。

これらは採否マージンでも未知のglobal optimumでもない。run途中の到達や未測定点の補間を作らない。test曲線/下側包絡/観測到達は事前固定点の記述で、未見性能を保証する選択結果ではない。実用設定はvalidationのみで選び、対応するtest品質・時間を別表に出す。

## 計時と資源

warm training（初期中心注入、train、finalize）、最終exact labels、Resident-input pipeline、cold import/JITを別々に保存する。`assign_seconds`はtrain splitに対する一度の`evaluate_centers(train)`だけを測る。この処理ではfinal exact labelsとtrain MSE/cluster countsが融合しているため、その全時間を記録し、純粋なassignment kernel時間とは呼ばない。syntheticのtrain group評価、validation/test評価、その他の診断は`evaluate_seconds`へ分け、`assign_seconds`と二重計上しない。実際に使ったevaluator thread数とbackendを記録する。現行計画値はCPU 16 threads、GPU 1 threadである。GPUの学習時間は前後同期したwall timeを主とし、CUDA eventは補助とする。resident pipelineでは評価を終えてからcentroidを保存するのではなく、学習直後に保存・fsyncして計時窓を閉じ、その後に評価する。input load/cache、初期中心準備とwhole-trial wall時間はmetadataに残す。

GPUのrunごとのallocator peakを取るため、warm-up後かつmeasured H2D transfer前にCUDA同期、`empty_cache()`、`reset_peak_memory_stats()`を行う。`max_memory_allocated`と`max_memory_reserved`をrunごとに保存する。NVMLのdevice使用量はCUDA contextや外部processを含む補助的な総使用量として別記し、run固有peakと同一視しない。worker RSS合計が運用上限を超えた場合はphaseをabortし、実行中runをinterruptedとして保全する。共有pageを含む合計RSS超過だけで個別runを`infeasible`にしない。個別GPU OOMは従来どおり`infeasible`とする。

CPU主計時は1 NUMA nodeの16物理coreへ固定し、CPU baseline同士とGPU jobを同時に計時しない。GPUは1trial1GPU、1物理CPU coreを割り当て、BLAS/Torch host threads=1。workerはphase中常駐し、(worker,method,cell)の初回に最小予算の捨てrunでwarm-upする。その時間はcold/warmup記録へ分離する。複数GPUの総throughputと1trial速度は分け、method割当をseedで巡回する。

sentinelはpilot版S0/R0、seed100で、GPU=e-means T8/Lloyd I10、CPU=NumPy e-means T8/Faiss Lloyd I10を使う。単独/同時とrepeat0–2は別measurement_idで保存する。CPU比較は従来どおりである。並列GPU sentinelはpilotに限り、8個の凍結済みGPU UUIDを参加者として固定し、1 participant・1 CLIにつき1 GPU、共有host memory reservation 48 GiBを使う。既存のshared phase lockとGPUごとのexclusive lockを維持する。任意の`measurement.sentinel`は`group_id`、`participants`、`members`、`timeout_seconds`を持ち、timeoutは300秒とする。groupは一度限りで、同じgroupを再開・再利用しない。

parallel sentinelではwarm-upの後、CUDA入力転送と同期を終えてtrain timerを始める直前にだけ`before_train` hookを呼び、file-backed barrierで同期する。全参加者が揃った後に共通の将来release時刻を発行し、その時刻まで待って計測を開始する。warm-up、通常main、main-isolatedではhookを呼ばず待機しない。実際のtrain timerと同じ境界でadapterが`time.time_ns()`の開始・終了時刻を記録する。GPUの終了時刻はCUDA同期後に取る。8 UUID全員の成功、同じinput/init hash、非abort cohort、および8本の学習時間窓の正の共通overlapを確認する。共通時間、開始ずれ、overlap比を保存する。わずかなoverlapだけで干渉がないと結論せず、sentinelの資格条件はpilot前に固定する。参加不足、worker失敗、timeout、stale group、hash不一致はgroup全体を資格なしとし、黙って単独実行へ切り替えない。実行中attemptはinterruptedとして残し、再測定は新しい`group_id`、新しいrootで8参加者すべてを揃える。

barrier waitはtrain timerから除外する一方、連続したResident-input pipelineには含め、`coordination_wait_seconds`として別記する。sentinelのpipeline値は通常のpipeline比較に混ぜず、sentinelの5%干渉判定はwarm `train_seconds`だけで行う。再現MSEの許容はGPU atol1e-6+rtol1e-4、CPU atol1e-6+rtol1e-5。計時差10%超は不安定として原因を調べる。これは数値/計時の検査で、手法間の品質合格幅ではない。pilot中にseed101の初回記録も保存し、最後に同seed101のsentinelを別measurement_idで再実行して比較する。

stockのS0 engineering campaignは48 runが個別に成功した一方、e-meansの同seed MSEは数値再現許容を超えた。個別statusを保持し、campaignはunqualifiedとする。新profileは全22条件の最大予算coverage44件と、8 GPU再資格720件を完了した。e-meansは該当数値identityごとに全反復のcentroid SHAが一致し、Lloydは既定許容内だった。正式pilotは360件すべて成功し、validationまでを評価した。環境、感度検証160件、計時フラグ、レビューの条件は[本評価プロトコル](../../report/protocol/README.md)に記録する。

R0とS0の代表予算点（e-means T=3,8、Lloyd I=10,50、MiniBatch pass=2,10）は全10seedでrepeat0–2を実行しseed内時間中央値を使う。他点はrepeat0。qualifiedな単独GPUと8 GPU同時sentinelのwarm `train_seconds`中央値が5%以上違えば、本計時の採用cohortをmain-isolatedにし単独GPUで行う。5%は測定干渉の検査値。seed100の適格48比較のうち16比較で5%以上の差が出たため、本評価はmain-isolatedを採用する。pilot終了後・主test前に設定を凍結し、元artifactを保持する。学習後の評価並列化を学習時間へ加算しない。

メモリ適合しないElkan等はOOMを再現し続けずinfeasibleと記録する。予算によるtimeoutを設定する場合はpilot実測に基づく運用上の監視値を凍結し、打切り値と成功時間を混同しない。固定の全体GPU時間上限は置かない。

## 完了条件

全22セルについて、適合する必須手法・予算点・10seedのrun台帳が埋まり、successful/failed/infeasible/unsupportedが区別されること。必須手法全体の欠落は部分完了とする。失敗を除いて優劣を作らず、未到達も含めて描画する。共通初期値、split、設定、上流版、実backend/dtype、失敗理由を追跡でき、別seed sentinelの再実行が再現許容範囲にあることを確認する。

最後に条件別の得意不得意、計時干渉、warm trainingとResident-input pipelineの差、未実証範囲を日本語レポートへまとめる。DEEP96/IVF等の追加は、この完了を置き換えない。
