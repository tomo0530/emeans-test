# e-means 現環境検証の統合プラン

基準日: 2026-11-17。作成: 親エージェントによる一次資料の照合と独立案の統合。**全実験は未実施**。以下の閾値、構成、予算は提案であり、性能結果ではない。

## 1. 推奨する進め方

**CPU FP32の最小確認 → V100 native FP32の適合確認 → 現代的な埋め込みで品質と時間を比較 → IVF検索で用途を判定**、の順を推奨する。V100で最適化カーネルが使えないことと、EMA更新則の有効性を分けて扱う。H100の論文倍率の再現は現環境の目標にしない。

原論文は2026-11-06公開。H100上の大きな高速化は著者の報告であり、現環境では品質・速度とも未確認である。[原論文v1](https://arxiv.org/abs/2611.09501v1) 現在の可視資源と設定は [環境記録](environment-and-sources.md) にまとめた。8枚のGPUや可視RAMは専有資源として予算化しない。

独立プランの本文は改変して一本化せず保存し、実行時の選択・修正をこの文書に置く。独立調査の全候補を実装することも目的にしない。

## 2. 最新調査から採用する情報

次の主要資料は親も一次ソースを確認した。各研究の性能主張を本repoの結果として使わない。

| 資料・公開/更新日 | 今回への反映 | 初期段階での扱い |
| --- | --- | --- |
| [Flash-KMeans](https://arxiv.org/abs/2603.09229v2)、2026-03-10初版 / 04-09改訂 | 更新則の改善とGPU実装の高速化を区別。最適化Lloydとの比較が必要 | V100での必須実機対照から外し、SM80以降の別条件の候補。旧Tritonの強制導入をしない |
| [SuperKMeans](https://arxiv.org/abs/2603.20009v1)、2026-03-20 | MSEだけでなく検索recallと実時間を測る。CPUの強い対照があり得る | Faiss Lloydの主比較後、CPU追加対照。recall早期停止あり/なしを分ける |
| [Stop Indexing at Full Precision](https://arxiv.org/abs/2608.14648v1)、2026-07-30 | 量子化・次元削減も別の速度要因 | 最初はFP32固定。量子化/PCAは後の一因子実験 |
| [VIBE v3](https://arxiv.org/abs/2505.17810v3)、2025-05-23初版 / 2026-08-23改訂 | 現代的な埋め込み、ID/OOD query、元metricの保持 | IVF用のデータ候補に採用。2026初出と記載しない |
| [Small Is Not Dead](https://arxiv.org/abs/2611.18238v1)、2026-11-13 | 小さく孤立した群をcountだけで復活対象にする限界 | 不均衡な合成分布、rare-mode保持率、群別MSEを探索評価に追加。著者のCPU proof of conceptをGPU対応と解釈しない |
| [s-means](https://github.com/suayptalha/s-means/tree/83c20d09621bf8be2b7ca896359d17c1cca54f0c)、2026-11-13公開の派生実装 | 初期化・復活・finalizationを切り分ける参考 | 主対照へ昇格させず保留。著者の大きな改善値は独立再現前の主張 |
| [PyTorch公式告知](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)、2026-09-04 | V100用のwheel選択は最新版無指定にしない | 2.14系列cu126等のSM70収録を実物で確認 |
| [Flash-GMM](https://arxiv.org/abs/2606.10896v1) / [Turbo-GMM](https://arxiv.org/abs/2611.02449v1)、2026-06 / 11 | soft assignmentや複数postingでは目的・容量も変わる | e-means対Lloydの直接比較から除外。IVF拡張が必要になった場合だけ検討 |

Flash-KMeansの固定版にはLICENSE本文Apache-2.0とclassifierのMITの表示不整合がある。利用時は実配布物のLICENSEを基準に確認する。[LICENSE](https://github.com/svg-project/flash-kmeans/blob/b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0/LICENSE.txt) FaissはMIT、SuperKMeansはMIT。データの許諾をコードの許諾から推定しない。

そのほかCRDS、multi-vector retrieval、キュレーション関連の候補と採否は [GPT調査](research-gpt-6.1-sol.md)、[Claude調査](research-claude-opus.md)、[Gemini調査](research-gemini-4-hydrogen-high.md) に残した。今回はRAG回答生成やLLM学習まで広げない。

## 3. 実行前に通す適合確認

1. **実行場所**: 開発コンテナの実体、mount、image digest、Python、lock、GPU割当を確認する。Codex環境の `docker` 不在だけでホストにDockerがないと断定しない。ホストPythonへ自動的に切り替えない。
2. **実装固定**: 読解したFaiss SHAは `c8f283867285af547db7a701c1b8dfa89631b3ea`。安定版候補v1.16.0とは別記し、実際のwheel/ソース、import元、hashを記録する。Faiss GPUをe-meansのGPU実行に必須としない。
3. **依存**: 候補はfaiss-cpu、NumPy、SM70を含むPyTorch cu126、必要時scikit-learn/h5py。具体版とlock差分は実装前に確定する。`uv add` やインストールを今回実行しない。`exclude-newer="1 week"` の対象artifactの公開時刻も確認し、release日やSHA固定で迂回しない。
4. **V100入口**: FP32 CUDA tensorに `QuantizedEmeans(d,k,quantizer=None,assignment_backend="native")` を使う候補を検査する。通常の `Emeans` はFP32入力をBF16へ変換するため、`assignment_backend="native"` だけでFP32が保たれるとは扱わない。要求値と `recipe` の実representation/backendを確認する。[dtype解決](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_torch.py#L279)、[量子化設定](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_quantized.py)
5. **実演算**: capability/arch listだけで終わらず、微小GEMM、割当、更新、revival、finalize、保存/再読込まで確認する。V100のBF16/FP8/CuTe/Triton経路は初期実験から外す。
6. **CUDA/driver**: driver535はCUDA12.x minor compatibilityの枠内だが、PTXや新機能などの条件は別にある。composeの12.8.1、wheelのCUDA、driver、SM70実演算を別々に確認する。[NVIDIA公式](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

失敗時はCPUでできる範囲を報告し、GPU性能は未判定とする。driver更新、image変更、PyTorchソースビルド、別GPUの有料調達はこのプランを理由に自動実行しない。

## 4. 検証データと段階

| 段階 | データと構成（提案） | 成果・停止条件 |
| --- | --- | --- |
| P0 適合・評価器 | 手計算できる微小配列と n=4,096,d=128,k=32、CPU/適合済みV100 | 形状、有限値、ラベル範囲、保存再読込、dtype/backendを確認。不一致で停止 |
| P1 最小比較 | 合成Gaussian混合 n=100k,d=128,k=256、Fashion-MNIST train50k/val10k/test10k,d=784,k=256 | 3 seed pilotで数値・計時を確認。画素データであり文章埋め込みの実証にはしない |
| P2 主評価 | VIBE `agnews-mxbai-1024-euclidean`、train250k/val10k/test10k、k=2,048、固定10 seed | 高次元テキストベクトルのIVF用中心学習としてH1/C1を評価。許諾・取得・分割を確定するまで開始しない |
| P3 用途評価 | P2の同じ中心、固定したbaseとpublisher query、Faiss IndexIVFFlat | exact top-100を再計算し、recall/latency/構築費用を判定 |
| P4 拡張 | 別modality、OOD、1M規模、rare-mode合成分布、追加baseline | P2/P3の結果と予算から選ぶ。8GPUや1B/10Bを最初の目標にしない |

Fashion-MNISTは [公式README](https://github.com/zalandoresearch/fashion-mnist) とMIT LICENSEを確認済み。ファイル取得とSHA-256は未実施。合成生成器は等方・異方・不均衡を分け、分布パラメータ、生成seed、学習seedを保存する。

VIBEの [固定版カタログ](https://github.com/vector-index-bench/vibe/blob/1e7f77666aaf1754d14b8d1072aa8d8be1bbedf9/README.md) はAGNewsを769,382点・1024次元・Euclideanと記載する。VIBEのHDF5は `train` が索引用corpus、`test` が検索queryであり、機械学習のtrain/testと同名でも役割が違う。[形式と用途](https://arxiv.org/html/2505.17810v3#A1.SS8) P2のMSE用holdoutはcorpusから別に作り、P3のqueryとは列名・IDを分ける。

VIBEはコードMIT、著者作成embeddingはCC BY4.0という説明があるが、用途はvector-index benchmarkで、元データには独自条件もある。[A.4–A.9](https://arxiv.org/html/2505.17810v3#A1.SS4) **配布embeddingと対象用途の許諾・revisionをmanifestに記録できることをP2の条件**にする。元文書・画像の取得やキュレーション後のLLM学習には流用しない。条件が満たせなければP1までを報告し、代替データはtestを見る前に選定し直す。

重複ベクトルをhashでグループ化し、群単位で分割する。前処理のfitはtrainのみ。実件数が減ったら記録し、test結果を見て都合よく差し替えない。corpusを縮小・除外した場合は元のneighborsを流用せず、対象baseの正解近傍を再計算する。

## 5. 主仮説と採用条件

すべて**事前固定する提案閾値**。初期pilotは別のデータ/seedにし、主testから閾値を調整しない。

- **H1（主仮説・品質）**: P2のCPU FP32条件で、e-means@5epochはローカルの全件Lloyd@25iterationに非劣性。対応seedごとの `delta = 100*(MSE_e/MSE_L-1)` の平均の95%CI上限が **+0.5%以下**なら支持、下限が+0.5%を超えれば反証、それ以外は保留。
- **C1（実務上の時間条件）**: 同じdeviceでの `t_L/t_e` の中央値が **2以上**、95%CI下限が **1超**。H1だけ成立した場合は速度目的の採用としない。GPU条件はCPUと分けて再判定する。
- **頑健性条件**: 実行失敗、非有限値、意図しないsubsampling、予算超過を隠さない。Lloyd@5/10や推奨設定の実務対照がe-meansを品質・費用で支配するなら、Lloyd@25に対する勝利だけでは採用しない。
- **用途条件**: P3で、val queryで選んだrecall目標0.95の設定を凍結。testのRecall@10差の95%CI下限 ≥−0.5 percentage point、p95 latency比の95%CI上限 ≤1.05、総構築時間中央値を20%以上短縮、をIVFの採用候補条件とする。未到達は未到達として記録する。

0.5%/2倍/20%は研究の報告値ではない。Astra案のより厳しい0.2%とFable案の0.5%などに差があるため、本統合では実務上の非劣性マージンとして0.5%を提案した。実装承認時に用途に合う閾値を固定する。別データや本番RAGへの一般化は別の検証が必要である。

探索では e-means T={1,2,3,5}、Lloyd={5,10,25}、初期化、revival有無、batch law、群別MSE、最終labelsの不一致、CPU/GPU経路を調べる。Tごとにbatch/scheduleが変わるので、長い一回の学習の途中を別Tの最終結果として使わない。探索の最良test結果でH1を差し替えない。

## 6. 公平な比較と独立評価

主CPU対照は同じ版のFaiss Lloyd、追加対照はMiniBatchKMeans、任意のSuperKMeans。GPUは適合済みFaiss GPU/cuMLを実務対照候補とし、同演算系のTorch Lloydは機構比較用に区別する。Faiss GPU wheelの配布主体・SM70対応・導入費用は未確認であり、必須依存にしない。

同じtrain行、共通の初期中心行列とhash、FP32、1 restart、明示した固定反復を用いる。Faissの点数上限を `ceil(n/k)` 以上にし、e-meansは上限なしとして、実際に見た行数を確認する。seed値の一致だけで初期値が一致したとは判断しない。既定設定同士の比較、全件比較、時間予算内の複数restartは別の表にする。

品質は保存した最終centroidに対して共通のexact L2評価器で再割当し、座標ごとの二乗和を点数で平均する。次元数では割らない。小規模FP64総当たりとの照合、tieの規則、集計FP64を設ける。`em.obj` と最後のepoch中のlabelsを最終MSEに使わない。`assign` は **(distances, labels)** の順である。[Numpy実装](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means.py)、[Torch実装](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_torch.py)

計時は、(1)初回import/JIT、(2)データ常駐後の初期化＋train＋finalize、(3)厳密な最終labels作成、(4)読込/転送/保存を含むE2E、を別々に測る。評価器時間を明示し、片方だけ課金しない。GPUは前後同期、CPUは同じthread/affinity、順序はseed内で交互化する。PyTorch allocatorの値だけでFaissのメモリを代表させず、peak RSS、NVML、scratchも記録する。

P2は事前固定のseed0–9。MSE相対差と時間log比をseed単位のpaired bootstrap（10,000再標本）で評価する。反復計時や各点を独立学習seedとして数えない。10seedを見て都合のよいseedを追加せず、CIが広ければ保留にする。timeout/OOMはstatusを残し、打切りデータを除外して勝率を出さない。

P3は **top-100 ground truth** を用意してRecall@1/10/100を評価する。top-10だけでRecall@100を計算しない。nprobeの選択はval queryのみ、固定nprobeとscan件数を揃えた比較も併記する。p50/p95、QPS、実scan数、list imbalance、indexサイズ、train/add/serialize時間を保存する。クラスタ内距離が良くても検索で勝てるとは結論しない。

## 7. 実施予算と成果物

P0/P1は8物理CPU core・wall time4時間、GPU適合0.5 GPU-hourを予約上限として提案する。P2/P3の初期上限は追加CPU wall time8時間、単V100で8 GPU-hours。これは所要時間の予測ではなく上限で、pilot後にtrial数×実測時間で再見積もりする。超過時はtestを見る前に実行範囲を縮小して計画版を更新する。

P2の入力250k×1024×FP32は約0.95GiB、中心2048×1024は約8MiBという算術見積もり。EMA状態、変換コピー、labels、chunk距離、workspaceは別に必要。GPU24GiB/RSS64GiB/1trial30分/実験保存20GiBを初期上限とし、実割当が小さければそちらに合わせる。全n×k距離行列は作らずchunk化し、学習batch変更と計算chunk変更を混同しない。

予定する配置は既存の規約に合わせる（**今回は実装しない**）。

- `experimentation/e_means/`: config、data取得/分割、runner、描画。
- `experimentation/dataset/e_means/`: raw/processed、license・hash・split manifest。大容量はGit管理外。
- `experimentation/outputs/e_means/<run_id>/`: 1runのmetrics、環境、中心、図、失敗の記録。
- 再利用する学習adapter/評価器は必要性が明確になった段階で `project_module/src/project_module/e_means/` へ。重い実験はunit testと分離する。

manifestにはplan版、repo/upstream SHA、dirty diff hash、wheel/lock/image hash、Python/CUDA/driver/arch、要求/実backend・dtype、全recipe、データ許諾/revision/hash、split/初期中心hash、seed、実処理行数、各時間/メモリ、評価器版、status/停止理由を残す。

最終レポートは参考Notion記事に合わせ、概要/採否、技術と比較選定、主/探索仮説、データと共通条件、MSE–epoch/時間、IVF recall–latency、代表失敗、実証範囲/未実証、費用と実務示唆、再現手順を備える。結果表は未実施としておき、論文や他モデルの推測で埋めない。

## 8. 独立案の扱いと親による修正

- Astra案のCPUから始める段階化と、実測できる小さなデータの選択を採用する。合成・画素データだけでは高次元埋め込みの採用根拠が不足するため、最新調査のVIBEを用途に合わせてP2/P3へ追加する。
- Fable案の「同じ割当カーネルによる機構比較」と「実務ライブラリ比較」の区別、短いLloydやsubsampling設定も含むPareto確認を採用する。配置案の `experiments/` は本repoの `experimentation/` へ読み替える。
- Claude調査のCPU優先と2026年PyTorch告知は採用する。`assign` の逆順表記、直接照合できていない速度値、保存済みH100 targetを無条件で主要基準にする提案は採用しない。
- GPT調査の版・公開日の区別、VIBEの2026改訂、最新のrevival研究や派生実装の隔離を採用する。
- Gemini調査のdtype/sampling/学習とlabels作成時間の区別は採用する。旧fastkmeans版の最新扱い、Faiss GPU必須と読める説明、データloaderの許諾からデータの利用可否を推定する記述、V100 FP16/Tritonの強制検証は採用しない。
- 独立案のtop-10 ground truthとRecall@100の組合せ、testを見てseed数を追加する判定手順は、上記のtop-100と固定seedへ統一する。

Gemini Carbonは同じモデルでの再試行を含む2回ともタイムアウトし、完成回答を取得できなかった。この統合案は、取得できた調査3件・独立プラン2件と親による一次資料の照合に基づき、Carbonの意見を含まない。[試行結果](plan-gemini-4-carbon-high.md) と [相談記録](consultation-record.md) に状況を記録した。
