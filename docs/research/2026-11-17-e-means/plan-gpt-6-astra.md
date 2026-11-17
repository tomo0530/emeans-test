# e-means 検証プラン — GPT-6 Astra 独立案

基準日: 2026-11-17 / 担当モデル: `gpt-6-astra` / reasoning effort: `xhigh`

状態: **調査と実験設計のみ。依存導入・データ取得・実験・実装は未実施。**

## 1. 概要と現時点の結論

CPU FP32 で学習効率を確認し、適合確認後に V100 native FP32 へ進む。少ない epoch で同等品質に達するかと、実時間を短縮できるかを別々に判定する。現時点で優位性は未実証。

論文はミニバッチの点数・座標和を指数移動平均で更新する手法を提案し、バッチサイズと更新率の設定則、クラスタの復活、最終中心の再計算を組み合わせる。論文の 1.4–2.6 epoch と GPU 10–45 倍という値は著者の比較条件での報告であり、本環境の予測値に使わない。特に H100 での測定と V100 の検証を混同しない。[論文 v1][paper]

① CPU のアルゴリズム比較、② V100 のコスト比較、③ H100 での追試を候補とし、①→②を推奨する。③は資源外。以下の閾値と予算は**実施前に固定する提案値**である。

## 2. 証拠、環境、実行前の適合条件

論文は `arXiv:2611.09501v1`、コードは SHA `c8f283867285af547db7a701c1b8dfa89631b3ea` に固定する。Faiss 本体は [MIT][faiss-license]。この版の動作と wheel 内の contrib の同一性は未確認。

| 項目 | 2026-11-17 の観測・設定 | 実施時に解消する条件 |
| --- | --- | --- |
| リポジトリ | `tomo0530/emeans-test`、`main`、既存変更あり。コードはテンプレート段階 | この文書以外は今回変更しない。実装開始時に作業範囲を承認する |
| Python・依存 | Python `>=3.10,<3.13`、GPU compose の指定は `3.12.15`。lock に torch/faiss/triton/scikit-learn の名を確認できない | インストール済み環境とは区別し、承認されたコンテナ内で実体・import パス・版を確認 |
| GPU | Tesla V100-SXM2-32GB ×8、driver `535.154.05` | GPU の割当・空き・共用負荷を確認。最初は 1 GPU のみ |
| CPU・RAM | Xeon E5-2698 v4、40 物理コア・80 論理 CPU、可視 RAM 約 504 GiB | 可視値は専有枠ではない。CPU 比較は同じ 8 物理コアを上限に固定 |
| 容量 | 作業領域空き約 909 GB、`/tmp` 側約 28 GB | 実行前に再測定し、データとキャッシュは作業領域へ |
| 実行場所 | `/.dockerenv` は存在、`docker` コマンドは見つからない | プロジェクトの開発コンテナにいる証拠にはならない。mount、image digest、`.venv_container` を確認 |
| CUDA 設定 | GPU compose は digest 固定の CUDA `12.8.1` runtime | 実際にロードされる PyTorch CUDA・cuBLAS と driver の組合せを確認 |

根拠は共通の読み取り観測と [pyproject.toml](../../../pyproject.toml)、[GPU compose](../../../context/gpu/docker-compose.yml)。割当・実行互換性は未確認。

### V100 の入口を間違えない

V100 は SM70 である。[NVIDIA 一覧][gpu] 固定 SHA の raw code では次の処理を確認した。

- `Emeans._resolve_input_dtype` は CUDA FP32 入力を既定で BF16 に切り替える。[dtype の解決処理][emeans-torch]
- `select_backend` は SM90 の対応形状・BF16 を CuTe、SM80 以上の対応形状を Triton とする。それ以外の native 選択では FP32 が許容される一方、SM70 の FP16/BF16 は auto で例外になる。必要なソフトウェアがない場合も、必ず他 backend に退避する設計ではない。[backend 選択処理][backend]
- `QuantizedEmeans` の **明示的な** `quantizer=None` は自動変換を無効にする。候補入口は FP32 入力に対する `QuantizedEmeans(d, k, quantizer=None, assignment_backend="native")`。`Emeans(..., assignment_backend="native")` だけでは既定の BF16 化を無効にできない。[量子化入口][quantized]

実動作は未確認。import、FP32 GEMM、割当・更新・復活・finalize、保存後の再評価を適合試験とし、`recipe` の dtype/backend を記録する。主比較では低精度・CuTe・Triton を使わない。

PyTorch 公式告知では 2.15 以降 cu126 wheel を廃止し、Volta 向け最終系列は 2.14。`2.14.x + cu126` を候補とし、Python 3.12/x86_64 artifact、SM70、必要 API の実在を確認する。[公式告知][pytorch]

driver 535 / CUDA 12.8 を一律非互換としない。CUDA 12.x は driver 525 以上で minor compatibility があるが、PTX・新機能には制約がある。[NVIDIA][cuda] 実 CUDA、コンテナ起動、SM70 cubin、GEMM を確認し、失敗時は CPU に限定。driver 更新・ソースビルド・image 変更は別途判断する。

開発コンテナが確認できなければ**実験を開始しない**。ホスト Python/uv/make への切替と依存追加には repo 規約上の承認が必要。候補版・取得元・license・lock 差分を準備し、`exclude-newer="1 week"` は維持する。

## 3. 主仮説と事前の判定基準

共通の品質指標は、保存した最終中心に対する `MSE = mean_i min_j ||x_i-c_j||²`。次元数では割らず、train/validation/test を別に集計する。

| 区分 | 仮説・測定 | 成功 / 失敗 / 保留の事前基準 |
| --- | --- | --- |
| **主仮説 H1: 学習効率** | 後述の固定 2 セルで、e-means 5 epoch は Lloyd 25 iteration に test MSE で非劣性 | 各セルの対応 seed 差 `δ=100×(MSE_e/MSE_L−1)` の平均と 95% CI を求め、両セルで上限 ≤ +0.2% なら支持。いずれかの下限 > +0.2% ならそのセルで反証。その他は保留 |
| 実用条件 C1: 時間 | H1 を満たす構成の、同じ device 上の実時間 | `t_L/t_e` の中央値 ≥2、かつ CI 下限 >1 を採用候補条件とする。E2E が悪化する場合は warm な反復学習用途に適用を限定 |
| 探索 H2: 約 1/10 epoch | e-means 2.5 epoch と Lloyd 25 の比較 | δ の上限 <0 なら当該セルで支持。平均値だけの勝敗や一番よい seed は採用しない |
| 探索 H3: 設定則と頑健性 | n/d/k、偏り、低次元、バッチ則、初期化、復活の影響 | 品質と時間の Pareto 曲線を提示。探索結果で H1 の条件を後付け変更しない |
| 探索 H4: IVF への有用性 | 同じ recall に必要な構築時間・検索時間 | recall と latency の非劣性を確認して初めて ANN 用途を検討。MSE からの推論だけでは支持しない |

H1 は CPU FP32 が必須、GPU は独立に再判定する。異なる device の速度比で C1 を通過させない。結論は検証したセルの範囲に限る。

## 4. データ、版、許諾、分割

| データ | 用途と固定方法 | 分割・前処理・許諾 |
| --- | --- | --- |
| 合成 `mixture-v1` | 主セル A: train 100,000、d=128、k=256。生成仕様と NumPy 版を固定 | 64 等重み Gaussian 成分、中心を `N(0,4²I)`、各点のノイズを `N(0,I)` とする。中心 seed=20261117、train/val/test の乱数系列を分離。val/test は各 20,000。自作データ、外部権利物なし |
| Fashion-MNIST 公式版 | 主セル B: d=784、k=256。公式 GitHub の `data/fashion` の gzip を取得対象とする | 元 train 60,000 を train 50,000 / val 10,000、公式 test 10,000 は封印。28×28 を FP32 flatten、255 で除算。学習済み特徴抽出器は使わない |
| SIFT1M / DEEP 系の実ベクトル | 埋め込みに近い領域への外的妥当性を確認する候補 | 配布元・版・明示的利用許諾・最小取得単位を確認後に追加。今回 SIFT 配布サイトの取得は timeout、許諾は未確認。自動採用しない |
| SPACEV1B | 100 次元の実テキストベクトルの拡張候補 | 公式仕様・O-UDA を確認、実ダウンロードは未確認。repo 全体の MIT と取り違えず、許諾制約との適合確認前は対象外 |
| DINO10B | 論文との対応関係を整理するだけ | 配布物・個別 license・取得可能性は今回未確認。全量実験は予算外 |

Fashion の取得口・MIT は [公式 README][fashion] / [LICENSE][fashion-license] を確認。版はファイル名と公開 MD5 で識別する。train images: `8d4fb7e6c68d591d4c3dfef9ec88bf0d`、test images: `bef4ecab320f06d8554ea6380940ec79`。取得時に commit、日時、SHA-256、license 原文を保存する。[SPACEV の仕様・license][spacev]

分割 seed=20261117。同一 raw-vector hash をグループ化し、train/val を分割する。公式 test と重なる学習側の群は除外し、実件数を記録。test 内重複は群単位で扱う。ラベルは分割層化だけに使用し、学習には渡さない。

公式 learn/base/query は維持し、base を縮小したら ANN 正解を再計算する。PCA・中心化・量子化校正は train のみで fit。球面版は別実験で共通の正規化・ゼロベクトル規則を定義する。画素・合成での成功を文章埋め込みへ一般化しない。

## 5. 最小から拡張する実験表

段階ごとに数値健全性と予算を確認する。**H1 を含め test は探索・設定選択が終わるまで封印**し、中間判断は val で行う。

| 段階 | 条件・比較 | 回数 | 通過条件・得られる判断 |
| --- | --- | --- | --- |
| E0 適合 | CPU import、微小配列、V100 FP32 経路、評価器の照合。n=4,096、d=16/128、k=16/64 | seed 0、時間測定は参考 | NaN/Inf なし、保存・再読込可、要求 dtype/backend と一致。GPU 不可なら CPU に限定 |
| E1 主比較 | 主セル A/B、CPU `NumpyEmeans` 5 epoch と Faiss Lloyd 25。全行・同一初期中心 | seed 0–9、2×2×10=40 学習 | H1 用の学習。最終 test 開封時に判定し、失敗セルも残す |
| E2 実時間 | A/B、V100 e-means native FP32 と適合済み GPU Lloyd、必要なら同演算系の Torch Lloyd を機構比較に追加 | まず seed 0–2 の pilot、続いて 0–9 | C1、メモリ、cold/warm、実装差を判定。対応する実用 baseline がない場合は限定比較と明記 |
| E3 収束と初期化 | e の T={1,2.5,3,5,10}、Lloyd={5,10,25,50}。random と共通 k-means++ 初期値 | pilot 3 seed、代表 1 セルを 10 seed | H2。各 T は独立学習。最適 budget を val で選び test は最後に評価 |
| E4 規模・頑健性 | 合成 n={100k,1M}、d={16,128,768}、k={256,1,024} のうち一因子ずつ。偏り、外れ値、quality/throughput、復活有無 | 最初 3 seed、重要条件だけ 10 seed | どの条件で優位性を失うか。実ベクトルは許諾・取得確認後に追加 |
| E5 IVF | Fashion 小規模で配線確認、次に許諾済み実ベクトル 1M、d=128 程度、nlist=1,024 を目標 | 10 学習 seed、固定 query | 同品質の検索まで短縮するか。実ベクトル未取得なら用途判断は保留 |

50 iteration は参照の収束確認用。H1 の参照は替えない。e のバッチ則・schedule は T に依存するため、10 epoch の途中を 2.5 epoch の結果で代用しない。[設定解決][base]

## 6. baseline の公平性と乱数制御

CPU baseline は Faiss Lloyd。既定 `max_points_per_centroid=256` は A を間引くため `ceil(n_train/k)` 以上を指定。e は `max_points_per_centroid=None`、`batch_law="throughput"`、更新率は公式則、復活閾値 0.1、`capture_assignments=False`、双方単一 restart。読取行数・実 iteration を検証する。Faiss が同一目的値で早期停止した場合は実回数を併記し、25 回未実行なら H2 の「1/10」は判定しない。[Faiss 設定][clustering]

seed 一致だけでは RNG 間で初期中心が一致しない。共通行 ID から生成した FP32 `init_centroids` を双方に渡し hash を保存。e の permutation・revival seed と処理点数も記録する。追加の MiniBatchKMeans は `partial_fit` へ同じ順序・バッチを渡す。別途 tuning する場合は val 探索予算を揃える。[API][minibatch]

k-means++ は共通初期値として両手法へ与え、初期化時間を E2E に含める。ライブラリ既定設定の比較は参考表に分離。Flash-KMeans の V100 対応は仮定しない。

GPU baseline は Faiss GPU の artifact・SM70・FP32・初期値注入を確認後に使用。不可なら同じ FP32/chunk の Torch Lloyd を機構比較に用い、CPU 基準と照合する。その結果で最速実装への優位性は主張しない。CPU の thread/affinity/BLAS は固定し、backend 差を算法だけの効果と混同しない。

## 7. 評価器、時間、統計、IVF

### 独立した品質評価

学習が返す labels や `obj` を最終品質として流用しない。e-means の labels は中心が移動している最終 pass で得たもの、`obj` は移動中の距離であり、最終中心への再割当とは異なる。[学習 API の説明][emeans-torch]

保存した FP32 中心と元入力を共通の exact L2 評価器で再割当する。`IndexFlatL2` で chunk 化し、距離集計は FP64。小規模で NumPy FP64 総当たりと MSE を照合する（`rtol=1e-5, atol=1e-7`）。tie は距離で判定し、不一致なら比較を中断する。

再割当後の空クラスタ率、サイズ分位点、最大占有率、不均衡度 `k×sum(n_j²)/n²` を測る。NaN/Inf、形状不整合、全ラベル同一、ゼロ・重複中心、入力の欠損・ゼロ分散列を検査する。

### 時間とメモリ

`core_train` は device 上にデータを置いた後の初期化・学習・finalize。`end_to_end` は読込・変換・転送・保存も含め、外部評価時間は別記する。cold 初回と warmup 後を分離し、GPU は前後で同期。手法の実行順を seed 内で交互化し、pilot は warm 3 反復を行う。反復を独立 seed と数えない。

time-to-quality は val の Lloyd@25 MSE +0.2% を満たす最小の事前指定 budget の時間とする。未到達は打切りとして残し、曲線の外挿で到達時刻を作らない。探索全体の計算費用も別記する。

peak RSS、GPU allocated/reserved、NVML peak、Faiss scratch を併記。GPU 利用率・温度・clock・同居プロセスも記録する。混雑試行は黙って除外せず、warm 時間の変動係数 >10% は不安定として再測定を検討する。

### 複数 seed と不確実性

固定 split の seed 0–9 を対応付け、MSE 相対差の平均・時間比の中央値を seed 単位で 10,000 回 bootstrap して 95% CI を求める。全 seed 値も保存し、CI が不安定なら保留。test を見た seed 追加は禁止。この CI はデータセット間の一般化を表さない。検索は学習 seed と query 重複群の二段階 bootstrap とする。

### IVF 下流評価

同一 Faiss `IndexIVFFlat` に各手法の中心を注入し、追加時に中心が再学習されていないことを hash で確認する。base vectors、順序、距離、nlist、検索 device を揃える。`nprobe={1,2,4,8,16,32,64,128}` を val query だけで走査し、Recall@10 ≥0.95 を達成する最小 nprobe を固定して test query へ進む。両者が達成できなければ到達不能と報告する。

正解は同一 base の exact top-100。Recall@1/10/100、QPS、単一 query の p50/p95、batch=128 throughput、スキャン候補数、index 容量を測る。学習・追加・保存時間も分離。ANN 採用条件は test recall 差の CI 下限 ≥−0.5 percentage point、p95 比の上限 ≤1.05、総構築時間の中央値を 20%以上短縮。外れ query・不均衡リストも調べる。RAG の回答品質は測定対象外。

## 8. 計算・保存予算と停止条件

以下は**未実測の見積**。FP32 の入力 `4nd`、中心 `4kd`、全距離 `4nk` byte。1M×128 は 0.48 GiB、1M×768 は 2.86 GiB。1M×16,384 の全距離 61 GiB は生成しない。距離 chunk 8,192×4,096 は 128 MiB だが workspace・複製が別途必要。

| 段階 | 予約上限の提案 | 保存量の目安 |
| --- | --- | --- |
| E0–E1 | 8 CPU core を使う wall time 4 時間、GPU 適合 0.5 時間 | 入力・中心・ログ合計 2 GiB 以下 |
| E2–E3 | V100 1 枚で合計 6 GPU-hours | warm/cold ログと代表中心で 5 GiB 以下 |
| E4–E5 | 適合・許諾後の別枠、最大 12 GPU-hours | 取得時 archive を含め追加 30 GiB 以下を目標 |

3 seed の pilot で epoch・初期化・評価・I/O 時間を測り、trial 数から再見積する。n/k/d 変更時も pilot で確認。単価未提示のため金額は未算定とし、実 GPU-hours・CPU-core-hours・GB-days を残す。

上限は GPU 24 GiB、RSS 64 GiB、trial 30 分、空きディスク 100 GiB。実割当が小さければそちらを優先する。OOM は記録し、共通 chunk 縮小を一度だけ試す。batch・精度・データ量の変更は別実験。互換性、許諾、評価器不一致、NaN/Inf、上限超過で停止する。8 GPU 分散は今回の予算外。

## 9. 再現メタデータと repo 内の配置案

manifest: 計画版、repo/upstream SHA・dirty diff、wheel/lock/image hash、Python/CUDA/driver/GPU UUID、CPU/BLAS/thread、data URL/版/license/hash、split 行 ID、前処理、初期中心 hash、全 seed、n/d/k、予算・実処理行数、要求/実 backend・dtype、`recipe`、chunk、停止理由、時間・メモリ、評価器版、成果物 hash。秘密情報・PII は保存しない。

実装承認後の配置は既存構成に合わせる。

- `experimentation/e_means/`: config、I/O、runner、描画 notebook。
- `experimentation/dataset/e_means/`: raw/processed、split/license manifest。大規模データは Git 管理外。
- `experimentation/outputs/e_means/<run_id>/`: metrics、manifest、最終中心・図、代表失敗。巨大 labels は原則保存しない。
- 再利用可能な評価・学習 adapter は `project_module/src/project_module/e_means/` へ。`Input/Output/Config`、`invoke`、dtype/shape 契約を持ち、I/O・時刻測定は外側に置く。
- `project_module/tests/`: 手計算 MSE、再割当、漏洩防止、再読込、不正入力、微小 CPU 比較。重い benchmark は通常テストから分離。

実装時は [データ規約](../../../docs/agents/ml-and-data.md)、[テスト規約](../../../docs/agents/testing.md)、[DS 契約](../../../docs/ds_integration_contract.md) に従い、開発コンテナで `make test`、`make fix`、`make all-lint` を実行する。今回は未実行。

## 10. 実験後レポートの目次と未実施表

参考の [技術比較記事](https://app.notion.com/p/3596fbf10b7280bd9d07e2ed15dec15a) と [State-Aware RAG 記事](https://app.notion.com/p/3a46fbf10b7280ceb429d4fa793a1a97) は構成の参考とする。技術的根拠や数値は流用しない。Notion への書込みは今回行わない。

1. 概要・採用判断・適用可能な条件
2. 技術の仕組みと比較対象の選択理由
3. 主仮説・探索仮説・事前判定基準
4. データ、環境、版、分割、公平性
5. 結果: MSE–epoch、MSE–時間、メモリ、IVF recall–latency
6. 代表 seed の収束、失敗・外れ値、結果の再現手順
7. 考察: 実証された範囲、反証、環境依存、未検証
8. コスト、保守性、採用・条件付き採用・見送り

| 実験 | train/test MSE・CI | core / E2E 時間 | peak RAM/VRAM | IVF recall / p95 | 判定 |
| --- | --- | --- | --- | --- | --- |
| E0 適合 | 未実施 | 未実施 | 未実施 | 対象外 | 未判定 |
| E1 主比較 | 未実施 | 未実施 | 未実施 | 対象外 | 未判定 |
| E2–E3 時間・収束 | 未実施 | 未実施 | 未実施 | 対象外 | 未判定 |
| E4 頑健性 | 未実施 | 未実施 | 未実施 | 対象外 | 未判定 |
| E5 IVF | 未実施 | 未実施 | 未実施 | 未実施 | 未判定 |

**採用判断:** H1・C1・再現性・メモリ条件を満たしたデータ/device で条件付き採用。H1 のみなら研究継続。V100 不適合は環境上の見送りであり算法の反証ではない。ANN は E5、RAG は許諾済み実埋め込み・業務データでの別検証まで保留。

## 参照一次資料

リンク先の閲覧日は 2026-11-17。可変 URL の最新性、未取得 artifact の利用可能性は保証していない。

[paper]: https://arxiv.org/html/2611.09501v1
[faiss-license]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/LICENSE
[emeans-torch]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_torch.py
[backend]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/torch_backend.py
[quantized]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_quantized.py
[base]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/base.py
[clustering]: https://raw.githubusercontent.com/facebookresearch/faiss/c8f283867285af547db7a701c1b8dfa89631b3ea/faiss/Clustering.h
[pytorch]: https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
[cuda]: https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html
[gpu]: https://developer.nvidia.com/cuda/gpus/legacy
[fashion]: https://github.com/zalandoresearch/fashion-mnist
[fashion-license]: https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/LICENSE
[spacev]: https://raw.githubusercontent.com/microsoft/SPTAG/main/datasets/SPACEV1B/README.md
[minibatch]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
