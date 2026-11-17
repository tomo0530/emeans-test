> 作成: Gemini 4 Hydrogen (High)。`agy models` で確認した `gemini-4-hydrogen-high` を指定して独立Web調査を実施。CLI終了コード0・JSONのSUCCESS・非空の最終回答を確認。本文冒頭の一時promptファイルへのリンクのみ [相談記録](consultation-record.md) に置換した。他の担当の回答は参照していない。実験は未実施。

> **親による検証注記**: 本回答は上流コードの直接読解が未完了で、最新の版・許諾・GPU対応について未確認の推測を含む。採用判断は [統合プラン](synthesis.md) を優先する。現行e-meansのV100候補は **明示的 `quantizer=None` のnative FP32** とし、FP16/Tritonの強制実行は初期対象にしない。GPU e-meansがFaiss GPUを内部利用するとの解釈、`fastkmeans v0.0.4` を最新とする記述、ローダのライセンスからデータ許諾を推定する記述は採用しない。driver535/CUDA12.8の組合せはminor compatibilityとPTX等の条件を切り分けて確認する。

---

[共通調査指示](consultation-record.md) に基づき、2026年11月17日時点の一次資料を中心とした独立調査結果を報告する（シェル実行・ファイル編集・環境導入は一切実施していない。確認済み事実と推論・未確認事項は明確に区別している）。

---

## 1. 推奨と重要な知見

### 最重要の環境制約（V100-SXM2 + driver 535.154.05 + CUDA 12.8.1）
1. **ドライバ 535.154.05 と CUDA 12.8.1 の整合性リスク**:
   - **確認済み事実**: NVIDIA CUDA 12.8 GA の付属推奨ドライバは Linux で `>= 570.26` である一方、CUDA 12.x Minor Version Compatibility の最低要件は Linux x86_64 で `>= 525.60.13`（かつ `< 580`）であり、ドライバ `535` 系は CUDA 12.2 世代のブランチである（[NVIDIA CUDA Compatibility / Release Docs](https://docs.nvidia.com) 検索結果より）。
   - **推論 / リスク**: Compose 設定の CUDA 12.8.1 / Python 3.12.15 環境がホスト側 `535.154.05` 上でそのまま動作するか（`cuda-compat` の有無や CUDA 12.8 新機能要求によるエラー有無）は、本調査時点では実機未検証のため**最優先のブロッカー確認項目**となる。
2. **Volta (`sm_70` / V100) 非推奨化・非対応化の潮流と `bfloat16` 非対応**:
   - **確認済み事実**:
     - V100 は Volta アーキテクチャ（Compute Capability `7.0` / `sm_70`）であり、ハードウェアのネイティブ `bfloat16` 演算・`fp8` 演算を持たない（`float16` Tensor Core および `float32` に依存）。
     - OpenAI Triton は v3.3 以降で Volta (`sm_70`)・Turing (`sm_75`) のネイティブ Tensor Core (`tl.dot`) サポートを廃止・非推奨化している（[triton-lang/triton#5066](https://github.com/triton-lang/triton/issues/5066)）。
     - PyTorch 2.7〜2.8 以降では CUDA 12.8/13.x への移行に伴い `sm_70` 標準ビルドの縮退が進んでおり、NVIDIA RAPIDS (`cuVS` / `cuML`) も CUDA 13.x 構成では `sm_75`（Turing）以上を必須としている（[rapidsai/cuvs](https://github.com/rapidsai/cuvs)）。
   - **重要結論**: `e-means` 原論文（H100 での 9.5×〜45.5× 高速化）や `Flash-KMeans`（H200 での 17.9×〜200× 高速化）、`SuperKMeans`（H100 NVL + CUDA 13.1.0）の結果を **V100-SXM2 の実証として外挿してはならない**。V100 では「Triton `tl.dot` の非動作／スカラーフォールバック」「`bfloat16` 不可に伴う `float16` / `float32` 限定」「80 logical CPU / 504GiB RAM とのCPU-GPU比較」が検証の主軸になる。

### 現環境で比較・検証すべき候補の優先順位
- **Tier 1（必須・コア比較群）**:
  1. **`e-means`**（`faiss.contrib.e_means`, Faiss v1.16.0 / [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）: 一次検証対象。
  2. **`Faiss` 標準 GPU/CPU KMeans**（`faiss.Kmeans` / `faiss-gpu-cu12`, [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)）: 同一ライブラリ内ベースライン。`faiss-gpu-cu12` ホイールは `sm_70` を含む構成が一般的で V100 適合性が高い見込み。
  3. **`fastkmeans`**（[AnswerDotAI/fastkmeans](https://github.com/AnswerDotAI/fastkmeans), v0.0.4）: `major >= 8`（Ampere 以上）でのみ Triton を有効化し、それ未満では **PyTorch フォールバック**に切り替わる設計のため、V100 上の「素の PyTorch GPU 実装」および `max_points_per_centroid=256` 既定挙動の対照群として非常に有用。
  4. **古典的 Mini-Batch K-Means**（[Sculley 2010](https://dl.acm.org/doi/10.1145/1772690.1772862) / scikit-learn 等）: 「ミニバッチ化単体」と「e-means の EMA＋スケーリング則＋dead-cluster revival」の差分を切り分ける対照群。
- **Tier 2（互換性ゲート通過時のみ比較）**:
  5. **`Flash-KMeans`**（[arXiv:2603.09229](https://arxiv.org/abs/2603.09229) / [svg-project/flash-kmeans](https://github.com/svg-project/flash-kmeans)）: 2026年時点の最有力 exact GPU Lloyd ベースラインだが、Triton カーネルと低精度（`float16` / `bfloat16`）前提のため、V100（`sm_70` + `float16` + 導入予定 Triton 版）でコンパイル・実測可能かを最初にスモークテストで判定し、不可なら理由を明記して除外する。
  6. **`SuperKMeans`**（[arXiv:2603.20009](https://arxiv.org/abs/2603.20009) / [cwida/SuperKMeans](https://github.com/cwida/SuperKMeans)）: 80 logical CPU / 約 504GiB RAM を活かした **CPU 比較（IVF 用クラスタリング・Early Termination by Recall）** として採用価値が高い。GPU 版は論文が CUDA 13.1.0 / H100 NVL 評価のため、`CMAKE_CUDA_ARCHITECTURES=70` と GCC 13+ / Clang 17+ / CMake 3.26+ で V100 向けビルドが通る場合のみ GPU 比較に加える。
  7. **`NVIDIA cuVS / cuML`**（[rapidsai/cuvs](https://github.com/rapidsai/cuvs) / [rapidsai/cuml](https://github.com/rapidsai/cuml/blob/main/LICENSE)）: `cuvs-cu12` が CUDA 12.8.1 / Python 3.12 / V100 で依存競合なく動く場合の参考ベースライン。構築コストや依存衝突が出る場合は除外。
- **除外対象**:
  - `DINO10B` 全量（100億ベクトルは単機検証には過大。ライセンス詳細も要確認）、`DINO-AD`（[arXiv:2602.03870](https://arxiv.org/abs/2602.03870), 医療異常検知特化）、`emeansR`（R/Rust ポート）、RAG特化クラスタリング（`TopClustRAG` 等）。

---

## 2. 関連研究比較表

| 名称 | 公開/更新日 | 直接URL | 主張 | コード | ライセンス | 現環境適合性 (V100x8 / CUDA12.8.1 / Py3.12) | 採否 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **e-means** (`arXiv:2611.09501`) | 2026-11-06 (論文) / 2026-11-10 (Faiss v1.16.0) | [arXiv:2611.09501](https://arxiv.org/abs/2611.09501) / [contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) | ミニバッチ EMA 更新＋スケーリング則＋dead-cluster 分割復活により、1.4〜2.6 エポックで Lloyd@25 以下の test MSE を達成し、GPU で 9.5×〜45.5× 高速化（H100 等評価） | `facebookresearch/faiss` (`contrib/e_means`) | MIT (Faiss リポジトリ準拠) | Faiss ベースで V100 (`sm_70`) 適合見込み。ただし CUDA 12.8.1 wheel/ソースビルドと PyTorch/Faiss 連携は未確認 | **採用（一次対象）** |
| **Flash-KMeans** (`arXiv:2603.09229`) | v1: 2026-03-10 / v2: 2026-04-09 | [arXiv:2603.09229](https://arxiv.org/abs/2603.09229) / [flash-kmeans](https://github.com/svg-project/flash-kmeans) | FlashAssign（距離行列非物質化）と Sort-Inverse 更新により exact Lloyd を H200 上で PyTorch 比最大 17.9×、cuML 比 33×、FAISS 比 200×超高速化 | `svg-project/flash-kmeans` (Triton) | Apache-2.0 | **要注意**: Triton カーネル + `float16`/`bfloat16` 前提。V100 は `bfloat16` 非対応かつ Triton 3.3+ で `sm_70` `tl.dot` 廃止のため動作未確認 | **条件付き採用**（`fp16` V100 スモークテスト通過時のみ） |
| **SuperKMeans** (`arXiv:2603.20009`) | 2026-03-20 (論文) | [arXiv:2603.20009](https://arxiv.org/abs/2603.20009) / [SuperKMeans](https://github.com/cwida/SuperKMeans) | 次元枝刈りと GEMM インターリーブ、Early Termination by Recall (ETR) により CPU で FAISS/sklearn 比最大 7×、GPU で cuVS 比最大 4× 高速化 | `cwida/SuperKMeans` (C++/Python, CUDA/cuBLAS) | MIT | 80 CPU / 504GiB RAM とは好相性。GPU は論文が CUDA 13.1.0 / H100 NVL 評価であり `sm_70` ビルド可否と GCC 13+/Clang 17+ 要件が未確認 | **CPU採用 / GPU条件付き**（`sm_70` ビルド成功時のみ） |
| **fastkmeans** | 2025-05 (v0.0.4) | [fastkmeans](https://github.com/AnswerDotAI/fastkmeans) | `torch`+`numpy` のみの軽量 k-means（FAISS/sklearn 互換 API、既定 `max_points_per_centroid=256`）。Ampere 以上では Triton、それ以外は PyTorch 実行 | `AnswerDotAI/fastkmeans` | Apache-2.0 | V100 (`major=7 < 8`) では自動的に PyTorch フォールバック経路になる設計で互換性は高い見込み | **採用**（PyTorch GPU ベースライン） |
| **Faiss (Kmeans / IVF)** | 2024-01 (論文) / 2026-11-10 (v1.16.0) | [arXiv:2401.08281](https://arxiv.org/abs/2401.08281) / [faiss releases](https://github.com/facebookresearch/faiss/releases) | 大規模ベクトル探索・量子化・クラスタリングの標準基盤。GPU/CPU 双方の `Kmeans` と IVF インデックスを提供 | `facebookresearch/faiss` (`faiss-gpu-cu12`) | MIT | `faiss-gpu-cu12` は `sm_70` を含むビルドが一般的で V100 適合見込み。v1.16.0 の Py3.12 / CUDA 12.8 互換は要確認 | **採用**（標準・IVF ベースライン） |
| **NVIDIA cuVS / cuML** | 2025–2026 (25.x / 26.x) | [rapidsai/cuvs](https://github.com/rapidsai/cuvs) / [rapidsai/cuml LICENSE](https://github.com/rapidsai/cuml/blob/main/LICENSE) | GPU ベクトル検索・k-means 基盤（cuVS は CAGRA/IVF 向け k-means 等、cuML は `KMeans`） | `rapidsai/cuvs`, `rapidsai/cuml` | Apache-2.0 | CUDA 12.x 系では `>= 7.0` (Volta) 対応の記述がある一方、最新世代は Ampere+ 最適化・CUDA 13 系で `sm_75+` 必須化。PyPI 導入競合リスクあり | **低優先 / 条件付き**（導入容易なら参考値、競合時は除外） |
| **Mini-Batch K-Means (Sculley)** | 2010-04 (WWW '10) | [DOI:10.1145/1772690.1772862](https://dl.acm.org/doi/10.1145/1772690.1772862) | Web スケール向けミニバッチ k-means の基礎アルゴリズム | scikit-learn 等 / 同等ミニバッチ実装 | BSD-3-Clause (sklearn 参考時) | CPU/GPU いずれでも基準実装が容易 | **採用**（EMA 無し対照） |
| **DINOv3 / DINO10B** | 2025-08-13 | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | 17B 画像プールから 5 階層 k-means で LVD-1689M を構築。DINO10B は 10B 枚の量子化パッチ記述子 | Meta FAIR 系 | DINOv3 専用ライセンス（非 Apache-2.0） | 10B 全量は 504GiB RAM / 8x V100 で非現実的。また DINOv3 は独自ライセンス制約に注意 | **比較実装外（文脈）** / DINO10B全量は**除外** |
| **Vo et al. 2024** | 2024-05 (arXiv / TMLR) | [arXiv:2405.15613](https://arxiv.org/abs/2405.15613) | 階層的 k-means と均衡サンプリングによる SSL 自動データキュレーション | 論文手法 | — | DINOv3 等データキュレーション評価設計の背景文献 | **文脈・評価設計参照** |
| **SemDeDup / FineWeb** | 2023-03 / 2024-06 | [arXiv:2303.09540](https://arxiv.org/abs/2303.09540) / [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) | 埋め込み k-means による意味重複除去・LLM コーパスキュレーション | 論文・データセット | — | データキュレーション用途のダウンストリーム評価観点として参照 | **文脈参照（比較対象外）** |
| **DINO-AD** | 2026-02 | [arXiv:2602.03870](https://arxiv.org/abs/2602.03870) | 医療画像異常検知向けの訓練不要 DINOv3 + k-means プロトタイプ構築 | 論文手法 | — | 汎用・大規模 GPU クラスタリング速度比較とは目的が異なる | **除外** |

*(データセットライセンス補足: ベンチマーク候補のうち **SIFT1M/BigANN** は INRIA Texmex 由来の研究公開データ、**DEEP1B** は Yandex Research（TFDS ローダは Apache-2.0 / 文書 CC-BY 4.0）、**SPACEV-1B** は Microsoft Bing の Big-ANN 向け公開データ、ANN-Benchmarks 等の HDF5/GT 派生は MIT 等のものが多いが、利用前に対象配布元の明示ライセンスを再確認すること。)*

---

## 3. e-means 原典の再現条件・注意点

> [!IMPORTANT]
> 本調査ではシェル実行および `raw` ファイル直接取得を行っていないため、検索で裏付けられた論文・リリース仕様と、ソースファイル実読が別途必要な code-level 詳細を切り分けて記載する。

1. **API と基本設計**:
   - **確認済み事実**: `e-means` は `faiss` v1.16.0（2026-11-10 リリース）の `contrib/e_means`（[GitHub](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)）に収録され、Faiss の標準 `Kmeans` の代替として使える設計である。更新則はミニバッチごとのクラスタ和・カウントに対する EMA（$S \leftarrow (1-\alpha)S + \alpha s$ 相当）、スケーリング則によるバッチサイズ $B$ とピークステップ幅 $\alpha_0$ の自動設定、コサイン減衰スケジュール、およびドナークラスタ分割による dead-cluster revival を含む（[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）。
   - **未確認（要実コード確認）**: クラス名・関数シグネチャ・PyTorch Tensor 入力可否・マルチ GPU 指定引数などの厳密な Python/C++ API（検索時にサードパーティ解説ブログが `Emeans(d, k)` 等を例示していたが、一次コード未読のため断定しない）。
2. **dtype / 割当（assignment）**:
   - **確認済み事実**: 論文の超大規模実験（DINO10B 等）は 8-bit 量子化等の埋め込みや GPU 高速カーネルで評価され、比較先の `Flash-KMeans` は `float16` / `bfloat16` 入力＋ `float32` 累算で Tensor Core を使う（[arXiv:2603.09229](https://arxiv.org/abs/2603.09229)）。
   - **現環境注意点**: V100 (`sm_70`) は **`bfloat16` 非対応**。そのため現環境検証では、入力・インデックス計算の `float32` 対 `float16`（および累算精度の差）を明確に分けて測定し、`bfloat16` 前提の他論文設定をそのまま持ち込まないこと。
3. **subsampling（サブサンプリング）**:
   - **確認済み事実**: Faiss 標準 `Kmeans` や `fastkmeans` は既定で `max_points_per_centroid = 256` のサブサンプリングを行う（[fastkmeans](https://github.com/AnswerDotAI/fastkmeans)）。一方 `e-means` はミニバッチ EMA によって全データ分布を 1.4〜2.6 エポックで効率的に走査する（[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）。
   - **再現上の罠**: $N/K \gg 256$ の条件で `faiss.Kmeans` の既定値（256点/セントロイドで打ち切り）のまま比較すると、Faiss 側はデータの大半を見ないため高速に見えても **test MSE が悪化する**、あるいは逆に不公平な比較になる。検証では **「デフォルト設定の比較」** と **「サブサンプリング上限を解除/統一した統制比較」** の2系統を必須とすべき。
4. **train MSE / test MSE と仕上げ処理**:
   - **確認済み事実**: 原論文は過学習やサブサンプリングバイアスを避けるため、学習データと独立な **held-out test MSE** を主指標とし、`Lloyd@25`（25エポック）の test MSE 到達を基準に比較している。また、最終エポックでの割当点の厳密平均への置換や後処理 EMA による仕上げ手法も報告されている（[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）。
   - **再現上の注意点**: 評価時は必ず学習セット・テストセットを分離し、さらに IVF 等のベクトル検索用途を想定する場合は MSE だけでなく **IVF `Recall@k` / QPS** や dead cluster 率（空クラスタ数）も併記すべきである。
5. **labels（クラスタ割当ラベル）**:
   - **注意点**: フルバッチ Lloyd（`Flash-KMeans` 等）は最終イテレーションの副産物として全学習点の最新 `cluster_ids` を直接得る。一方、ミニバッチ更新の `e-means` で全学習点の厳密な最終 `labels` が必要な場合、学習完了後のセントロイドに対する **全点1パスの最近傍割当計算** が追加で必要になる。比較時は **「セントロイド学習のみの時間」** と **「学習＋全点ラベル割当までの総時間」** を区別して記録すること。
6. **時刻測定（wall-clock timing）**:
   - **確認済み事実**: 原論文の GPU 高速化（9.5×〜45.5×）は H100 等（NVMe SSD ストリーミング構成含む）での実測である。
   - **計測規律**: V100 実測時は、(a) `torch.cuda.synchronize()` による非同期カーネル完了同期、(b) 初回 Triton/CUDA JIT ウォームアップ時間の分離、(c) H2D/ディスク読み込み I/O を含む End-to-End 時間とオンデバイス計算時間の分離、(d) 評価用 test MSE 計算時間の除外、を徹底する必要がある。

---

## 4. 見つかった2026一次資料とそれ以前の基礎文献の区別

### 2026年の一次資料（本調査で特定）
- **e-means**: *e-means: outperforming k-means (in one-tenth the epochs)*（Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard, Matthijs Douze, Hervé Jégou; Meta FAIR & ENPC; **2026-11-06** 公開: [arXiv:2611.09501](https://arxiv.org/abs/2611.09501) / [GitHub contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) / Faiss v1.16.0 **2026-11-10** リリース: [PyPI faiss-cpu](https://pypi.org/project/faiss-cpu/) / プロジェクトページ: [e-means.github.io](https://e-means.github.io/)）
- **Flash-KMeans**: *Flash-KMeans: Fast and Memory-Efficient Exact K-Means*（**2026-03-10** v1 公開, **2026-04-09** v2 改訂: [arXiv:2603.09229](https://arxiv.org/abs/2603.09229) / [svg-project/flash-kmeans](https://github.com/svg-project/flash-kmeans)）
- **SuperKMeans**: *A Super Fast K-means for Indexing Vector Embeddings*（Leonardo Kuffo, Sven Hepkema, Peter Boncz; CWI & ETH Zurich; **2026-03-20** 公開: [arXiv:2603.20009](https://arxiv.org/abs/2603.20009) / [cwida/SuperKMeans](https://github.com/cwida/SuperKMeans)）
- **DINO-AD**（参考確認・比較除外）: *Training-free unsupervised anomaly detection*（**2026-02** 公開: [arXiv:2602.03870](https://arxiv.org/abs/2602.03870)）

### 2025年以前の先行・基礎文献
- **2025年**:
  - **DINOv3**（**2025-08-13** 公開: [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)）— LVD-1689M の 5 階層 k-means キュレーション。
  - **fastkmeans**（Benjamin Clavié, Benjamin Warner; v0.0.4 **2025-05** リリース: [AnswerDotAI/fastkmeans](https://github.com/AnswerDotAI/fastkmeans)）。
- **2024年以前**:
  - **The Faiss library**（Douze et al., **2024-01**: [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)）
  - **Automatic Data Curation for Self-Supervised Learning**（Vo et al., **2024-05**: [arXiv:2405.15613](https://arxiv.org/abs/2405.15613)）
  - **FineWeb**（Penedo et al., **2024-06**: [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)） / **SemDeDup**（Abbas et al., **2023-03**: [arXiv:2303.09540](https://arxiv.org/abs/2303.09540)）
  - **Web-scale k-means clustering**（D. Sculley, **2010-04** WWW '10: [DOI:10.1145/1772690.1772862](https://dl.acm.org/doi/10.1145/1772690.1772862)）— ミニバッチ k-means の原典。ほか古典基盤として Lloyd (1982)、k-means++ (2007)、VQ-VAE EMA (2017)。

---

## 5. 未確認事項・矛盾・具体的な次の行動

### 未確認事項
1. `facebookresearch/faiss` の `contrib/e_means` 実ソースファイル一覧、正確なクラス/関数 API、および `faiss-gpu-cu12` v1.16.0 ホイールへ同梱されているか（あるいはリポジトリ取得が必要か）。
2. ホストドライバ `535.154.05` 上で compose 設定の `CUDA 12.8.1 / Python 3.12.15` コンテナを起動した際、PyTorch・CUDA カーネル・`faiss-gpu-cu12` がエラーなく GPU を認識・実行できるか。
3. コンテナ内の PyTorch / Triton バージョンが `sm_70` (V100) バイナリを保持しているか、特に Triton カーネル（`Flash-KMeans`）が V100 `float16` で動作するか。
4. `SuperKMeans` の GPU コードが `CMAKE_CUDA_ARCHITECTURES=70` かつ環境内 GCC/Clang/CMake でコンパイル可能か。
5. `DINO10B` の具体的なデータセット入手先・完全な配布ライセンス条項。

### 矛盾・緊張関係
- **ドライバ世代とコンテナ CUDA 世代のギャップ**: `535.154.05`（CUDA 12.2 世代）に対し compose は `CUDA 12.8.1`（GA 推奨ドライバ `570.26+`）。Minor Version Compatibility 範囲内とはいえ、互換層未導入または一部 API 要求で失敗するリスクがある。
- **2026年研究の最新 GPU 前提と V100 (`sm_70`) の世代ギャップ**: 2026年の主要論文（`e-means`=H100, `Flash-KMeans`=H200/Triton `tl.dot`/`bfloat16`, `SuperKMeans`=H100 NVL/CUDA 13.1.0）はいずれも Volta 切り捨て後の環境を前提としている。
- **既存 `uv` workspace との依存整合**: `Python >=3.10,<3.13` の既存ワークスペースに対し、`sm_70` 互換を維持した PyTorch / Faiss と最新研究コードを共存させるには、環境分離または固定バージョンの検証が不可欠。

### 具体的な次の行動（推奨実行順）
1. **ステップ1（実行互換性スモークテスト）**: compose コンテナ（CUDA 12.8.1 / Python 3.12.15）内で GPU 認識、`torch.cuda.get_device_capability()`（`(7, 0)` 確認）、最小 `fp16`/`fp32` 行列積、および導入予定 `triton` バージョンでの小規模カーネル動作を確認する（失敗時は `cuda-compat` 確認または CUDA 12.2〜12.6 へのベース変更要否を判断）。
2. **ステップ2（`contrib/e_means` 原典の直接読解）**: `facebookresearch/faiss` (`v1.16.0` / `main`) の `contrib/e_means` ファイル群を直接確認し、正確な API・`dtype`・サブサンプリング仕様・CPU/GPU 切替・`labels` 生成方法を確定する。
3. **ステップ3（比較候補の互換性ゲート）**: `faiss-gpu-cu12`（必要なら `CUDA_ARCHITECTURES="70-real"` ビルド）、`fastkmeans`（PyTorch フォールバック確認）、`Flash-KMeans`（V100 `fp16` コンパイル可否）、`SuperKMeans`（CPU ビルド＋GPU `sm_70` ビルド可否）の順で動作可否を切り分ける。
4. **ステップ4（統一プロトコルでの V100 小〜中規模ベンチマーク）**: `SIFT1M` / `DEEP1M` 等の許諾明確な公開ベンチマーク（および想定用途に近い `d=128..1024` 埋込）を用い、**(a) `dtype` (`fp32`/`fp16`)**、**(b) `max_points_per_centroid=256` 有無**、**(c) held-out test MSE と IVF `Recall@10`**、**(d) `torch.cuda.synchronize()` 厳守の「学習のみ」および「学習＋全点ラベル割当」壁時計時間**を 1x V100 および 8x V100 で実測する。
