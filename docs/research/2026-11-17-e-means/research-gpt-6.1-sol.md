# e-means の現環境検証に向けた独立調査

調査担当: GPT-6.1 Sol。基準日・閲覧日: **2026-11-17**。他担当の成果物は参照していない。公開一次資料、公式ドキュメント、GitHub のソースと read-only API を調べた。実験、依存追加、環境変更、コード実装、Git 操作は実施していない。

## 1. 検証方針に影響する結論

1. **V100 では、まず native FP32 のアルゴリズム検証を設計する。** NVIDIA の表で V100 は SM70。e-means の現行高速経路は Triton が SM80 以降、CuTe が SM90 向けである。さらに `Emeans` の CUDA FP32 入力は自動的に BF16 表現へ切り替わるため、単に FP32 配列を渡すだけでは V100 の入口として不十分。固定した上流の `QuantizedEmeans(..., quantizer=None, assignment_backend="native")` を適合確認候補とする。ただし、これはソースから判断した候補であり、この環境での成功実績ではない。[NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus/legacy)、[e-means 入力型解決](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_torch.py#L279)、[バックエンド選択](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/torch_backend.py#L490)、[明示的量子化設定](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_quantized.py#L94)
2. **論文の高速化倍率を V100 の予測値にしない。** e-means の原論文は 2026-11-06 公開。1.4–2.6 epoch で Lloyd 25 epoch の損失を超え、GPU 時間を 10–45 倍短縮したと著者が報告している。この調査の実測ではない。公式再現手順は H100 80GB を指定し、GPU 時間の移植可能性を H100 に限定している。[原論文](https://arxiv.org/abs/2611.09501v1)、[再現手順](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/benchs/e_means/README.md)
3. **最初の比較は Faiss Lloyd、MiniBatchKMeans、e-means を中心にする。** GPU ライブラリ対照は Faiss GPU と cuML CUDA 12 系を適合確認後に追加。Flash-KMeans は2026年の重要比較対象だが、V100 に対する現行 Triton の正式対応を確認できないため、同 GPU での必須比較にはしない。[Triton 対応表](https://github.com/triton-lang/triton/blob/4b346ae4c225f75299cb9d9bea4238a2b6ef8276/README.md#compatibility)
4. **ANN とキュレーションを別々に判定する。** MSE、IVF の recall/検索時間、選択したデータの下流学習品質は同一の指標ではない。2026年の関連研究も目的関数、量子化、分布外クエリ、埋め込み前処理の影響を示している。後述の評価は本調査からの検証提案であり、e-means の実証結果ではない。

## 2. 上流の版、公開日、再現時の固定点

以下のコミット日は GitHub の committer date。`updated_at` や検索エンジンのクロール日を研究の公表日として使っていない。API には基準日までの `until` 条件を指定した。

| 対象 | 確認した版・日付 | 意味 |
|---|---|---|
| e-means 論文 | arXiv `2611.09501v1`、2026-11-06 | 基準日時点の arXiv 提出履歴は v1。速度・品質の主張はこの版に帰属させる |
| Faiss 最新リリース | `v1.16.0`、tag SHA `6ccfd5f9be2283ebc037564bb45c6ba63b71e96a` | release API の公開時刻は **2026-11-10 16:03:21 UTC**。本文の changelog 見出しは **2026-11-09**。両者を区別する |
| Faiss main の調査固定点 | `c8f283867285af547db7a701c1b8dfa89631b3ea`、2026-11-16 | 本文中の実装の読解はこの SHA。リリースと main を同一扱いしない |
| e-means ディレクトリの直近変更 | `2b648f0605f2ef3d14c4705350f8a0b214463878`、2026-11-11 | CPU の sums 高速化を含む。v1.16.0 より後の変更なので CPU 時間比較に影響し得る |
| 再現スクリプト追加 | `fc1095a9314f892938bf83f59e7ac966b5095def`、2026-11-09 | 論文の一部図表を再測定する公開手順がある |
| API の変更例 | 2026-11-06 `277b3c76b7a1507915eb8f5010dcb367cbc95494` | `niter` → `nepochs`。古い紹介例のコピーに注意する |

出典: [論文提出履歴](https://arxiv.org/abs/2611.09501v1)、[v1.16.0 リリース](https://github.com/facebookresearch/faiss/releases/tag/v1.16.0)、[タグ参照 API](https://api.github.com/repos/facebookresearch/faiss/git/ref/tags/v1.16.0)、[基準日までの main](https://api.github.com/repos/facebookresearch/faiss/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[e-means 更新履歴](https://api.github.com/repos/facebookresearch/faiss/commits?path=contrib%2Fe_means&until=2026-11-17T23%3A59%3A59Z&per_page=30)。

公式再現 README には「まだ Faiss リリースに含まれない」と残る一方、v1.16.0 のリリース本文には e-means と再現スクリプトが入っている。README の手順が想定した `faiss-cpu 1.15.1` に contrib だけをコピーする方法と、v1.16.0 の配布物を使う方法を混在させない。実際の配布 wheel の内容・ハッシュと import 元は実行前に確認する。[再現 README](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/benchs/e_means/README.md)、[リリース内容](https://github.com/facebookresearch/faiss/releases/tag/v1.16.0)

## 3. 2026年の関連研究・実装の選別

表の「採用」は検証プラン上の候補という意味であり、インストール・性能確認済みという意味ではない。コードのライセンスと、論文・データのライセンスは区別する。

### 3.1 直接比較・近接する拡張候補

| 研究・直接 URL | 公表・更新 | 関連性と著者の主張 | 公開物・ライセンス・再現性 | 比較採否 |
|---|---|---|---|---|
| [e-means](https://arxiv.org/abs/2611.09501v1) / [Faiss 実装](https://github.com/facebookresearch/faiss/tree/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means) | 2026-11-06 | 固定データに対する EMA mini-batch 更新。batch/decay の経験的則、初期化、復活、終了処理を組み合わせる | [コード MIT](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/LICENSE)。NumPy 参照版、Torch 版、分散版、再現スクリプトあり | 主対象。V100 は native FP32 と CPU 参照から品質を確認 |
| [Flash-KMeans](https://arxiv.org/abs/2603.09229v2) / [コード](https://github.com/svg-project/flash-kmeans) | v1 2026-03-10、v2 04-09。調査固定 SHA `b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0` は08-28 | Lloyd の exact assignment を IO/集約カーネルで高速化。H200 で評価。e-means の更新則自体とは異なる改善軸 | `0.3.1`。公開の benchmark/example と Torch fallback がある。[実 LICENSE は Apache-2.0](https://github.com/svg-project/flash-kmeans/blob/b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0/LICENSE.txt)だが、[pyproject の classifier は MIT](https://github.com/svg-project/flash-kmeans/blob/b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0/pyproject.toml)。配布物の表示不整合を記録する | H100/A100 以降の別環境で重要な速度対照。V100 の必須対照から保留。Torch fallback の結果を Triton の速度再現と呼ばない |
| [A Super Fast K-means for Indexing Vector Embeddings](https://arxiv.org/abs/2603.20009v1) / [SuperKMeans](https://github.com/cwida/SuperKMeans) | 2026-03-20。固定 SHA `f00b55615ffea1ff2ce521de20de1f9479982618` は10-13 | 高次元の不要な距離計算を pruning。ANN recall による早期停止も提案。GPU 評価は H100 NVL | [MIT](https://github.com/cwida/SuperKMeans/blob/f00b55615ffea1ff2ce521de20de1f9479982618/LICENSE)、benchmark/構築手順あり。GPU 論文環境は CUDA13.1 なので V100 対応の証拠にはならない | CPU の追加対照として有用。早期停止あり/なしを区別する。GPU はビルド対象・実装の SM70 適合確認まで保留 |
| [Stop Indexing at Full Precision](https://arxiv.org/abs/2608.14648v1) / [同実装](https://github.com/cwida/SuperKMeans) | arXiv ページの提出日は **2026-07-30**。VLDB 2026 workshop | clustering 前の次元削減・量子化・pruning が IVF 品質に与える影響を調査。arXiv ID の月だけで日付を推定しない | SuperKMeans の MIT コードで SQ8/LVQ/RaBitQ の公開経路がある。実データの個別条件は別確認 | 主比較が揃った後の量子化アブレーション。V100 に FP8/INT8 の新世代カーネルを仮定しない |
| [Flash-GMM](https://arxiv.org/abs/2606.10896v1) / [IBM 公式コード](https://github.com/IBM/Flash-GMM) | 2026-06-09。固定 SHA `375d857a5d2e3bb337340718aba69b05f2733a08` は09-30 | soft GMM を IVF coarse quantizer に使い境界点を複数リストへ割り当てる。MSE 最小化と目的が異なる | [Apache-2.0](https://github.com/IBM/Flash-GMM/blob/375d857a5d2e3bb337340718aba69b05f2733a08/LICENSE)。公式 README の検証 GPU は A100/H100/RTX5080。論文 v1 は ANN 10M、kernel 100M まで、billion ANN は将来課題と明記。後日の README の billion streaming 記述とは出典を分ける | IVF 品質重視の第2段階。学習時間に加え重複 posting による index サイズ増加も比較。V100 正式対応は未確認 |
| [Turbo-GMM](https://arxiv.org/abs/2611.02449v1) / [コード](https://github.com/mominabbass/turbo-gmm) | 2026-11-02。固定 SHA `7dfe42ee1229c1bcc4c86f1278f740946180ded2` は11-04 | GMM の融合・メモリ削減。同じ初期化、固定反復、数値品質を揃えた比較を重視 | [Apache-2.0 の公開リポジトリ](https://github.com/mominabbass/turbo-gmm)。README の計測 GPU は A100。`diag`/`spherical` のみ、再計測手順と測定ログあり | GMM 拡張に進む場合だけ採用。e-means の直接 Lloyd 対照にはしない。V100 動作未確認 |
| [s-means 公式リポジトリ](https://github.com/suayptalha/s-means) / [著者ブログ](https://suayptalha.github.io/s-means-blog/) | repo 作成2026-11-13。固定 SHA `83c20d09621bf8be2b7ca896359d17c1cca54f0c` も同日 | e-means の初期化、distortion に基づく復活、終了処理を変更。T4/L4/A100 での MSE 改善を著者が主張 | [MIT](https://github.com/suayptalha/s-means/blob/83c20d09621bf8be2b7ca896359d17c1cca54f0c/LICENSE)。README と raw JSON 結果あり。GPU 9設定の多くは合成データで、実データ covtype の改善は小さい。`benchmarks/` の公開一覧は results と summarize_results.py。生成・計時条件の完全な再現手段は追加確認が必要 | 非査読の最新派生候補として記録。標準埋め込み・同一初期値の独立再現を先に行い、主対照に昇格させる。T4 の結果から V100 を保証しない |

補足の一次資料: [Flash-KMeans の基準日前 commit](https://api.github.com/repos/svg-project/flash-kmeans/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[SuperKMeans の基準日前 commit](https://api.github.com/repos/cwida/SuperKMeans/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[Flash-GMM の基準日前 commit](https://api.github.com/repos/IBM/Flash-GMM/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[Turbo-GMM の基準日前 commit](https://api.github.com/repos/mominabbass/turbo-gmm/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[s-means の基準日前 commit](https://api.github.com/repos/suayptalha/s-means/commits?until=2026-11-17T23%3A59%3A59Z&per_page=1)、[s-means の公開 benchmark 一覧](https://api.github.com/repos/suayptalha/s-means/contents/benchmarks?ref=83c20d09621bf8be2b7ca896359d17c1cca54f0c)。

### 3.2 ANN・データキュレーションの設計に効く2026年研究

| 研究 | 公開日・一次 URL | 実装・データ・ライセンス確認 | 本検証との関係・採否 |
|---|---|---|---|
| MV-IVF: Multi-Vector Retrieval via Multi-Vector Clustering | 2026-11-06、[論文](https://arxiv.org/abs/2611.09040v1) | MVSIC の実装を論文中で説明。今回の調査では著者所有の配布先とコードライセンスを確定できなかった。BEIR/LoTTE と ColBERTv2 を使うが、各データの条件は未監査 | Chamfer/MaxSim を対象にするため、single-vector L2 の e-means と同じ表で優劣を付けない。将来の multi-vector 用途の参考。最初の実験対象から除外 |
| On Representation Redundancy in Large-Scale Instruction Tuning Data Selection / CRDS | 2026-02-14、[論文](https://arxiv.org/abs/2602.13773v1)、[公式コード](https://github.com/tdano1/CRDS) | コード Apache-2.0、直近 SHA `45abd6fe72c383c72d187cd9eba38f8975b4a4d6` は03-06。README に tulu-3-sft-mixture、GSM8k/MBPP、8プロセスの実行例。データ・encoder のライセンスはコードから継承されないため個別確認が残る | whitening/表現の冗長性が選択品質へ影響する。e-means の高速化だけでキュレーション品質を保証しない根拠。主比較では埋め込みを固定し、前処理を変える場合は別実験 |
| Active Learning with Low-Rank Structure for Data Selection | arXiv 2026-06-14、[論文](https://arxiv.org/abs/2606.16045v1)、[ICML2026/PMLR](https://proceedings.mlr.press/v306/cohen-addad26a.html) | 公開一次ページで論文・実験を確認したが、公式の実行リポジトリと実装ライセンスを確認できなかった。独立再現にはその入手か明示的再実装が必要 | clustering に基づく代表性と low-rank 構造の適合は異なる。初期段階の必須 baseline にはしない。キュレーション評価に uniform sampling と同予算の下流比較を置く設計上の参考 |
| VIBE: Vector Index Benchmark for Embeddings | **初版2025-05-23、v3更新2026-08-23**、[論文](https://arxiv.org/abs/2505.17810v3)、[公式コード](https://github.com/vector-index-bench/vibe) | [コード MIT](https://github.com/vector-index-bench/vibe/blob/1e7f77666aaf1754d14b8d1072aa8d8be1bbedf9/LICENSE)。多くの埋め込みは CC BY4.0、GloVe は PDDL1.0。モデル/元データの条件は別。ID/OOD 分割と ground truth を配布 | 初出2026と誤記しない。IVF の実用性評価に採用候補。benchmark 用埋め込みをキュレーション後の学習データとして扱わない。論文 Appendix A.6 は用途を vector-index benchmark と説明 |

出典補足: [CRDS README と入力準備](https://github.com/tdano1/CRDS/blob/45abd6fe72c383c72d187cd9eba38f8975b4a4d6/README.md)、[VIBE データ・ライセンス・再現手順](https://arxiv.org/html/2505.17810v3#A1.SS4)。

### 3.3 古いが必要な対照、名前が似る候補

- **Faiss Lloyd**: e-means の公開論文上の主対照。上流 Faiss 自体は [MIT](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/LICENSE)。2026年リリースを使うことと、Lloyd の新規研究であることは別。25反復、同一初期中心、同一学習行、subsampling 無効、metric/precision を記録する。
- **scikit-learn MiniBatchKMeans**: Sculley 系の標準 mini-batch 対照。原法は2010年であり2026年研究ではない。コードは [BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)。[公式 API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)の `tol`、`max_no_improvement`、`reassignment_ratio`、`batch_size` を明示する。paper matching と現行安定版の結果を混ぜない。
- **AnswerDotAI/fastkmeans**: [公式コード](https://github.com/AnswerDotAI/fastkmeans)、引用年2025。`0.5.0`、直近 SHA `596bff0a2f91df998d30bd3a5e607c237dfa251c` は **2025-06-27**。2026研究としては数えない。[設定](https://github.com/AnswerDotAI/fastkmeans/blob/596bff0a2f91df998d30bd3a5e607c237dfa251c/pyproject.toml)は Apache-2.0。Torch/NumPy の経路を持ち、Triton の自動選択は BF16 対応判定に依存。GPU 既定 dtype FP16、256点/centroid の sampling、早期停止があるため、比較時は明示して揃える。[実装](https://github.com/AnswerDotAI/fastkmeans/blob/596bff0a2f91df998d30bd3a5e607c237dfa251c/fastkmeans/kmeans.py#L45)。V100 の任意追加対照とし、元の RTX4090 の倍率は転用しない。
- **Komorebi660/Fast-KMeans**: [別のコード](https://github.com/Komorebi660/Fast-KMeans)。repo 作成は2026-03-20だが、README が引用する [RetroInfer](https://arxiv.org/abs/2505.02922) は2025年。既定 metric が inner product で、A100 の小規模 batched workload が中心。GitHub metadata の license は null、root 一覧にも LICENSE を確認できず、利用可能なライセンスを確定できないため採用保留。AnswerDotAI 版と混同しない。[metadata](https://api.github.com/repos/Komorebi660/Fast-KMeans)
- **cuML / cuVS**: 2026年の公式安定版をライブラリ対照として使う。cuML `v26.10.00` は2026-10-08公開、[Apache-2.0](https://github.com/NVIDIA/cuml/blob/v26.10.00/LICENSE)。[cuML リリース](https://github.com/NVIDIA/cuml/releases/tag/v26.10.00)。cuML の `max_samples_per_batch` は距離計算のバッチ幅であり、EMA centroid 更新を意味しない。cuVS の hierarchical k-means と通常の flat k-means を同じ対照にしない。[公式 API](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.KMeans/)、[SuperKMeans 論文の比較上の注意](https://arxiv.org/html/2603.20009v1#S5.SS2)

## 4. V100 ×8、driver535 で残る適合条件

環境前提は親セッションから与えられた **V100-SXM2 32GB ×8、driver535.154.05、当 Codex 環境では Docker コマンドなし**。本担当は環境を再調査・変更していない。

| 層 | 一次資料で分かること | 現環境に関する判断 |
|---|---|---|
| GPU | V100=SM70。[NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy) | SM80/SM90 前提の実装を直接使えるとは判断しない |
| PyTorch | 2026-09-04 の公式告知は **2.14 が Volta 向け既成 wheel の最終版**、2.15 から CUDA12.6 wheel を廃止すると説明。[公式告知](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432) | e-means 再現 README の `pip install torch` をそのまま使わない。2.14以前の SM70 を含む CUDA12 系 wheel を第一候補にし、実際の arch list と演算の起動を確認。具体的 lock は環境構築時に確定 |
| CUDA | CUDA13 では Volta の offline compilation/library support を削除。[13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/#deprecated-architectures) | CuTe 用の `cu13` 手順は V100 用の環境構築手順にならない |
| driver | CUDA12.x minor compatibility は driver>=525。ただし新機能、PTX、NVCC target に制約。[NVIDIA 互換性](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) | 535だからCUDA12系の全構成が成功すると断定しない。特に JIT/PTX は独立に確認 |
| Triton | 基準日前 main README の正式 NVIDIA 対象は CC8.0+。[固定 README](https://github.com/triton-lang/triton/blob/4b346ae4c225f75299cb9d9bea4238a2b6ef8276/README.md#compatibility) | 過去版 Triton の SM70 対応情報を現在の e-means/Flash-KMeans 対応と同一視しない |
| e-means | 自動 FP32→BF16、SM80未満の半精度自動経路を拒否する条件をソースで確認 | 明示的な量子化なし/native FP32候補と NumPy 参照版を先に試す。成功未確認 |
| cuML | RAPIDS26.10 の CUDA12列は toolkit12.2–12.9、driver535+、Volta70を掲載。CUDA13列は Turing75以降。[公式版別表](https://docs.nvidia.com/datascience/platform-support/#rapids-26-10) | **CUDA12の適合候補**。OS/glibc、Python、wheel、CUDA libraries が未確認なので動作保証ではない。26.12 nightly を安定版として採用しない |
| 複数GPU | e-means に分散・centroid sharding 実装がある | ×8で256GBの単一配列を自動的に扱えるわけではない。まず単GPU、その後同期コストと分散品質を測る |

FP32の入力だけでも、1024次元×1M行は4.096GB、10M行は40.96GB（十進単位）。後者は32GBカード1枚を超える。この値は形状と dtype からの算術であり使用メモリの実測ではない。centroid、EMA統計、labels、割当 scratch、入力変換、allocator を加味して規模を決める。8枚への展開前に native の chunk 幅とピークを確認する。

Docker が呼べない現状態からホスト Python/uv/make に切り替えることは、本調査の範囲に含めない。実験はユーザーが利用できる dev container の経路を確定した後に行う。

## 5. データ候補と取得上の不確実性

| 候補 | 一次資料・公開条件 | 初期検証への適否 |
|---|---|---|
| DINO10B の100k/1M prefix | e-means 論文 Appendix K は1024次元の byte-quantized DINOv3 patch、nested subset、100k query と各規模の ground truth、データ **CC BY4.0** を説明。[論文](https://arxiv.org/html/2611.09501v1#A11)、[著者配布先](https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/) | 高次元の本命。ただし今回 directory URL は403だった。個別ファイルへの range access、manifest、checksum、subset と ground truth の対応を後で確認。未取得を取得済みと扱わない |
| VIBE の小規模埋め込み | [公式 repo](https://github.com/vector-index-bench/vibe)、[ライセンス説明](https://arxiv.org/html/2505.17810v3#A1.SS4)。train/query/100-NN ground truth を持ち、metric がデータに付随 | ANN 評価の第一候補。最初はL2の同分布、次にOOD。cosine/IP を L2 へ無断で置換しない。個別ソースの条件を記録 |
| DEEP* prefix | [公式再現README](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/benchs/e_means/README.md)は96次元、10M学習+評価行の部分取得方法を記載 | 低めの次元でアルゴリズムを比較するには有用。mirror の存在だけでは元データの利用条件を確認したことにならない。ライセンス未確定につき取得の必須対象にしない |
| SPACEV、SimSearchNet++、SIFT | 再現READMEはprefix/heldoutの部分取得とfingerprint検証を説明 | 異なる次元・dtypeを追加する後続候補。配布先、利用条件、heldout固定を個別に確認してから採用 |
| 合成 Gaussian/異方性/不均衡/外れ値 | 外部の顧客データは不要。生成器と seed を保存する設計 | 小規模の数値確認と failure-mode 検査向け。実埋め込みへの一般化や ANN/キュレーションの性能証拠には使わない |

データの CC BY とコードの MIT/Apache/BSD は異なる許諾体系。埋め込みが公開されていても元画像・原文を再配布してよいとは推論しない。VIBE は入力に原文・画像を含まない benchmark 用の埋め込みであり、キュレーション後の学習結果を見るには別途許諾の確認できる元データが必要。

## 6. 実行前に具体化する検証プラン

以下は提案であり、この担当による実装・実行はしていない。

### 段階0: 適合性と数値の確認

- dev container を使える実行経路を決め、wheel/ソースの SHA・hash、Python、Torch、CUDA、driver、Faiss、GPU arch、バックエンドを保存する。依存追加と環境変更は別の承認済み作業で扱う。
- 小規模の固定配列で NumPy参照、native FP32 を同じ初期中心から動かし、centroid/label の shape、有限値、label範囲、割当数の総和、空クラスタを確認する。完全一致を要求せず、共通の高精度評価器で MSE を計算する。
- `recipe` の実際の表現、batch law、batch幅、decay、最終化、backend、epoch数を保存する。初回compile、warm-up、本計測を分ける。

### 段階1: e-means の品質と所要時間

- 主対照は固定した Faiss Lloyd 25反復、MiniBatchKMeans、e-means 1/1.5/2/2.5/3/5 epoch。同じデータ・学習行・初期中心・seedから始める。random初期化とk-means++は別系列にし、初期化費用を測る表も用意する。
- Faiss/fastkmeansの点数制限や MiniBatchKMeans/cuMLの早期停止を明示して揃える。まず全点・固定budgetの比較、その後に各ライブラリの推奨既定設定も別表で測る。
- 最終centroidに対する **train MSE と holdout MSE**、同じローカル Lloyd@25 target への到達epoch/秒、未到達、ピークGPUメモリ、入力転送を測る。著者配布のH100時間をV100の分母にしない。
- e-means の返す training labels は最後のepoch中の割当で、最後のcentroidの最近傍と一致しないことがある。比較用 MSE は共通の exact reassignment から求め、追加passの費用を別記する。[公開API説明](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/README.md#quick-start)
- 次元は少なくとも低次元、96/128程度、768/1024程度を分け、`n/k`、cluster imbalance、空クラスタ率、極小クラスタ率を報告する。seed はpilotを少数で行い、比較表は同一seed群で中央値とばらつきを示す。
- native同士の比較で更新則の差を見てから、Faiss GPU/cuML の利用可能な最適経路による実務上の比較を行う。精度とカーネルが同時に変わる比較は、その条件を表記する。

### 段階2: Faiss IVF/ANN への効果

- coarse centroidだけを差し替え、同じデータ・同じ `nlist`・同じ `IndexIVFFlat`・同じ精度で作る。量子化を加える前の段階で差を見る。
- recall@1/10/100、検索latency/QPS、build時間、indexサイズ、実際にscanしたvector数、list imbalanceを測る。固定 `nprobe` と固定 scan予算の両方を見る。クラスタ不均衡で同じ `nprobe` の仕事量が変わるためである。
- ID query と OOD query を分け、元のmetricと同じexact ground truthを使う。Flash-GMM のmulti-assignmentを加えるなら postingの重複率・容量を費用として含める。
- 大規模e-meansの内部IVF近似割当と、完成後の検索用IVFを混同しない。前者は学習誤差と速さの近似、後者はユーザー検索の精度を評価する。[e-means のIVF説明](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/ivf_flat.md)

### 段階3: データキュレーションとスケール

- 同じ埋め込み、同じ選択数で uniform random、Lloyd代表点、e-means代表点を比較する。centroidを架空の学習例として使わず、元データIDへ対応する実例を選ぶ。重複排除と品質フィルタは各方式で共通にする。
- 学習条件を固定した下流指標、少数群/rare modeの保持、重複率、分布coverage、選択時間を評価する。MSE改善だけで「良い学習データ」と結論しない。whitening/PCAは別の要因として後から比較する。
- 単GPUが再現できてから2/4/8 GPUへ進む。global batchとGPU数の増加を分けて測り、all-reduce、転送、I/O、GPU時間合計、学習品質の変化を記録する。1B/10Bの再現は初期目標にしない。

## 7. 不足・未確認事項

- 本担当は現環境の import/演算を一切試していない。native FP32 を含め、V100での e-means 動作・速度は未確認。
- e-means以外の **2026年の固定データ向けEMA mini-batch clustering** で、同等の大規模実埋め込み評価・公開実装・再現手順が揃う独立論文は、本調査範囲では追加確認できなかった。11月13日のs-meansは最新の公開派生実装として分けた。古い VQ-VAE、Sculley、Staleness-Reduction を2026研究に繰り上げていない。
- Flash-KMeans、SuperKMeans GPU、Flash-GMM、Turbo-GMM のSM70成功例・正式対応は確認できない。新世代GPUの計測から推定しない。
- DINO10Bの個別配布ファイル、rangeアクセス、checksumの取得成功は未確認。DEEP*/SPACEV/SimSearchNet++等の利用条件は未確定。
- CRDS等の下流実験を追加するには、元データ、encoder、学習モデルの許諾と必要メモリを別途確認する必要がある。
- MV-IVF、low-rank selection の著者公式実行コードとライセンスを確定できなかった。「存在しない」ことまでは主張しない。
- README、paper、releaseで日付・環境・対応範囲が食い違う箇所は上記の通り残る。基準日以降の将来の安定版や、nightlyの予定を既成事実として扱っていない。
