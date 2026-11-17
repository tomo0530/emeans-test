# e-means 検証設計の独立調査: 配布物・データ・比較対象

調査日: 2026-11-17。一次資料の読取調査であり、承認済み設計・性能結果ではない。既存 `2026-11-17-e-means/synthesis.md` は提案として読んだ。初期調査は peer 本文を与えず独立に行い、後続協議で親から Fable の要旨と確認事実が共有された。以下の配布物・API・データの事実は一次資料に照合した。

## 現時点の結論

**Python 3.14 と V100 は両立する配布候補がある。ただし V100 で e-means の最速カーネルは使えず、native FP32 の実行確認が必要。** Python 対応、CUDA wheel の SM70 対応、e-means の表現形式、実環境での演算成功は別の判定である。

主目的が「同等のクラスタ品質に対する時間と、その得意・不得意の条件」であれば、固定 `e-means@5 vs Lloyd@25` 一点より、品質対時間の曲線が適している。CPU の主対照は Faiss Lloyd、追加派生対照は MiniBatchKMeans と Elkan を推奨する。GPU の native Torch Lloyd は更新則の機構比較に有用だが、最適化された全 GPU k-means 実装の代表とは呼ばない。

データは制御可能な合成分布に加え、高次元画像 embedding の VIBE Landmark、原著に近い低次元画像 embedding の DEEP1M、高次元テキスト embedding の VIBE AGNews を候補とする。画像主評価は Landmark、原著との接点を確認する条件は DEEP1M が適している。Fashion-MNIST は生画素なので画像 embedding の代用にはしない。AGNews は配布 embedding の CC BY 4.0 と元ニュース本文の条件を区別する必要があり、主目的がクラスタリングだけになったため、VIBE の想定用途との対応を設計に明記したい。

## 1. 調査時点で共有された決定と、未決定の事項

親セッションから通知されたユーザー決定:

- Python は `>=3.14,<3.15` 固定。テンプレートの設定は検証に合わせて変更可。
- 主目的はクラスタ品質と学習時間の条件別比較。IVF 検索は追加評価。
- k-means 派生手法も比較する。
- 特定用途に限定せず、text embeddings / image embeddings / 制御可能な synthetic を扱う。
- 可視 8 V100 の使用を許可。固定の時間上限を先に置かず、十分な検証を優先する。
- 固定の品質非劣性幅を採否条件にせず、品質と速度の曲線を重視する。

したがって既存案の MSE +0.5%、速度 2 倍、10 seed、8 GPU-hours、後に候補となった 24 時間 / 192 GPU-hours は、この文書で採用条件にしない。原著アルゴリズムを書き直す計画も置かない。

未決定なのは、具体的な条件表、各条件の独立 seed 数、曲線の予算点、データ採否、中心の初期化を統制する比較と既定値比較の比重、CPU スレッド配分、SuperKMeans を追加する価値と依存条件である。8 枚使用の許可は、8 GPU を使う分散学習の必要性を意味しない。

## 2. Python 3.14 と配布物

### 公開 artifact の確認

以下は配布サイトの metadata とファイル一覧の確認。インストール、依存解決、Python import は未実施。

| 対象 | 確認できた配布物・公開日時 | 判定 |
| --- | --- | --- |
| faiss-cpu 1.16.0 | `faiss_cpu-1.16.0-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`、19,485,185 bytes、2026-11-10 16:40:58.511478 UTC | Linux の cp314 専用 wheel ではなく stable ABI wheel。通常の CPython 3.14 の候補 |
| PyTorch 2.14.1 cu126 | `torch-2.14.1+cu126-cp314-cp314-manylinux_2_28_x86_64.whl`、2026-09-30 20:04:15 UTC | Python 3.14 / Linux x86_64 の実 artifact がある |
| NumPy 2.5.4 | `numpy-2.5.4-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`、2026-10-10 20:03:49 UTC | CPU 経路の候補。調査時最新 2.5.5 は 11/15 配布で cooldown 内 |
| h5py 3.16.0 | `h5py-3.16.0-cp314-cp314-manylinux_2_28_x86_64.whl`、2026-03-06 13:48:41 UTC | VIBE の HDF5 読取用候補 |
| scikit-learn 1.9.1 | `scikit_learn-1.9.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`、9,152,030 bytes、2026-09-10 18:33:17 UTC | MiniBatchKMeans / Elkan の候補、BSD-3-Clause |

一次資料: [Faiss PyPI JSON](https://pypi.org/pypi/faiss-cpu/1.16.0/json)、[PyTorch cu126 index](https://download.pytorch.org/whl/cu126/torch/)、[NumPy 2.5.4 metadata](https://pypi.org/pypi/numpy/2.5.4/json)、[h5py metadata](https://pypi.org/pypi/h5py/3.16.0/json)、[scikit-learn metadata](https://pypi.org/pypi/scikit-learn/1.9.1/json)。Python の stable ABI は指定した最低版以降との互換性を提供するが、一般的な拡張の振る舞いまで実証するものではない。[Python 3.14 C API Stability](https://docs.python.org/3.14/c-api/stable.html)（参照日 2026-11-17）

Faiss wheel の公称 SHA-256 は `876abdd551465c9a81c27eda6f42cbe76d17592acb582834e9ec53b763ff5724`。末尾 65,536 bytes の HTTP Range を読み、ZIP directory に `faiss/contrib/e_means/e_means.py`、`e_means_torch.py`、`e_means_quantized.py`、`torch_backend.py` などの同梱を確認した。これは全体 hash の再計算や import 検証ではない。[artifact 詳細](https://pypi.org/project/faiss-cpu/1.16.0/#files)

PyTorch 対象 wheel の公称 SHA-256 は `eb7564fdaccd2920ed78b4e3c08ca4a180fcaf5d4b3ce3068b1968bce4f5834b`。PEP 658 metadata は `Requires-Python: >=3.10`、CUDA toolkit 12.6.3、cuDNN 9.10.2.21、Linux/Python<3.15 で `triton~=3.8.0` を指定する。wheel 名、metadata、GPU 対応方針を照合したが、CUDA バイナリ内の cubin を分解して SM70 を実査してはいない。[公式 metadata](https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp314-cp314-manylinux_2_28_x86_64.whl.metadata)

**cooldown の注意点は具体的な時刻にある。** 2026-11-17 11:41 UTC の観測時点では Faiss wheel 配布から 7 日未満だった。この wheel が通常の `exclude-newer="1 week"` の候補になるのは 2026-11-17 16:40:58.511478 UTC より後、JST では 11/18 01:40:58 より後である。uv は release 日でなく artifact の upload time を比較する。日付だけを根拠に迂回しない。[uv 設定仕様](https://docs.astral.sh/uv/reference/settings/#exclude-newer)（参照日 2026-11-17）

調査中のローカル設定は root / `project_module` とも `>=3.10,<3.13`、Dockerfile と GPU compose は Python 3.12.15 のままだった。これは今後整合させる設定であり、Python 3.14 要件が変更されたという意味ではない。

### SM70 と e-means API を別に扱う

PyTorch 公式の 2026-09-04 告知は、cu126 を配布する 2.14 が Volta / SM70 の最後の既成 wheel 系列と説明する。2.15 の cu13 系へ更新して V100 が動くとは扱わない。[PyTorch 公式告知](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)。2.14.1 自体の公開日は 2026-09-30。[release](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)

Faiss v1.16.0 の API では、CPU は `NumpyEmeans` と NumPy / faiss-cpu で成立する。GPU e-means は PyTorch 側で動くため、Faiss GPU は必須ではない。CuTe は Hopper、Triton 最速経路は Ampere 以降が対象。[v1.16.0 README](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/contrib/e_means/README.md)（release 2026-11-10）

通常の `Emeans` は CUDA FP32 入力を既定で BF16 表現へ変換する。V100 の試行候補は **`QuantizedEmeans(..., quantizer=None, assignment_backend="native")` と FP32 tensor**。`assignment_backend="native"` だけを指定して FP32 が保たれるとは限らない。epoch 引数は読解した版では `nepochs`、`assign` の返り値は distances, labels。train の最終 epoch 中ラベルを最終中心の厳密な割当と同一視しない。[Torch 実装](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/contrib/e_means/e_means_torch.py)、[quantized 実装](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/contrib/e_means/e_means_quantized.py)

backend 選択は FP32 native を許す一方、SM80 未満の FP16/BF16 auto 経路は成立しない。V100 ではまず native FP32 の assignment、統計更新、revival、finalize、再割当の実行を確認する必要がある。[backend 実装](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/contrib/e_means/torch_backend.py)

driver 535 は CUDA 12.x minor compatibility の最低 R525 以上という範囲に入る。ただし PTX や新機能には制約があり、compose の CUDA 12.8.1、wheel 内 CUDA 12.6、driver、実演算成功を同じ指標にしない。[NVIDIA 公式仕様](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)（参照日 2026-11-17）

### 推薦する提供経路

1. **主評価: faiss-cpu 1.16.0 wheel 同梱版。** 配布物・hash・import 元を記録し、NumPy CPU と PyTorch native GPU を比較する。安定公開版として再現しやすく、原著の再実装は不要。
2. **CPU の版差感度: 必要なら固定 main `c8f283867285af547db7a701c1b8dfa89631b3ea` を別環境で評価。** 公式比較では v1.16.0 より 19 commit 先で、CPU の `np.add.at` を SciPy sparse 積へ置き換える `_add_rows_at` が追加されている。SciPy がない場合は旧経路へ戻るため、SciPy 有無も実験条件である。scikit-learn を追加すると SciPy も依存として入る点を記録する。[固定版比較](https://github.com/facebookresearch/faiss/compare/v1.16.0...c8f283867285af547db7a701c1b8dfa89631b3ea)
3. main を採るなら固定 checkout から正式に package/build した環境を使う。wheel の native Faiss と別 checkout の Python パスを `sys.path` や上書き copy で混在させない。source build はユーザー禁止事項ではないが、ビルド・版固定・検証の追加作業になる。主評価と版差評価の結果を混ぜない。

## 3. データ候補

### VIBE AGNews: 高次元 text embedding / IVF 追加評価

- artifact: `vector-index-bench/vibe` の `agnews-mxbai-1024-euclidean.hdf5`。HF revision `07b387891a221b7b073b83d2f752b76462e5fa03`、repo 最終更新 2026-08-22。3,157,100,864 bytes。LFS 公称 SHA-256 `acb2712c1717e1309e316c20701e534bc4d3015917e6b7667baed2b3592218d0`。[HF 配布元](https://huggingface.co/datasets/vector-index-bench/vibe)、[tree metadata](https://huggingface.co/api/datasets/vector-index-bench/vibe/tree/main?recursive=false&expand=false)
- 規模: corpus 769,382 × 1,024、FP32、Euclidean。既存 MXBAI embedding を使うためモデル推論不要。[配布カタログ](https://github.com/vector-index-bench/vibe#evaluation)
- 公開・非 gated を HF API で確認。通常取得に HF_TOKEN は必須ではない。固定 revision の単一 HDF5 を取得すればよく、VIBE 全フレームワークや元ニュース本文は不要。[repo metadata](https://huggingface.co/api/datasets/vector-index-bench/vibe)
- `train` は索引用 corpus、`test` は 1,000 query。`neighbors` / `distances` は各 query の top-100。MSE 用の training/validation/test と名前を分ける。corpus を縮小・削除したらその base に対する ground truth を再計算する。[VIBE データ形式 A.8](https://arxiv.org/html/2505.17810v3#A1.SS8)（v3 公開 2026-08-23）
- 生成コードは `sentence-transformers/agnews` の description を用い、1,000 行を乱数 seed 42 で query へ分ける。標準の 120k/7.6k AG News 分類セットではない。時間・ニュース配信元・同一記事系列で group 分離した設計ではなく、raw 文書 ID がない embedding 配布物だけでは意味的重複の監査に限界がある。[生成コード](https://github.com/vector-index-bench/vibe/blob/main/vibe/datasets.py)、[元の整形データ card](https://huggingface.co/datasets/sentence-transformers/agnews)（card 更新 2024-04-30）
- VIBE コード MIT、作者作成 embedding CC BY 4.0。著者は用途を vector-index benchmark と記載し、元データの多くには独自 non-commercial 条件があると説明する。AGNews の元配布ページは本調査では timeout し、原文条件の独立照合は完了していない。embedding の許諾を元記事本文・embedding モデル学習・データ選別後の別学習へ拡張しない。[VIBE A.4–A.9](https://arxiv.org/html/2505.17810v3#A1.SS4)、[AG 元配布ページ](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

設計上の判断: 現代的 text embedding の候補として有用。ただし clustering 単独の主評価なら、vector index の中心学習のベンチマークとして扱うことを明記するか、用途条件が明確な別 corpus を選ぶ。CC BY 自体の attribution / 変更明示等は保持する。[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)（参照日 2026-11-17）

### VIBE Landmark: 高次元 image embedding の主候補

- artifact: 同じ VIBE revision `07b387891a221b7b073b83d2f752b76462e5fa03` の `landmark-nomic-768-normalized.hdf5`。2,341,733,696 bytes、LFS 公称 SHA-256 `9e870c21f3a33cea1adb9055ce0e7af0988557b5e4e21485a661f40365be286a`。corpus 760,757 × 768 と query 1,000、既存の正規化 Nomic Vision embedding。公開・非 gated なので通常取得に HF_TOKEN は必須ではない。[HF artifact metadata](https://huggingface.co/api/datasets/vector-index-bench/vibe/tree/main?recursive=false&expand=false)、[公式カタログ](https://github.com/vector-index-bench/vibe#evaluation)（repo 更新 2026-08-22）
- VIBE の `landmark_loader` は Google Landmark の **index** 画像を使う。GLDv2 の公式説明は index 761,757 枚の全画像を CC0 または Public Domain と記載し、metadata は CC BY 4.0。train 側の画像条件と混同しない。GLDv2 index 公開は 2019-05、v2.1 は 2019-09。[VIBE 生成コード](https://github.com/vector-index-bench/vibe/blob/main/vibe/datasets.py)、[Google Landmark 公式](https://github.com/cvdfoundation/google-landmark#google-landmarks-dataset-v2)
- コード MIT、VIBE 配布 embedding CC BY 4.0、embedding model `nomic-ai/nomic-embed-vision-v1.5` は Apache-2.0 を別々に記録する。元画像やモデルを取得・実行せず、既存 embedding を使うので encoder 費用はゼロ。[VIBE dataset card](https://huggingface.co/datasets/vector-index-bench/vibe)、[Nomic 公式 model card](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- 生成処理は index 画像全体から seed 42 の random split で query 1,000 を分ける。ランドマーク ID 単位の group holdout ではない。クラスタ汎化評価の validation/test は corpus から別に固定し、query は任意 IVF 評価用に保持する。意味的な重複や近接画像の独立性までは保証しない。
- 正規化入力で通常の Euclidean SSE を比較することは可能で、`spherical=True` は必須ではない。中心も単位長へ再正規化すると目的関数が変わる。付属 cosine 近傍表を、変形した入力・縮小した base・異なる距離尺度へそのまま流用しない。

設計上の判断: 現代的な高次元画像の主候補。元画像の条件が ImageNet や AGNews より明確で、再 embedding を不要にできる。ただし VIBE 作者が示す vector-index benchmark という想定用途は AGNews と共通であり、中心学習との関係を記録する。

### DEEP1M: 原著に近い image embedding の最小候補

- Yandex 原配布は GoogLeNet の image descriptors を PCA で 96 次元化し L2 正規化した既存 embedding。L2 距離を使う。元画像・モデル推論の取得は不要。[Yandex 原著者の説明](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)（2021-04-26）
- 1B artifact は 384,000,000,008 bytes あるが、e-means の 1M 条件は最初の 1,000,000 行を学習、次の 10,000 行を評価に使う。必要 prefix は **387,840,008 bytes**。原著コードは評価 10,000 行を seed 1234 で 5k validation / 5k test に分け、fingerprint を検査する。[Faiss paper_data](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/benchs/e_means/paper_data.py)、[download source](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/benchs/e_means/paper_download.py)（release 2026-11-10）
- 原配布 URL は Yandex の `base.1B.fbin`。上流 e-means が参照する HF mirror は `suchun/deep1b-mirror`、revision `d0df68376fa478c558f2204fbc77b418431f4453`、2026-05-22 作成・更新、public / non-gated。HF_TOKEN は必須ではない。取得時は Range が守られたこと、byte 数と fingerprint を検査し、1B 全体へ黙って拡張しない。[mirror card](https://huggingface.co/datasets/suchun/deep1b-mirror)、[metadata](https://huggingface.co/api/datasets/suchun/deep1b-mirror)
- 公開データ提供を含む Big ANN 公式カタログは Yandex DEEP を CC BY 4.0 と記載する。mirror のラベルだけを権利根拠にしない。[NeurIPS 2021 Big ANN datasets](https://big-ann-benchmarks.com/neurips21.html#datasets)（2021 年公開、参照日 2026-11-17）
- 原著の prefix 分割をそのまま使うなら再現性に有利だが、一般化を測る random/group split とは別条件。1M の索引に対して 1B / 10M 用の既存近傍表を転用しない。IVF 用 10k query file は追加 3,840,008 bytes で、対象 base の正解近傍を別に作る。

設計上の判断: 原著との接点を持つ低次元 image embedding の対照候補。AGNews / Landmark より低次元・小容量だが、96 次元圧縮特徴なので、これだけで高次元 image embedding 全体を代表しない。H100 の所要時間は V100 へ移せない。[原著再現 README](https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/benchs/e_means/README.md)

### Fashion-MNIST: 安価な補助・数値確認候補

- Zalando 原配布。train 60k / test 10k、28×28 grayscale、10 class。`train-images-idx3-ubyte.gz` 約 26 MB、`t10k-images-idx3-ubyte.gz` 約 4.3 MB、label gzip 2 本は合計約 34 KB。公式 GitHub `data/fashion` または公式 download URL を使う。HF_TOKEN 不要。[公式 README](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data)（2017 年公開、参照日 2026-11-17）
- 公式 repo は MIT を示し、データも同 repo で配布する。[LICENSE](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE)
- 生画素を flatten / 固定 scale するだけなら embedding 生成は不要。ただし neural image embedding ではない。既存の train から validation を分け、公式 test を探索に使わない。ラベルは補助解釈に使えるが、k=256 のクラスタと10服種を一対一にする根拠にはならない。
- 全 70k の FP32 化は 219,520,000 bytes、約209 MiB。最小動作・距離評価・保存再読込に向く。現代的 embedding に対する性能の主証拠にはしない。

今回のユーザー要件なら、Fashion-MNIST を第4の主評価データとして加える必要はない。小規模数値確認は制御可能な synthetic で代替できる。

### CPU / V100 費用の評価方法

実測なしに分・GPU-hour は予測しない。上記は既存ベクトルまたは画素を使うため、encoder 推論費用はゼロ。バッチ数、k、precision、評価回数、thread 配置が未確定なので、データ容量が収まることと計算が安いことを同一視しない。

| 入力 | FP32 ベクトルのみ | dense assignment 1 pass の主乗算量例、k=2,048 |
| --- | --- | --- |
| AGNews 全 corpus | 3,151,388,672 bytes ≈2.935 GiB | 約3.23×10^12 FLOP |
| AGNews 250k 部分集合 | 1,024,000,000 bytes | 約1.05×10^12 FLOP |
| Landmark 全 corpus | 2,337,045,504 bytes ≈2.177 GiB | 約2.39×10^12 FLOP |
| DEEP 1M | 384,000,000 bytes | 約3.93×10^11 FLOP |
| Fashion-MNIST 全70k | 219,520,000 bytes | 約2.25×10^11 FLOP |

FLOP は単純な `2*n*k*d` の算術見積りで、更新・revival・top-k・メモリ帯域・実装 overhead・評価器を含まない。全候補はベクトル自体なら V100 32GB に十分小さいが、距離行列は chunk 化が必要。8 GPU の独立 trial 並列化は総経過時間を減らす可能性があり、1 trial の speedup には数えない。CPU 比較は GPU trial の読込や BLAS と競合しない配分で行う。

## 4. 比較対象の推薦

| 手法 | 同じ目的への比較可能性 | Python 3.14 / V100 / license・費用 | 推薦 |
| --- | --- | --- | --- |
| Faiss CPU Lloyd | FP32 Euclidean SSE の基準。既定の点数上限・再開始数を明示する | faiss-cpu wheel を共用、MIT | 必須 |
| MiniBatchKMeans | mini-batch 更新の強い既存対照。同じ最終 SSE を外部評価 | sklearn cp314 wheel、CPU、BSD-3-Clause | 必須の派生対照 |
| KMeans / Elkan | triangle inequality による exact k-means 高速化。明確に分離したクラスタ等で意味がある | sklearn を共用、CPU。追加 `(n,k)` 配列が必要 | 代表条件で追加。大きい n,k はメモリ見積り後 |
| native Torch Lloyd | native FP32 e-means と device / arithmetic を揃え、EMA 更新則の効果をみる | 同じ torch cu126、V100 実行確認が必要 | GPU の機構対照。最速実用 GPU 対照とは区別 |
| SuperKMeans 0.2.0 | flat/f32、同じデータ量、同じ停止予算なら候補。既定 hierarchical / sampling / recall 停止は別設定 | cp314 wheel 無し。source build は可能性あり。MIT 本体とは別に Eigen/MPL2 と任意 FFTW/GPL がある | 条件付き追加。現在の repo の copyleft 禁止を独断で越えない |
| Flash-KMeans 0.3.1 | exact Lloyd の GPU 最適化対照として意義あり | Python metadata は>=3.8だが現行 Triton は NVIDIA CC8.0+。V100 は外れる。LICENSE 本文 Apache-2.0、classifier MIT は不整合 | V100 主比較から除外。別 GPU を研究する場合の候補 |

MiniBatchKMeans は `tol=0` だけでは early stop が止まらず、`max_no_improvement=None` も必要。`compute_labels=True` は最後の全件ラベル評価を追加する。共通 initial centers、n_init、実処理 batch/行数、最終評価費用を揃える。[公式 API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)（1.9.1、参照日 2026-11-17）

Elkan は追加 `(n,k)` 配列を使い、250k×2,048 の FP32 配列だけで約2.05 GB。高次元ほど常に速いとは言えないため、条件別曲線に向く。[公式 KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

Faiss の標準 `max_points_per_centroid=256` は **n > 256×k** で学習点数に影響する。例えば n=250k, k=2,048 は上限524,288未満なので、この条件だけで sampling が生じるとは言えない。全条件で意図した学習点数を設定・記録する。[公式 Python clustering wrapper](https://github.com/facebookresearch/faiss/blob/v1.16.0/python/extra_wrappers.py)

Flash-KMeans の依存は torch / triton で、一般的な Python 下限だけでは V100 対応を示さない。[pyproject](https://github.com/svg-project/flash-kmeans/blob/main/pyproject.toml)、[Triton 対応ハードウェア](https://github.com/triton-lang/triton#compatibility)、[LICENSE 本文](https://github.com/svg-project/flash-kmeans/blob/main/LICENSE.txt)（各参照日 2026-11-17）

### GPU 機構対照を置く価値と最小範囲

実用比較と機構比較を二層にする案を支持する。CPU の既成ライブラリ比較は実用性を測り、V100 の共通 native FP32 assignment を使う e-means / Lloyd 比較は、この演算環境での更新則の差を調べる。後者から最速の GPU k-means 全般に対する優位性は主張しない。

既存の `faiss.contrib.torch.clustering` はそのままでは native Torch 対照にならない。通常の検索は `faiss.knn`、GPU subclass は `faiss.knn_gpu` を呼ぶ。さらに既存 `contrib.clustering.kmeans` は各反復で割当を CPU NumPy に戻し、`np.bincount` と count の GPU 転送を行う。この上流 loop を流用するだけでは、転送条件の異なる対照になり得る。[Torch wrapper](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/torch/clustering.py)、[既存 Lloyd loop](https://github.com/facebookresearch/faiss/blob/v1.16.0/contrib/clustering.py)

最小の機構対照は、固定版の上流 assignment helper を共有し、全件の sum/count から中心を更新する Lloyd loop と明示的な空 cluster 処理に限定する。共通 initial centers、FP32、入力順、chunk、同じデータ常駐条件を使い、e-means 本体は再実装しない。検証は小規模データで1反復の割当・和・count・中心を独立参照と照合し、重複点・空 cluster・最終中心での MSE、CUDA 同期を含む計時を確認する。空 cluster の復活方針は手法差として記録し、機構差が EMA だけだと言い切らない。

この対照の正しさ・転送範囲を確認できなければ、GPU 結果は e-means の実行可否・設定感度・品質時間曲線までに留める。CPU と GPU の時間差を e-means のアルゴリズム speedup と呼ぶより、結論の範囲を限定する方が適切である。

### SuperKMeans の source build を読むと何が分かるか

0.2.0 は 2026-08-28 公開、PyPI の cp314 wheel は無い。build 設定自体も cp39–cp313 まで。source build を初期範囲から外すことは設計上の選択肢であり、ユーザーの禁止事項ではない。[PyPI metadata](https://pypi.org/pypi/superkmeans/0.2.0/json)、[配布元 SHA の pyproject](https://github.com/cwida/SuperKMeans/blob/1610da9bf3877cfcfa71a6ee8bec893c1c568864/pyproject.toml)

- `scikit-build-core>=0.3.3`, `pybind11>=2.10.0` なので、Python3.14 を正式対応する pybind11 3.x を選べる。pybind11 3.0.0 の対応追加は 2025-07-10。ただし SuperKMeans 全体のビルド成功は未検証。[pybind11 changelog](https://pybind11.readthedocs.io/en/stable/changelog.html#version-3-0-0-july-10-2025)
- CMake≥3.26、OpenMP、BLAS が必要。portable x86 は AVX2/FMA、非portable は既定で `-march=native`。今回 `lscpu` では avx2/fma が可視だった。GPU は既定 OFF。
- NumKong / cpuinfo / ruy を main/master から FetchContent する箇所があり、SuperKMeans の SHA 固定だけでは全ビルド入力が固定されない。BLAS 不在時には OpenBLAS source build へ進む。これらは導入費用と再現性の具体的な追加作業である。[CMake](https://github.com/cwida/SuperKMeans/blob/1610da9bf3877cfcfa71a6ee8bec893c1c568864/CMakeLists.txt)
- FFTW は `SKMEANS_SKIP_FFTW=ON` で除外できるが、Eigen は直接依存。Eigen は MPL2 の file-level copyleft、FFTW は GPL。現在の AGENTS.md の「copyleft 禁止」との整合を先に決める必要がある。[submodules](https://github.com/cwida/SuperKMeans/blob/1610da9bf3877cfcfa71a6ee8bec893c1c568864/.gitmodules)、[Eigen 公式](https://libeigen.gitlab.io/pages/licensing_faq/)、[FFTW 公式](https://www.fftw.org/faq/section1.html#isfftwfree)（FFTW FAQ 更新 2014-03-04）
- 既定では n>100k で hierarchical、flat では sampling_fraction が 0.3、点数上限 256/cluster、early termination が有効。MIT 本体の「100×」等を、そのまま全件 FP32 の同条件結果とみなさない。[Python API](https://github.com/cwida/SuperKMeans/blob/1610da9bf3877cfcfa71a6ee8bec893c1c568864/python/superkmeans/__init__.py)

## 5. 既存案の批判的評価と代替

既存案の良い点は、CUDA の dtype/backend、最終中心での独立再割当、共通初期値、学習と E2E 計時、IVF query と MSE holdout の区別を明示していること。ただし次を更新する必要がある。

1. P2 の単一 n,d,k と AGNews だけでは「できる条件・できない条件」を十分説明できない。合成データで n / d / k / 分離度 / 不均衡を制御し、実データで外的な傾向を確認する。異なる実データ間の差を d の因果効果とは呼ばない。
2. e-means@5 と Lloyd@25 の非劣性一点を主結果にしない。複数予算点の最終 MSE 対実時間、複数の品質目標への到達時間、未到達を含む Pareto を示す。目標や曲線予算点を test 結果から都合よく選ばない。
3. 10 seed / bootstrap 10,000 回 / 特定 CI 幅を機械的な完成条件にしない。独立 seed と同一 seed の反復計時を区別し、pilot のばらつきから必要な反復数と coverage を事前に設計する。固定品質幅を置かないというユーザー決定を反映する。
4. GPU と CPU の時間差は、更新則以外に hardware とライブラリ実装の差を含む。同じ device の比較と「この環境で最も実用的な選択」の比較を別の表にする。8 GPU 並列時の CPU/IO 競合も記録する。
5. IVF は任意評価になったので、初期実装に recall tuning / query split / 全 top-100 再生成 / RAG 評価を必須化しない。意味的検索品質と ANN recall も別の概念である。
6. 時間上限がないことは終了条件がないことを意味しない。事前の条件表、独立反復数、失敗時の扱い、曲線の停止規則を満たしたら終了し、無期限に手法・データを追加しない。

比較可能なアプローチ:

| 案 | 内容 | 得られる答え / 費用 |
| --- | --- | --- |
| A. 最小の更新則・実用比較 | synthetic + DEEP1M、CPU: e-means/Faiss/MiniBatch/Elkan、GPU: native e-means/Lloyd | 小さく因果を整理しやすい。高次元 text embedding の証拠は薄い |
| B. 条件範囲を広げた比較（現在の主目的に推薦） | A に高次元 Landmark と許諾確認済み text embedding を加え、n,d,k/分布の少数代表条件を層別化。独立 trial を8GPUに配る | 得意不得意の境界と再現ばらつきを示せる。全直積ではなく検証したい相互作用を選ぶ |
| C. IVF へ接続する追加評価 | B の凍結した中心を使い固定 base/query で IndexIVFFlat、構築時間と recall/latency を評価 | 用途の有効性を追加確認。主クラスタ評価の完了条件にはしない |

KISS: dataset 読取、method 呼出、共通評価、結果保存の少数の境界で足りる。YAGNI: 最初から分散 e-means、一般的な plugin registry、全 VIBE runner、外部 API encoder を作らない。DRY: 距離尺度・split manifest・最終 MSE・計時規約は共通化する。SOLID: 各 adapter は共通の最終 centroid と計時記録へ変換し、method 固有の early stop や labels の意味を評価器に漏らさない。これらは設計提案であり、コードの生成指示ではない。

PIE については一般的な唯一の定義を仮定しない。親が確認したユーザー指定参考 repo の §0.3.4 にある `Proper Indentation, English etc` という可読性原則として扱う。[参考設計](https://github.com/tomo0530/self-evolve-agent-test/blob/main/docs/plan/design/00_overview.md)。本担当はこの参考本文を独立取得しておらず、この定義だけ親からの確認事項として記録する。

## 6. ユーザーが決めることと、調査・実装で解くこと

| ユーザーの意図・設計判断 | こちらで事実確認・検証する事項 |
| --- | --- |
| text / image / synthetic の必須3種に対する具体的 coverage と追加候補（技術判断は委任済み） | 配布 artifact、件数、dimension、license、revision、download 条件 |
| practical default と条件を揃えた更新則比較の解釈（親は二層構成を提案中） | 各 API の初期化・sampling・early stop・finalization の実際 |
| CPU/GPU の評価対象と試行の完成条件 | Python3.14 wheel/import、SM70/GEMM/native e-means、CPU SIMD/threads |
| SuperKMeans を含める価値と現 repo license 方針 | Eigen/FFTW の依存経路、build 入力固定、CPython3.14 build 成否 |
| IVF 追加評価へ進むか | query/base の対応、exact ground truth、recall/latency 実装 |

既に回答された主目的、8GPU使用、固定時間上限を置かない方針、固定品質マージンを置かない方針を聞き直さない。依存の導入可否や具体的な設定変更は、設計の合意後に具体的な差分として扱う。

## 7. 後続協議の追記: 主テキストを arXiv にする案

親から Carbon の独立調査要旨が共有され、主 text を VIBE arXiv、主 image を VIBE Landmark、DEEP96 を原著接点の追加条件とする提案があった。この選定を支持する。上記の独立候補調査は残し、AGNews を採る必然性はないと判断する。

追って本担当も一次資料を照合した。`arxiv-nomic-768-normalized.hdf5` は corpus 1,344,643 × 768、既存正規化 embedding。固定 HF revision `07b387891a221b7b073b83d2f752b76462e5fa03` の artifact は 4,135,431,488 bytes、公称 SHA-256 `8be0993b978b0d0ef023d21d878251a5ed09e058adb25994553c08388d37d414`。VIBE の public / non-gated 配布物で encoder 推論・HF_TOKEN は通常不要。[固定 tree metadata](https://huggingface.co/api/datasets/vector-index-bench/vibe/tree/07b387891a221b7b073b83d2f752b76462e5fa03?recursive=false&expand=false)、[公式カタログ](https://github.com/vector-index-bench/vibe#evaluation)（参照日 2026-11-17）

生成コードは `ejaasaari/arxiv-abstracts-2025-04-13` の text を使い、`nomic-embed-text-v1.5` に `clustering:` prefix を与え、layer normalization と L2 normalization を行う。モデルを再実行する必要はない。[生成コード](https://github.com/vector-index-bench/vibe/blob/main/vibe/datasets.py)

arXiv 公式 API 規約は descriptive metadata を CC0 とし、脚注で abstract を含むと明記する。論文本文・PDF の個別ライセンスと区別でき、今回の abstract embedding では AGNews 元ニュースの未照合条件を避けられる。VIBE の embedding CC BY 4.0 は別に保持する。[arXiv 公式 API 規約と脚注](https://info.arxiv.org/help/api/tou.html)（公開日表示なし、参照日 2026-11-17）

条件設計では arXiv と Landmark を同じ d=768 でも異なる分布として扱う。同じ Nomic 系モデルを用いた2データだけで全 embedding family を代表するとは主張せず、synthetic の d / 分布操作と DEEP96 の追加条件で解釈範囲を補う。通常 L2 SSE を共通目的とし、単位入力だから spherical k-means と同じだとは扱わない。

## 調査範囲の記録

read-only Git、設定の限定読取、一次サイト browse、公開 package / HF metadata、Faiss wheel 末尾 ZIP directory のみ取得。データ本体の download、Python/uv/make、インストール、source build、実験、外部メッセージ送信、commit/PR は行っていない。HF_TOKEN と env ファイルの値は読んでいない。変更は本ファイルのみ。既存の未コミット変更は保持した。
