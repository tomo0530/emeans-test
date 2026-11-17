# Gemini 4 Carbon (High) の独立データセット調査

2026-11-17。`agy --model gemini-4-carbon-high --sandbox --mode plan --print-timeout 15m --output-format json` が終了0、status SUCCESS、非空 response、stderr空で完了。847.57秒。Highはモデルvariant。CLI JSONのmodelフィールドはなく、起動指定をモデル選択の証跡とする。以下は成功した response の原文。調査開始後のユーザー決定やpeerの回答は入力に含まれていない。

## 親が照合した採用点と修正

- arXiv要旨のCC0は、回答中のS3全文ページでなく [arXiv API利用条件](https://info.arxiv.org/help/api/tou.html) のdescriptive metadata節と脚注（abstractを明記）で確認。論文全文の許諾とは別。Nomic textはVIBE生成コード上v1.5。配布embeddingはHF cardのCC BY4.0、コードMITを区別する。
- HF revision `07b387891a221b7b073b83d2f752b76462e5fa03` のtree metadataを親が確認: arxiv HDF5は4,135,431,488 bytes、LFS SHA256 `8be0993b978b0d0ef023d21d878251a5ed09e058adb25994553c08388d37d414`。Landmark Nomic HDF5は2,341,733,696 bytes、LFS SHA256 `9e870c21f3a33cea1adb9055ce0e7af0988557b5e4e21485a661f40365be286a`。全体ファイルはまだ取得していない。
- 正規化したデータ点同士の近傍順位と、長さを制約しないL2 k-means中心への割当を区別する。入力が単位ベクトルでも通常L2 k-meansとspherical k-meansの最適化は同一ではない。主比較は通常L2に統一する。
- V100のe-meansはnative FP32で確認する。FP16推奨、GTを数秒で作れるという時間推測、商用可能性1%による判断規則は採用しない。
- Nomicの事前学習集合の重なりを今回のクラスタリング手法間の不公平と自動的に扱わない。配布ベクトルを全手法で固定し、厳密重複group分割を自前確認する。意味的重複の除去完了とは主張しない。
- AGNewsとQdrant DBpediaは追加の元データ条件・規約があるため、主データに必須化しない。arXiv Nomicを主文章データ、別途照合したLandmark Nomicを主画像データとする親の選択を設計に反映する。

---

## 1. 推薦

主目的（商用/非商用、クラスタ単体評価かIVF検索評価までか）および予算が未確定のため、単一候補への決め打ちを避け、条件別の優先順位を提案する。

* **第一推薦（デフォルト最有力）: VIBE `arxiv-nomic-768-normalized`**
  * **[事実]** `e-means` 論文（arXiv:2611.09501v1）で評価されたVIBE収録データであり、元データ（arXiv書誌・要旨: CC0-1.0）→埋め込みモデル（Apache-2.0）→配布HDF5（CC-BY-4.0）まで非商用制限やShareAlikeを含まない。`train`（約134万件, $d=768$）に加え `test`（1,000件）と厳密top-100近傍GTを同梱する。
  * **[推論]** L2正規化済みのため、Euclidean MSE最小化とCosine/内積IVF検索の近傍順位が数学的に一致し、用途未確定段階の基準データとして最も手戻りが少ない。
* **条件付き推薦A（非商用研究確定・非正規化 $L_2$ 重視）: VIBE `agnews-mxbai-1024-euclidean`**
  * **[事実]** `e-means` 論文でも使用された $d=1024$・非正規化Euclideanデータ（GT同梱）だが、元データAG Newsが研究・非商用限定を明記する。
  * **[推論]** ベクトルノルムが一定でない一般的な $L_2$ 空間でのEMA更新・dead-cluster revival検証に有効であり、非商用と確定すれば最有力となる。
* **条件付き推薦B（極小DL開始・高次元 $d=1536$ 重視）: `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`**
  * **[事実]** 26分割Parquet（1シャード約367MB・約3.85万件）で配布され、全体DL不要で小さく始められる。ただしHF配布版はquery/GT非同梱で、元データ由来のCC-BY-SAとOpenAI規約が絡む。
* **小規模synthetic併用の意義**: **[推論]** 既知の真のクラスタ中心を持つ小規模合成データ（例: $n=10^4\text{--}5\times10^4, d=1024$ の混合ガウス等）を最初に併用すれば、通信ゼロ・権利制約ゼロでPython 3.14やV100（SM70）上の数値精度・EMA収束・バグ有無を、実embedding特有の異方性やhubnessと切り離して検証できる。

---

## 2. 比較表

| 項目 | 候補1: `arxiv-nomic-768-normalized` (VIBE) | 候補2: `agnews-mxbai-1024-euclidean` (VIBE) | 候補3: `Qdrant/dbpedia-entities-openai3-...-1536-1M` |
| :--- | :--- | :--- | :--- |
| **実artifact / 形式** | **[事実]** `arxiv-nomic-768-normalized.hdf5`（`/train`, `/test`, `/neighbors`, `/distances`, `/avg_distances`） | **[事実]** `agnews-mxbai-1024-euclidean.hdf5`（候補1と同一HDF5内部構造） | **[事実]** `0.parquet`〜`25.parquet` 計26ファイル（列: `_id`, `title`, `text`, `...-embedding`） |
| **`n` / `d` / `metric`** | **[事実]** `train`: 1,344,643 / `test`: 1,000、$d=768$、L2正規化済み（`any`: L2/IP/Cosine対応） | **[事実]** `train`: 769,382 / `test`: 1,000、$d=1024$、`euclidean`（非正規化L2） | **[事実]** `train`: 1,000,000 / `test`: なし、$d=1536$、正規化済み（Cosine/IP/L2対応） |
| **容量** | **[推論]** `train` 生配列（FP32）約4.13GB（実HDF5個別容量は未確認；リポジトリ全体約271GB **[事実]**） | **[推論]** `train` 生配列（FP32）約3.15GB（実HDF5個別容量は未確認） | **[事実]** 全体約9.55GB **[推論]** 1シャードあたり約367MB（約38,461件） |
| **元データ / モデル / 配布条件** | **[事実]** 元: arXiv要旨（CC0-1.0）/ モデル: Nomic Embed（Apache-2.0）/ 配布: CC-BY-4.0（コードMITとは別） | **[事実]** 元: AG News（**非商用・研究限定**）/ モデル: `mxbai-embed-large-v1`（Apache-2.0）/ 配布カード表記: CC-BY-4.0 | **[事実]** 元: `BeIR/dbpedia-entity`（CC-BY-SA-4.0、本文同梱）/ モデル: OpenAI API規約（競合モデル開発禁止・分類整理例外）/ 配布カード表記: MIT |
| **取得方法 / 環境変数** | **[事実]** HF公開（非gated）。単一HDF5をHTTP/`huggingface_hub`取得。`HF_TOKEN`必須ではない | **[事実]** 候補1と同じ（非gated、`HF_TOKEN`任意） | **[事実]** HF公開（非gated）。必要シャードのみ単体HTTP取得可。`HF_TOKEN`任意 |
| **分割 / GT / subset時再計算** | **[事実]** corpus/query分離・厳密top-100 GT同梱。**[推論]** `train` subset時は既存GTが無効化されるためIVF評価にはGT再計算が必須 | **[事実]** 候補1と同じ（top-100 GT同梱）。**[推論]** subsetでのIVF評価時はGT再計算が必須 | **[事実]** corpusのみ（GT非同梱）。**[推論]** 別シャードからqueryを分離し `IndexFlat` 等でGT計算が必要 |
| **データ漏洩リスク** | **[事実]** VIBEで重複除去・分割済。一方Nomic事前学習（arXiv:2402.01613）に `S2ORC Title Abstract` 3,544万対を含む | **[事実]** VIBEで重複除去・分割済。`mxbai`（7億対超学習）のAG News含有有無は非公開 | **[推論]** 自前分割時に重複エンティティ除去が必要。OpenAI学習データのDBpedia含有は非公開 |

---

## 3. 重要な根拠と公式URL/公開または更新日

コードのライセンス（Faiss/VIBEのMIT等）からはデータの利用許諾を一切推定せず、各層の一次情報のみを根拠とする。

1. **`e-means` 一次情報**
   * 論文: https://arxiv.org/abs/2611.09501v1 （初版提出: **2026-11-06** **[事実]**）
   * 実装: https://github.com/facebookresearch/faiss/tree/main/contrib/e_means （該当ディレクトリの個別最終更新日は **[未確認]**）
   * **[事実]** 論文の評価にVIBE（AG-News, arXiv等）やDBpedia-OpenAI系が含まれ、Lloyd法25 epoch相当のtest MSEを約1.4〜2.6 epochで達成している。
2. **VIBE（候補1・候補2）と上流ライセンス・漏洩根拠**
   * 論文: https://arxiv.org/abs/2505.17810 （v1提出: **2025-05-23**、v2改訂: **2026-08-04** **[事実]**）
   * データ配布: https://huggingface.co/datasets/vector-index-bench/vibe （データライセンス: **CC-BY-4.0** **[事実]**、個別ファイルの最終更新日は **[未確認]**）／コード: https://github.com/vector-index-bench/vibe （MIT）
   * 上流原典: arXivメタデータ・要旨は https://info.arxiv.org/help/bulk_data_s3.html にて **CC0-1.0** と明記 **[事実]**。一方AG News（ https://huggingface.co/datasets/fancyzhx/ag_news ）はAntonio Gulli原典に基づき **研究・非商用限定** と記載されており **[事実]**、**[推論]** VIBE側のCC-BY-4.0表記のみを根拠に商用利用するのはリスクがある。
   * 学習データ重複: `nomic-ai/nomic-embed-text-v1`（ https://huggingface.co/nomic-ai/nomic-embed-text-v1 、Apache-2.0）の論文 https://arxiv.org/abs/2402.01613 Table 7は `S2ORC Title Abstract`（35,438,592対）や `AGNews`（409,600対）の事前学習含有を開示 **[事実]**。`mixedbread-ai/mxbai-embed-large-v1`（ https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 、Apache-2.0）は学習対の詳細内訳を非公開 **[事実]**。**[推論]** 固定ベクトル上のクラスタMSE・ANN recall評価では、corpus/query間の重複が除去されていればモデル事前学習重複が直接の不正リークにはならないが、分布の凝集度には影響し得る。
3. **Qdrant DBpedia（候補3）の根拠**
   * 配布先: https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M （作成: **2024-02**、カード表記: MIT **[事実]**、最終更新日 **[未確認]**）
   * 上流条件: 抽出元の https://huggingface.co/datasets/BeIR/dbpedia-entity は **CC-BY-SA-4.0** **[事実]** であり、Parquet内に `title`/`text` を含むため **[推論]** テキスト同梱再配布時はShareAlike条件に留意が必要。またOpenAI利用規約（ https://openai.com/policies/terms-of-use/ ）は出力による競合AIモデル開発を禁じる一方、データの分類・整理目的を例外として許容する **[事実]**。
4. **Subsetting時のGT再計算とCPU / V100（SM70）最小利用法**
   * **[推論]** クラスタ品質（train/test MSE）と速度のみの評価なら近傍GTは不要で、`train` 先頭5万〜10万件のスライスのみで即実行できる。IVF検索（recall@$k$）をsubset上で測る場合は、subsetに対し `test` 1,000件との厳密全探索（`IndexFlatL2` / `IndexFlatIP`）でtop-$k$ GTを再計算する（$n=10^5$ 規模ならCPU/V100で数秒以内）。
   * **[推論]** 共有・専有未確認のノードでは、まずCPUスレッド制限（例: 4〜8物理コア）または単一GPU（`CUDA_VISIBLE_DEVICES=0`、V100 32GB 1枚）で始めるのが安全である。3候補ともフルデータ（FP32で約3.2〜6.1GB）がV100 1枚の32GB VRAMに十分収まる。なおV100（SM70）はFP32/FP16に対応するが **ハードウェアBF16 Tensor Core非対応** **[事実]** のため、FP32またはFP16（累積FP32）を選択する **[推論]**。設定済みと申告のある `HF_TOKEN` は秘密値を扱わず環境変数経由のままレート制限緩和のみに用いればよい **[推論]**。

---

## 4. 未確認点

1. **要件・予算**: ユーザーの主目的（商用利用有無、IVF検索評価の要否）および許容通信量・ディスク容量・GPU時間。
2. **配布メタデータ詳細**: VIBE各 `.hdf5` 個別ファイルの正確なバイト容量・更新日時、および `arxiv-nomic-768-normalized` 生成モデルのマイナー版表記（一次情報間で `nomic-embed-text-v1` と `v1.5` の記述が混在）。
3. **Python 3.14互換性**: Python `>=3.14,<3.15` 固定環境における `faiss`（`contrib/e_means` 含む）・`torch`・`triton`・`h5py`・`pyarrow` の公式wheel提供有無。
4. **計算機専有とSM70カーネル挙動**: V100×8 / RAM 504GiB環境の専有可否・ローカル空き容量、および `contrib/e_means` のGPUカーネルがSM70＋driver 535上でFP16/FP32フォールバック動作するか。

---

## 5. 次の具体的な確認

1. **設計協議での用途確定**: 商用可能性が1%でも残る場合は `agnews` を外して `arxiv-nomic-768-normalized` を主軸とし、極小DL優先でQdrant版を使う場合はベクトル列のみ抽出して `title`/`text` を扱わない運用を確認する。
2. **Python 3.14＋V100最小疎通確認（データDL前）**: Python 3.14環境で `faiss.contrib.e_means` と読み込み系（`h5py` または `pyarrow`）が導入可能か、合成データ（$n=10^4, d=1024$, FP32/FP16）がCPU少数コアおよびV100 1枚で完走するかを最初に確認する。
3. **単一ファイルのサイズ確認と段階評価**: Hugging Face Hub APIで対象1ファイルの実サイズを確認した上で取得し、`n=5万` subset（MSE・速度確認）→ 必要に応じsubset上で厳密GT再計算（IVF確認）→ フル1M級評価の順に進める。
