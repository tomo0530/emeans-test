<!-- Independent response preserved below. -->

> 作成: Claude Fable、指定 effort: xhigh。CLI正常終了と最終JSONのsuccessを確認。実モデル識別子: `claude-fable-5-2`。共通一次資料を渡したEvidence modeで、他担当の回答は参照していない。本文はモデルの独立案。根拠の追確認、配置・評価手順の修正、および最新研究の採否は [統合プラン](synthesis.md) を参照する。実装・実験は未実施。

---

# e-means 検証プラン（独立案・基準日 2026-11-17）

**凡例**: 【事実】渡された証拠に記載／【推論】証拠からの導出／【提案】本プランの設計判断／【未確認】ゲートで確認する点。**性能値はすべて未実測**です。

**出典**
- [P] https://arxiv.org/abs/2611.09501v1 （HTML: https://arxiv.org/html/2611.09501v1 ）
- [R] https://github.com/facebookresearch/faiss/tree/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means （README.md、torch_backend.py:490-541、e_means_torch.py:256-305、e_means_quantized.py）
- [E] 共通環境観測（2026-11-17）

**前提**: 本セッションでは原典URLを再取得しておらず、根拠は渡された抜粋のみです。README抜粋は "Several GPUs" の表の途中で切れています。

## 1. 提案の結論

採否は次の6点を測れば判断できます【提案】。

| # | 測るもの | 判断への寄与 |
|---|---|---|
| M1 | V100で実際に選択されるbackendとrepresentation、エラー有無 | そもそも検証可能か |
| M2 | 最終centroidに対する独立実装・FP32厳密割当のheld-out MSE | 品質 |
| M3 | train＋finalizeの壁時計時間、厳密assign込み時間、GPU/ホストのピークメモリ | コスト |
| M4 | seed間ばらつき、空クラスタ、imbalance、NaN | 安定性 |
| M5 | IVFでのrecall対コスト曲線と索引構築時間 | 用途価値（MSEとは別評価） |
| M6 | 依存のpin可能性、`exclude-newer`との衝突、license | 導入コスト |

最大のリスクは性能ではなく適合性です。V100では自動選択でTriton/CuTeに入らず、FP16/BF16は拒否され、FP32のPyTorch fallbackだけが残ると読めます（§5）。したがって互換性確認（P0）を最重要ゲートとし、通過しなければ「技術の否定」ではなく「現環境では判定不能」と報告します。

## 2. 原典の主張と未確認点

| 区分 | 内容 | 出典 |
|---|---|---|
| 事実（論文の主張） | EMAミニバッチ更新、cosineスケジュール、cluster revival、finalizationを提案。見出しは約1.4–2.6 epoch、10–45倍高速化 | [P] https://arxiv.org/abs/2611.09501v1 |
| 事実 | 論文のGPU計測はH100 80GBと最適化backend。held-out MSEとLloyd@25が参照基準 | [P] |
| 事実 | Sec3.2はlast-epoch-mean finalizationとslow EMAを区別し、finalizationの対象が結果に効くと述べる | [P] |
| 事実（READMEの主張） | Lloyd比で約10倍（epochと壁時計）かつ低loss、追加ハイパラなし。`nepochs=5`で通常Lloyd収束lossを下回る | [R] README |
| 事実 | TritonはA100以降、CuTeはH100/H200のBF16のみ。PyTorch backendはfallbackで「かなり遅い」 | [R] README |
| 事実 | 返却labelsは最終epoch中のもので、最終centroidの最近傍とは限らない。`em.obj`はcentroid移動中の誤差で最終誤差ではない | [R] README |
| 事実 | GPU経路はNaN/infを検査しない。`max_points_per_centroid`超過分はlabel=-1 | [R] README |
| 推論 | 論文のGPU値はBF16＋CuTeの可能性が高く、V100のFP32 fallbackとは精度もカーネルも異なる。壁時計の主張は移植不可。epoch効率はハード非依存に近く、V100でも検証可能 | — |

**未確認（すべてゲート化）**
- 見出し数値の定義と条件（どのn/k/d、何に対する倍率か）。
- 論文のアルゴリズムとpin版API（c8f2838）の差。
- `train()`が初期centroidを受け取れるか。
- 基底`Emeans`の`_automatic`の既定値（§5）。
- 必要な最小PyTorch版と、`benchs/e_means`にLloyd参照実装があるか。
- マルチGPU要件（README抜粋が途切れている）。
- Flash-KMeansとSuperKMeans（2026）は、出典・実装・licenseを新規確認するまで比較に入れません [P付録]。

## 3. 仮説と判定規則（実験前に固定）

以下の閾値はすべて【提案】で、論文値でも実測値でもありません。

**主仮説 H1（確証）**: V100 1枚・FP32・同一train集合・同一初期centroids・同一割当カーネル（§7のレーンA）で、e-means既定設定（`nepochs=5`）は全件Lloyd 25反復に対し、主要4セル（2データセット × k∈{4096, 16384}）すべてで次を満たす。
- (a) test MSE相対差 (e-means − Lloyd)/Lloyd の95%CI上限が +0.5% 以下（非劣性）。
- (b) 時間比 T_Lloyd@25 / T_e-means@5 の95%CI下限が 2.0 以上。
- (c) 事前登録したLloyd格子（全件/サブサンプル × niter）の中に、e-means@5より速く、かつtest MSEが0.5%以上低い構成がない。

(c)を置く理由は、Lloyd@25が弱い比較点になり得るためです（例えばLloyd@10がほぼ同品質なら(b)は形式的に通ってしまう）【推論】。

**棄却**: いずれかの主要セルで、(a)のCI下限が +0.5% を超える、または(b)のCI上限が 1.2 未満。
**保留**: 上記以外。seedを20まで一度だけ延長し、その時点の判定を最終とします。
マージン0.5%は、P1のseed間ばらつきを見た後、確証データ取得前に一度だけ見直せます（見直しは記録）。

**探索仮説（判定には使わず報告のみ）**

| ID | 問い |
|---|---|
| E1 | Lloyd@25のtest MSEに並ぶ最小`nepochs`はいくつか（論文の約1.4–2.6 epochのepoch効率部分を照合） |
| E2 | 高次元で`batch_law`のqualityとthroughputの差 |
| E3 | 最終epoch labelsと厳密assignの不一致率。「T epoch＋返却labels」対「T−1 epoch＋assign」（README主張） |
| E4 | `current_centroids()`、`finalize()`、Lloyd 1反復追加の3者のMSE差（Sec3.2） |
| E5 | 空クラスタ、imbalance、`n_splits`をLloydと比較 |
| E6 | 正規化埋込みでの`spherical=True` |
| E7 | CPU経路（`NumpyEmeans`）対`faiss.Kmeans` CPU |
| E8 | V100で`assignment_backend="triton"`を明示した場合に動くか（適合性の記録のみ） |
| E9 | `DistributedEmeans`の8×V100スケーリング（P2成功後のみ） |

**最終判定**

| 判定 | 条件 |
|---|---|
| 採用候補 | H1成立、レーンBの実用baselineにPareto支配されない、P3で非劣性 |
| 条件付き | H1成立だがレーンBで優位なし。SM80以上の環境での再検証を提案 |
| 見送り | H1棄却 |
| 判定不能 | P0不通過、またはタイムアウトやOOMが多発 |

## 4. フェーズ表

GPU時間は上限設定であり、所要時間の予測ではありません（未実測）。

| フェーズ | 目的 | 内容 | 通過ゲート | 上限【提案】 |
|---|---|---|---|---|
| P0 互換性確認 | 動く経路の確定 | §5の確認、原典の未確認点の解消、license確認、事前登録configの確定 | G0: FP32経路で`train`/`assign`が完走し、recipeを記録できる | 0.5 GPU時間 |
| P1 smoke | 正しさと計測系の検証 | 合成データ（d=16/64/128、n=20万、k=500程度）とSIFT小規模。独立MSE評価器の相互検証、timing手順、seed再現性 | G1: `em.assign`距離と独立評価器の相対誤差1e-4以下、label一致99.9%以上、NaNなし | 2 GPU時間 |
| P2 高次元埋込み比較 | H1の確証 | §6のデータ、§7のbaseline、§8の評価。1セル1 seedのパイロットで総時間を再見積り後に本実行 | G2: H1判定が確定 | 40 GPU時間 |
| P3 用途評価 | IVFでの価値 | §9 | G3: 非劣性判定 | CPU中心、10時間相当 |
| P4 拡張（任意） | 適用範囲 | n=500万〜1000万（batch loop）、マルチGPU、spherical | 別途承認 | 別途 |

## 5. V100適合確認とfallback

**コードから読める挙動**（[R] torch_backend.py:490-541、e_means_torch.py:256-305、e_means_quantized.py）

- 【事実】auto選択は `capability >= (8, 0)` のときだけTriton、`capability == (9, 0)` かつBF16のときだけCuTe。
- 【事実】FP16/BF16でSM80未満だと "requires an NVIDIA SM80 or later GPU" のRuntimeError。
- 【事実】`_resolve_input_dtype`は `_automatic` が真でCUDAのFP32入力ならBF16へキャスト。`QuantizedEmeans`はquantizer省略時に `_automatic=True`、`quantizer=None` 明示でnative FP32のまま。
- 【推論】V100はVolta（capability (7,0)、既知情報でP0で実測）なので、autoではFP32→PyTorch backendのみ。
- 【推論】`QuantizedEmeans(d, k)` にCUDA FP32を渡すと、BF16キャスト後にRuntimeErrorになる。
- 【未確認】基底`Emeans`の`_automatic`既定値。真なら同じエラーになるため `QuantizedEmeans(quantizer=None)` を使う。

**P0で記録する項目**【提案】
1. `torch.__version__`、`torch.version.cuda`、`torch.cuda.get_device_capability(0)`、`torch.cuda.get_arch_list()`（sm_70を含むか）。
2. FP32 CUDA入力での `em.recipe["assignment_backend"]` と `["representation"]`。期待はPyTorch backendとfloat32。
3. 負例として、BF16入力および`QuantizedEmeans`既定の例外メッセージ。
4. `assignment_backend="native"` 明示時と "triton" 明示時（E8）の成否。
5. `faiss.contrib.e_means` がインストール済みwheelに含まれるか。
6. README式（batch size、α0）と `em.recipe` の値の突合。

**fallbackの順序**: F1 GPU FP32 native → F2 sm_70を含む別のPyTorch/CUDA組合せ（e-meansの最小版要件が許す場合）→ F3 CPU経路（`faiss.knn`、FP32。READMEはn×k>1e10でGPU推奨なので規模を縮小）→ F4 停止し「判定不能、SM80以上が必要」と報告。

**driver / CUDA / PyTorch / Dockerの条件**

| 項目 | 現状 | 不確実性と確認 |
|---|---|---|
| driver | 535.154.05【事実 E】 | CUDA 12.x系はminor互換で動く見込みだが、私の記憶ベースで未検証。https://docs.nvidia.com/deploy/cuda-compatibility/ で確認 |
| ベースイメージ | nvidia/cuda:12.8.1-cudnn-runtime（digest固定）、Python 3.12.15【事実 E】 | PyTorch wheelは自前のCUDAランタイムを同梱するため、実際に効くのはdriverとwheelの組合せ【推論】 |
| CUDA 13系 | CuTe用依存は `[cu13]`【事実 R】 | V100では対象外。CUDA 13はVolta非対応と記憶しているが未検証。cu13ビルドのPyTorchは避ける |
| 依存 | uv.lockにtorch/faiss/triton/scikit-learnなし【事実 E】 | `exclude-newer="1 week"` により、e_meansを含むfaissリリースが未公開または除外される可能性。代替はpin SHAのcontribを別ディレクトリに置く方法（相対importで単体importできるか未確認）。依存変更は別途承認 |
| Docker | `/.dockerenv`あり、`docker`コマンドなし【事実 E】 | 既存コンテナ内の可能性。GPU composeの開発コンテナと同一か、`.venv_container`の実在は未確認。コンテナ内からcomposeは起動できない前提で、実行場所をP0で確定 |
| 資源 | GPU使用許可・空きは未確認。共有ホスト。/tmpは28G【事実 E】 | 実行前に `nvidia-smi --query-compute-apps` で占有確認。キャッシュとデータは作業ディスクに置く |

## 6. データ

すべて候補であり、入手・存在・license・revisionはP0で確認します【未確認】。

| 役割 | 候補 | d | 入手先 | license |
|---|---|---|---|---|
| smoke | 合成（ガウス混合、seed固定生成） | 16/64/128 | 生成スクリプト | 不要 |
| 古典ANN | SIFT1M | 128 | http://corpus-texmex.irisa.fr/ | 要確認 |
| テキスト埋込みA | Wikipedia系の公開埋込み（例: Cohere/wikipedia-22-12系） | 768 | Hugging Face Hub | 要確認 |
| テキスト埋込みB | DBpedia OpenAI埋込み1M（例: KShivendu/dbpedia-entities-openai-1M） | 1536 | Hugging Face Hub | 要確認 |
| 代替 | 許諾の明確なコーパスとオープンな埋込みモデルで自前生成 | 768–1024 | — | コーパスとモデルの双方を確認 |

- **選定理由**【提案】: `batch_law="auto"` はd≤16で挙動が変わる [R] ため、高次元（throughput則側）を主対象にします。単一分布への過適合を避けるため、埋込みモデルの異なる2種を使います。SIFTはANN文献との比較用です。論文の使用データとの対応はP0で確認します。
- **採用条件**: 社内研究利用を許す明示licenseがあること。revisionとsha256を記録できること。満たさなければ代替に差し替えます。
- **分割**: 固定seedでシャッフルし、完全重複をhashで除去した後、train 100万、validation 10万、test 10万。IVFのクエリはtestから1万件。
- **用途**: validationは探索的な設定選択のみ。testは設定凍結後に一度だけ評価します。
- **サイズ根拠**（計算値であり実測ではない）: FP32で100万×768次元は約3.1GB、100万×1536次元は約6.1GBで、V100 32GBにデータ・centroid・label（8 byte/点 [R]）が載る見込み。1000万×768次元は約31GBで載らないためP4（batch loop）に回します。
- **k**: 1024、4096、16384（1クラスタあたり約977、244、61点）。全件とサブサンプルの差が出る領域と、READMEのGPU推奨域（n×k>1e10）の境界をまたぎます。

## 7. baselineの公平性

**比較対象と選定理由**【提案】
- **レーンA（確証用）**: 同一割当カーネルのLloyd。e-meansと同じPyTorch FP32割当を使い、更新則だけを変えます。カーネル差を除いてアルゴリズム差を取り出すためです。上流`benchs/e_means`に参照実装があればそれを優先し、なければ自作して小規模で`faiss.Kmeans`と軌跡一致を検証します。
- **レーンB（採用現実性）**: `faiss.Kmeans` CPU（必須）と、導入可能ならfaiss GPU。現環境で実際に使う代替だからです。
- **探索**: scikit-learnのMiniBatchKMeans（同系統のミニバッチ法、小規模のみ）。

| 公平性の軸 | 取り決め |
|---|---|
| 全件/サブサンプリング | {e-means, Lloyd} × {全件, 256点/centroid上限} の2×2。faiss既定は256点/centroid上限のサブサンプルと記憶（pin版Clustering.hで確認）。e-meansは`max_points_per_centroid`で同じ上限 [R] |
| 初期centroids | 確証は共有初期値。trainから固定seedでk点を抽出し、両手法に同一行列を渡す。e-meansは `begin(n, init)` のbatch loop [R]。`train()`との等価性はP1で確認。併走でライブラリ既定初期化も記録 |
| precision | 両者FP32固定。`recipe["representation"]` を記録。BF16はV100では不可（§5） |
| epochs/restarts | e-meansは nepochs∈{1,2,3,5,8}、Lloydは niter∈{2,5,10,25,50}。B・α0がTに依存する [R] ため、途中経過を流用せずnepochsごとに独立実行。再起動は両者1回。等時間での複数再起動は探索扱い |
| finalization | e-meansは`finalize()`後のcentroids、Lloydは最終割当の平均。finalization時間は学習時間に含める |
| assignコスト | 「学習のみ」と「学習＋最終centroidへの厳密assign」を両手法とも別々に報告。e-meansの返却labelsをLloydの厳密labelsと同列に扱わない |
| 計測条件 | データは事前にGPU上へ置く [R]。`torch.cuda.synchronize()` で区切る。初回実行（コンパイル・初期化）は分離報告。CPUスレッド数は両者同一に固定。手法の実行順は交互化。対象GPUに他プロセスがあれば再実行 |

## 8. 評価プロトコル

- **独立MSE評価**: 両ライブラリとは別のコードで、最終centroidへのFP32厳密最近傍を求め、float64で集計します。小規模ではNumPy float64の素朴実装と一致することを確認します。`em.obj` [R] やfaissのobjは使いません。label=-1の点も評価に含めます。
- **対象**: train、validation、testそれぞれのMSE。主要指標はtestです。
- **時間・メモリ**: train、finalize、厳密assign、初回実行の各秒数。`torch.cuda.max_memory_allocated`、ピークRSS、実行前後のホスト負荷。
- **seedとCI**: 確証構成（e-means@5、全件Lloyd@25）は対応のある10 seed。曲線用の構成は5 seed。95%CIは対応差のt区間とbootstrap（1万回）の両方を出し、保守的な側で判定します。seedの一覧は事前に固定します。
- **タイムアウト**: 1実行の上限は、パイロットでの全件Lloyd@25所要時間の3倍（最大30分）【提案】。超過はstatus=timeoutとして記録して除外せず、時間は打切りデータとして扱います。OOMは`assignment_chunk_size`と`centroid_chunk_size`（メモリと速度のみに影響 [R]）を調整して再試行し、履歴を残します。
- **入力検査**: GPU経路は非有限値を検査しない [R] ため、投入前に `isfinite` を必須とします。

## 9. IVF用途（別評価）

MSEが低いだけで改善とは断定しません。リストの偏りが探索コストを変え得るためです【推論】。

- **構成**: 各手法のcentroidsをcoarse quantizerに固定した `IndexIVFFlat`（CPU、nlist=k）。量子化器を事前投入すれば学習が省略される挙動はP0で確認します。centroid以外は同一パイプラインです。
- **正解**: 総当たりの厳密top-10。
- **指標**:
  - recall@1/10/100、nprobe∈{1,2,4,…,128}。
  - 1クエリあたりの距離計算数（ndis）。共有ホストのノイズを受けない指標として併記します。
  - QPS（単スレッドとバッチ）、P50/P95/P99レイテンシ。
  - 索引構築時間（centroid学習＋add）、imbalance factor、最大リスト長。
- **判定**【提案】: recall@10が0.90と0.95になる点で、ndisとQPSがLloyd由来の索引に対し非劣性（−3%以内）であること。優越は構築時間込みのParetoで述べます。
- **注意**: クエリがtest由来の場合、分布はデータベースと同一です。実クエリを持つデータでの確認は探索扱いとします。

## 10. 実装予定配置・再現メタデータ・予算・停止/拡張

**配置**（今回は作成しません）
```
docs/research/emeans/        計画と最終レポート（日本語）
experiments/emeans/README.md AI向け手順（英語）
experiments/emeans/configs/  事前登録config（仮説・格子・閾値。実行前にコミット）
experiments/emeans/src/      env_probe / data / baselines / run_cluster / eval_mse / eval_ivf / aggregate
experiments/emeans/results/  run単位のJSONL（大容量はgit管理外、集計のみ管理）
```
結果schemaは既存依存のpydanticで定義します。

**再現メタデータ**
- リポジトリのSHAとdirty有無、faiss commit（c8f283867285af547db7a701c1b8dfa89631b3ea）と版。
- torch / CUDA / driver、GPU型番とcapability、CPUとスレッド設定。
- イメージdigest、uv.lockのhash、`exclude-newer`の基準日。
- データのrevision、sha256、分割seed、config hash、seed。
- `em.recipe` 全体、`iteration_stats`、`n_splits`、開始・終了時刻。

**予算上限**【提案】: P0〜P3合計で約52 GPU時間＋CPU 10時間相当、作業は約1週間。パイロット後の予測が上限を超える場合は、niter=50、k=1024、曲線用seed（5→3）の順に削ります。確証構成のseedは5未満にしません。

**停止基準**
- G0が全fallbackで不通過。
- G1で正しさの不一致。
- 5 seed時点で全主要セルが有意に劣り、かつ遅い（無益性による停止）。
- 予算超過。

**拡張基準**: H1成立でP3へ進みます。P3も非劣性ならP4と、SM80以上のGPUでのTriton経路検証を提案します。

## 11. 最終レポート目次と結果schema（未実施）

1. 概要と結論（採否と条件）
2. 技術解説（EMAミニバッチ、cosineスケジュール、revival、finalization）
3. 競合手法と選定理由（Lloyd、faiss、ミニバッチ系、未採用候補とその理由）
4. 実験目的と仮説（主仮説、探索仮説、事前登録した判定規則と変更履歴）
5. 共通baselineと公平性
6. データ、pinした実装、設定、指標
7. 結果
   1. 互換性
   2. smoke
   3. MSE対時間
   4. IVF
   5. 代表的なトレースと失敗例
8. 考察（実証できた主張と未検証の主張の分離、論文値との関係）
9. コスト・時間対品質のPareto
10. 採用基準と推奨アクション
11. 再現手順と付録

```json
{"run_id": null, "phase": null, "status": "未実施",
 "env": {"repo_sha": null, "faiss_commit": null, "torch": null, "torch_cuda": null, "driver": null,
         "gpu": null, "capability": null, "image_digest": null, "threads": null},
 "data": {"name": null, "revision": null, "license": null, "sha256": null,
          "n_train": null, "n_val": null, "n_test": null, "d": null, "split_seed": null},
 "method": {"name": null, "lane": null, "k": null, "nepochs_or_niter": null, "subsample_cap": null,
            "init": null, "seed": null, "dtype": null, "recipe": null},
 "cost": {"train_s": null, "finalize_s": null, "exact_assign_s": null, "cold_start_s": null,
          "peak_gpu_gb": null, "peak_rss_gb": null},
 "quality": {"mse_train": null, "mse_val": null, "mse_test": null, "label_mismatch": null,
             "n_empty": null, "imbalance": null, "n_splits": null},
 "ivf": {"nprobe": null, "recall_at_10": null, "ndis": null, "qps": null, "p95_ms": null, "build_s": null}}
```

| セル（データ×k） | 手法/設定 | test MSE（平均±95%CI） | 学習時間[s] | 時間比 | H1 (a)/(b)/(c) |
|---|---|---|---|---|---|
| 未実施 | 未実施 | 未実施 | 未実施 | 未実施 | 未実施 |
