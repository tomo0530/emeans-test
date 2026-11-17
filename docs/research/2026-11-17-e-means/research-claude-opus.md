> 作成: Claude Opus、指定 effort: xhigh。CLI正常終了・最終JSON success・permission_denialsなしを確認。主回答モデル: `claude-opus-5-5`。WebSearch/WebFetchによる独立調査で、他担当の回答は参照していない。本文は独立回答を保持。実装・実験は未実施。

> **親による検証注記**: APIの返却順、データ許諾、論文値の解釈には下記の修正が必要。実行判断は [統合プラン](synthesis.md) を優先する。固定SHAの `assign` の返却順は **(distances, labels)** であり、本文の逆順表記は採用しない。データセットは個別の一次ライセンス確認を完了するまで取得対象を確定しない。著者の保存済みH100/BF16 targetは条件一致時の参考照合に限り、V100/FP32での主要分母にはローカルで再計測したLloydを使う。要約ツール由来で直接確認できなかった速度値は採用しない。

---

# e-means（arXiv:2611.09501）をV100環境で検証するための調査報告（2026-11-17時点）

## 1. 推奨と重要な知見

**結論：** この環境では、論文にあるH100上のGPU速度（10〜45×）を再現することはできません。結果を論文値と比べることも避けるべきです。検証は次の3本に絞ることを勧めます。

- **(A) 同じマシンのCPU同士の比較（優先度：最高）**
  - 比較対象は`NumpyEmeans`と`faiss.Kmeans`（25反復、subsampling無効）です。
  - 論文のCPU主張（サイト表記ではFaiss比8〜15×）は、同じ機械の上で比として検証できます。
- **(B) ハードに依存しない品質の再現**
  - 固定されたtest split上でtest MSEを測り、`paper_targets.json`に入っている論文のFlash@25目標値（seed 10〜19）を何epochで下回るか（crossing）を確認します。
  - READMEには「test errorはどのハードでも再現できる」とあります。
- **(C) V100固有のGPU比較**
  - e-means（FP32 native経路）と、同じくFP32のLloyd（Faiss-GPU／素朴なPyTorch実装／fastkmeansのTorch backend）を比べます。
  - 結果は「V100・FP32 native経路での比」として、論文値とは分けて報告します。

**確認済みの事実**
- **e-meansのGPUカーネル選択：** CuTe経路は能力値が(9,0)ちょうどでBF16のときだけ、Triton経路は(8,0)以上が条件です。`assignment_backend="auto"`はV100（7,0）でFP32ならnative（PyTorch）に落ちますが、BF16/FP16だと`RuntimeError`になります（[torch_backend.py](https://raw.githubusercontent.com/facebookresearch/faiss/main/contrib/e_means/torch_backend.py)）。READMEでも、V100/T4で使えるのはFP32 nativeだけです（[README](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)）。
- **GPUでの既定の型変換：** FP32入力は既定で`CastQuantizer(d, torch.bfloat16)`によりBF16へ変換されます。FP32のまま使うには`QuantizedEmeans(d,k,quantizer=None)`を指定します（README）。
  - 【推論】V100で`Emeans`を既定のまま使うと、上のRuntimeErrorになる可能性が高いです。最初の動作確認でこれを確かめるべきです。
- **ベンチスクリプトの前提：** `paper_xb.py`には`assignment_backend="cute"`、`flash-kmeans==0.3.1`の厳密一致、`FLOAT32_MAX_BYTES=60GiB`（80GB GPU前提）が固定で書かれています（[paper_xb.py](https://raw.githubusercontent.com/facebookresearch/faiss/main/benchs/e_means/paper_xb.py)）。
  - 【推論】V100ではそのままでは動かず、リポジトリの外にラッパーを書く必要があります。
- **Tritonの対応GPU：** 公式要件は「Compute Capability 8.0+」です（[triton](https://github.com/triton-lang/triton)）。Tritonベースのe-means Triton経路、Flash-KMeans、fastkmeansのTriton backendは、いずれもV100では公式サポート外です。
- **PyTorchのVolta対応：**
  - 2.11からcu128ビルドにVolta（sm_70）が含まれなくなりました（2026-01-15告知、[dev-discuss](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)）。
  - 2.15からはcu126の配布も止まり、Voltaを含むprebuilt wheelは2.14が最後です（2026-09-04告知、[dev-discuss](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)）。
  - PyPIの公開日：2.14.1は2026-09-30、2.15.0は2026-10-28です（[PyPI](https://pypi.org/project/torch/#history)）。
  - 論文の環境はPyTorch 2.14、再現確認の環境は2.15.0です。
  - **→ V100ではtorch 2.14.x＋cu126を選ぶのが現実的です。** compose設定のCUDA 12.8.1に合わせてcu128 wheel（2.11以降）を入れると、sm_70向けのカーネルが入っていません。
- **ドライバ535：** CUDA 12.xのminor version compatibility（ドライバ525以上）の範囲内です。ただしPTXだけのバイナリは古いドライバでは動きません（[NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)）。sm_70のSASSを含むバイナリが必要です。
- **GPUメモリ：** 100M規模の実行はGPU上で20〜51GBを使います（[bench README](https://raw.githubusercontent.com/facebookresearch/faiss/main/benchs/e_means/README.md)）。V100単体（32GB）では1M／10M規模が現実的です。

## 2. 関連研究の比較表

| 名称 | 公開/更新日 | 主張 | コード/ライセンス | 現環境での適合性 | 採否 |
|---|---|---|---|---|---|
| [e-means論文](https://arxiv.org/abs/2611.09501) | v1 2026-11-06（版はv1のみ） | EMA＋mini-batch。1.4〜2.6 epochでLloyd@25に到達。GPUで10〜45× | 論文はCC BY 4.0。コードはFaiss内（MIT、リポジトリのライセンスに準ずると推定） | CPU経路は可。GPUはFP32 nativeのみ | **主対象** |
| [faiss contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)＋[benchs/e_means](https://github.com/facebookresearch/faiss/tree/main/benchs/e_means) | commitは2026-11-02〜11-11（[履歴](https://github.com/facebookresearch/faiss/commits/main/contrib/e_means)）、v1.16.0は2026-11-09/10 | 実装と論文再現スクリプト | MIT（[PyPI](https://pypi.org/project/faiss-gpu/)） | 毎日のように変更中。cute固定部分は改変が必要 | **採用（commitを固定）** |
| [Small Is Not Dead](https://arxiv.org/abs/2611.18238) | 2026-11-13 | 距離を考慮したクラスタ復活。小さく孤立した群でtest誤差が10〜80%減 | コード公開なし。CC BY 4.0 | CPUでの概念実証のみ | 保留（必要な場合のみ） |
| [Flash-KMeans](https://arxiv.org/abs/2603.09229)／[GitHub](https://github.com/svg-project/flash-kmeans) | v1 2026-03-10、v2 04-09。PyPI 0.3.1は2026-06-03 | 厳密なLloyd。H200で最大17.9× | GitHubはApache-2.0、PyPIの分類子はMIT（不一致） | Triton（CC8.0以上）が前提で、調整済み構成はH200/H100/A100/GB10のみ | **実行は除外**。保存済みの目標値だけ使う |
| [fastkmeans](https://github.com/AnswerDotAI/fastkmeans) | 2025（PyPI 0.5.0は2025-05-14） | Triton版は単一GPUのfaiss比で4〜5× | Apache-2.0 | Torch backendは動く見込み（未確認）。既定で1centroidあたり256点に間引く | 条件付きで採用（間引き無効、tol=-1） |
| [SuperKMeans](https://arxiv.org/abs/2603.20009)／[GitHub](https://github.com/cwida/SuperKMeans) | 2026-03-20。Faiss v1.14.2で統合 | CPUでFAISS比最大7×、GPUでcuVS比4× | MIT | CPUで使える | 2番手のCPUベースライン（任意） |
| Faiss k-means（Lloyd） | [v1.16.0](https://github.com/facebookresearch/faiss/releases) | 基準 | MIT | CPUは可。GPU wheelがsm_70を含むかは未確認 | **主ベースライン** |
| cuML KMeans | 論文はRAPIDS 25.02.01を使用 | 基準 | 未確認 | 26.x系でのVolta対応は未確認 | 任意 |
| [Staleness-Reduction mbatch](https://jjiantong.github.io/assets/pdf/tnnls23_kmeans.pdf) | TNNLS（DOI 2023） | mini-batch比で40〜130×速く目標到達 | 公式コードは未確認 | — | 除外（論文側も再実装） |
| [VIBE](https://arxiv.org/abs/2505.17810)／[HF](https://huggingface.co/datasets/vector-index-bench/vibe) | 2025-05-23 | 埋め込みANNベンチ | コードはMIT。HFカードはCC BY 4.0（元データごとの条件は別途） | 1M級なら可 | 採用（ImageNet-CLIPとIVF評価） |
| [big-ann系データ](https://big-ann-benchmarks.com/neurips21.html) | NeurIPS'21 | DEEP/SIFT/SPACEV/SSNPP | DEEPはCC BY 4.0、SIFTはCC0、SPACEVはO-UDA、**SSNPPはCC BY-NC 2.0** | 1M/10Mなら可 | 採用（SSNPPは非商用条件に注意） |
| [Vo+ 2024 データキュレーション](https://arxiv.org/abs/2405.15613) | 2024-05 | 階層k-meansでデータを均衡化 | 未確認 | — | 背景（下流評価の候補） |
| [Summer-22B](https://arxiv.org/pdf/2603.00173) | 2026-03（arXiv IDからの推定、本文は精読していない） | GPU mini-batch k-meansの応用 | — | — | 除外（応用例にとどまる） |

## 3. e-means原典の再現条件と注意点

- **API**
  - `Emeans(d,k,nepochs=5,batch_law="auto",seed=1234,…)`の`train(x)`は、centroidと最終passのlabelを返します。
  - `assign(x)`は（label、二乗距離）を返します。
  - CPUでNumPy配列を渡すと`NumpyEmeans`が使われ、`faiss.knn`（k=1、FP32）で割り当てます（README）。
  - nepochsは小数も指定できます（例：2.5）。crossingの探索では1.0、1.2、1.4…と予算を刻みます。
- **論文ベンチの設定（paper_xb）**
  - 初期centroidは、CUDA generatorによる`torch.randperm`の先頭k行です。
  - batch_lawは、2M行までが`throughput`、それより大きいと`quality`です。
  - **README側の`auto`は次元で決まり、d≤16で`quality`になります。** 基準が違うので明示的に指定してください。
- **dtypeと割り当て**
  - 論文のGPU割り当てはBF16（CuTe）で、EMA状態とバッチ和はFP32です。
  - Appendix Jによると、BF16は割り当ての0.1〜4%を変えますが、最終誤差への影響はわずかです。FP8/INT8はBF16から±0.13%以内です（[論文HTML](https://arxiv.org/html/2611.09501)）。
  - V100で使えるのはFP32だけです。FP32割り当てとBF16割り当ての差の大きさは、論文のどこにも明記がない点に注意してください。
- **subsampling**
  - e-meansの`max_points_per_centroid`は既定でNone（間引きなし）です。使うと、学習に使われなかった点のlabelは-1になり、scaling lawは間引き後の件数で計算されます。
  - 論文はFaissの自動subsamplingを無効にしています。fastkmeansも既定で256点に間引くので、無効にする必要があります。
- **train/test MSE**
  - test MSEは、固定した評価用20k行の後半を使い、最近傍centroidまでの二乗距離の平均です。探索はFP32・TF32無効で行い、距離はfloat64で差分から再計算します。
  - train誤差（Figure 3）は、各手法の最終passの割り当てで測った誤差です。
  - `em.obj`はcentroidが動いている途中の誤差で、**最終誤差ではありません**。
- **labels**
  - 最終passのlabelは、最終centroidと一致しないことがあります。正確なlabelが必要なら`assign()`をやり直します。
- **時間の測り方**
  - 1回の実行ごとに新しいプロセスを立てます。データ読み込み・初期化・GPU転送・採点は時間に含めません。
  - 計測しないwarm-upを先に1回走らせます（Flashは2反復）。
  - 計測は`torch.cuda.synchronize()`で挟み、2秒未満なら3回の中央値を取ります。
  - speedupは、Flash@25の時間をcrossing時点のe-meansの時間で割った値です。
- **Faissベースラインの版**
  - 論文はFaiss 1.12.0を使っています。v1.14.0でk-meansにearly stoppingが追加されたので、反復回数を固定してください（【推論】既定値は要確認）。

## 4. 2026年の一次資料と、それ以前の基礎文献

- **2026年の一次資料（日付を確認済み）**
  - e-means論文（11-06）、contribのcommit群（11-02〜11-11）、Faiss v1.16.0（11-09/10）
  - Small Is Not Dead（11-13）
  - Flash-KMeans（03-10／04-09、PyPI 06-03）
  - SuperKMeans（03-20）
  - PyTorchのVolta関連告知（01-15、09-04）
  - Deep1Bミラー（2026-05-22にミラー作成、[HF](https://huggingface.co/datasets/suchun/deep1b-mirror)）
- **2025年以前の基礎文献**
  - fastkmeans（2025）、VIBE（2025）、Vo+ 2024、Staleness-Reduction（TNNLS）、[Nested mini-batch（NeurIPS 2016）](https://proceedings.neurips.cc/paper/2016/hash/8d317bdcf4aafcfc22149d77babee96d-Abstract.html)、big-ann（2021）、Sculleyのmini-batch k-means（2010）
- 2026年のデータキュレーション研究で、e-meansとの比較に使える一次資料は見つかりませんでした。件数合わせのために追加はしていません。

## 5. 未確認事項・矛盾・次の行動

**矛盾と未確認事項**
1. **速度の数字がソースごとに違います。** 論文要旨は「SOTA比GPU 10〜45×」、サイトは「Flash比10〜45×・Faiss CPU比8〜15×」、READMEは「約10×」です。論文HTMLの後半（付録）を要約ツール経由で読んだところでは「GPU 8〜23×／CPU 9〜15×」でしたが、本文の表で直接は確認していません。
2. **Faissの版の記述が食い違っています。** bench READMEは「e_meansはまだリリースに入っていない」としていますが、v1.16.0のリリースノートにはe-meansの機能が並んでいます。
3. **demo出力の0.3423が二重の意味で使われています。** 「seed 10のFlash@25」と「Lloyd基準」の両方として書かれています。
4. **Flash-KMeansのライセンス表記が一致しません**（Apache-2.0とMIT）。
5. **未確認の点：**
   - Faiss-GPUの公式wheel（1.16.0）がsm_70を含むか
   - cuML 26.xがVoltaに対応しているか
   - VIBEの元データごとの利用条件
   - BF16とFP32の割り当てによる差の大きさ

**次の行動（すべて未実施）**
1. 環境を確定します。torch 2.14.x＋cu126を入れ、`get_device_capability()==(7,0)`と`get_arch_list()`にsm_70が含まれることを確認します。あわせてfaissのcommit（#5849以降）を固定します。
2. 動作確認として、`QuantizedEmeans(quantizer=None)`がFP32 nativeで動くこと、`Emeans`を既定のまま使うとエラーになるかを見ます。
3. (A)のCPU比較を、DEEP\* 1M、ImageNet-CLIP 1M、SIFT/SPACEV 10Mで行います（ダウンロードは4.5GBと21.7GB）。
4. (B)のcrossingを、`paper_targets.json`の目標値に対してseed 10〜19で行います。あわせて、自前のFP32 Lloyd@25も局所的な基準として測ります。
5. (C)のV100 GPU比較を行い、必要ならIVF Recall@10とnprobeの関係も調べます。

**除外するもの：** Flash-KMeansの実機実行、CuTe/BF16/FP8/INT8経路、DINO10B（約10TB、8×H100前提）、単一V100での100M規模。

Sources:
- https://arxiv.org/abs/2611.09501 ・ https://arxiv.org/html/2611.09501 ・ https://e-means.github.io
- https://github.com/facebookresearch/faiss/tree/main/contrib/e_means ・ https://raw.githubusercontent.com/facebookresearch/faiss/main/contrib/e_means/README.md ・ https://raw.githubusercontent.com/facebookresearch/faiss/main/contrib/e_means/torch_backend.py
- https://github.com/facebookresearch/faiss/tree/main/benchs/e_means ・ https://raw.githubusercontent.com/facebookresearch/faiss/main/benchs/e_means/README.md ・ https://raw.githubusercontent.com/facebookresearch/faiss/main/benchs/e_means/paper_xb.py
- https://github.com/facebookresearch/faiss/commits/main/contrib/e_means ・ https://github.com/facebookresearch/faiss/releases ・ https://github.com/facebookresearch/faiss/blob/main/INSTALL.md ・ https://pypi.org/project/faiss-gpu/
- https://arxiv.org/abs/2611.18238
- https://arxiv.org/abs/2603.09229 ・ https://github.com/svg-project/flash-kmeans ・ https://pypi.org/project/flash-kmeans/#history
- https://github.com/AnswerDotAI/fastkmeans ・ https://pypi.org/project/fastkmeans/
- https://arxiv.org/abs/2603.20009 ・ https://github.com/cwida/SuperKMeans
- https://github.com/triton-lang/triton
- https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290 ・ https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432 ・ https://pypi.org/project/torch/#history
- https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- https://docs.rapids.ai/platform-support/
- https://arxiv.org/abs/2505.17810 ・ https://github.com/vector-index-bench/vibe ・ https://huggingface.co/datasets/vector-index-bench/vibe
- https://big-ann-benchmarks.com/neurips21.html ・ https://huggingface.co/datasets/suchun/deep1b-mirror
- https://jjiantong.github.io/assets/pdf/tnnls23_kmeans.pdf ・ https://proceedings.neurips.cc/paper/2016/hash/8d317bdcf4aafcfc22149d77babee96d-Abstract.html
- https://arxiv.org/abs/2405.15613 ・ https://arxiv.org/pdf/2603.00173
