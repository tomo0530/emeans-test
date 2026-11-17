# Claude Fable の独立調査・設計意見

2026-11-17。`claude --model fable --effort xhigh` の成功した最終 `result` を以下にそのまま保存。実モデルは `claude-fable-5-2`。Web補助には `claude-haiku-5-5` の利用記録もある。実行・インストールはしていない。調査開始後にユーザーが確定した8GPU使用・時間上限なし・品質時間曲線重視・文章/画像/合成データという条件は、この独立回答の入力には含まれていない。後続協議で補う。

## 親による確認・留保

- Python 3.14対応のscikit-learn配布物もAstraが確認しており、CPU対照をFaiss Lloydだけに限定する根拠はない。
- Faiss既定の点数上限は `256*k`。n=250,000,k=2,048では524,288点を超えないため、既定値によって必ずsubsamplingされるという読み方は誤り。主比較ではいずれも上限と実処理件数を明示する。
- cosine向け正規化データでも、通常L2 k-meansの比較は可能。`spherical=True`は中心も正規化する別の目的・制約となり、主L2比較へ自動的に混ぜない。
- SIFTは画像記述子であり、現代の学習済み画像埋め込みの代替とは扱わない。データの元条件は配布コードの許諾から推定しない。
- 実行時間・CPU/GPU速度・GPU動作の推定は未測定。元データのライセンス原文に接続できていない箇所は未確認のまま扱う。
- Astraはfaiss-cpu1.16.0 wheelのZIP directoryにe_meansの同梱を確認済み。release/main/wheelの同一性は版別に記録する。

以下がFableの回答本文。

---

# e-means検証 設計協議(2026-11-17調査)

【確】一次資料で確認、【推】推論、【未】未確認。実行はしていません。PIEは定義未提示のため適用していません。

## 結論
- Python 3.14・V100・driver 535で成立を確認できた既製実装どうしの比較は、**CPU上の`faiss.Kmeans`(Lloyd25)対`NumpyEmeans`**だけです。GPUのLloyd基準はpipで入らず、自作になります。
- V100で検証できるのは「少ないepochで同等MSE」までで、論文のGPU高速化倍率は検証できません。
- 既存案の規模は1 epochが約1秒以下(推定)で、3段階パイロットと実時間「2倍」基準は費用対効果が低いです。

## 既存案の盲点
1. **GPU Lloyd基準がない。** faiss-gpu-cuvs 1.16.0はCUDA 13向け[S9]。CUDA 13.xはdriver 580以上[S8]でsm_70非対応[S6]。`DatasetAssignGPU`もFaiss GPU必須[S4]【確】。Flash-KMeansのV100動作は【未】。
2. **論文のGPU倍率はH100前提。** H100・BF16対Flash-KMeans FP16の値で、再現READMEも実時間比較はH100のみ有効と明記[S1][S3]【確】。
3. **k/nが論文と別条件。** 論文のVIBEはMSEがK=16k全件、RecallがK=4√N・20 seeds[S1]。250k×k2048は法則のfit範囲内(n≥1e5、n/k≥8)ですが、論文値と直接比べられません【推】。
4. **FashionMNISTはfit範囲外(n<1e5)。** 論文も小規模・低次元で法則が弱いと明記[S1]。どちらの結果でも判断材料になりにくいです【推】。
5. **5 epochの1点比較。** 論文指標は「Lloyd@25到達epoch数」(VIBEで1.6–2.6)[S1]。scheduleがTに依存するためT別の再実行が必要で、途中の`obj`は使えません[S2]【推】。
6. **Faissの暗黙subsample。** 論文は無効化[S1]。全件×k2048は既定上限(256点/centroid、記憶に基づく)に当たります【推】。
7. **Recall基準がノイズ域の恐れ。** queryは1000件、GTは`train`全件への100-NN[S5]【確】。−0.5ppは標準誤差と同程度になり得ます。部分集合だけ索引化するとGTは使えません【推】。
8. **版ずれ。** contrib/e_meansは11/2–11/11に22件以上のcommitで既定値も変更、mainはv1.16.0より先(2b648f0)。c8f2838はIVF-RaBitQ-SSD追加(#5746)で、1.16.0に含まれるかは不明[S2]【確/未】。

同初期中心は可能です。`NumpyEmeans`/`Emeans`と`faiss.Kmeans`の`train`が`init_centroids`を受けます[S2][S4]【確】。

## (1) データ候補

| | A: VIBE agnews-mxbai-1024-euclidean | B: VIBE glove-200-cosine | C: BIGANN(SIFT) 1M |
|---|---|---|---|
| 取得元 | HF `vector-index-bench/vibe`、3.16GB[S5] | 同、956MB[S5] | fbaipublicfiles。`paper_download.py --results 1M`は計約4.5GB、指紋検証と固定split付き[S3] |
| 形状 | 769,382×1024、L2 | 1,192,514×200、cosine | uint8 128d、L2 |
| split/GT | query 1000、100-NN同梱 | 同左 | query 10K。GT配布は10M以上、1Mは要計算[S12] |
| license | embedding CC BY 4.0、model Apache-2.0。元AG corpusは独自の非商用条件(VIBE論文記載)[S5] | PDDL 1.0(VIBEカード記載)[S5] | CC0[S12] |
| 論文での使用 | Table 11、付録L(Recall) | Table 11 | 本文、`paper_targets.json`に基準値[S3] |

- いずれも既存embeddingで生成不要です。HF revisionは`07b387891a221b7b073b83d2f752b76462e5fa03`(閲覧時最新)。Bは正規化と`spherical=True`が必要です。Cのscriptは他データ(SSN++はCC BY-NC)も取得します。
- env: gated表示はなく、HF_TOKENは必須でない見込みです【未】。他は`HF_HOME`、`OMP_NUM_THREADS`、`CUDA_VISIBLE_DEVICES`程度です。
- 費用【推】: 1 epoch≈2·n·k·d flop。A全件×k16384で約2.6e13、CPU実効1 TFLOPS仮定でLloyd25が約11分/seed。論文READMEの「Faiss CPU 1Mで約2分/seed(32コア)」[S3]のd比換算では12–25分/seed、10 seedsで2–4時間。V100は実効5–10 TFLOPS仮定で数秒/epoch、3.2GBで1枚に収まります。
- AG corpus・GloVe・TEXMEXの原ページは接続できず未確認です。DEEPは法則のfit元で独立検証に不向きです[S1]。

## (2) Python 3.14での実施可能性
- **CPU: 可能な見込み。** faiss-cpu 1.16.0(2026-11-10)は`cp310-abi3` manylinux x86_64 wheelでclassifierに3.14[S7]。h5py 3.16.0にcp314 wheel[S10]。v1.16.0タグにcontrib/e_meansあり[S2]【確】。wheelへの同梱は【未】。
- **GPU: wheelはあるが実動作は未確認。** `torch-2.14.1+cu126-cp314-cp314-manylinux_2_28_x86_64`が存在[S11]。2.14が最後のCUDA 12.xで、2.15以降はVolta用wheelなし[S6]。CUDA 12.xの最低driverは525で535は条件内[S8]。論文もPyTorch 2.14[S3]。V100のFP32は`TorchBackend`になり、BF16/FP16はRuntimeError[S2]【確】。wheel内のsm_70、glibc≥2.28、V100での動作は【未】。
- **固定:** `torch==2.14.1`(cu126 index)と`faiss-cpu==1.16.0`。
- **代替:** GPU不可ならCPUのみで完結します。sourceビルド(`TORCH_CUDA_ARCH_LIST`、[S6])は重く非推奨です。

## (3) アプローチ
1. **CPU最小再現(推奨)。** Aで`faiss.Kmeans` Lloyd25(subsample無効)対`NumpyEmeans` T∈{1,2,3,5}、同初期中心、評価器は1つ(再割当してFP64でMSE)。論文Table 11のCPU比較(9–15×、64コア)に対応します[S1]。
2. **1+V100動作確認。** `QuantizedEmeans(quantizer=None, assignment_backend="native")`がCPU版と同等MSEかを1 GPUで確認します。GPU Lloydは作りません。
3. **IVF志向。** IVFFlatを固定して中心だけ差し替え、Recall@10を走査点数一致で比較しimbalanceも記録します。目的がIVF導入の場合のみです。

注意点:
- `em.obj`は移動中の値で、`train`のlabelは最終中心と一致しない場合があります[S2]。両手法とも同じ評価器で再割当します。
- 実時間はwarm-up(demoは学習0.52秒に対し約15秒[S3])、thread過剰割当、GPU共有で崩れます。主指標はepoch数、実時間は副次が安全です。
- 同nprobe比較はリスト偏りで走査量が変わり不公平です。
- 過剰設計: 8GPU、分散、IVF近似割当、量子化器、FashionMNIST、3段階化。GPU推奨の目安はn·k>1e10で、既存案は5e8です[S2]。

## (4) 先に聞くべき決定事項
1. **主目的と対象規模。** 論文主張の再現か、IVF導入判断か。後者なら本番のn・d・k・距離と商用利用の有無(Aの元データは非商用条件)。
2. **基準と合格条件。** 比較をCPU(Faiss Lloyd25)で行ってよいか、GPU比較が必須で自作Lloydを許容するか。MSE余裕(論文主張は0%以下、案は+0.5%)と、速度をepoch数と実時間のどちらで測るか。
3. **予算と実行許可。** 時間上限、GPU 1枚の専有可否、ホスト/devコンテナでのPython実行可否、3–5GBの取得可否。

## Sources(閲覧2026-11-17)
- [S1 論文v1(2026-11-06)](https://arxiv.org/abs/2611.09501v1)
- [S2 contrib/e_means(main、〜2026-11-11)](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means)
- [S3 再現README](https://github.com/facebookresearch/faiss/tree/main/benchs/e_means)
- S4 同repo `faiss/python/extra_wrappers.py`、`contrib/torch/clustering.py`(main)
- [S5 VIBE論文v3(2026-08-23)](https://arxiv.org/abs/2505.17810)、[HF](https://huggingface.co/datasets/vector-index-bench/vibe)
- [S6 PyTorch告知(2026-09-04)](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)、[RFC(2026-07-17)](https://github.com/pytorch/pytorch/issues/190385)
- [S7 faiss-cpu](https://pypi.org/project/faiss-cpu/)
- [S8 CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [S9 faiss-gpu-cuvs(2026-11-10)](https://pypi.org/project/faiss-gpu-cuvs/)
- [S10 h5py(2026-03-06)](https://pypi.org/project/h5py/)
- [S11 cu126 index](https://download.pytorch.org/whl/cu126/torch/)
- [S12 big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html)
