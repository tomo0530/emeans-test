# e-means 本評価プロトコル

> 本評価前レビュー提出版。テストデータの評価開始前に、OpusとSolの承認を記録する。

## 目的と比較の範囲

同一のFP32ベクトル・split・外部初期中心を使い、最終中心への独立したexact squared L2 MSEとwarm training timeの曲線を比較する。固定の品質合格幅を設けず、速くなる条件と品質が悪化する条件を両方残す。論文のH100速度倍率の再現や、CPU/GPU差をアルゴリズムだけの効果と解釈することは目的に含めない。

主手法はFaiss CPU Lloyd、NumPy e-means、scikit-learn MiniBatchKMeans、Elkan、Torch e-means、薄いTorch Lloydの6手法。薄いTorch Lloydは共通の上流native assignmentを用いる対照実装であり、最適化されたGPU Lloydの最高性能を代表しない。全手法1 restart、trainの厳密重複を除いた候補から一様非復元抽出した共通初期中心を使う。

[論文v1の実験](https://arxiv.org/html/2611.09501v1)はH100 80GB、CPU64物理coreを使い、Faiss/Flashの25反復や10 seed中央値による図を報告している。今回はV100、CPU16物理core、Faiss CPU@100を相対品質の基準とし、品質をseed間算術平均で集約する独自の統制比較である。[上流READMEの固定revision](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/README.md)にはGPUごとの経路が記されているが、このGit commitとインストール済みwheelのsourceが同一とは主張しない。実測したV100のTorch/native FP32経路とwheelのhashを記録する。

実装・依存の資格確認対象は `bf73f55e569ed856a82c63c64725d58f693e04bf`。本評価前の凍結commitでは、数値実装・runner・依存を変更しない。Pythonは3.14、V100 32GB ×8、CPU計時はNUMA node 0の16物理core、GPU workerは1物理coreとする。

## 凍結する条件と集計

- VIBEの文章・画像埋め込み各5条件、合成12条件の合計22条件。実データはrevision・raw SHA256・厳密重複groupを分離したsplitを固定し、validation/test各10,000行。合成はseedごとの生成と学習を名前付きstreamで分離する。
- seed 0–9の10組。実データのdata_seedは42、合成のdata_seedは組seed。同一組ではデータと初期中心を共有する。
- Lloyd/Elkanは1,2,5,10,25,50,100反復、e-meansは1,2,3,5,8,12 epoch、MiniBatchは0.25,0.5,1,2,5,10,25全件相当pass。総予算ごとに独立して完走し、長いrunの途中を短い予算の結果へ転用しない。
- 880個の条件・手法・予算の組、8,800 seed job、代表点の追加720計時を合わせて本評価9,520 run（core 4,480、stress 5,040）。CPU 6,420、GPU 3,100。
- R0/S0の代表2予算点は全10 seedで3回計時し、seed内の時間中央値を先に取る。他はrepeat 0。品質は常にrepeat 0。
- 主図はtest MSEの10 seed算術平均とwarm時間中央値。bootstrap seed 1729、10,000回、seed indexを全手法・予算・両軸で共有した95%点ごとCI。曲線全体や事後選択した最良点の同時保証ではない。
- 相対品質は同じcell/seedのFaiss Lloyd@100を基準とする。補助水準0,0.1,0.5,1,2,5,10%への到達時間は、条件を満たす観測済み予算点の時間中央値の最小値。未測定点の補間、未知の最適値、採否の品質マージンを意味しない。
- 主点は10/10成功かつ有限を必要とする。部分成功点は欠落と成功数を表示し、成功seedだけで順位を作らない。failed/infeasible/unsupported/unreached/reference_undefinedを区別する。

## GPU profileと採用する計時cohort

両GPU手法へ `native-fp32-deterministic-v1` を共通適用し、両手法ともnative FP32で実行する。e-meansは明示的に`quantizer=None`を渡す。共通設定は`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True, warn_only=False)`、CUDA graph policy=`upstream_auto`である。上流のbatch/schedule/revival、seed、精度、予算を変更しない。主結果はこの再現性profileの測定として表示し、stock defaultの速度・品質へ一般化しない。実測graph replay数は正式runで未観測のためnullであり、eligibilityをreplay数と呼ばない。

既定5%規則で **main-isolated** を選択した。seed100の適格48比較中16比較でparallel/isolated時間差が5%以上で、すべてS0・両GPU手法・8 GPUに対応する。seed101は選択に使っていない。S0ではe-meansが約5.71–14.41%、Lloydが約10.35–24.55%長く、実R0ではe-meansは3.09%以下、Lloydは0.52%以下だった。本評価はGPU 0で1 trialずつ計時し、CPU baselineと重ねない。8 GPU利用可と、同時に8 GPUで正式計時することは区別する。

warm trainingはCUDA前後同期のwall時間。副次の `resident-input-pipeline-v1` は検証済みresident input/initからadapterを呼び、centroid保存・fsyncまで。入力load/hash、初期化、warmup、評価、台帳、NVML初期化を含めない。sentinelのbarrier待ちはwarmから除外するがpipelineに含むため、そのpipeline時間を通常runの速度比較へ混ぜない。

## 実施済み資格確認と計時上の限界

| 証拠 | 実施結果 | 解釈 |
| --- | --- | --- |
| 最大予算GPU coverage | 全22条件×2手法、44 success、test=null | 各形状の実行適合。全予算・全seedの主性能を保証しない |
| GPU再現・干渉資格 | 720 success、18数値identity group、54個の実8 GPU共通時間窓 | qualified=true、共通overlap比最小0.9075。e-meansは各groupでcentroid SHA一致、Lloydは既定許容内 |
| CPU sentinel | 36 success、12 group×3反復、test=null | 全groupでcentroid SHA一致。実workerの32 native threadすべてのaffinityはCPU0–15 |
| 正式pilot | 360 success、CPU 243・GPU 117、120集計点、欠損0、test=null | 9組でmanifestと初期中心が共通。各run keyを再計算し、個別の実行環境を照合。中断1件を履歴に保持し、再開後の成功を採用 |
| CPU起動修正後テスト | 201 PASS、GPU10 deselected、format/type/lint PASS | CPU spawn時のaffinity継承と親の正常・例外時復元を含む。GPU10は変更前の同一GPU経路の証拠を別記 |

正式pilotではGPU 4–7で117件を実行した。全8 GPUの適合は別のsentinelで確認した。正式pilotの実測最大値はworker RSS 9,278,234,624 bytes、NVML total-device 3,626,631,168 bytes、PyTorch allocator allocated 2,198,714,880 bytes、reserved 2,904,555,520 bytes。観測範囲が異なるため各指標を別々に示す。最大予算GPU coverageのallocator最大値はallocated 4,739,140,096 bytes、reserved 5,465,178,112 bytes。

GPUの10%超repeat range/medianフラグは7件で、すべて単独S0 e-means（seed100のGPU0/2/4、seed101のGPU0/2/4/7）。絶対rangeは約0.00956–0.01201秒。CPUにもarxiv R0/Faiss I10/seed100の1件があり、16.2552,16.3126,19.5330秒、range/median約20.09%、絶対range約3.2778秒。同じ入力・初期値・処理量・最終中心を確認したが、原因は確定していない。

監視はPython threadのRSS/NVML、間隔0.1秒。監視、host scheduling、GPU clock、graph captureのどれかを原因と断定しない。短時間GPUだけの問題とも断定しない。元の計時値とフラグを保持し、事前固定の3反復中央値・10 seed CI・isolated cohortで報告する。再現許容、予算、監視を有利になるよう調整しない。この扱いは本評価前のOpus/Solレビューの判断対象である。

8 GPUのNVML UUIDを順に `CUDA_VISIBLE_DEVICES` へ指定し、別々の新規プロセスでCUDA allocationを行った。全8件でCUDAから見えるデバイスは1台で、CUDA propertiesのUUIDは指定UUIDと一致した。全件Tesla V100-SXM2-32GB、compute capability 7.0。異なるPID名前空間間のNVML PID突合はできなかったため、実デバイスの対応をこの照合で確認した。

## stock実装と感度検証の扱い

stock S0 engineering campaignは48個別runすべてsuccessだった一方、e-meansの同seed数値再現が許容外となった。個別statusを書き換えず、campaignをunqualifiedとして残す。数値profile選択はこの再現性要件に基づき、良いMSEや速い時間から選び直さない。

固定した感度検証はS0とarxiv R0、seed103–107、e-means T8/Lloyd I10、stock 5反復/deterministic 3反復の160 run。seedごと同じ物理GPUに対応させ、順序はstock先3組・deterministic先2組。主結果へpoolせず、各seed内中央値からの対応時間比の中央値と品質の分布・centroid SHA数を記述する。5組の小標本、順序の3/2差、seedとGPUの交絡、trajectory/revival差があるため、純粋なkernel overheadの因果推定とは呼ばない。

全160件の入力・設定・結果を照合した。detのe-meansは各seed・条件の3反復で中心NPYが一致し、Lloydはvalidation MSEが既定許容内だった。以下は各seedで反復の中央値を比較した時間比（det / stock）の、5組間の要約。

| 条件 | 手法・予算 | 中央値 | 最小–最大 |
| --- | --- | ---: | ---: |
| S0 | e_means 8 | 1.4574 | 1.4163–1.4838 |
| S0 | lloyd 10 | 1.9129 | 1.7300–2.0676 |
| arXiv R0 | e_means 8 | 1.0167 | 1.0141–1.0199 |
| arXiv R0 | lloyd 10 | 1.1065 | 1.1029–1.1139 |

時間比の数表は[感度検証CSV](stock-sensitivity.csv)。stockのCUDA workspace設定は未設定、detは `:4096:8` と記録された。5組の観測値は比較した条件と予算点に限る。

## パラメータ・版の再現

条件/手法/予算880個の設定を別JSONへ展開し、requestedなNone/autoと上流wheelが解決するeffective recipeを区別する。Faiss/sklearnのconstructor params、e-meansのbatch/alpha/schedule/revival/chunk、seed/epoch stream、config/source/wheel/lock SHA256を保存する。GPU上の実際のFP32 cast後のtraceを、CPU上で計算したlogical scheduleのhashから推定しない。

固定wheelの `init_usage=2**-16` は論文の値と異なり、係数にも版差がある。利用したwheelの規則を正本とする。NumPy e-means recipe内backendはfaiss。正式pilot360件とcoverage44件（手法・形状・予算・seedで258組）の成功時の記録を照合し、一致を確認した。本評価880設定は固定したスカラー値から計算した。実験中のloss・時間・メモリ・revival回数は実測値を記録する。

faiss-cpu 1.16.0は承認済みcooldown例外。通常の1 weekを維持し、既存templateのcertifi例外も別に残る。実ベクトルのみを取得しencoderは利用しない。HF_TOKEN以外のbenchmark認証は追加していない。

## 検証記録と再現手順

[パラメータ索引](resolved-parameters/index.json)に、2件の設定、13件のパラメータ表、生成スクリプト、実装・wheelのSHA-256を記録した。凍結前のGit HEADは資格確認に使った実装commitを指す。本評価のrun keyにはレビュー承認後の凍結commitを使用する。

同じlockのGPU環境で、ルートから以下を実行するとスカラー値を再生成できる。出力先には新規ディレクトリを指定する。実測値との照合は `--compare-pilot-root` で保存済みのpilotとcoverageのルートを渡す。

```bash
UV_NO_SYNC=1 uv run --extra gpu python docs/report/protocol/tools/export_parameter_snapshot.py \
  --output-root /tmp/emeans-resolved-parameters
```

[証拠のSHA-256一覧](evidence.json)は、計測出力と資格確認に使用した保存済みファイルを識別する。全160件の感度検証の行別記録、正式pilot、資格確認、計時フラグの生値は実行環境に保持する。

レビュー承認後、凍結commitからcore 4,480件・stress 5,040件を実行する。終了時に別の測定IDでseed101のCPU/GPU sentinelを各18件実行し、同入力・初期中心の開始時記録と照合する。GPU側は本評価と同じGPU 0を使用する。GPU 0の開始時記録は既存の単独GPU資格に含まれ、6組×3反復の成功を確認済み。CPU/GPUの既定の数値許容、e-meansの中心SHA一致、warm中央値の10%超変化の調査を適用する。その後、全22条件の図表、群別損失、クラスタサイズ・空クラスタ、メモリ、観測点の到達時間と未到達、失敗を日本語の報告へまとめる。
