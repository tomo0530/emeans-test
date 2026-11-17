# GPU再現性profileの判断と資格確認

更新日: 2026-11-18。対象はGPU実験の実行profileであり、e-meansのアルゴリズム、数値許容、実験条件を変更する判断ではない。profileのruntime/config変更と正式runnerのfresh-process確認は完了した。8 GPU資格確認、全22条件GPU preflight、干渉判定、本評価開始は未完了である。

## 判断

GPUの主比較2手法（`torch_e_means` と `torch_lloyd`）には、同じV100/native FP32の再現性profileを適用する方針とする。選択理由は、既定の同seed再現性要件を満たす共通profileが必要なためであり、観測MSEの良し悪しや速度を選択理由にしない。

採用するprofileは `native-fp32-deterministic-v1`、計画版は `2026-11-18-cuda-deterministic-v1`。設定は `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`torch.use_deterministic_algorithms(True, warn_only=False)`、CUDA graph policy=`upstream_auto`。workspace環境変数はworker起動時にTorch/CUDA/cuBLASを初期化する前に設定し、Torch flagはwarm-up前に有効化する。4 config（smoke/pilot/core/stress）はこのplan revisionと`gpu_deterministic_algorithms: true`を指定している。両GPU手法に共通して適用し、上流source、algorithm、FP32精度は変更しない。

主比較の22条件、6手法、10 seed、予算点、既定の数値再現許容は維持する。stock defaultで得られた結果は別profileの独立findingとして必ず保持し、deterministic profileの結果へ混ぜない。主図は「再現性を要求したV100/native FP32 Torch profileでの品質–時間比較」と表示し、上流Torch既定設定の速度・品質へ一般化しない。

## 根拠

### stock campaign

freeze `cc87943ca725baa9a1bedbc8f909debdc4d46088` のS0 engineering campaign `s0-probe-20261117T151502-cace2d25` では48 runがすべてsuccessで終了した。6 matched trialすべてで8 GPUの入力・初期中心hash、共通descriptor、正の実行窓overlapを確認した。一方、同一seedのe-means T8 MSEは約176.35–187.81に分かれ、事前の数値再現許容を満たさず、campaignの資格判定は失敗した。Lloyd I10はこの許容内だった。

runのsuccess statusとcohort資格失敗は両立する。個別の成功artifactはsuccessのまま保持し、campaign全体をunqualifiedとして扱う。このstock結果は正式な性能比較の一部に置き換えたり、profileをまたいで集約したりしない。

### 単GPU診断

単GPU S0の診断では、workspace設定だけ、またはCUDA graphを止めるだけでは再現しなかった。deterministic modeを有効にした場合はgraph autoとgraph offの両方で、各3反復のMSE・centroid hash・revival数が一致した。

| 診断profile | validation MSE範囲 | 3反復のcentroid hash | warm train中央値 |
| --- | ---: | --- | ---: |
| stock | 180.93577–187.80997 | 不一致 | 0.05456 s |
| workspaceのみ | 180.35377–185.74885 | 不一致 | 0.05624 s |
| workspace・graph off | 184.77899–184.93301 | 不一致 | 0.12640 s |
| workspace・deterministic・graph auto | 176.5076490463 | 一致 | 0.07994 s |
| workspace・deterministic・graph off | 176.5076490463 | 一致 | 0.16784 s |

さらに、strict deterministic・graph autoを正式runnerとは別の診断経路で2つのfresh processから各3回実行した。全6回でMSE=`176.5076490463257`、centroid SHA=`b030519197b0ebfc860cf0b4af2195c00c199aa7bd58bc6dd380599a1f993079`、revival数165が一致し、各反復で実graph replay数95を記録した。これはprofileを選ぶ根拠となる診断証拠であり、全22条件・8 GPU・正式runnerの資格確認ではない。

workspace・deterministic・graph autoの診断warm train中央値はstockの約1.47倍だった。これは単一GPU S0診断の時間で、正式速度比較ではない。正式profileの速度・メモリへの影響は本計測で改めて評価する。

### 新profile source・正式runnerの検証

2026-11-18の新profile sourceでは、CPU suiteが199 PASS / 10 deselected、明示GPU suiteが10 PASS / skipなしだった。全体lintはformat 83 files、mypy 29 source、ruffのすべてが成功した。これらはprofileを含む作業sourceの検証である。

正式runner経路では2つのfresh processでGPUの両手法を各3回、計12 run実行し、すべてsuccessした。`torch_e_means`と`torch_lloyd`はそれぞれ6/6でcentroid SHAが一致した。e-meansのMSE=`176.5076490463257`とSHAは診断結果と一致した。実行時のactual stateはdeterministic algorithms=true、warn-only=false、workspace=`:4096:8`、TF32=false、matmul precision=`highest`、`fill_uninitialized_memory`=trueだった。

この検証は新profileのsourceと正式runnerでのfresh-process再現を確認したものだ。8 GPU間、S0/大形状の再資格、全22条件のpreflightを代替せず、正式速度比較でもない。

## 原因に関する限界

最有力の仮説は、CUDA FP32の`index_add`集約順による微小差がe-meansの離散的なassignment/revival経路で増幅されること。上流の固定版e-means実装にindex-addがあり、制御診断の結果とも整合する。ただし最初に差が生じる演算を単離していないため、確定原因とはしない。CUDA graphはworkspace-only等の非deterministic条件でもhash不一致が残り、deterministic条件ではgraph auto/offの結果が一致したが、これをgraph raceの不存在証明とはしない。

[PyTorch 2.14 API](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html) は、CUDA `index_add`をdeterministic modeで決定的な動作へ切り替える演算に含め、代替実装がない演算は`warn_only=False`で例外にすると説明する。このflagだけでapplication全体の再現性を保証するものではない。[PyTorch 2.14 reproducibility note](https://docs.pytorch.org/docs/2.14/notes/randomness.html) は、release・commit・platformをまたぐ完全再現を保証しないと記載している。

[CUDA 12.6 cuBLAS reproducibility](https://docs.nvidia.com/cuda/archive/12.6.3/cublas/index.html#results-reproducibility) は、複数streamでcuBLASのworkspace選択が変わる状況に対する設定として`:4096:8`を示す。これはworkspace設定の役割であり、`index_add`等の演算順を単独で固定する根拠ではない。

## 実装・記録契約

- `ResourceConfig.gpu_deterministic_algorithms: bool = False`を要求値として持つ。作業中の4実験config（smoke/pilot/core/stress）は明示的に`true`を設定する。CPU手法にはTorch設定を適用しない。
- profile名、plan revision、config/profile hashをrun identityに含める。既存stock rootへprofile変更後のrunをresume・上書きしない。
- requestedとactualの両方をenvironment artifactへ保存する。少なくともdeterministic algorithmsの有効状態と`warn_only`、workspace env、`fill_uninitialized_memory`、FP32 matmul設定、graph policy、Torch/CUDA/cuBLAS/driver/GPU UUID/SM、dtype/backendを記録する。formal adapterはinstance限定の観測wrapperでupstream/effective eligibilityを記録し、upstreamの戻り値を変えず、`finally`で一時属性を削除する。`stats.graph_policy="upstream_auto"`、`stats.graph_eligibility`（同じ観測値を示す`upstream_eligible`/`effective_eligible`）、`graph_actual_replays=null`を保存する。replay数はformal runnerでは未計測とし、実replay数95は診断で観測した値としてだけ報告する。
- worker起動時およびrun開始前に要求値と実状態を照合する。不一致やprofile driftはrun failureへ変換せず、既存attemptを保全してphaseをabortする。`warn_only`への変更、CPU/別backendへの自動fallback、run単位・cell単位のgraph切替はしない。
- 主図・stock所見・再現性の限界を別々に報告する。stockとdeterministicのpilot値を同じprofile/cohort/CIへpoolせず、stockの結果をdeterministic profileの速度へ読み替えない。

正式e-means runはeligibilityのみを観測し、実replay countは未観測（null）とする。Lloydはこのe-means graph経路の対象外である。coverageでsuccess以外が1件でも出た場合は次の条件へ進まず停止する。

## 主評価前に残る資格確認

次の項目は未実施であり、完了するまで主testを開始しない。

1. **全条件coverage:** 非主seed 100、両GPU手法の最大予算を使う全22条件の実行表・coverage準備は完了しているが、GPU preflight実行は未着手。決定性未対応演算やcapture制約による失敗を記録し、例外cellだけ設定を変えない。
2. **8 GPU・S0/大形状再資格:** 新profileでfresh processを用い、S0と大形状を含む固定条件について各GPUで3反復を行う。e-meansは全GPU・全反復のcentroid SHA一致、Lloydは従来の数値許容を資格条件とする。単独・8 GPU同時条件で固定入力・初期中心、実overlap、時間干渉、allocator/NVML/host memoryも再判定する。S0の診断だけで大形状や全条件の適合を推定しない。
3. **干渉と採用cohort:** 新profileによる単独・8 GPUの速度/資源を比較し、5%判定に基づくcohortを主test前に凍結する。
4. **主test前gate:** 正式runner検証、profile適合、8 GPU資格、干渉に基づくcohortを揃えてprotocolを凍結し、Opus/Solの限定reviewを通す。

再資格の代表条件はpilot S0、arxiv R0、landmark R0（T8/I10、seed100/101）。単独は各GPUで3反復、並列は8 GPUを揃えたfresh process世代を3回使う。境界条件はarxiv Rk-large（n500k,d768,k8192）、arxiv Rn-full（n1,324,643,d768,k2048）、Sd（n100k,d1024,k256）で、seed100、T12/I100、8 GPUのfresh 3世代とする。最大予算coverageは全22条件で別に実施する。

5%判定はseed100の同じGPU・条件・手法を対応付け、単独3反復の中央値と並列3世代の中央値の相対差の絶対値を比較する。いずれかが5%以上なら保守的にmain-isolatedを採用する。seed101は開始・終了時の再現性照合用として保持し、この選択には混ぜない。数値資格に失敗したcampaignからcohortを採用しない。


既存の通常181 CPU test、明示10 GPU test、format/type/lintの結果はprofile導入前sourceの証拠であり、profile sourceでの199 CPU / 10 GPU検証とは区別する。準備済み全22条件coverageのGPU実行、8 GPU/S0・大形状再資格、cohort干渉判定、主test前gateはいずれもpendingである。

## CPU実行資源の起動監査

本評価前に別の資源監査を行い、CPU workerの主threadをspawn後に固定するだけでは、NumPy import時に既に作られたOpenBLAS threadへ制限が及ばないことを確認した。実workerの修正前は主threadだけCPU0–15、残る15 threadはCPU0–79を使える状態だった。Linuxのaffinityはthread単位で、子へ継承されるため、worker起動前に親の呼出threadを一時的にCPU0–15へ制限し、起動後に親を復元する修正を加えた。[Linux sched_setaffinity仕様](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)

実spawnの回帰試験では全16 threadがCPU0–15を継承し、実trialのOpenBLASも16 threadだった。正常時と起動例外時の双方で、親の元affinityと3種のthread数環境変数を復元した。GPU経路・数学・profile設定は変更していない。各runの主thread affinity記録は維持し、固定ライブラリ下での全thread継承はこの起動試験で確認する。

発見時点で正式CPU計時・主testは未開始。先行GPU coverageは6条件・12 successを工程証拠として保全し、実行中CLIを終えてからsourceを修正した。source identityが変わるため、全22条件coverageは新しいfreeze commitと新rootからやり直す。途中の旧rootを完了扱いしたり、異なるsourceの台帳へresumeしたりしない。

## 主test前に固定する限定感度調査

stockとの違いは、主曲線へ混ぜず、S0とarxiv R0、非主seed103–107、両GPU手法T8/I10の固定点で記述する。stockは各5反復、deterministicは各3反復とし、合計160 run。同seed・同条件では同じ物理GPU・input・初期中心を使い、profileごとにfresh processで実行する。順序はseed103/105/107がstock→deterministic、104/106が逆順（5組なので3対2）と事前固定する。結果を見ても主profileを選び直さない。

実行計画JSONのSHA256は `eb0e271ab7b9378b1b754f2fe025ca1cbf4844b12f598520f8b06aba5f08c3eb`。正式run artifactのconfig snapshotと入力hashで追跡する。品質分布と速度費用の限定的な診断として扱い、分布の裾・有意な偏り・全22条件への一般化を主張しない。

全22条件の最大予算によるcoverageに加え、全132個のe-means condition/budget点のfitted recipeを静的に確認した。batch sizeは512–131072、steps/passはすべて3以上だった。これは計算経路の説明資料であり、全132点を実行した証拠や、実graph replayの保証ではない。
