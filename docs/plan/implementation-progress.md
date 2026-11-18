# 実装進捗

更新日: 2026-11-18。目的・条件の正本は[設計索引](README.md)、実施項目は[実装計画](2026-11-17-e-means-implementation-plan.md)。実装検証は性能比較の結論ではありません。

| Batch | 現在の状態 | 確認した証拠 | 残作業 |
|---|---|---|---|
| B1: 環境と契約 | 完了・ゲート承認済み | Python3.14.9/Faiss1.16/Torch2.14.1+cu126、契約23件成功、SM70 native FP32 | 凍結環境で実験 |
| B2-A: データと初期中心 | 完了・ゲート承認済み | データ13＋初期化9件成功。raw2件SHA照合、実10セル・合成12条件×10 seedを準備 | 主評価で利用 |
| B2-B: adapter | 完了・ゲート承認済み | CPU17件・V100 GPU10件成功。6手法CLI成功、転送同期後hookと実train窓 | 正式pilotで規模確認 |
| B2-C: 評価・分析 | 完了・ゲート承認済み | 評価14・解析42・sentinel資格18件成功。frozen hash/manifest、新scope照合 | 実測の集計・報告 |
| B3: runner・smoke | 完了・旧sourceのゲート承認済み | 通常suite181成功、明示GPU10成功、全体format/type/lint、実worker強制終了と復旧。すべてprofile導入前sourceの証拠 | 新profileの8GPU再資格・全条件preflightはB4で実施 |
| B4: pilot・凍結・主評価・報告 | 予備検証完了 | GPU coverage44件・GPU資格720件・CPU sentinel36件・正式pilot360件・感度160件が成功。8 GPUのCUDA UUID照合も通過 | 本評価前レビュー・条件凍結、本評価9,520件、再現性の再確認、報告・最終レビュー |

## 最終ゲートと検証

[実装ゲート記録](../reviews/2026-11-18-e-means-implementation.md)のとおり、Opus xhighとGPT-6.1 Sol xhighはprofile導入前のB1–B3 sourceを承認しました。旧sourceの通常suiteは181 PASS / GPU10 deselected、明示GPU suiteはV100で10 PASS・skipなし、format81 files / mypy29 source / ruffが成功しました。新profile sourceでは通常suite199 PASS / GPU10 deselected、明示GPU suite10 PASS・skipなし、format83 files / mypy29 source / ruffが成功しています。いずれのゲートも8 GPU再資格や主test開始承認ではありません。

親はCLI全6手法のsuccess、共通初期中心、有限MSE、pilot test=null、strict completeを再読込しました。frozen key・3input hash・manifest原文・実効GPU1 thread・owned memory peak・実train窓・新計時scopeを照合しています。CPU/GPU workerを実際に強制終了した試験では、最初の失敗を保持して後続を再生成・成功させ、resumeでもattempt履歴を保持しました。

barrierのlocal8 processによる実作業窓と2独立ledgerのadapter hookは検証済みです。これは実8GPUのS0/R0 sentinel資格の代わりではなく、実機cohortの共通窓・入力一致・非abortをpilotで監査します。model reviewのtimeoutや空回答は承認へ数えていません。

## 計時と再現性

主GPU比較には、両Torch手法共通の`native-fp32-deterministic-v1`を適用します。4 configは`2026-11-18-cuda-deterministic-v1`と`gpu_deterministic_algorithms=true`を設定し、既存の計時scope、条件・seed・予算・許容値は維持します。stock S0の48 runはすべて個別successですが、e-means同seed MSEの分散が数値許容を超えてcampaign資格は不合格です。成功artifactを失敗へ書き換えず、stockの独立findingとして残します。診断ではworkspace設定だけ/graph offだけでは再現せず、deterministic+workspaceでは2 fresh processes×3反復のMSEとcentroid hashが一致しました。原因はFP32 index-add差からrevivalへ増幅する仮説に留まり、未確定です。S0診断で約1.47倍だったwarm train時間は正式速度結果に使いません。詳細は[GPU再現性profile記録](../research/2026-11-18-gpu-reproducibility.md)を参照してください。

正式runnerのfresh-process検査は12件が成功した。後続の全22条件coverage44件と8 GPU再資格720件も成功した。seed100の干渉判定では48比較中16比較で差が5%以上だったため、本評価の計時は単独GPUとする。正式pilot360件の全入力・初期中心・環境を照合し、テスト評価がないことを確認した。詳細は[本評価プロトコル](../report/protocol/README.md)に記録する。

主warm trainingとMSE、条件・seed・予算点は維持します。secondaryのe2e_secondsはresident-input-pipeline-v1へ明示改定し、resident inputからtrain呼出し→中心fsyncを連続計測します。入力ロード・初期化・全診断込みwallはmetadataで別保存します。train exact labels＋MSE融合はassign_seconds、群別とheldoutの診断はevaluate_secondsです。旧scopeなしartifactを新図表へ混ぜません。

parallel sentinelは8 UUID・single-use group・timeoutをfreezeし、転送同期後のhookで待ち合わせます。groupの再使用やabortを成功へ変換しません。frozen identity/input相違と共有host境界はphase abortとし、残りのjobをterminal failedへ変えません。専用freeze commit・同じ親CPU affinity・GPU extraとlockを維持し、phase中にcommit/source編集/依存同期をしません。

公開vector以外のencoder・原文書・原画像は使いません。追加API keyは不要でHF_TOKENは任意です。data/centroid/run logはGit管理外です。一般cooldown1週と既存のcertifi例外は維持しました。追加例外はユーザー承認済みの固定版Faiss 1.16.0です。

B4の完了までgoalを完了にしません。[profile変更ゲート](../reviews/2026-11-18-gpu-profile.md)もOpus/Solが承認しました。次の2ゲートは、pilot後・主test観測前のprotocol freezeと、本評価後の科学的報告についてOpus/Solへまとめて依頼します。

## CPU affinity起動監査の追補

CPU workerをspawnする前に親の呼出threadを指定16コアへ一時pinし、NumPy import時の全OpenBLAS threadへ継承させました。実workerは16/16 threadがCPU0–15となり、正常・例外時とも親maskと環境値を復元します。全201 CPU test・全体品質が通過し、Opus/Solもこの変更を承認しました。GPU経路・手法・設定は変えていません。

発見時点で正式CPU計時と主testは未開始でした。旧GPU preflight6条件・12 successを先行証拠として保存しました。新しいsourceと新rootで全22条件を実行し、44件が成功しました。
