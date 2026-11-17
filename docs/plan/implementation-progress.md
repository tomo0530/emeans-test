# 実装進捗

更新日: 2026-11-18。目的・条件の正本は[設計索引](README.md)、実施項目は[実装計画](2026-11-17-e-means-implementation-plan.md)。実装検証は性能比較の結論ではありません。

| Batch | 現在の状態 | 確認した証拠 | 残作業 |
|---|---|---|---|
| B1: 環境と契約 | 完了・ゲート承認済み | Python3.14.9/Faiss1.16/Torch2.14.1+cu126、契約23件成功、SM70 native FP32 | 凍結環境で実験 |
| B2-A: データと初期中心 | 完了・ゲート承認済み | データ13＋初期化9件成功。raw2件SHA照合、実10セル・合成12条件×10 seedを準備 | 主評価で利用 |
| B2-B: adapter | 完了・ゲート承認済み | CPU17件・V100 GPU10件成功。6手法CLI成功、転送同期後hookと実train窓 | 正式pilotで規模確認 |
| B2-C: 評価・分析 | 完了・ゲート承認済み | 評価14・解析42・sentinel資格18件成功。frozen hash/manifest、新scope照合 | 実測の集計・報告 |
| B3: runner・smoke | 完了・ゲート承認済み | 通常suite181成功、明示GPU10成功、全体format/type/lint、実worker強制終了と復旧 | 専用freeze commitからpilot |
| B4: pilot・凍結・主評価・報告 | 未完了 | 主22セル・6手法・10 seed、9,520 timing runの設定被覆確認。正式pilot・主test評価はまだ | 実8GPU資格、干渉判定、主cohort凍結gate、全本評価、報告・最終gate |

## 最終ゲートと検証

[実装ゲート記録](../reviews/2026-11-18-e-means-implementation.md)のとおり、Opus xhighとGPT-6.1 Sol xhighがB1–B3からpilotへの移行を承認しました。通常suiteは181 PASS / GPU10 deselected、明示GPU suiteはV100で10 PASS・skipなし。全体format81 files、mypy29 source、ruffが通過しています。

親はCLI全6手法のsuccess、共通初期中心、有限MSE、pilot test=null、strict completeを再読込しました。frozen key・3input hash・manifest原文・実効GPU1 thread・owned memory peak・実train窓・新計時scopeを照合しています。CPU/GPU workerを実際に強制終了した試験では、最初の失敗を保持して後続を再生成・成功させ、resumeでもattempt履歴を保持しました。

barrierのlocal8 processによる実作業窓と2独立ledgerのadapter hookは検証済みです。これは実8GPUのS0/R0 sentinel資格の代わりではなく、実機cohortの共通窓・入力一致・非abortをpilotで監査します。model reviewのtimeoutや空回答は承認へ数えていません。

## 計時と再現性

主warm trainingとMSE、条件・seed・予算点は維持します。secondaryのe2e_secondsはresident-input-pipeline-v1へ明示改定し、resident inputからtrain呼出し→中心fsyncを連続計測します。入力ロード・初期化・全診断込みwallはmetadataで別保存します。train exact labels＋MSE融合はassign_seconds、群別とheldoutの診断はevaluate_secondsです。旧scopeなしartifactを新図表へ混ぜません。

parallel sentinelは8 UUID・single-use group・timeoutをfreezeし、転送同期後のhookで待ち合わせます。groupの再使用やabortを成功へ変換しません。frozen identity/input相違と共有host境界はphase abortとし、残りのjobをterminal failedへ変えません。専用freeze commit・同じ親CPU affinity・GPU extraとlockを維持し、phase中にcommit/source編集/依存同期をしません。

公開vector以外のencoder・原文書・原画像は使いません。追加API keyは不要でHF_TOKENは任意です。data/centroid/run logはGit管理外です。一般cooldown1週は維持し、Faiss1.16 wheelだけユーザー承認済み0日例外を使用しました。

B4の完了までgoalを完了にしません。次の2ゲートは、pilot後・主test観測前のprotocol freezeと、本評価後の科学的報告についてOpus/Solへまとめて依頼します。
