# e-means B1–B3 実装ゲート

判定日: 2026-11-18 JST。対象: feature/e-means-validationのB1–B3実装と追加修正。B4 pilotへ進むゲートであり、本評価結果の承認ではない。

## 判定

- Claude Opus（実応答model claude-opus-5-5）、effort xhigh: **承認**。最後の確認は提示した変更hot pathに基づくEvidence Mode。process exit0、result/success、is_error=false、nonempty result、permission_denialsなしを確認した。216,718 ms。回答のないtimeoutは承認へ数えていない。
- GPT-6.1 Sol、effort xhigh: **承認**。独立した初回code reviewと変更範囲の再確認を実施。最後にsentinel関連26テストを独立実行し、smokeのsource/lock/versions一致も確認した。
- 複雑な同期境界はGPT-6 Astra highとも協議。Fable/Astraの設計ゲートは先行する設計commitで完了しており、今回の最終code gateは上の2者。

## 解消した重要事項

1. GPU spawn前のOMP/MKL/OpenBLAS制限、Torch実効1 thread、NVML UUIDによるCUDA割当、ハードウェア/driver/runtime snapshotとrun identity照合。
2. CPU/GPU共通台帳でgpu extraを固定。独立集計でrun key、config/manifest/init hash、manifest原文・型・識別・shape・件数を照合。
3. parallel sentinelの転送同期後hook、single-use group・8 UUID・死活lock・入力一致・共通future release・timeout/abort。実学習のwall/monotonic窓と資格監査を追加。
4. frozen identity/input/既存artifact相違はphase abort。未実施jobへterminal failedを作らず、in-flightはinterrupted。通常の手法失敗・実worker crashは保持。
5. secondaryをResident-input pipelineへ明示改定。train→中心fsyncを評価前に計測し、load/init/診断/全wallを分離。train exact labels＋MSEはassign_seconds、残る群別・heldout診断はevaluate_seconds。
6. warmup後のCUDA allocator reset、run固有allocated/reserved peakと補助NVMLの分離。multiworker host合算境界は全手法infeasibleへ帰属せずphase abort。

主warm training、MSE、22セル、6手法、10 seed、予算点、対応bootstrapは維持した。新secondaryはplan_revision=2026-11-17-timing-v2 / e2e_scope=resident-input-pipeline-v1で識別し、旧load/評価込みE2Eと同義に扱わない。

## 最終検証

| 対象 | 親が確認した結果 |
|---|---|
| 通常suite | 181 PASS、GPU10 deselected |
| 明示GPU suite | V100で10 PASS、skipなし |
| 全体品質 | format81 files、mypy29 source、ruffすべてPASS |
| CLI smoke | CPU4/GPU2の全6成功、共通初期中心、有限MSE、pilot test=null、strict complete |
| artifact再読込 | frozen key/3input hash/manifest、新timing scope、実clock、実効thread、GPU allocation peakを検査 |
| 実worker強制終了 | CPU/GPUとも最初のjob failed、再生成後の次job success、resumeでも履歴を保持 |
| 追加同期試験 | local8 processの共通release/作業窓、2独立ledgerの実adapter hook、timeout/stale/重複/入力不一致/死活喪失/再使用拒否 |
| 数値資格確認 | validationのみ、real d768/k8192等の10,000行、32行FP64との相対誤差約1.04e-7以下、3chunkでMSE一致 |
| secret / staging | secret scanで検出なし。skills-lock.json/.agents/.claudeとdata/run artifactは対象外 |

再現コマンドはREADMEと実装計画を参照。詳細なローカル台帳は.superpowers/sdd/2026-11-17-e-means-implementation-plan/、実smokeはGit管理外のexperimentation/outputs/e_means/smoke-final-v4等に保持する。

runner初期3件は未実装REDを記録した。一部の後続テストは事後に加えた回帰であり、全テストについてtest-firstの歴史を主張しない。追加の重要修正は失敗再現から回帰へ進めた。

## pilotで残る確認

実8 GPUのS0/R0で、共通実学習窓と十分なoverlap、同じinput/init hash、非abortの全参加を確認する。監査にはinput/data/manifest/initの4hashを必ず渡す。member固有hardware snapshotがdescriptor mismatchを起こさないことも初回に確認する。5%干渉判定・10%不安定判定・数値再現許容を確認して、主test観測前にcohortと全設定を凍結し、Opus/Solのprotocol gateへ渡す。本評価9,520 timing runと最終報告は未実施であり、完了扱いにしない。
