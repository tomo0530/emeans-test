# GPU再現性profileの変更ゲート

判定日: 2026-11-18 JST。比較元は `cc87943ca725baa9a1bedbc8f909debdc4d46088`。対象は両GPU手法共通のstrict deterministic runtime、設定・identity・実状態監査、e-meansのeligibility観測と関連回帰。正式pilotへ進むcode gateであり、主test開始の承認ではない。

- GPT-6.1 Sol xhigh: **APPROVED**。変更を独立確認し、CPU差分テスト16件と正式runnerの12 artifactを再読込。source一致、profile、手法ごとのSHA一致を確認した。
- Claude Opus xhigh（実応答 `claude-opus-5-5`）: **APPROVED**。提示したdiff・ログ・証拠文書に基づくEvidence Mode。exit0 / result-success / is_error=false / nonempty result / permission_denialsなし。231,857 ms。ワークスペースの直接探索をしたとは扱わない。

| 検証 | 結果 |
| --- | --- |
| 通常suite | 199 PASS / GPU10 deselected |
| 最終GPU suite | V100で10 PASS、skipなし |
| 全体品質 | format83 files / mypy29 source / ruff PASS |
| 正式runner工程試験 | fresh 2 process、各GPU手法3反復、計12 success。各手法6/6 centroid SHA一致、pilot test=null |
| e-means再現 | MSE 176.5076490463257、SHA b030519197b0ebfc860cf0b4af2195c00c199aa7bd58bc6dd380599a1f993079。別診断と一致 |
| actual state | deterministic=true、warn_only=false、workspace=:4096:8、TF32=false、matmul=highest、fill_uninitialized_memory=true |
| graph観測 | upstream/effective eligibilityを同じ返値で保存。measured actual replay countはnull、実95 replayの観測は別診断に限定 |

契約・spawn環境・profile identity・drift・未対応演算・CPU-only importの回帰は未実装失敗から確認した。stockの48 successと数値資格失敗は保持する。追加profileは再現性を理由に選び、品質・速度から選び直さない。

Opusの非blocking注記は文書化した。coverageではsuccess以外があれば次の条件へ進まず停止する。Lloydはe-means CUDA graph経路の対象外なので、そのeligibility/replay欄を持たない。その他の算法や上流sourceを変更していない。

このレビュー時点では、全22条件seed100最大予算coverage、8GPU fresh3のS0/R0/大形状再資格、5%干渉判定、固定160 runのstock感度、正式pilotは未実施。これらを揃えた別のpre-test protocol gateが必要である。主22条件・6手法・10 seed・9,520 timing runの完了を先取りしない。

詳細判断は[再現性profile記録](../research/2026-11-18-gpu-reproducibility.md)。ローカル証拠は `.superpowers/sdd/2026-11-17-e-means-implementation-plan/profile-*` に保持する。

## CPU起動affinityの追補ゲート

比較元 `1e36ee9953474b9efdccd43e9fb9b2f38b740d9a`。正式CPU計時の前に、spawn後の主thread pinではimport済みOpenBLAS補助threadを制限できない実測不備を確認した。修正は`_cpu_phase.start`の起動前mask適用と親の復元だけで、他のrunner定義・GPU経路は同一である。

- GPT-6.1 Sol xhigh: **APPROVED**。独立2回帰試験で全16 threadのCPU0–15継承と正常・例外時の復元を再確認。
- Claude Opus xhigh（実応答 `claude-opus-5-5`）: **APPROVED**。提示diff・証拠に基づく判断で、再テストしたとは扱わない。最初の300秒timeout・空回答は不採用。絞った再試行はexit0/result-success/nonerror/nonempty/denialsなし、31,399 ms。

親の全体検証は201 PASS / GPU10 deselected、format83 files / mypy29 source / ruff PASS。runner対象55件もPASS。実spawnは修正前1 threadのみCPU0–15・残り15がCPU0–79、修正後16/16がCPU0–15だった。実trialのOpenBLAS16と、正常・起動例外時の親mask/env復元を確認した。GPU専用10件は変更のないGPU経路に対する既存証拠であり、今回のCPU差分で新たに実行したとは数えない。

旧GPU coverage6条件・12 successは工程証拠として保持した。新freezeで実施した全22条件の結果は[本評価プロトコル](../report/protocol/README.md)に記録する。resource_trackerが起動時maskを継承しても待機主体であること、Linux以外は今回の資源契約の対象外であることはOpusのminor注記として記録する。承認範囲はcommitとpilot/preflightまでで、本評価開始の別protocol gateは維持する。
