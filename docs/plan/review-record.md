# 設計・計画のレビュー記録

対象: 設計3文書、実装計画、root/module AGENTS、e-means workflow、調査の採用判断と参照整合。過去の独立調査本文は履歴資料であり、現在の実装仕様とは区別する。

早期の設計相談を最終承認とは数えない。初回は同じ設計・計画セットを4者へ並列に提示し、下記指摘を一括修正した。変更範囲の最終確認を終え、4者ともAPPROVEとなった。承認対象は設計・計画であり、未実行の実装や実験の成功ではない。

| レビュワー | 起動指定 / 実モデル | 初回判定 |
| --- | --- | --- |
| Astra | gpt-6-astra / xhigh / fresh context | CHANGES_REQUIRED: 統計契約1件 |
| Fable | fable / xhigh / claude-fable-5-2 | CHANGES_REQUIRED: 契約・計時・pilot・gate等。288.280秒、成功result |
| Opus | opus / xhigh / claude-opus-5-5 | CHANGES_REQUIRED: 契約・統計・再試行等。276.148秒、成功result |
| Sol | gpt-6.1-sol / xhigh / fresh context | CHANGES_REQUIRED: 統計、data realization、再計時、空group4件 |

Claude2件はexit0、result/success、is_error=false、非空result、permission_denials=[]を確認。native2件は指定reviewファイルと最終報告を確認した。

## 一括修正と判断

1. 主軸をtest MSEのseed平均、warm train時間のseed中央値とした。再計時はseed内中央値→seed間集約。共通seed再標本化のpercentile CIを固定し、対応seedのFaiss@100比、集約/seed別到達、未到達・不完全・基準未定義を分けた。Fable初回の時間平均案に対して、Opus/Sol/Astraが支持する時間中央値を採用する。
2. loaderにcondition/data_seedを加え、配列・group・manifestをprepare時に物化する。ExperimentConfig/DatasetManifest/RunRecord/RunMetricsと乱数streamをB1で凍結し、並列担当の別schemaを防ぐ。
3. timing_repeat/measurement_idをrun keyへ追加し、計時反復・GPU別sentinel・retry attemptを区別する。採用cohortは主test前に固定する。
4. resumeはpending/interruptedのみ。failed/infeasible/unsupportedを終端とし、明示retryは履歴を保持、主集計の失敗を成功で置換しない。
5. pilotのtest評価/集計を拒否し、実データpilotと主trainが重なり得ることを明記した。rare groupの件数0はnull/empty_groupで記録する。
6. sentinel、数値再現許容、warmup、Elkan全22セル、既定paramsの固定、追加panelを主範囲外とする点を明示した。
7. 実装後・protocol凍結後・最終レポートのOpus/Sol gateを計画checkboxへ落とし、4者承認後の今回pushとskills除外をworkflowへ明記した。
8. Fableのtorch index日時欠落の懸念を一次資料で確認した。[公式cu126 index](https://download.pytorch.org/whl/cu126/torch/)に対象wheelのdata-upload-time=2026-09-30T20:04:15Zがあり、今回は懸念に該当しない。B1で実取得元を再確認する。

判断・修正案はSolへ実行中に共有し支持を確認。Opusの具体修正案も取り込み、変更後の全体を両者とFable/Astraへまとめて最終確認する。文言だけの改善や範囲外の追加を理由に工程を増やさない。

## 最終判定

| レビュワー | 最終判定 / 確認 |
| --- | --- |
| Astra xhigh | APPROVE、重要な未決事項なし。統計、改訂契約、31bit seedと計画整合を確認 |
| Fable xhigh | APPROVE。131.854秒、claude-fable-5-2、成功result。初回8項目の解消を確認 |
| Opus xhigh | APPROVE。93.966秒、claude-opus-5-5、成功result。初回8項目の解消を確認 |
| GPT-6.1 Sol xhigh | APPROVE。初回4項目、data/repeat識別、空group、signed-safe seedを確認 |

Fable/Opusの最終入力は一括修正v2。両者は非blockingとしてFaiss signed seedの31bit化を推奨し、再レビュー不要とした。並行してAstra/Solが同じ点を指摘したため、コミット前に `SHA256先頭4byte big-endian & 0x7fffffff` と対応テスト記述へ先行修正した。Astra/Solはこのv3を読んでAPPROVE。負seedのclock seedingを避ける範囲修正で、他の実験値は変えていない。

非blockingの実装申し送りは、B1 tiny fixtureで凍結config snapshot/job台帳のrun-root上の形式を決めること、B3でmain-isolatedと複数device指定の矛盾を拒否すること、親停止で残るrunningと検知済みworker crashをテストで区別すること。既存のB1契約凍結/B3状態検証の中で実施し、独立した追加工程は作らない。

## コミット範囲と確認

今回作成した設計・計画・調査、エージェント規約、README導線、先に依頼された初期化監査記述/関連ファイルの除去をまとめる。READMEの既存tmux追記、env/container/依存設定、ライブラリ削除等のユーザー変更は含めない。`skills-lock.json`・`.agents/`・`.claude/`は明示的に除外する。

Markdown非空/コードフェンス/参照先、CLAUDE単一import、実装checkbox未実行、diff whitespace、ステージ内容と秘密情報を検査する。READMEの既存workflowリンク3種類の不整合はコピー元由来で、新規不整合がないことを比較する。文書だけのためmake/Python/uvや実験は実行しない。ローカルGitHub認証を使えないため、同じtreeを接続済みGitHub経由でcommit/pushし、ローカルとremoteのcommit/treeを照合する。

今回の実装は未着手。全実装計画checkboxは未実行のまま。
