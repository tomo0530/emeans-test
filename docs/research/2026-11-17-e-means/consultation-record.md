# 独立相談の方法・実行記録

基準日: 2026-11-17。ユーザー指定のモデル/effortを使用。代替モデルへの切替は行っていない。調査結果・提案は実測の代わりにしない。

## 担当と実行条件

| 役割 | 指定 | 起動時の指定/観測 | 保存先 |
| --- | --- | --- | --- |
| 独立プラン | GPT-6 Astra / xhigh | subagent `model=gpt-6-astra, reasoning_effort=xhigh, fork_turns=none` | [独立案](plan-gpt-6-astra.md) |
| 独立プラン | Claude Fable / xhigh | CLI `--model fable --effort xhigh`、最終JSONのmodelUsage `claude-fable-5-2` | [独立案](plan-claude-fable.md) |
| 独立プラン | Gemini 4 Carbon (High) | `agy models` で表示名を解決し `gemini-4-carbon-high` を指定 | [未取得・試行結果](plan-gemini-4-carbon-high.md) |
| 独立調査 | GPT-6.1 Sol / xhigh | subagent `model=gpt-6.1-sol, reasoning_effort=xhigh, fork_turns=none` | [独立調査](research-gpt-6.1-sol.md) |
| 独立調査 | Claude Opus / xhigh | CLI `--model opus --effort xhigh`、主回答のmodelUsage `claude-opus-5-5` | [独立調査](research-claude-opus.md) |
| 独立調査 | Gemini 4 Hydrogen (High) | `agy models` で表示名を解決し `gemini-4-hydrogen-high` を指定 | [独立調査](research-gemini-4-hydrogen-high.md) |

GeminiのHighはモデル表示名のvariantとして選択した。別の `--effort` を推測で足していない。Gemini最終JSONにはcanonical model名がなく、指定IDと成功結果を実行証拠として記録する。

Claude Opusのusageには `claude-haiku-5-5` も現れる。Web取得・検索の補助処理を含むCLIの記録で、独立報告として保存したのはOpus呼出しの最終 `result` のみ。補助応答や途中のthinkingを独立したOpus意見として転載していない。

## 独立性

- GPTの両担当はfresh contextで起動し、他の成果物を読まず、自分の担当Markdownのみ作成するよう指示した。
- プラン3件に渡した共通資料は、環境観測、Notion参考記事の報告構成、固定候補の上流コード・README、原論文のURLと事実抜粋。ほかの担当の回答は渡していない。
- プラン作成と最新調査を並行して依頼した。**各プランは最新調査3件の結論を読み合わせて作った合議案ではない**。新たに得た研究の採否は、全体を見た親の [統合プラン](synthesis.md) に記録した。
- Astra/GPT調査には途中でPyTorchとNVIDIAの一次資料URLを追加した。他モデルの分析や提案は渡していない。
- Claudeは `--safe-mode --no-session-persistence` でセッション継続をしない。FableはEvidence modeで `--tools=""`、Opusの独立研究は読み取りのWebSearch/WebFetchだけを許可した。
- Geminiは新規のheadless呼出しを各1件、共通資料だけを置いたscratch directoryで実施。shell実行、ファイル編集、外部書込みを禁じ、`--sandbox --mode plan` とした。回答ファイルをそのworkspaceに置いていない。
- 親は重要な採用候補を原論文・固定ソース・公式licenseへ戻って照合した。CLIのSUCCESSは内容の正しさを保証しないため、訂正は各文書の冒頭と統合プランに明示した。

## 指示した内容

調査担当には、2026年を重視したEMA/mini-batch clustering、GPU k-means、Faiss/IVF、埋め込み・データキュレーションの一次資料を調べ、公開日・更新日、直接URL、実装・データの許諾、GPU条件、採否、未確認点を示すよう依頼した。2025以前の基礎文献と2026年の新規研究/改訂を区別し、件数合わせで無関係な研究を増やさないことを指定した。

プラン担当には、主仮説と探索仮説、公平なbaseline、データ/分割、実際のAPI・dtype/backend、V100適合、時間/品質/メモリ、複数seedと不確実性、IVF下流、概算予算、停止条件、repo配置、参考Notion形式の報告目次を求めた。性能の捏造、H100値のV100への外挿、実装・依存変更・実験実行を禁じた。

## 終了結果の確認

| 呼出し | 終了・回答検査 |
| --- | --- |
| Claude Fable | exit 0、`type=result, subtype=success, is_error=false`、非空result、permission_denials=[]。1turn、約371.5秒 |
| Claude Opus | exit 0、同じsuccess条件、非空result、permission_denials=[]。47turn、約481.0秒 |
| Gemini Hydrogen | exit 0、`status=SUCCESS`、非空response、stderr空。約602.2秒 |
| Gemini Carbon | 初回・再試行ともexit 3、JSONのERROR・空response、print_timeout。再試行は出力上限超過も記録。完成回答なし、途中出力は未採用 |
| GPT-6.1 Sol | 親への完了報告と担当ファイル実在・本文を確認 |
| GPT-6 Astra | 親への完了報告と担当ファイル実在・本文を確認 |

Claudeは600秒のtimeout、Geminiは20分のprint timeoutを指定。Carbonの初回は20分でタイムアウトしたため、同じモデルで一度だけ再試行。原典・環境の抜粋をpromptへ直接渡し、ファイル探索・Web取得を求めず、完成プランの出力に範囲を絞った。ほかの回答は渡していない。再試行も20分の制限に達し、出力上限超過のerrorと空responseが返った。usageのoutput_tokens=65,536、thinking_tokens=65,533を記録したが最終回答ではない。未成功の途中出力を成果物として扱わない。ローカルのask-geminiスキル（`.agents/skills/ask-gemini/SKILL.md`） の “Retry once only” に従い追加試行は停止した。生JSON/認証状態/セッション情報はrepoへ保存せず、最終回答と必要最小限のモデル情報だけを保存する。

## 文書変更と検証範囲

ユーザーの追加指示に従い、AGENTSの初期化・設定監査ゲート、対応する状態ファイル・監査ガイド、およびREADME日英/テンプレート運用文書の参照を削除した。通常のDocker開発手順、秘密情報・依存追加・破壊的操作の規則は維持した。`CLAUDE.md` 2件は `@AGENTS.md` 一行のまま。

このタスクはMarkdown/指示文書の変更のみ。e-meansの依存導入、データ取得、学習、ベンチマーク、コミット、PR、Notion更新、GitHub設定変更は実施していない。`make test/fix/all-lint` は実行しておらず、それらの成功を主張しない。文書に対しては参照切れ、初期化ゲートの残存、差分の空白を別途確認する。

README日英には既存のworkflowリンク切れが各3種類あり、HEADとの比較で今回の追加ではないと確認した。最終確認では新規Markdown10件の実在・非空、コードフェンス、全ローカルリンク、CLI成功/失敗記録の一致、CLAUDEの1行import、初期化関連2ファイルの削除を確認し、問題0件。旧初期化ファイル名へのrepo内参照は0件。`git diff --check` も終了コード0だった。今回の変更で新しい参照切れは検出されていない。
