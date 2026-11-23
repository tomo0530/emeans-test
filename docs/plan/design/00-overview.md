# e-means 検証設計: 目的と方針

版: 1.0 / 2026-11-17。設計・計画のみ。実測結果は未取得。

## 目的とユーザーが決めたこと

現環境で、e-means が従来の k-means とその派生手法に対し、同等のクラスタ品質を短い時間で得られる条件と、得られない条件を明らかにする。論文の H100 上の倍率を V100 の期待値にはしない。

| 決定 | 内容 |
| --- | --- |
| 主結果 | 品質と実行時間の曲線。MSE +0.5%や速度2倍といった固定の合否幅は置かない |
| 対象 | 特定用途に限定しない文章埋め込み、画像埋め込み、制御可能な合成分布 |
| 計算資源 | 既存 V100 32GB ×8を使用。固定の全体時間上限を設けず、事前の条件表と反復数で完了を定める |
| Python | `>=3.14,<3.15`、通常の CPython。free-threaded 版は対象外 |
| テンプレート | 初期化・設定監査不要。AGENTS/Makefile/環境設定は検証に合わせて更新可 |
| 自律実行 | 依存追加、コマンド、コミット、外部送信等の個別再確認は不要。現時点の実装禁止を優先する |
| 今回の納品 | 設計・実装計画・調査/協議・レビュー記録を作り、指定レビュー承認後にコミットしてpush。実装は開始しない |

技術的な組み立てはユーザーから委任された。以下の具体的なデータ、条件表、seed数、構成は、Astra/Fableとの協議を踏まえた設計者の判断であり、個々をユーザーが指定したとは記載しない。

## 比較したアプローチと選択

| 案 | 得られるもの | 費用・限界 |
| --- | --- | --- |
| A: 実用比較＋計算条件を揃えた比較 | 公開実装の選択材料と、e-meansの更新/バッチ/scheduleの効果を切り分ける材料 | GPU上の薄いLloyd対照と、その数値検証が追加で必要 |
| B: 公開実装の実用比較に集中 | 手法・条件の広い比較 | 実装やdeviceの差を更新則の効果として解釈できない |

**Aを採用する。** ユーザーの判断委任後、Astra xhigh と Fable xhigh が支持した。[Astra](../../research/2026-11-17-e-means-design/astra-feasibility-and-datasets.md)、[Fable追加協議](../../research/2026-11-17-e-means-design/fable-followup-discussion.md)。最初のFable最小CPU案は、時間・GPU予算が広がる前の提案として保存する。

実用比較では公開実装をCPU/GPU別に示し、実装＋deviceを含む環境内の選択材料も別表で示す。機構比較では同じV100、FP32、native assignmentを使う。ただし時間差には更新処理・batch/schedule・メモリ処理の実装も含まれるため、数学的更新則だけの因果効果や最速GPU実装に対する優越とは断定しない。

## 範囲

必須のCPU手法は Faiss Lloyd、NumPy e-means、MiniBatchKMeans、Elkan。GPUはnative FP32 e-meansと薄い全件Torch Lloyd対照。Elkanも全22セルを対象にし、凍結資源境界に不適合ならinfeasibleを残す。全手法を同じ外部MSE評価器で測る。

主文章データは VIBE arXiv–Nomic、主画像データは VIBE Landmark–Nomic。既存ベクトルを使い、encoderや有料APIを呼ばない。合成分布でn/d/kと分布形状を分けて変え、実データ間の違いだけから次元の因果効果を論じない。データ採否の根拠と具体条件は [実験仕様](01-experiment-protocol.md) に置く。

SuperKMeansは本体MITでも依存Eigen/MPL2等とビルド固定の問題があるため初期必須比較にしない。Flash-KMeansの現行Triton経路はV100を対象にしない。これらを未実行のまま性能比較表に並べず、除外理由を示す。[互換性と依存の調査](../../research/2026-11-17-e-means-design/astra-feasibility-and-datasets.md)

追加候補はDEEP96、CPUスレッド数・初期化・e-means revivalの感度、Faiss mainのCPU版差。IVF検索・分散学習・量子化・他GPU調達は主評価の完了条件に含めない。追加は主条件を凍結した後の別実験版とし、主結果を置き換えない。

## 設計原則

- **SOLID**: データ準備、手法adapter、独立評価、run管理、集計の責務を分ける。最小の関数契約で接続し、全adapterに不要な能力を要求しない。
- **KISS**: YAML設定、少数のPythonモジュール、ファイル成果物で足りる。汎用plugin基盤やサービス群を作らない。
- **YAGNI**: 8GPUは独立trialに配る。分散k-meansやIVF基盤は初期実装に持ち込まない。
- **DRY**: 距離、split、初期中心、MSE、計時、artifact schemaの正本を共通化する。実験値の正本は設定とmanifest、説明上の正本は本仕様。
- **PIE**: ユーザー指定の[参考設計](https://github.com/tomo0530/self-evolve-agent-test/blob/main/docs/plan/design/00_overview.md)での可読性原則（Proper Indentation, English etc）として適用する。意味のある命名、浅い制御構造、前提と理由の記述を優先する。

## 失敗と完了

結果が「e-meansが負ける」「一部条件でしか有利でない」「GPU非適合」でも有効な成果とする。自己申告obj、異なる初期値、隠れたsubsampling、CPU/GPU混在倍率、失敗runの除外によって勝ちを作った場合は検証の失敗である。

実行段階では環境適合→pilot→条件表/設定凍結→本評価→独立集計/レポートを行う。必要条件が未実証のときは部分完了を明記する。GPU適合失敗をCPU結果だけで充足したと扱わない。今回の完了はドキュメントの整合・指定レビュー・コミット/pushまで。
