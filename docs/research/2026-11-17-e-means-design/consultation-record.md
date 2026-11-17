# 調査・協議の実行記録

2026-11-17。過去のCarbonプラン取得失敗とは別の、新しいデータ選定依頼である。

| 担当 | 起動指定 / 実モデル | 結果と範囲 |
| --- | --- | --- |
| 独立調査＋設計協議 | gpt-6-astra / xhigh / fresh context | 公開artifact、Python3.14/SM70、データ、派生手法を調査。後でFable要旨とユーザー決定を共有し二層比較を支持 |
| Claude独立調査 | fable / xhigh / safe-mode / WebSearch,WebFetchのみ | exit0、result/success、is_error=false、非空result、permission_denials=[]。実モデルclaude-fable-5-2、Web補助にclaude-haiku-5-5。543.096秒、41turn |
| Claude追加協議 | fable / xhigh / safe-mode / toolsなし | 同じ成功契約、claude-fable-5-2。128.662秒、1turn。新しいユーザー条件とAstra/親の確認事実からAを支持 |
| Geminiデータ調査 | gemini-4-carbon-high / sandbox / plan / 15m | exit0、SUCCESS、非空response、stderr空。847.570秒。モデルIDはagy modelsと起動引数で確認、応答JSONにcanonical model欄なし |
| 計画執筆 | gpt-6.1-terra / xhigh / fresh context | 設計を参照して実装計画のみ執筆。コード・設定・依存・実験は未実施 |

Claudeはrepo root、Carbonは秘密情報を含まないscratch directoryから実行。初回調査はpeerの回答を渡さず、後続協議は共有した情報を明記した。CLIはtimeout、成功JSON、最終本文を検証し、途中出力を回答に数えない。認証情報・生セッションログは保存しない。

## 親の照合と判断

CarbonのarXiv Nomic推薦を採用し、元データCC0の正しい一次出典を[arXiv API規約のmetadata節と脚注](https://info.arxiv.org/help/api/tou.html)へ差し替えた。VIBEコードMIT、embedding CC BY4.0、元metadata CC0、NomicモデルApache2を区別する。Landmarkのindex由来はVIBE生成コードと[Google公式README](https://github.com/cvdfoundation/google-landmark#index-image-licenses)で照合した。

固定HF revision `07b387891a221b7b073b83d2f752b76462e5fa03` のtree APIで両HDF5のsize/LFS SHAを確認。本文の全体download/hash再計算は未実施。arXivは4,135,431,488 bytes、Landmarkは2,341,733,696 bytes。

AstraのCPython3.14 wheel調査を採用。Fable最初の「CPU対照はFaissだけ」という絞込みは、sklearn wheelの存在と予算拡張を踏まえ修正。通常L2とsphericalの相違、Faiss点数上限、FP16/SM70、推定時間を実測としないことを、各回答冒頭に訂正した。

AGNewsは元ニュースの条件に未確認点があるため必須外。DBpediaの第三者embeddingはShareAlike/API規約の確認を増やすため必須外。DEEP96は公開CC BY4.0の低次元圧縮image特徴として追加候補。主データのためにencoder/APIキーを導入しない。

## 参考repoと設計の大きさ

[State-Aware RAG](https://github.com/tomo0530/state-aware-rag-test/blob/main/README.md)のproject_module/configs/experimentation/docs分離、[Dreaming](https://github.com/tomo0530/dreaming-test/blob/main/README.md)の小さいrunner構成を参照。[自己進化Agent設計](https://github.com/tomo0530/self-evolve-agent-test/tree/main/docs/plan/design)はwhat/how/implementationの区別とPIE定義を参照する。サービスや8分冊の構成を機械的に転用せず、今回は設計3文書＋実装計画とした。

本調査・今回の執筆では実装、uv add、Python/import、make、データ本体download、source build、ベンチマークを行っていない。依存とコンテナの既存変更を保持する。文書の最終承認は [レビュー記録](../../plan/review-record.md) にまとめる。
