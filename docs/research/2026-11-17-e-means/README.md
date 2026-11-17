# e-means 技術検証の調査・計画

基準日: **2026-11-17**。現環境で検証するための独立調査3件と独立プラン3件を依頼した記録。**成功回答は調査3件・プラン2件。Gemini Carbonは2回のタイムアウトにより未取得**。実験・依存導入は未実施。

最初に [統合プラン](synthesis.md) と [環境・一次資料](environment-and-sources.md) を読むと、実施順序、仮説、判定基準、データ、比較条件、停止条件を確認できる。各モデルの独立回答は原案として残し、親による照合・訂正・採否を統合プランと各回答冒頭に記載した。

| 役割 | 指定モデル | effort / variant | 成果物 |
| --- | --- | --- | --- |
| 独立プラン | GPT-6 Astra | xhigh | [プラン](plan-gpt-6-astra.md) |
| 独立プラン | Claude Fable | xhigh | [プラン](plan-claude-fable.md) |
| 独立プラン | Gemini 4 Carbon | High | **未取得**: [試行結果](plan-gemini-4-carbon-high.md) |
| 独立調査 | GPT-6.1 Sol | xhigh | [最新研究・実装の調査](research-gpt-6.1-sol.md) |
| 独立調査 | Claude Opus | xhigh | [最新研究・実装の調査](research-claude-opus.md) |
| 独立調査 | Gemini 4 Hydrogen | High | [最新研究・実装の調査](research-gemini-4-hydrogen-high.md) |

[相談記録](consultation-record.md) に、起動ID、CLIの完了確認、共通資料、独立性、各回答の制限、再試行、文書検証を記載した。独立プラン作成時に、ほかのプランや調査担当の結論は渡していない。最新調査を踏まえた最終的な選択は親の統合案である。

現時点の検討結果は、**CPU FP32 → 適合確認済みのV100 native FP32 → 埋め込み → IVF検索**の段階化。e-meansの既定CUDA入力変換とV100の対応範囲を確認する必要があり、H100での論文倍率をこの環境の予測値には使わない。

参考にしたNotionは [新規技術検証](https://app.notion.com/p/2548dc874e2a428da4bab958bf52e116)、[自己進化AIエージェント比較](https://app.notion.com/p/3596fbf10b7280bd9d07e2ed15dec15a)、[State-Aware RAG](https://app.notion.com/p/3a46fbf10b7280ceb429d4fa793a1a97)。報告構成の参考として読み、Notionへの書込みは行っていない。
