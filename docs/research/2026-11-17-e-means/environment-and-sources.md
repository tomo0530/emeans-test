# 環境観測と共通の一次資料

観測日: 2026-11-17（JST）。この記録は調査・実験設計用であり、e-meansのインストール、学習、性能測定は実施していない。

## 現環境で確認した事実

| 項目 | 観測 | 解釈・未確認点 |
| --- | --- | --- |
| リポジトリ | `tomo0530/emeans-test`、作業開始時 `main` | テンプレート構成。既存の未コミット変更があり、本調査では保持 |
| GPU | `Tesla V100-SXM2-32GB` ×8、driver `535.154.05` | `nvidia-smi` の可視値。8枚の専有権・空き容量・使用許可を意味しない |
| GPU世代 | V100 = compute capability 7.0 | [NVIDIA公式表](https://developer.nvidia.com/cuda/gpus/legacy)。上流のSM80/SM90経路と区別 |
| CPU | Xeon E5-2698 v4 @2.20GHz、2 socket、計40物理core/80 logical CPU | `lscpu`。利用可能なCPU quota、共有負荷、NUMA制約は未確認 |
| RAM | MemTotal 528,251,236 kB（約504 GiB） | `/proc/meminfo`。共有ホストの可視値 |
| 空き容量 | repo側 約909 GB、`/tmp` 側 約28 GB | `df -h` の一時点。データと大きい一時ファイルは前者に予算を取る |
| 実行場所 | `/.dockerenv` あり、`docker` コマンドなし | 既存コンテナ内と推測。想定開発コンテナへの接続・ホストDockerの有無は未確定 |
| Python指定 | root / project_module とも `>=3.10,<3.13` | [root設定](../../../pyproject.toml)、[module設定](../../../project_module/pyproject.toml) |
| GPU compose指定 | CUDA `12.8.1-cudnn-runtime-ubuntu24.04`、Python `3.12.15` | [compose](../../../context/gpu/docker-compose.yml)。実際の実行中image/versionを証明しない |
| 依存状況 | `uv.lock` にNumPyあり。torch/faiss/triton/scikit-learnのpackage名は見当たらず | ロックの検索結果。実環境への導入有無とは別 |
| 依存方針 | `exclude-newer = "1 week"`、依存追加は別途承認 | 既存設定を変更しない。API調査の新規コードとPyPI配布日も分けて記録 |

確認に使ったのは `git status`、`git branch --show-current`、`git remote get-url origin`、`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`、`lscpu`、`/proc/meminfo`、`df -h`、設定ファイルの読み取り。Gitは実行ユーザーと所有者が異なるため、対象repoに限定した `git -c safe.directory=/home/tomotaka.harada/emeans-test ...` を使用し、global設定は変更していない。

## 上流を固定する根拠

- 原論文: [e-means: outperforming k-means (in one-tenth the epochs), arXiv:2611.09501v1](https://arxiv.org/abs/2611.09501v1)。2026-11-06公開。論文のGPU実験はH100 80GBであり、V100の実測結果ではない。
- 今回読んだFaiss main: [`c8f283867285af547db7a701c1b8dfa89631b3ea`](https://github.com/facebookresearch/faiss/commit/c8f283867285af547db7a701c1b8dfa89631b3ea)（commit日時2026-11-16）。
- 安定版候補: [Faiss v1.16.0](https://github.com/facebookresearch/faiss/releases/tag/v1.16.0)。release APIの公開日時は2026-11-10、本文changelog日付は2026-11-09。release日、tag、main SHA、PyPI artifactの中身を同一視しない。
- [e_means README（固定SHA）](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/README.md)、[再現benchmark群](https://github.com/facebookresearch/faiss/tree/c8f283867285af547db7a701c1b8dfa89631b3ea/benchs/e_means)。
- [Faiss LICENSE](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/LICENSE)はMIT。これは別配布のデータセットや第三者コードの許諾を代替しない。

## 最初に解消すべき互換性

固定SHAの [`Emeans._resolve_input_dtype`](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_torch.py#L279) は、通常のCUDA FP32入力を自動BF16変換する。[backend選択](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/torch_backend.py#L490)はSM80未満のFP16/BF16自動経路を拒否する。したがって「CUDAが見えるのでQuick startをそのまま実行できる」とは判断しない。[明示的 `quantizer=None`](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/contrib/e_means/e_means_quantized.py) のnative FP32経路とNumPy CPU経路を適合確認候補にする。これらも今回の実行確認はない。

PyTorchは2026-09-04の[公式告知](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)で、2.15以降のcu126既成バイナリ廃止とVoltaへの影響を説明している。[2026-01-15の告知](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)は2.11のcu128でVoltaを対象外とする表を掲載している。[公式の過去版案内](https://pytorch.org/get-started/previous-versions/)には2.14.1のcu126取得方法が掲載されている。候補はcu126/SM70を含む2.14以前の実物artifactから選ぶが、e-meansが使う演算/APIも満たすことを事前確認する。最新版の無指定インストールは再現手順にしない。

driver535とcomposeのCUDA12.8という数値だけで成否を断定しない。[NVIDIAのminor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)にはCUDA12.xのR525以降という枠があるが、新機能・PTXに制約がある。image起動、対象wheelのSM70収録、実際の小演算、e-meansの適合確認を別々に記録する。

## 参考Notion記事から借りる報告構成

- [新規技術検証一覧](https://app.notion.com/p/2548dc874e2a428da4bab958bf52e116)
- [自己進化AIエージェントフレームワークを試してみた](https://app.notion.com/p/3596fbf10b7280bd9d07e2ed15dec15a)
- [State-Aware RAGを試してみた](https://app.notion.com/p/3a46fbf10b7280ceb429d4fa793a1a97)

接続済みNotion経由で取得し、記事の概要、比較対象の選定、実験条件、仮説、結果、限界、実務上の示唆を参照した。特に主仮説と探索仮説の区別、同じ評価条件、品質と費用の関係、失敗例、「実証した範囲／未実証」の切り分けを今回の設計へ反映する。他技術の記事にある性能値はe-meansの証拠として使用しない。Notionへの作成・更新は行っていない。

## 今回の変更・実行範囲

調査Markdownと、ユーザーが指定したリポジトリ初期化・設定監査の必須記述の削除のみ。通常のDocker開発手順と安全規則は維持する。依存追加、データセット取得、学習、ベンチマーク、ホストPython/uv/make、Git commit/PR、GitHub設定変更は実施していない。検証プランにあるコマンド・閾値・所要時間は、今後の提案または見積りである。
