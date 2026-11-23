# DS検証コード インテグレーション契約

## 1. 目的

本ドキュメントは、DS が作成した検証コードを、MVP 開発や本番開発へ
安全に受け渡すための最小契約を定める。

目的は、Notebook や検証スクリプトをそのまま渡すのではなく、以下の形で
受け渡せるようにすること。

```text
Pydantic Input
  -> invoke(...)
  -> Pydantic Output
```

この契約は、LLM、ML モデル、ルールベース、前処理、後処理、外部 API
呼び出しのいずれにも適用できることを目指す。

特に重要なのは、DS ロジック本体とプロダクト環境依存を分離することで
ある。DS はロジックの中身と入出力契約に責任を持ち、ENG はその契約を
安定したインターフェースとして利用できる状態を作る。

## 2. 基本原則

- 入力は Pydantic モデルで定義する
- 出力は Pydantic モデルで定義する
- 処理本体は `invoke(input_data: Input) -> Output` を基本形とする
- 処理固有の設定は `Config` として分離する
- モデル、設定、外部リソースの解決は処理本体から分離する
- 外部システム依存は adapter または注入される下位コンポーネントで吸収する
- 接続に変換が必要なら mapper を明示的に挟む
- 破壊的変更はバージョンを上げる

## 3. 責務分界

DS 検証コードをプロダクトへ搭載する際は、DS と ENG の責務境界を明示する。

| 領域 | 主責任 | 例 |
| --- | --- | --- |
| DS ロジックの内部実装 | DS | 特徴量生成、推論、スコアリング、ルール判定、後処理 |
| DS ロジックの入出力仕様 | DS 主導、ENG レビュー | 入力データ型、出力データ型、境界条件 |
| DS ロジックが必要とする依存 I/F | DS・ENG 共同 | モデル読み込み I/F、マスタ参照 I/F、外部 API 呼び出し I/F |
| 検証環境向けの依存 I/F 具象実装 | DS | ローカルファイル、検証用 CSV、検証用モデル |
| プロダクト環境向けの依存 I/F 具象実装 | ENG | S3、DB、Secrets Manager、Feature Store |
| DS ロジックへの依存注入方法 | ENG 主導、DS レビュー | factory、設定ファイル、環境別 provider |
| プロダクト側の呼び出し設計 | ENG | API、バッチ、ワーカー、非同期処理、キャッシュ、リトライ |
| 品質評価 | DS 主導、ENG 協力 | 精度評価、回帰テスト、受け入れ基準、テストデータ |

DS がプロダクトアーキテクチャ全体を設計する必要はない。一方で、ENG が
取り込みやすい形で DS ロジックを提供する責任は DS 側にある。

## 4. 受け渡し単位

DS から受け渡す単位は、Notebook ではなく、次の 4 点を備えた処理部品と
する。

- `Input` Pydantic モデル
- `Output` Pydantic モデル
- `Config` Pydantic モデル
- `invoke()` を持つ実装

最小イメージ:

```python
from pydantic import BaseModel


class StepInput(BaseModel):
    text: str


class StepOutput(BaseModel):
    normalized_text: str


class StepConfig(BaseModel):
    lowercase: bool = True


class Step:
    def __init__(self, config: StepConfig) -> None:
        self.config = config

    def invoke(self, input_data: StepInput) -> StepOutput:
        text = input_data.text
        if self.config.lowercase:
            text = text.lower()
        return StepOutput(normalized_text=text)
```

## 5. 標準インターフェース

### 5.1 `BaseInvokable`

DS ロジックは、プロダクトから呼び出し可能な `BaseInvokable` として提供する
ことを標準とする。

```python
from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel, contravariant=True)
OutputT = TypeVar("OutputT", bound=BaseModel, covariant=True)


class BaseInvokable(ABC, Generic[InputT, OutputT]):
    InputType: ClassVar[type[BaseModel]]
    OutputType: ClassVar[type[BaseModel]]

    @abstractmethod
    def invoke(self, input_data: InputT) -> OutputT:
        """Run logic for one logical input."""
        raise NotImplementedError
```

DS ロジックは、この `BaseInvokable` を継承し、型引数で入力型・出力型を
指定して実装する。

```python
class FooInput(BaseModel):
    text: str


class FooOutput(BaseModel):
    normalized_text: str


class FooLogic(BaseInvokable[FooInput, FooOutput]):
    def invoke(self, input_data: FooInput) -> FooOutput:
        return FooOutput(normalized_text=input_data.text.lower())
```

非同期処理が必要な場合は、同期版と混ぜずに `BaseAsyncInvokable` として分ける。

```python
class BaseAsyncInvokable(ABC, Generic[InputT, OutputT]):
    InputType: ClassVar[type[BaseModel]]
    OutputType: ClassVar[type[BaseModel]]

    @abstractmethod
    async def ainvoke(self, input_data: InputT) -> OutputT:
        """Run async logic for one logical input."""
        raise NotImplementedError
```

サブクラスは、継承時の型引数と `invoke()` の型ヒントで入力型・出力型を
明示する（例: `class FooLogic(BaseInvokable[FooInput, FooOutput])`）。
契約が求める「入出力型の明示」はこれで満たされる。

サブクラスでの `InputType` / `OutputType` への代入は省略してよい。代入する
のは、実行時に schema を参照する用途（API 化、テスト自動化、registry 化）が
ある場合のみである。未代入のまま参照すると `AttributeError` になる。

なお、`base.py` 側の注釈が `ClassVar[type[BaseModel]]` なのは、`ClassVar` に
TypeVar を含められないという型仕様上の制約による（`BaseFactory.ConfigType`
も同様）。サブクラスを書く側が意識する必要はない。

本テンプレートには、`BaseInvokable` / `BaseFactory` の共有実装が
`project_module/src/project_module/base.py` として最初から同梱されている。
本ドキュメントのコード例は仕様の提示であり、実装時は同梱のものを
import して用いる（全コンポーネントが同一の基底クラスを共有することで、
registry 化や `isinstance` による統一的な取り扱いが可能になる）。

### 5.2 Input / Output

DS は、`invoke()` の入力と出力それぞれについて Pydantic モデルクラスを
定義する（命名は `FooInput` / `FooOutput` のように用途がわかる名前とする）。

必須:

- `pydantic.BaseModel` を継承していること
- 各フィールドの型、必須/任意、デフォルト値が定義から読み取れること
- ロジックを 1 レコード、バッチ、ファイルのどの粒度で呼び出すかが分かること
- `dict[str, Any]` のような無型のフィールドを用いないこと（型付けが困難な場合は
  docstring で補う。詳細は [`docs/agents/ml-and-data.md`](./agents/ml-and-data.md)）

推奨:

- `Field(...)` に説明を付ける
- `ge`, `le`, `min_length`, `max_length` などの制約を付ける
- 単位やタイムゾーンを docstring または field description に書く
- 欠損値、判定不能、低信頼度の表現方法を明示する

例:

```python
from pydantic import BaseModel, Field


class PricePredictionInput(BaseModel):
    maker: str
    model: str
    grade: str | None = None
    year: int = Field(ge=1900)
    mileage: int = Field(ge=0)


class PricePredictionOutput(BaseModel):
    predicted_price: int
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
```

### 5.3 `invoke`

処理の公開インターフェースは原則として `invoke` に統一する。

標準形:

```python
def invoke(self, input_data: InputModel) -> OutputModel: ...
```

必要に応じて、実行制御のための `context` を追加してよい。

```python
def invoke(
    self,
    input_data: InputModel,
    context: RunContext | None = None,
) -> OutputModel: ...
```

ただし、`context` にはモデル固有パラメータを入れない。モデル名、閾値、
プロンプト、辞書パスなどの処理固有設定は `Config` に置く。なお、
`RunContext` は例示であり、必要になった段階でプロジェクト側で定義する。

### 5.4 Config

処理固有の設定値は `Config` に閉じ込める。

例:

- LLM: `model_name`, `temperature`, `max_output_tokens`
- ML: `model_name`, `model_version`, `threshold`, `device`
- Rule: `dictionary_name`, `dictionary_version`, `score_threshold`

共通インターフェース側には置かない。

### 5.5 `BaseFactory`

DS ロジックの生成には Factory を用いる。Factory は、生成に必要な `Config`
を受け取り、`BaseInvokable` に準拠した実行可能コンポーネントを返す。

```python
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)
# InputT が反変のため、bound の型引数は BaseModel ではなく Any にする
# （strict mypy では型引数の省略も [type-arg] エラーになる）
InvokableT = TypeVar("InvokableT", bound=BaseInvokable[Any, Any])


class BaseFactory(ABC, Generic[ConfigT, InvokableT]):
    ConfigType: ClassVar[type[BaseModel]]

    @abstractmethod
    def build(self, config: ConfigT) -> InvokableT:
        """Build an invokable instance from config."""
        raise NotImplementedError
```

Factory は DS ロジックと実行環境の接合点である。ただし、Factory 自体を
検証環境用・プロダクト環境用に安易に分けない。まずは Factory に注入する
下位コンポーネントの具象実装を差し替える。

Factory を分けるのは、環境ごとに組み立てる object graph が本質的に異なる
場合に限る。

例:

- プロダクト環境では監視、キャッシュ、リトライ用 wrapper を追加する
- 検証環境とプロダクト環境で依存コンポーネントの構成が大きく異なる
- プロダクト環境では複数の外部リソースや secret 解決を組み合わせる
- テナント、環境、実験条件によって生成する object graph が大きく変わる

## 6. 依存関係と外部 I/O

DS ロジック本体は、モデルファイルの保存場所、設定値の取得元、外部 API、
DB、S3、Secrets Manager などを直接知らないようにする。

検証環境とプロダクト環境では、依存先が必ず変わる。したがって、DS ロジック
本体は具象の保存場所や実行基盤に直接依存せず、必要な依存を抽象 I/F として
受け取る。

例:

```python
from abc import ABC

from pydantic import BaseModel, Field


class ModelLoadInput(BaseModel):
    name: str
    version: str


class PriceModelArtifact(BaseModel):
    """学習済み価格モデル。推論再現に必要な情報を型で明示する（§9）。"""

    artifact_version: str
    coefficients: dict[str, float]
    trained_at: str


class ModelLoadOutput(BaseModel):
    artifact: PriceModelArtifact
    model_version: str
    metadata: dict[str, str] = Field(default_factory=dict)


class BaseModelLoader(
    BaseInvokable[ModelLoadInput, ModelLoadOutput],
    ABC,
):
    InputType = ModelLoadInput
    OutputType = ModelLoadOutput
```

検証環境では、DS がローカルファイルシステム向けの `BaseModelLoader`
実装を用意できる。プロダクト環境では、ENG が S3、モデルレジストリ、DB
などに対応した `BaseModelLoader` 実装を用意できる。

上位の Factory は、低レベルな I/O 依存ではなく、抽象化された loader や
provider に依存する。

```python
class PricePredictionLogicConfig(BaseModel):
    model_name: str
    model_version: str
    threshold: float = Field(ge=0.0, le=1.0)


class BasePricePredictionLogic(
    BaseInvokable[PricePredictionInput, PricePredictionOutput],
    ABC,
):
    InputType = PricePredictionInput
    OutputType = PricePredictionOutput


class PricePredictionLogic(BasePricePredictionLogic):
    def __init__(
        self,
        artifact: PriceModelArtifact,
        config: PricePredictionLogicConfig,
    ) -> None:
        self.artifact = artifact
        self.config = config

    def invoke(self, input_data: PricePredictionInput) -> PricePredictionOutput:
        return PricePredictionOutput(
            predicted_price=1234567,
            confidence=0.82,
            reason_codes=["similar_market_price", "mileage_adjustment"],
        )


class PricePredictionFactory(
    BaseFactory[PricePredictionLogicConfig, BasePricePredictionLogic],
):
    ConfigType = PricePredictionLogicConfig

    def __init__(self, model_loader: BaseModelLoader) -> None:
        self.model_loader = model_loader

    def build(
        self,
        config: PricePredictionLogicConfig,
    ) -> BasePricePredictionLogic:
        model_output = self.model_loader.invoke(
            ModelLoadInput(
                name=config.model_name,
                version=config.model_version,
            )
        )
        return PricePredictionLogic(
            artifact=model_output.artifact,
            config=config,
        )
```

この形にすると、検証環境とプロダクト環境の差分は Factory の差分ではなく、
Factory に注入される下位コンポーネントの具象実装差分として表現できる。

なお、fitted な scikit-learn モデルのように Pydantic で表現できない
実行時オブジェクトを受け渡す必要がある場合は、任意型の `object` で
逃げるのではなく、必要な操作を `Protocol` として定義した上で
`model_config = ConfigDict(arbitrary_types_allowed=True)` を併用するか、
Pydantic 境界の外側の内部依存（コンストラクタ注入のみで受け渡す）として
扱う。

## 7. Mapper / Adapter 契約

### 7.1 Mapper

前段の `Output` が後段の `Input` と一致しない場合、暗黙変換を避け、
mapper を 1 つの処理部品として明示する。

```text
StepAOutput -> MapperInput
MapperOutput -> StepBInput
```

これにより、どこでスキーマ変換しているかを明示できる。

### 7.2 Adapter

外部 I/O は処理本体から分離する。

分離対象の例:

- S3 読み書き
- DB 読み書き
- API 呼び出し
- Queue / Batch / Workflow 連携
- 認証・認可

処理本体は、できるだけ論理入力を受け取り、論理出力を返す形に寄せる。

```text
External Input
  -> Adapter
  -> Pydantic Input
  -> invoke()
  -> Pydantic Output
  -> Adapter
  -> External Output
```

## 8. データ契約

Pydantic モデルだけでは足りない場合、あわせて論理データ契約を定める。

最低限明記するもの:

- カラム名 / フィールド名
- 型
- 必須/任意
- Null 許容可否
- 主キー相当
- 時刻の意味とタイムゾーン
- 数値の単位
- 列挙値の定義

重要なのは、物理形式ではなく論理形式を固定すること。

物理形式の例:

- CSV
- Parquet
- DB row
- API request / response

これらは adapter 層で吸収してよい。

## 9. Artifact 契約

学習済みモデルや中間成果物を受け渡す場合は、artifact にも契約を持たせる。

最低限必要な情報:

- artifact の entrypoint
- load 方法
- 依存バージョン
- 学習時の主要設定
- 入力特徴量定義
- 出力の意味

最低限の原則:

- artifact は再読込可能であること
- artifact だけで推論再現に必要な情報が追えること
- artifact format の破壊的変更はバージョンを上げること

## 10. エラー・判定不能・副作用

### 10.1 エラーと判定不能

モデルやロジックは常に正常な答えを返せるわけではない。契約では、少なく
とも以下の状態をどう表現するかを明示する。

- 入力不正
- 前提条件不足
- 推論不能
- 信頼度不足
- 外部依存の失敗
- タイムアウト
- モデルロード失敗
- 想定外エラー

特に、「例外」と「業務上の判定不能」を分ける。入力 schema 違反や外部依存
障害は例外として扱う余地がある。一方で、モデルが十分な信頼度で判定できない
状態は、`Output` 上の状態として表現することを検討する。

### 10.2 副作用

`invoke()` は、可能な限り副作用を小さくする。

避ける例:

- `invoke()` 内で直接 DB に書き込む
- `invoke()` 内で直接ファイルを保存する
- `invoke()` 内でグローバルな環境変数を変更する
- `invoke()` 内で外部 API を暗黙的に呼ぶ
- `invoke()` 内でログ以外の永続的な状態変更を行う

副作用が必要な場合は、その責務を DS ロジックに持たせるべきか、プロダクト
側に持たせるべきかを明示的に設計する。

## 11. バージョニング

次の 3 つは分けて管理する。

- 処理ロジックのバージョン
- Input / Output 契約のバージョン
- artifact format のバージョン

ルール:

- `Input` または `Output` の破壊的変更は契約バージョンを上げる
- artifact の破壊的変更は artifact version を上げる
- 同名の処理部品の意味を変えない

セマンティックバージョニングまでは求めない。破壊的変更が生じたときに
`v1` → `v2` と整数で上げる運用で十分である。

- Input / Output 契約: モデルクラス名に付す（例: `FooInputV2`）
- artifact format: 保存先に付す（例: `artifacts/foo_model_v2/`）
- 処理ロジック: 破壊的変更なら別クラス（例: `FooLogicV2`）として並存させ、
  移行完了後に旧版を削除する

## 12. テスト契約

最低限、以下を用意する。

### 12.1 Contract テスト

- `Input` が妥当な値を受け付ける
- `Input` が不正値を拒否する
- `Output` が想定 schema を満たす

### 12.2 Step 単体テスト

- `invoke()` が期待どおりの出力を返す
- `Config` の違いが出力に反映される
- 境界値、判定不能、低信頼度ケースの挙動が確認できる

### 12.3 Integration テスト

- adapter から `Input` へ変換できる
- `invoke()` の結果を downstream が受け取れる
- artifact save / load 後も推論できる
- 検証環境向け依存実装を差し替えても契約が維持される

モデル更新やルール更新がある場合は、代表ケースのスナップショットテストや
回帰テストを用意し、既存挙動がどの程度変わったかを確認できるようにする。

## 13. 成果物構成

DS から ENG に渡す成果物は、少なくとも import 可能なモジュール、テスト、
README を含む。

構成例:

```text
project/
  README.md
  pyproject.toml
  src/
    ds_logic/
      __init__.py
      base.py
      logic.py
      schemas.py
      dependencies.py
  tests/
    test_logic.py
    fixtures/
      sample_input.json
      sample_output.json
  artifacts/
    README.md
  configs/
    default.yaml
```

README には少なくとも以下を含める。

- 何をするロジックか
- 呼び出し方
- 入力仕様
- 出力仕様
- 依存関係
- 初期化方法
- サンプルコード
- テスト実行方法
- 既知の制約
- プロダクト搭載時に ENG と相談が必要な点

## 14. 受け渡しチェックリスト

DS が ENG へ渡す時点で、最低限以下を満たすこと。

- `Input` Pydantic モデルがある
- `Output` Pydantic モデルがある
- `Config` Pydantic モデルがある
- `invoke()` の公開シグネチャがある
- 実行単位が明示されている
- 期待入力例がある
- 期待出力例がある
- 不正入力時の挙動が分かる
- 判定不能や低信頼度の表現方法が分かる
- 外部依存の抽象 I/F と検証環境向け具象実装がある
- artifact がある場合は保存形式と load 方法が分かる
- 既知の制約が明記されている
- 最低限のテストがある

## 15. 実務上の推奨

### 15.1 まずは薄く作る

最初から巨大な共通基盤にしない。

まずは以下だけで十分である。

- `Input`
- `Output`
- `Config`
- `invoke`

プロダクト搭載が見えてきた段階で、Factory、依存 I/F、artifact 契約、
回帰テストを追加する。

### 15.2 Notebook は説明資料、モジュールが本体

Notebook は検証過程の共有には有用だが、インテグレーション契約にはしない。

ENG へ渡す本体は、import 可能なモジュールとテストである。

### 15.3 暗黙仕様を禁止する

次の情報がコード外の口頭説明にしかない状態を避ける。

- 必須入力
- 出力の意味
- 単位
- 前処理前提
- artifact の読み方
- downstream 接続条件
- エラーと判定不能の違い

### 15.4 変換は明示する

前処理、列名変換、集約、型変換、後処理を暗黙に混ぜない。

接続用の変換は mapper または adapter として明示する。

## 16. この契約で目指す状態

最終的には、DS 検証コードが次の形で渡せる状態を目指す。

```text
Config
  -> Factory
  -> BaseInvokable
  -> Pydantic Input
  -> invoke(...)
  -> Pydantic Output
  -> mapper / adapter
  -> next Pydantic Input
  -> invoke(...)
  -> Pydantic Output
```

この形になっていれば、MVP 実装でも本番実装でも、差分は主に下位依存の
具象実装、adapter、orchestration に閉じ込めやすい。

## 17. 想定運用

本契約は、検証開始時点から一律に強制するものではない。フェーズに応じて適用範囲を
広げていくことを想定する。

### 17.1 フェーズ運用

- **探索フェーズ**（notebook 上での試行錯誤）
  契約は適用しない。`Input` / `Output` / `Config` の形を意識せず、自由に検証してよい。
- **プロダクト搭載の目処が立った段階**
  notebook 上のロジックを `project_module/` へ切り出すタイミングで、初めて本契約の形
  （`Input` / `Output` / `Config` + `invoke()`）に揃える。
- **既存ロジックの移行**
  本契約の策定前に書かれた `project_module/` 配下のロジックに対して、一括での全面
  改修は求めない。プロダクト搭載が具体化したタイミングで、該当するロジックのみ
  段階的に本契約へ寄せる。
- **ENG への引き渡し前**
  引き渡し前に、本契約 §14 の受け渡しチェックリストを満たしているかを確認する。

### 17.2 日常運用

- `project_module/` に変更を加えた際は、変更したロジックが本契約に沿っているかを
  軽くセルフレビューすることを推奨する。
- ただし、コミット単位や CI 上での機械的な強制ゲートにはしない。検証速度を落とす
  強制力は持たせず、任意のタイミングで 1 コマンド相当の確認ができる状態にとどめる。

### 17.3 支援ツール

- notebook のロジックを `project_module/` へ切り出す際は `/ds-extract` を用いる。
- 既存の `project_module/` ロジックが本契約に準拠しているかの確認、および必要な
  リファクタの提案には `/ds-contract-check` を用いる。リファクタの適用は常にユーザー
  の確認を経て行う。
- いずれも Claude Code スキルである。`disable-model-invocation` などの frontmatter は
  Claude Code 固有の制御情報である。他のコーディングエージェントは frontmatter を
  実行制御として解釈せず、`.claude/skills/*/SKILL.md` の本文を手順書として読み、
  記載された手順を直接実行する。
