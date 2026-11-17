"""Shared base interfaces defined by the DS integration contract.

See ``docs/ds_integration_contract.md`` (section 5). Every component
under ``project_module`` imports these classes instead of redefining
them, so all DS logic shares one runtime interface (``isinstance`` /
registry friendly).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel, contravariant=True)
OutputT = TypeVar("OutputT", bound=BaseModel, covariant=True)


class BaseInvokable(ABC, Generic[InputT, OutputT]):
    """Smallest callable unit handed off from DS to ENG.

    Notes
    -----
    ``ClassVar`` cannot contain type variables, so ``InputType`` /
    ``OutputType`` are annotated as ``type[BaseModel]`` and narrowed by
    assignment in concrete subclasses.
    """

    InputType: ClassVar[type[BaseModel]]
    OutputType: ClassVar[type[BaseModel]]

    @abstractmethod
    def invoke(self, input_data: InputT) -> OutputT:
        """Run logic for one logical input."""
        raise NotImplementedError


ConfigT = TypeVar("ConfigT", bound=BaseModel)
# InputT is contravariant, so the bound is parameterised with Any
# (a ``[BaseModel, BaseModel]`` bound would reject concrete logics).
InvokableT = TypeVar("InvokableT", bound=BaseInvokable[Any, Any])


class BaseFactory(ABC, Generic[ConfigT, InvokableT]):
    """Builds an invokable component from its config.

    Environment differences (local files vs. S3 / DB) are expressed by
    the lower-level ``BaseInvokable`` implementations injected into the
    factory, not by swapping the factory itself.
    """

    ConfigType: ClassVar[type[BaseModel]]

    @abstractmethod
    def build(self, config: ConfigT) -> InvokableT:
        """Build an invokable instance from config."""
        raise NotImplementedError
