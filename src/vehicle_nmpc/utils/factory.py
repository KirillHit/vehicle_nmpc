"""Shared factory helpers for configured registries."""

from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, MutableMapping

TBase = TypeVar("TBase")


class ConfigSpec(Protocol):
    """Protocol for configs that can be built via registries."""

    name: str
    params: Mapping[str, object]


class ConfiguredBase(ABC):
    """Base class for components that validate their typed config."""

    Config: ClassVar[type[object]]

    def __init__(self, cfg: object) -> None:
        """Validate and store a typed configuration instance."""
        if not isinstance(cfg, self.Config):
            msg = (
                f"{self.__class__.__name__} expects {self.Config.__name__}, "
                f"got {type(cfg).__name__}"
            )
            raise TypeError(msg)
        self._cfg = cfg


def build_configured_instance(
    cfg: ConfigSpec,
    registry: Mapping[str, type[TBase]],
    base_cls: type[TBase],
    error_cls: type[Exception],
    kind: str,
    *args: object,
) -> TBase:
    """Instantiate a registry entry after validating and config."""
    if not hasattr(cfg, "name") or not hasattr(cfg, "params"):
        missing = [attr for attr in ("name", "params") if not hasattr(cfg, attr)]
        msg = f"Invalid config for {kind}: missing attribute(s) {', '.join(missing)}"
        raise error_cls(msg)
    component_class = registry.get(cfg.name)

    if component_class is None:
        available = ", ".join(sorted(registry)) or "<empty>"
        msg = f"Unsupported {kind} name: {cfg.name}. Available {kind}s: {available}"
        raise error_cls(msg)

    if not issubclass(component_class, base_cls):
        msg = f"Registered class must inherit {base_cls.__name__}, got: {component_class!r}"
        raise TypeError(msg)

    try:
        component_cfg = component_class.Config(**asdict(cfg), **cfg.params)
    except TypeError as exc:
        msg = f"Invalid config for {kind} '{cfg.name}' ({component_class.Config.__name__}): {exc}"
        raise error_cls(msg) from exc

    return component_class(component_cfg, *args)


def register_in_registry(
    name: str,
    registry: MutableMapping[str, type[TBase]],
    base_cls: type[TBase],
    error_cls: type[Exception],
    kind: str,
) -> Callable[[type[TBase]], type[TBase]]:
    """Register a class in a registry with validation."""

    def decorator(cls: type[TBase]) -> type[TBase]:
        if not issubclass(cls, base_cls):
            msg = f"Registered class must inherit {base_cls.__name__}, got: {cls!r}"
            raise TypeError(msg)

        if name in registry:
            msg = f"{kind.capitalize()} name '{name}' is already registered"
            raise error_cls(msg)

        registry[name] = cls
        return cls

    return decorator
