"""Foundry Local SDK startup helpers with WebGPU registration and CPU fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from foundry_local_sdk import Configuration, FoundryLocalManager


WEBGPU_EP = "WebGpuExecutionProvider"
CPU_EP = "CPUExecutionProvider"
_EP_STATUS: "EpSetupStatus | None" = None


@dataclass
class EpSetupStatus:
    attempted: bool = False
    webgpu_registered: bool = False
    status: str = ""
    registered_eps: list[str] = field(default_factory=list)
    failed_eps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ModelLoadStatus:
    alias: str
    model_id: str
    device: str
    execution_provider: str
    used_fallback: bool
    errors: list[str] = field(default_factory=list)


def initialize_manager(app_name: str) -> FoundryLocalManager:
    """Initialize Foundry Local and register GPU EPs when possible."""
    FoundryLocalManager.initialize(Configuration(app_name=app_name))
    manager = FoundryLocalManager.instance
    ensure_webgpu_registered(manager)
    return manager


def ensure_webgpu_registered(manager: FoundryLocalManager) -> EpSetupStatus:
    """Register WebGPU for this SDK process; fall back cleanly on failure.

    The WinML SDK exposes GPU model variants only for EPs registered in the
    current process. The SDK's single-name registration currently fails on this
    machine, so the robust path is to register all discoverable EPs and then
    check whether WebGPU is available.
    """
    global _EP_STATUS
    if _EP_STATUS is not None:
        return _EP_STATUS

    status = EpSetupStatus()
    try:
        discovered = manager.discover_eps()
        if any(ep.name == WEBGPU_EP and ep.is_registered for ep in discovered):
            status.webgpu_registered = True
            status.status = "WebGPU already registered."
            _EP_STATUS = status
            return status

        status.attempted = True
        result = manager.download_and_register_eps()
        status.status = result.status
        status.registered_eps = list(result.registered_eps)
        status.failed_eps = list(result.failed_eps)

        discovered = manager.discover_eps()
        status.webgpu_registered = any(
            ep.name == WEBGPU_EP and ep.is_registered for ep in discovered
        )
        if not status.webgpu_registered:
            status.errors.append("WebGpuExecutionProvider did not register.")
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
        status.errors.append(f"{type(exc).__name__}: {exc}")

    _EP_STATUS = status
    return status


def runtime_info(model: Any) -> tuple[str, str]:
    runtime = getattr(getattr(model, "info", None), "runtime", None)
    if runtime is None:
        return "unknown", "unknown"
    return str(runtime.device_type), str(runtime.execution_provider)


def select_variant_by_ep(model: Any, execution_provider: str) -> bool:
    for variant in model.variants:
        _, ep = runtime_info(variant)
        if ep == execution_provider:
            model.select_variant(variant)
            return True
    return False


def _download(model: Any, progress_callback: Callable[[float], None] | None) -> None:
    if not model.is_cached:
        model.download(progress_callback)


def load_model_with_webgpu_fallback(
    manager: FoundryLocalManager,
    alias: str,
    progress_callback: Callable[[float], None] | None = None,
    prefer_webgpu: bool = True,
) -> tuple[Any, ModelLoadStatus]:
    """Load a model by alias, preferring WebGPU but falling back to CPU."""
    model = manager.catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Model not found: {alias}")

    errors: list[str] = []
    if prefer_webgpu and select_variant_by_ep(model, WEBGPU_EP):
        try:
            _download(model, progress_callback)
            model.load()
            device, ep = runtime_info(model)
            return model, ModelLoadStatus(alias, model.id, device, ep, False, errors)
        except Exception as exc:
            errors.append(f"{model.id}: {type(exc).__name__}: {exc}")
            try:
                model.unload()
            except Exception:
                pass

    if not select_variant_by_ep(model, CPU_EP):
        raise RuntimeError(
            f"No CPU fallback variant found for {alias}. GPU errors: {'; '.join(errors)}"
        )

    _download(model, progress_callback)
    model.load()
    device, ep = runtime_info(model)
    return model, ModelLoadStatus(alias, model.id, device, ep, True, errors)
