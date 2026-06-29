"""Foundry Local SDK startup helpers — CPU-first, with an opt-in WebGPU path.

GPU status on this build (foundry-local-sdk-winml 1.2.3 / onnxruntime-genai
0.14.1): GPU inference is NOT available.
  * CUDA: the GenAI CUDA companion (onnxruntime-genai-cuda.dll) fails to load and
    has no Blackwell sm_120 kernels.
  * WebGPU: the shipped onnxruntime-genai.dll raises
    "WebGPU execution provider is not supported in this build."
So models are loaded on CPU. Registering GPU execution providers is therefore
both useless (no working GPU GenAI runtime) and harmful (it makes the SDK
catalog expose GPU variants and sort one first, so an unguarded
``get_model(alias).load()`` would pick a GPU variant and crash). We keep the
WebGPU machinery behind the opt-in ``AUTOFLASH_TRY_WEBGPU`` env var so it will
"just work" if a future SDK build adds WebGPU GenAI support, but by default we
register nothing and select the CPU variant explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from foundry_local_sdk import Configuration, FoundryLocalManager


WEBGPU_EP = "WebGpuExecutionProvider"
CPU_EP = "CPUExecutionProvider"
TRY_WEBGPU_ENV = "AUTOFLASH_TRY_WEBGPU"
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


def webgpu_opt_in() -> bool:
    """True only if the user opted into the (currently non-functional) WebGPU path."""
    return os.environ.get(TRY_WEBGPU_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def initialize_manager(app_name: str) -> FoundryLocalManager:
    """Initialize Foundry Local. GPU EPs are registered only when opted in."""
    FoundryLocalManager.initialize(Configuration(app_name=app_name))
    manager = FoundryLocalManager.instance
    if webgpu_opt_in():
        ensure_webgpu_registered(manager)
    return manager


def ensure_webgpu_registered(manager: FoundryLocalManager) -> EpSetupStatus:
    """Register WebGPU for this SDK process; fall back cleanly on failure.

    Only called when ``AUTOFLASH_TRY_WEBGPU`` is set. The WinML SDK exposes GPU
    model variants only for EPs registered in the current profile. The SDK's
    single-name registration is broken (the core reports "Unknown EP
    bootstrapper name(s)"), so the only working path is to register all
    discoverable EPs and then check whether WebGPU is available.
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
    prefer_webgpu: bool | None = None,
) -> tuple[Any, ModelLoadStatus]:
    """Load a model by alias on CPU (the working path), or WebGPU when opted in.

    With WebGPU opted in (``AUTOFLASH_TRY_WEBGPU``), the WebGPU variant is tried
    first and falls back to CPU on failure. By default WebGPU is skipped and the
    CPU variant is selected explicitly — required because, once GPU EPs are
    registered, the catalog sorts a GPU variant first and the default selection
    would otherwise be an unloadable GPU variant.
    """
    if prefer_webgpu is None:
        prefer_webgpu = webgpu_opt_in()

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

    # CPU path (default). Select CPU explicitly; only fall back to the catalog
    # default if no CPU variant is exposed (e.g. EPs not registered at all).
    select_variant_by_ep(model, CPU_EP)
    _download(model, progress_callback)
    model.load()
    device, ep = runtime_info(model)
    return model, ModelLoadStatus(alias, model.id, device, ep, bool(errors), errors)
