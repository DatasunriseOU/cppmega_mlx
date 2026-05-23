"""Step 4.b: production launcher for the named Mamba3 FP8 train block.

This module turns the compiled
``mamba3_m2rnn_attention_fp8_train_block`` ``JITKernel`` into a Python
callable that takes named MLX arrays for the real ABI inputs / outputs /
gradients and runs the single-launch fused PrimFunc on Metal through the
``tvm_ffi`` bridge.

The launcher is fail-closed:

* The PrimFunc must carry the ``tl.fusion.physical_abi.logical_to_physical``
  and ``tl.fusion.physical_abi.physical_buffer_shapes`` attrs that the
  Path C descriptor schedule emits. Without them we cannot pack logical
  buffers into the physical banks and we refuse to launch.
* Missing real ABI inputs raise immediately rather than silently zero-filling.
* Internal scratch buffers listed in ``tl.fusion.internal_scratch_abi_buffers``
  are allocated as zero-initialised MLX arrays. Their gradients (also
  listed there) are seeded to zero too.

The launcher reuses the ``tilelang.compile`` ``tvm_ffi`` adapter that
already ships native MLX-array support on Metal; no monkeypatch, no
runtime shim, no python-side substitution: we lift the contract straight
out of the compiled PrimFunc and hand the bank buffers back to the
production adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

__all__ = [
    "PathCBankPlacement",
    "PathCAbiManifest",
    "PathCLaunchResult",
    "load_path_c_abi_manifest",
    "launch_mamba3_fp8_train_block",
    "Mamba3Fp8TrainBlockLauncher",
]


# Manifest attrs we always need from the compiled PrimFunc. Optional ones
# (``tl.fusion.internal_scratch_abi_buffers``,
# ``tl.fusion.train_step_loss_cotangent_abi``) are tolerated as absent for
# schedules that have no dtype-banked scratch / no train-step loss seam.
_REQUIRED_PRIMFUNC_ATTRS = (
    "tl.fusion.physical_abi.logical_to_physical",
    "tl.fusion.physical_abi.physical_buffer_shapes",
)


def _decode_attr(prim_func: Any, key: str) -> Any:
    raw = prim_func.attrs.get(key)
    if raw is None:
        raise ValueError(
            f"compiled PrimFunc is missing required Path C ABI attr {key!r}; "
            "this launcher is fail-closed and refuses to guess the layout"
        )
    return json.loads(str(raw))


@dataclass(frozen=True)
class PathCBankPlacement:
    """Where one logical buffer lives inside a physical ABI bank."""

    bank: str
    dtype: str
    offset: int
    size: int
    shape: tuple[int, ...]
    logical_shape: tuple[int, ...]

    @classmethod
    def from_attr(cls, info: Mapping[str, Any]) -> "PathCBankPlacement":
        return cls(
            bank=str(info["bank"]),
            dtype=str(info["dtype"]),
            offset=int(info["offset"]),
            size=int(info["size"]),
            shape=tuple(int(x) for x in info["shape"]),
            logical_shape=tuple(int(x) for x in info.get("logical_shape") or info["shape"]),
        )


@dataclass(frozen=True)
class PathCAbiManifest:
    """Decoded view of the Path C ABI attrs on the compiled PrimFunc."""

    entry_symbol: str
    param_order: tuple[str, ...]
    logical_to_physical: Mapping[str, PathCBankPlacement]
    bank_shapes: Mapping[str, tuple[int, ...]]
    bank_dtypes: Mapping[str, str]
    internal_scratch_buffers: tuple[str, ...]
    cotangent_seed_buffers: tuple[str, ...]
    top_level_scratch_shapes: Mapping[str, tuple[int, ...]]
    top_level_scratch_dtypes: Mapping[str, str]


def load_path_c_abi_manifest(prim_func: Any) -> PathCAbiManifest:
    """Decode the Path C ABI manifest from a compiled PrimFunc."""

    for attr in _REQUIRED_PRIMFUNC_ATTRS:
        if prim_func.attrs.get(attr) is None:
            raise ValueError(
                f"compiled PrimFunc is missing required Path C ABI attr {attr!r}"
            )

    entry = prim_func.attrs.get("global_symbol")
    if entry is None:
        raise ValueError("compiled PrimFunc is missing global_symbol")

    logical_to_physical_raw = _decode_attr(
        prim_func, "tl.fusion.physical_abi.logical_to_physical"
    )
    bank_shapes_raw = _decode_attr(
        prim_func, "tl.fusion.physical_abi.physical_buffer_shapes"
    )
    scratch_attr = prim_func.attrs.get("tl.fusion.internal_scratch_abi_buffers")
    scratch_raw = (
        json.loads(str(scratch_attr)) if scratch_attr is not None else []
    )

    cotangent_raw = prim_func.attrs.get("tl.fusion.train_step_loss_cotangent_abi")
    cotangent_buffers: tuple[str, ...]
    if cotangent_raw is None:
        cotangent_buffers = ()
    else:
        cotangent_doc = json.loads(str(cotangent_raw))
        cotangent_buffers = tuple(cotangent_doc.get("logical_cotangent_buffers") or ())

    logical_to_physical = {
        name: PathCBankPlacement.from_attr(info)
        for name, info in logical_to_physical_raw.items()
    }

    bank_shapes = {
        name: tuple(int(x) for x in shape) for name, shape in bank_shapes_raw.items()
    }
    bank_dtypes = {placement.bank: placement.dtype for placement in logical_to_physical.values()}

    param_order = tuple(
        getattr(p, "name", str(p)) for p in prim_func.params
    )

    top_level_shapes: dict[str, tuple[int, ...]] = {}
    top_level_dtypes: dict[str, str] = {}
    # Top-level scratch is every prim_func buffer that is not one of the
    # three dtype banks and not already placed inside a bank via the
    # logical_to_physical map. Internal-scratch members whose data lives
    # outside the dtype banks (e.g. per-brick activations like
    # ``mamba3_delta``) end up in here too.
    for var, buffer in prim_func.buffer_map.items():
        logical = getattr(buffer, "name", None) or str(var)
        if logical in bank_shapes:
            continue
        if logical in logical_to_physical:
            continue
        top_level_shapes[logical] = tuple(int(x) for x in buffer.shape)
        top_level_dtypes[logical] = str(buffer.dtype)

    return PathCAbiManifest(
        entry_symbol=str(entry),
        param_order=param_order,
        logical_to_physical=logical_to_physical,
        bank_shapes=bank_shapes,
        bank_dtypes=bank_dtypes,
        internal_scratch_buffers=tuple(scratch_raw),
        cotangent_seed_buffers=cotangent_buffers,
        top_level_scratch_shapes=top_level_shapes,
        top_level_scratch_dtypes=top_level_dtypes,
    )


# --- packing helpers -----------------------------------------------------


_DTYPE_MAP: dict[str, Any] = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int8": mx.int8,
    "uint8": mx.uint8,
}


def _to_mx_dtype(name: str) -> Any:
    try:
        return _DTYPE_MAP[name]
    except KeyError as err:
        raise ValueError(f"unsupported Path C ABI dtype {name!r}") from err


def _flat_into_bank(
    bank: mx.array,
    placement: PathCBankPlacement,
    value: mx.array,
) -> mx.array:
    """Return ``bank`` with ``value`` flattened into ``placement``'s slot."""

    if int(value.size) != placement.size:
        raise ValueError(
            f"logical buffer size mismatch at offset {placement.offset}: "
            f"expected {placement.size}, got {int(value.size)}"
        )
    typed = value.astype(_to_mx_dtype(placement.dtype))
    flat = typed.reshape((placement.size,))
    indices = mx.arange(placement.offset, placement.offset + placement.size, dtype=mx.int32)
    return bank.at[indices].add(flat - bank[indices])


def _bank_slot_view(
    bank: mx.array, placement: PathCBankPlacement
) -> mx.array:
    """Slice ``placement``'s region out of ``bank`` and reshape logically."""

    flat = bank[placement.offset : placement.offset + placement.size]
    return flat.reshape(placement.logical_shape)


# --- public launcher ------------------------------------------------------


@dataclass
class PathCLaunchResult:
    """Logical outputs and gradients returned by the launcher."""

    forward: dict[str, mx.array]
    parameter_grads: dict[str, mx.array]
    state_after: dict[str, mx.array]


_FORWARD_OUTPUT_LOGICAL_BUFFERS: tuple[str, ...] = (
    "attention_out",
    "hidden_after_m2rnn",
    "lse",
)


_STATE_AFTER_LOGICAL_BUFFERS: tuple[str, ...] = (
    "mamba_state",
    "scan_state",
    "m2rnn_conv_state",
)


class Mamba3Fp8TrainBlockLauncher:
    """One-shot, fail-closed launcher for the fused Path C train block."""

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled
        self._kernel = compiled.compiled.artifact
        self._prim_func = self._kernel.prim_func
        self.manifest = load_path_c_abi_manifest(self._prim_func)

    # --- inspection ------------------------------------------------------

    @property
    def real_abi_inputs(self) -> tuple[str, ...]:
        """Logical buffer names that callers MUST provide as inputs.

        This is the full set of external buffers that flow INTO the
        fused kernel as data sources: trainable parameters (declared as
        ``declared_required_real_abi_inputs``) plus the forward
        activation seed (``hidden``) and recurrent state carriers
        (``mamba_state``, ``scan_state``, ``m2rnn_conv_state``). Without
        the activation seed the kernel reads zeros and the rest of the
        fused fwd+bwd pipeline silently produces zeros for every
        downstream buffer.
        """

        contract = self._compiled.compiled.plan.schedule_contract
        declared = set(contract.declared_required_real_abi_inputs)
        forward_state = {
            name
            for name in self.manifest.logical_to_physical
            if name in ("hidden", "mamba_state", "scan_state", "m2rnn_conv_state")
        }
        ordered: list[str] = list(contract.declared_required_real_abi_inputs)
        for name in ("hidden", "mamba_state", "scan_state", "m2rnn_conv_state"):
            if name in forward_state and name not in declared:
                ordered.append(name)
        return tuple(ordered)

    @property
    def cotangent_seed_buffers(self) -> tuple[str, ...]:
        """Logical ``*_grad`` buffers callers seed with the loss VJP."""

        return self.manifest.cotangent_seed_buffers

    @property
    def parameter_grad_buffers(self) -> tuple[str, ...]:
        """Logical ``*_grad`` buffers the kernel writes for trainable parameters."""

        scratch = set(self.manifest.internal_scratch_buffers)
        seeds = set(self.cotangent_seed_buffers)
        return tuple(
            sorted(
                name
                for name in self.manifest.logical_to_physical
                if name.endswith("_grad") and name not in scratch and name not in seeds
            )
        )

    @property
    def state_buffers(self) -> tuple[str, ...]:
        return _STATE_AFTER_LOGICAL_BUFFERS

    @property
    def forward_outputs(self) -> tuple[str, ...]:
        return _FORWARD_OUTPUT_LOGICAL_BUFFERS

    # --- launch ----------------------------------------------------------

    def __call__(
        self,
        *,
        real_abi_inputs: Mapping[str, mx.array],
        cotangent_seeds: Mapping[str, mx.array],
    ) -> PathCLaunchResult:
        manifest = self.manifest

        # Catch missing real ABI inputs up-front.
        missing_inputs = tuple(
            name for name in self.real_abi_inputs if name not in real_abi_inputs
        )
        if missing_inputs:
            raise ValueError(
                f"launch_mamba3_fp8_train_block: missing real ABI inputs {missing_inputs!r}; "
                "this launcher is fail-closed"
            )
        unexpected_inputs = tuple(
            name
            for name in real_abi_inputs
            if name not in manifest.logical_to_physical
        )
        if unexpected_inputs:
            raise ValueError(
                f"unknown logical buffer name(s) in real_abi_inputs: {unexpected_inputs!r}"
            )

        missing_seeds = tuple(
            name for name in self.cotangent_seed_buffers if name not in cotangent_seeds
        )
        if missing_seeds:
            raise ValueError(
                f"missing cotangent seed buffers {missing_seeds!r}; "
                "Path C train block needs them to compute parameter gradients"
            )

        # Allocate the three physical ABI banks.
        banks: dict[str, mx.array] = {
            bank: mx.zeros(shape, dtype=_to_mx_dtype(manifest.bank_dtypes[bank]))
            for bank, shape in manifest.bank_shapes.items()
        }

        # Pack real ABI inputs + cotangent seeds into the banks.
        for name, value in {**real_abi_inputs, **cotangent_seeds}.items():
            placement = manifest.logical_to_physical[name]
            banks[placement.bank] = _flat_into_bank(banks[placement.bank], placement, value)

        # Allocate scratch as zero-initialised MLX arrays sized per slot.
        # Internal scratch entries live either inside a dtype bank (uses
        # logical_to_physical) or as a stand-alone top-level buffer
        # (uses top_level_scratch_shapes).
        scratch_buffers: dict[str, mx.array] = {}
        for name in manifest.internal_scratch_buffers:
            if name in manifest.logical_to_physical:
                placement = manifest.logical_to_physical[name]
                scratch_buffers[name] = mx.zeros(
                    placement.logical_shape, dtype=_to_mx_dtype(placement.dtype)
                )
            elif name in manifest.top_level_scratch_shapes:
                scratch_buffers[name] = mx.zeros(
                    manifest.top_level_scratch_shapes[name],
                    dtype=_to_mx_dtype(manifest.top_level_scratch_dtypes[name]),
                )
            else:
                raise ValueError(
                    f"internal scratch buffer {name!r} has no placement in either "
                    "logical_to_physical or top-level buffer_map"
                )

        # Allocate top-level scratch buffers (per-brick activations that
        # the schedule keeps outside the dtype-banked ABI), skipping any
        # that the dtype-scratch loop above has already allocated.
        top_level_buffers: dict[str, mx.array] = {
            name: mx.zeros(
                manifest.top_level_scratch_shapes[name],
                dtype=_to_mx_dtype(manifest.top_level_scratch_dtypes[name]),
            )
            for name in manifest.top_level_scratch_shapes
            if name not in scratch_buffers
        }

        # Build the full positional argument tuple in ``prim_func.params`` order.
        positional: list[mx.array] = []
        for param_name in manifest.param_order:
            logical_name = param_name[: -len("_handle")] if param_name.endswith("_handle") else param_name
            if logical_name in banks:
                positional.append(banks[logical_name])
            elif logical_name in scratch_buffers:
                positional.append(scratch_buffers[logical_name])
            elif logical_name in top_level_buffers:
                positional.append(top_level_buffers[logical_name])
            else:
                raise ValueError(
                    f"compiled PrimFunc has an unexpected parameter {param_name!r} "
                    f"not present in banks, dtype scratch, or top-level scratch"
                )

        # Launch.
        self._kernel(*positional)
        # Force the lazy MLX graph to materialise so subsequent reads see the writes.
        mx.eval(*positional)

        # Extract logical outputs from the banks.
        forward = {
            name: _bank_slot_view(
                banks[manifest.logical_to_physical[name].bank],
                manifest.logical_to_physical[name],
            )
            for name in _FORWARD_OUTPUT_LOGICAL_BUFFERS
            if name in manifest.logical_to_physical
        }
        state_after = {
            name: _bank_slot_view(
                banks[manifest.logical_to_physical[name].bank],
                manifest.logical_to_physical[name],
            )
            for name in _STATE_AFTER_LOGICAL_BUFFERS
            if name in manifest.logical_to_physical
        }
        parameter_grads = {
            name: _bank_slot_view(
                banks[manifest.logical_to_physical[name].bank],
                manifest.logical_to_physical[name],
            )
            for name in self.parameter_grad_buffers
        }
        return PathCLaunchResult(
            forward=forward,
            parameter_grads=parameter_grads,
            state_after=state_after,
        )


def launch_mamba3_fp8_train_block(
    compiled: Any,
    *,
    real_abi_inputs: Mapping[str, mx.array],
    cotangent_seeds: Mapping[str, mx.array],
) -> PathCLaunchResult:
    """Functional shortcut around :class:`Mamba3Fp8TrainBlockLauncher`."""

    launcher = Mamba3Fp8TrainBlockLauncher(compiled)
    return launcher(
        real_abi_inputs=real_abi_inputs,
        cotangent_seeds=cotangent_seeds,
    )
