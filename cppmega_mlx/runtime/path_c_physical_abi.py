"""Physical ABI helpers for generated Path C fused schedules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any


_DTYPE_NBYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "uint16": 2,
    "int16": 2,
    "float32": 4,
    "uint32": 4,
    "int32": 4,
    "float64": 8,
    "uint64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class PathCPhysicalAbiBinding:
    """One logical buffer packed into a physical dtype bank."""

    logical_name: str
    bank: str
    dtype: str
    offset: int
    shape: tuple[int, ...]
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def nbytes(self) -> int:
        return self.size * _DTYPE_NBYTES.get(self.dtype, 0)


@dataclass(frozen=True)
class PathCPhysicalAbiBankSpec:
    """One caller-owned physical bank required by a generated Path C ABI."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    elements: int
    nbytes: int
    logical_buffers: tuple[str, ...]


@dataclass(frozen=True)
class PathCPhysicalAbiBankOwner:
    """Validated caller/model-owned physical bank buffers.

    The owner is a structural wrapper around buffers allocated by the caller. It
    never packs logical tensors into banks and never allocates tensors itself.
    """

    owner_name: str
    bank_specs: tuple[PathCPhysicalAbiBankSpec, ...]
    buffers: Mapping[str, Any]
    binding_payload: dict[str, Any]
    hidden_packing_performed: bool = False
    no_hidden_allocation_policy: bool = True

    @property
    def required_bank_buffers(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.bank_specs)


@dataclass(frozen=True)
class PathCLogicalBufferOwner:
    """Caller/model-owned logical buffers for direct Path C chain execution.

    This wrapper records references to existing logical tensors. It never
    reshapes, packs, casts, or allocates buffers to satisfy a generated ABI.
    """

    owner_name: str
    buffers: Mapping[str, Any]
    hidden_packing_performed: bool = False
    no_hidden_allocation_policy: bool = True


def compose_path_c_logical_buffer_owner(
    owner_name: str,
    *owners: PathCLogicalBufferOwner | Mapping[str, Any] | None,
) -> PathCLogicalBufferOwner:
    """Combine existing logical-buffer owners without touching tensor storage."""

    sentinel = object()
    buffers: dict[str, Any] = {}
    buffer_sources: dict[str, str] = {}
    hidden_packing_performed = False
    no_hidden_allocation_policy = True

    for index, owner in enumerate(owners):
        if owner is None:
            continue
        if isinstance(owner, Mapping):
            owner_buffers = owner
            source_name = f"mapping_{index}"
            owner_hidden_packing = False
            owner_no_hidden_allocation = True
        else:
            owner_buffers = getattr(owner, "buffers", None)
            source_name = str(getattr(owner, "owner_name", f"owner_{index}"))
            owner_hidden_packing = bool(
                getattr(owner, "hidden_packing_performed", False)
            )
            owner_no_hidden_allocation = bool(
                getattr(owner, "no_hidden_allocation_policy", True)
            )
        if owner_buffers is None:
            raise TypeError(
                "Path C logical owner entries must be PathCLogicalBufferOwner "
                "instances or mappings"
            )
        hidden_packing_performed = (
            hidden_packing_performed or owner_hidden_packing
        )
        no_hidden_allocation_policy = (
            no_hidden_allocation_policy and owner_no_hidden_allocation
        )
        for raw_name, value in owner_buffers.items():
            name = str(raw_name)
            existing = buffers.get(name, sentinel)
            if existing is not sentinel and existing is not value:
                raise ValueError(
                    f"conflicting logical buffer {name!r}: "
                    f"{buffer_sources[name]} and {source_name} provide "
                    "different objects"
                )
            buffers[name] = value
            buffer_sources.setdefault(name, source_name)

    return PathCLogicalBufferOwner(
        owner_name=owner_name,
        buffers=buffers,
        hidden_packing_performed=hidden_packing_performed,
        no_hidden_allocation_policy=no_hidden_allocation_policy,
    )


def normalize_physical_abi_map(
    mapping: Mapping[str, Any],
) -> tuple[PathCPhysicalAbiBinding, ...]:
    """Normalize a generated logical-to-physical ABI map."""

    bindings: list[PathCPhysicalAbiBinding] = []
    for logical_name, raw in sorted(mapping.items()):
        if not isinstance(raw, Mapping):
            raise TypeError(f"ABI mapping for {logical_name!r} must be a mapping")
        shape = tuple(int(dim) for dim in raw.get("shape", ()))
        size = int(raw.get("size", prod(shape) if shape else 1))
        bindings.append(
            PathCPhysicalAbiBinding(
                logical_name=str(logical_name),
                bank=str(raw["bank"]),
                dtype=str(raw["dtype"]),
                offset=int(raw["offset"]),
                shape=shape,
                size=size,
            )
        )
    return tuple(bindings)


def validate_physical_abi_map(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate bank capacity and range disjointness for a physical ABI map."""

    errors: list[str] = []
    bindings = normalize_physical_abi_map(mapping)
    bank_capacity: dict[str, int] = {}
    for bank, shape in bank_shapes.items():
        dims = tuple(int(dim) for dim in shape)
        bank_capacity[str(bank)] = int(prod(dims)) if dims else 1

    ranges_by_bank: dict[str, list[tuple[int, int, str]]] = {}
    for binding in bindings:
        expected_size = int(prod(binding.shape)) if binding.shape else 1
        if binding.size != expected_size:
            errors.append(
                f"{binding.logical_name}: size {binding.size} does not match "
                f"shape product {expected_size}"
            )
        capacity = bank_capacity.get(binding.bank)
        if capacity is None:
            errors.append(f"{binding.logical_name}: unknown bank {binding.bank!r}")
            continue
        if binding.offset < 0:
            errors.append(f"{binding.logical_name}: negative offset {binding.offset}")
        if binding.size < 0:
            errors.append(f"{binding.logical_name}: negative size {binding.size}")
        if binding.end > capacity:
            errors.append(
                f"{binding.logical_name}: range [{binding.offset}, {binding.end}) "
                f"exceeds {binding.bank} capacity {capacity}"
            )
        ranges_by_bank.setdefault(binding.bank, []).append(
            (binding.offset, binding.end, binding.logical_name)
        )

    bank_used_elements: dict[str, int] = {}
    bank_used_bytes: dict[str, int] = {}
    for bank, ranges in ranges_by_bank.items():
        ranges.sort()
        bank_used_elements[bank] = sum(end - start for start, end, _ in ranges)
        for previous, current in zip(ranges, ranges[1:], strict=False):
            previous_start, previous_end, previous_name = previous
            current_start, current_end, current_name = current
            if current_start < previous_end:
                errors.append(
                    f"{bank}: {previous_name} [{previous_start}, {previous_end}) "
                    f"overlaps {current_name} [{current_start}, {current_end})"
                )
        bank_used_bytes[bank] = sum(
            binding.nbytes for binding in bindings if binding.bank == bank
        )

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "logical_buffer_count": len(bindings),
        "bank_count": len(bank_capacity),
        "bank_capacity_elements": bank_capacity,
        "bank_used_elements": bank_used_elements,
        "bank_used_bytes": bank_used_bytes,
    }


def _bank_order(
    bindings: tuple[PathCPhysicalAbiBinding, ...],
    bank_shapes: Mapping[str, Any],
) -> list[str]:
    banks_in_map = {binding.bank for binding in bindings}
    ordered = [str(bank) for bank in bank_shapes if str(bank) in banks_in_map]
    seen = set(ordered)
    for binding in bindings:
        if binding.bank in seen:
            continue
        ordered.append(binding.bank)
        seen.add(binding.bank)
    return ordered


def _bank_dtypes(
    bindings: tuple[PathCPhysicalAbiBinding, ...],
) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for binding in bindings:
        previous = dtypes.setdefault(binding.bank, binding.dtype)
        if previous != binding.dtype:
            raise ValueError(
                f"bank {binding.bank!r} mixes dtypes {previous!r} and {binding.dtype!r}"
            )
    return dtypes


def _is_direct_zero_copy_binding(
    binding: PathCPhysicalAbiBinding,
    bank_capacity: Mapping[str, int],
) -> bool:
    return (
        binding.bank == binding.logical_name
        and binding.offset == 0
        and binding.size == bank_capacity.get(binding.bank)
    )


def plan_physical_abi_runtime_bridge(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify how a generated physical ABI can be bound at runtime.

    This helper intentionally does not pack or allocate tensors.  A banked ABI
    can be called only when the caller already owns the physical bank buffers.
    """

    validation = validate_physical_abi_map(mapping, bank_shapes)
    bindings = normalize_physical_abi_map(mapping)
    bank_capacity = {
        str(bank): int(prod(tuple(int(dim) for dim in shape))) if tuple(shape) else 1
        for bank, shape in bank_shapes.items()
    }
    required_banks = _bank_order(bindings, bank_shapes)
    logical_by_bank = {
        bank: [
            binding.logical_name
            for binding in bindings
            if binding.bank == bank
        ]
        for bank in required_banks
    }
    try:
        bank_dtypes = _bank_dtypes(bindings)
    except ValueError as exc:
        validation = {**validation, "status": "failed"}
        validation["errors"] = [*validation.get("errors", []), str(exc)]
        bank_dtypes = {}
    if validation["status"] != "ok":
        return {
            "status": "invalid_physical_abi_map",
            "reason": "physical ABI map failed validation",
            "errors": list(validation["errors"]),
            "logical_tensor_binding_supported": False,
            "prepacked_bank_binding_supported": False,
            "required_bank_buffers": required_banks,
            "logical_buffers_by_bank": logical_by_bank,
            "bank_dtypes": bank_dtypes,
            "no_hidden_allocation_policy": True,
        }

    direct_bindings = all(
        _is_direct_zero_copy_binding(binding, bank_capacity)
        for binding in bindings
    )
    if direct_bindings:
        return {
            "status": "direct_logical_tensor_binding_supported",
            "reason": (
                "each logical buffer is already a whole kernel buffer, so the "
                "runtime bridge can pass caller-owned tensors directly"
            ),
            "logical_tensor_binding_supported": True,
            "prepacked_bank_binding_supported": True,
            "required_bank_buffers": required_banks,
            "logical_buffers_by_bank": logical_by_bank,
            "bank_dtypes": bank_dtypes,
            "no_hidden_allocation_policy": True,
        }

    return {
        "status": "prepacked_bank_buffers_required",
        "reason": (
            "banked physical ABI groups logical buffers into dtype banks; the "
            "runtime bridge refuses to pack or copy model tensors implicitly "
            "and requires caller-owned bank buffers"
        ),
        "logical_tensor_binding_supported": False,
        "prepacked_bank_binding_supported": True,
        "required_bank_buffers": required_banks,
        "logical_buffers_by_bank": logical_by_bank,
        "bank_dtypes": bank_dtypes,
        "no_hidden_allocation_policy": True,
    }


def physical_abi_bank_specs(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
) -> tuple[PathCPhysicalAbiBankSpec, ...]:
    """Return the physical bank specs in generated kernel argument order."""

    validation = validate_physical_abi_map(mapping, bank_shapes)
    if validation["status"] != "ok":
        detail = "; ".join(str(error) for error in validation["errors"])
        raise ValueError(
            "physical ABI bank specs are not valid"
            + (f": {detail}" if detail else "")
        )
    bindings = normalize_physical_abi_map(mapping)
    bridge_plan = plan_physical_abi_runtime_bridge(mapping, bank_shapes)
    bank_dtypes = dict(bridge_plan.get("bank_dtypes", {}))
    logical_by_bank = dict(bridge_plan.get("logical_buffers_by_bank", {}))
    specs: list[PathCPhysicalAbiBankSpec] = []
    for bank in bridge_plan["required_bank_buffers"]:
        shape = tuple(int(dim) for dim in tuple(bank_shapes[bank]))
        dtype = str(bank_dtypes[bank])
        elements = int(prod(shape)) if shape else 1
        specs.append(
            PathCPhysicalAbiBankSpec(
                name=str(bank),
                shape=shape,
                dtype=dtype,
                elements=elements,
                nbytes=elements * _DTYPE_NBYTES.get(dtype, 0),
                logical_buffers=tuple(
                    str(name)
                    for name in logical_by_bank.get(bank, ())
                ),
            )
        )
    if not specs and bindings:
        raise ValueError("physical ABI bank specs are empty for non-empty mapping")
    return tuple(specs)


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in tuple(shape))


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).rsplit(".", 1)[-1]


def validate_physical_abi_runtime_bindings(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
    buffers: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate caller-supplied physical bank buffers without allocating."""

    bridge_plan = plan_physical_abi_runtime_bridge(mapping, bank_shapes)
    required_banks = list(bridge_plan["required_bank_buffers"])
    bank_dtypes = dict(bridge_plan.get("bank_dtypes", {}))
    if not buffers:
        return {
            "status": "not_bound",
            "reason": "no caller-owned physical bank buffers were supplied",
            "errors": [],
            "ordered_kernel_buffers": required_banks,
            "missing_bank_buffers": required_banks,
            "unexpected_buffers": [],
            "shape_mismatch_buffers": [],
            "dtype_mismatch_buffers": [],
            "bridge_plan": bridge_plan,
        }

    provided = {str(name): value for name, value in buffers.items()}
    missing = [bank for bank in required_banks if bank not in provided]
    unexpected = sorted(name for name in provided if name not in set(required_banks))
    errors = [
        f"{bank}: missing caller-owned bank buffer"
        for bank in missing
    ]
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []

    for bank in required_banks:
        if bank not in provided:
            continue
        expected_shape = tuple(int(dim) for dim in tuple(bank_shapes[bank]))
        actual_shape = _shape_of(provided[bank])
        if actual_shape != expected_shape:
            shape_mismatches.append(bank)
            errors.append(
                f"{bank}: shape {actual_shape} does not match expected {expected_shape}"
            )
        expected_dtype = bank_dtypes.get(bank)
        actual_dtype = _dtype_of(provided[bank])
        if actual_dtype != expected_dtype:
            dtype_mismatches.append(bank)
            errors.append(
                f"{bank}: dtype {actual_dtype!r} does not match expected {expected_dtype!r}"
            )

    return {
        "status": "ok" if not errors else "failed",
        "reason": (
            "caller supplied the physical bank buffers in kernel argument order"
            if not errors
            else "caller-supplied physical bank buffers do not satisfy the generated ABI"
        ),
        "errors": errors,
        "ordered_kernel_buffers": required_banks,
        "missing_bank_buffers": missing,
        "unexpected_buffers": unexpected,
        "shape_mismatch_buffers": shape_mismatches,
        "dtype_mismatch_buffers": dtype_mismatches,
        "bridge_plan": bridge_plan,
    }


def make_physical_abi_bank_owner(
    owner_name: str,
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
    buffers: Mapping[str, Any],
) -> PathCPhysicalAbiBankOwner:
    """Validate and wrap caller/model-owned physical ABI bank buffers."""

    provided = {str(name): value for name, value in buffers.items()}
    binding_payload = validate_physical_abi_runtime_bindings(
        mapping,
        bank_shapes,
        provided,
    )
    if binding_payload["status"] != "ok":
        detail = "; ".join(str(error) for error in binding_payload["errors"])
        raise ValueError(
            "physical ABI bank owner is not executable"
            + (f": {detail}" if detail else "")
        )
    return PathCPhysicalAbiBankOwner(
        owner_name=str(owner_name),
        bank_specs=physical_abi_bank_specs(mapping, bank_shapes),
        buffers=provided,
        binding_payload=binding_payload,
    )


def physical_abi_runtime_kernel_args(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
    buffers: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return caller-owned physical bank buffers in generated kernel order."""

    payload = validate_physical_abi_runtime_bindings(
        mapping,
        bank_shapes,
        buffers,
    )
    if payload["status"] != "ok":
        detail = "; ".join(str(error) for error in payload["errors"])
        raise ValueError(
            "physical ABI runtime bindings are not executable"
            + (f": {detail}" if detail else "")
        )
    return tuple(buffers[name] for name in payload["ordered_kernel_buffers"])


def physical_abi_full_runtime_kernel_args(
    mapping: Mapping[str, Any],
    bank_shapes: Mapping[str, Any],
    kernel_buffer_order: tuple[str, ...],
    buffers: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Return full generated kernel args: physical banks plus scratch buffers."""

    provided = {str(name): value for name, value in buffers.items()}
    bridge_plan = plan_physical_abi_runtime_bridge(mapping, bank_shapes)
    bank_buffers = {
        bank: provided[bank]
        for bank in bridge_plan["required_bank_buffers"]
        if bank in provided
    }
    physical_abi_runtime_kernel_args(mapping, bank_shapes, bank_buffers)
    missing = [name for name in kernel_buffer_order if name not in provided]
    if missing:
        raise ValueError(
            "physical ABI runtime bindings are not executable: "
            + "; ".join(
                f"{name}: missing caller-owned kernel buffer"
                for name in missing
            )
        )
    return tuple(provided[name] for name in kernel_buffer_order)



def logical_bank_view(
    abi_map: Mapping[str, Any],
    bank_buffers: Mapping[str, Any],
    logical_name: str,
) -> Any:
    """Return a bank-resident view of ``logical_name`` as an ``mx.array``.

    The view is a single slice of the appropriate bank, reshaped to the
    logical_shape attribute of the ABI map entry. The function performs no
    allocation, no cast, and no copy; if the logical name is not in the ABI
    map or the bank buffer is missing, a ``ValueError`` is raised.
    """

    info = abi_map.get(logical_name)
    if not isinstance(info, Mapping):
        raise ValueError(
            f"logical buffer is not in physical ABI: {logical_name!r}"
        )
    bank_name = str(info.get("bank", ""))
    if bank_name not in bank_buffers:
        raise ValueError(f"bank buffer is not bound: {bank_name!r}")
    bank = bank_buffers[bank_name]
    if not hasattr(bank, "__getitem__"):
        raise TypeError(f"bank buffer {bank_name!r} is not subscriptable")
    offset = int(info.get("offset", 0) or 0)
    size = int(info.get("size", 1) or 1)
    view = bank[offset : offset + size]
    logical_shape = tuple(
        int(dim)
        for dim in tuple(info.get("logical_shape", info.get("shape", (size,))))
    )
    if logical_shape and tuple(int(dim) for dim in tuple(view.shape)) != logical_shape:
        reshape = getattr(view.__class__, "reshape", None)
        if reshape is None:
            try:
                import mlx.core as mx  # local import to keep optional

                view = mx.reshape(view, logical_shape)
            except Exception as exc:
                raise TypeError(
                    f"bank view for {logical_name!r} cannot be reshaped to "
                    f"{logical_shape}"
                ) from exc
        else:
            view = view.reshape(logical_shape)
    return view


def write_into_bank_slot(
    abi_map: Mapping[str, Any],
    bank_buffers: Mapping[str, Any],
    logical_name: str,
    value: Any,
) -> None:
    """Write ``value`` into the bank slot for ``logical_name`` in place.

    Both ``value.shape`` and the logical_shape of the ABI entry are honored:
    the value is flattened to the 1-D slice slot. This is the explicit,
    caller-visible bridge from a parameter tensor to its bank slot; the
    function performs one slice-assignment into a pre-allocated bank buffer
    and never allocates a new bank.
    """

    info = abi_map.get(logical_name)
    if not isinstance(info, Mapping):
        raise ValueError(
            f"logical buffer is not in physical ABI: {logical_name!r}"
        )
    bank_name = str(info.get("bank", ""))
    if bank_name not in bank_buffers:
        raise ValueError(f"bank buffer is not bound: {bank_name!r}")
    bank = bank_buffers[bank_name]
    offset = int(info.get("offset", 0) or 0)
    size = int(info.get("size", 1) or 1)
    expected_dtype = str(info.get("dtype", ""))
    actual_dtype = _dtype_of(value)
    if expected_dtype and actual_dtype != expected_dtype:
        raise ValueError(
            f"value for {logical_name!r} has dtype {actual_dtype!r}, "
            f"expected {expected_dtype!r}"
        )
    # Flatten the value into a 1-D array of length ``size``.
    expected_size = int(prod(tuple(int(dim) for dim in tuple(getattr(value, "shape", ())))))
    if expected_size != size:
        raise ValueError(
            f"value for {logical_name!r} has size {expected_size}, "
            f"expected {size}"
        )
    try:
        import mlx.core as mx  # local import to keep optional
    except ImportError as exc:
        raise TypeError(
            "write_into_bank_slot requires mlx.core for slice assignment"
        ) from exc
    flat = mx.reshape(value, (size,))
    bank[offset : offset + size] = flat


__all__ = [
    "PathCPhysicalAbiBinding",
    "PathCPhysicalAbiBankOwner",
    "PathCPhysicalAbiBankSpec",
    "PathCLogicalBufferOwner",
    "compose_path_c_logical_buffer_owner",
    "make_physical_abi_bank_owner",
    "normalize_physical_abi_map",
    "physical_abi_bank_specs",
    "physical_abi_full_runtime_kernel_args",
    "logical_bank_view",
    "physical_abi_runtime_kernel_args",
    "write_into_bank_slot",
    "plan_physical_abi_runtime_bridge",
    "validate_physical_abi_map",
    "validate_physical_abi_runtime_bindings",
]
