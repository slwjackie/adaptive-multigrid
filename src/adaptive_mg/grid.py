"""Grid-shape and coarsening utilities for structured multigrid."""
from __future__ import annotations

from typing import Literal, TypeAlias

GridShape: TypeAlias = tuple[int, int]
CoarseningKind = Literal["full", "semi_x", "semi_y", "alternating"]


def as_shape(value: int | GridShape) -> GridShape:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError("grid shape must have two entries")
    return (int(value[0]), int(value[1]))


def unknowns(shape: int | GridShape) -> int:
    nx, ny = as_shape(shape)
    return nx * ny


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def is_nested_size(n: int) -> bool:
    return n >= 1 and is_power_of_two(n + 1)


def validate_root_shape(shape: int | GridShape, coarsest_n: int = 3) -> GridShape:
    nx, ny = as_shape(shape)
    if coarsest_n < 1 or not is_nested_size(coarsest_n):
        raise ValueError("coarsest_n must be of the form 2**l-1")
    for name, n in (("nx", nx), ("ny", ny)):
        if n < coarsest_n or not is_nested_size(n):
            raise ValueError(
                "nested structured MG requires each root dimension to be 2**L-1; "
                f"got {name}={n}, coarsest_n={coarsest_n}"
            )
        current = n
        while current > coarsest_n:
            current = (current - 1) // 2
        if current != coarsest_n:
            raise ValueError(
                f"dimension {name}={n} cannot reach coarsest_n={coarsest_n} exactly"
            )
    return (nx, ny)


def can_coarsen(n: int, coarsest_n: int) -> bool:
    return n > coarsest_n


def coarsen_dimension(n: int, coarsest_n: int) -> int:
    if not can_coarsen(n, coarsest_n):
        return n
    if n < 3 or n % 2 == 0:
        raise ValueError(f"nested coarsening requires odd dimension >=3; got {n}")
    return (n - 1) // 2



def full_coarsening_depth(n: int, coarsest_n: int = 3) -> int:
    """Number of nested transfer levels from ``n`` to ``coarsest_n``."""

    if not is_nested_size(n) or not is_nested_size(coarsest_n) or n < coarsest_n:
        raise ValueError("n and coarsest_n must be nested sizes with n >= coarsest_n")
    depth = 0
    current = n
    while current > coarsest_n:
        current = coarsen_dimension(current, coarsest_n)
        depth += 1
    if current != coarsest_n:
        raise ValueError(f"n={n} cannot reach coarsest_n={coarsest_n}")
    return depth

def terminal(shape: GridShape, coarsest_n: int) -> bool:
    return shape[0] <= coarsest_n and shape[1] <= coarsest_n


def next_shape(
    shape: int | GridShape,
    kind: CoarseningKind,
    coarsest_n: int,
    *,
    level_index: int = 0,
) -> GridShape:
    """Return the next nested shape.

    ``semi_x`` repeatedly coarsens x before y; ``semi_y`` does the reverse.
    ``alternating`` coarsens one direction per level.  All policies eventually
    reduce both dimensions to ``coarsest_n`` so the terminal direct solve stays
    small.
    """

    nx, ny = as_shape(shape)
    if terminal((nx, ny), coarsest_n):
        return (nx, ny)

    x_ok = can_coarsen(nx, coarsest_n)
    y_ok = can_coarsen(ny, coarsest_n)

    if kind == "full":
        return (
            coarsen_dimension(nx, coarsest_n) if x_ok else nx,
            coarsen_dimension(ny, coarsest_n) if y_ok else ny,
        )
    if kind == "semi_x":
        if x_ok:
            return (coarsen_dimension(nx, coarsest_n), ny)
        return (nx, coarsen_dimension(ny, coarsest_n))
    if kind == "semi_y":
        if y_ok:
            return (nx, coarsen_dimension(ny, coarsest_n))
        return (coarsen_dimension(nx, coarsest_n), ny)
    if kind == "alternating":
        prefer_x = level_index % 2 == 0
        if prefer_x and x_ok:
            return (coarsen_dimension(nx, coarsest_n), ny)
        if (not prefer_x) and y_ok:
            return (nx, coarsen_dimension(ny, coarsest_n))
        if x_ok:
            return (coarsen_dimension(nx, coarsest_n), ny)
        return (nx, coarsen_dimension(ny, coarsest_n))
    raise ValueError(f"unknown coarsening kind: {kind}")


def coarsening_ratios(fine: GridShape, coarse: GridShape) -> tuple[int, int]:
    ratios: list[int] = []
    for nf, nc in zip(fine, coarse):
        if nc == nf:
            ratios.append(1)
        elif nc == (nf - 1) // 2:
            ratios.append(2)
        else:
            raise ValueError(f"non-nested transfer shape: fine={fine}, coarse={coarse}")
    if ratios == [1, 1]:
        raise ValueError("transfer must coarsen at least one direction")
    return (ratios[0], ratios[1])
