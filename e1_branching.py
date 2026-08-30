#!/usr/bin/env python3
"""One-electron angular E1 line strengths and branching ratios.

This module treats spin-orbit-resolved *one-electron subshells*
``|n l s j m_j>`` with fixed ``s = 1/2``.  It does not describe arbitrary
many-electron terms such as ``2S+1 L_J``.  In particular, ``j`` here is the
total angular momentum of one electron, not its spin projection.

The base quantity returned by :func:`angular_line_strength` is the angular
line strength summed over every initial and final magnetic substate and all
three spherical photon polarizations.  The radial integral is factored out.
This complete-multiplet sum is symmetric on reversing a transition.  It is
different from both (a) a per-particle rate, which averages over populated
initial substates, and (b) a filled-subshell/empty-subshell integrated
intensity, which additionally depends on occupations and vacancies.

E1 is spin independent in this model.  The large ``L3 -> M5`` strength
relative to ``L3 -> M4`` comes from angular-momentum recoupling (the Wigner
6-j symbol), not simply from final-state degeneracies.  The resulting 9:1
intensity ratio is therefore not the degeneracy ratio 6:4.  Real measured
branching ratios can differ because of radial functions, photon energies,
nonuniform populations, multiplets, configuration interaction, crystal
fields, covalency, polarization, and other many-body effects.

The Wigner 3-j and 6-j symbols are evaluated locally with the Racah factorial
formulas.  Only the Python standard library is required.

Tuple order always denotes electron motion, ``from_state -> to_state``::

    >>> parse_subshell("L3")
    Subshell(n=2, l=1, j=Fraction(3, 2), label='L3')
    >>> round(angular_line_strength(("L3", "M5")).angular_strength, 12)
    2.4
    >>> branching_ratios([("L3", "M5"), ("L3", "M4")]).relative_ratio
    (9, 1)

Spectroscopic fluorescence names such as ``L3-M5`` conventionally put the
core hole first; the emitting electron actually travels in the reverse
direction, ``M5 -> L3``.

Advanced callers can attach explicit per-channel corrections::

    transition = Transition(
        "L3", "M5",
        CorrectionFactors(radial_strength_factor=0.97,
                          photon_energy=700.0,
                          energy_weighting="absorption"),
    )
    result = angular_line_strength(transition)

``energy_weighting='absorption'`` applies an oscillator-strength-like
factor proportional to omega; ``'emission'`` applies the omega**3 factor of
a spontaneous E1 rate; and ``'angular'`` applies no energy factor.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unittest
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from math import factorial, gcd
from typing import Any, Iterable, Mapping, Sequence


__all__ = [
    "Subshell",
    "CorrectionFactors",
    "Transition",
    "LineStrengthResult",
    "BranchingResult",
    "AbsorptionEdgeResult",
    "parse_subshell",
    "e1_allowed",
    "wigner_3j",
    "wigner_6j",
    "angular_line_strength",
    "branching_ratios",
    "absorption_edge_ratios",
]


_SHELLS = "KLMNOPQ"
_ORBITALS = "spdfghi"
_RATIO_TOLERANCE = 1.0e-10


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer; got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _half_integer(value: Any, name: str, *, nonnegative: bool = True) -> Fraction:
    """Convert an integer/half-integer representation to an exact Fraction."""
    try:
        if isinstance(value, bool):
            raise TypeError
        if isinstance(value, Fraction):
            result = value
        elif isinstance(value, int):
            result = Fraction(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError
            twice = round(2.0 * value)
            if not math.isclose(2.0 * value, twice, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError
            result = Fraction(twice, 2)
        elif isinstance(value, str):
            result = Fraction(value.strip())
        else:
            raise TypeError
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"{name} must be an integer or half-integer; got {value!r}"
        ) from exc
    if result.denominator not in (1, 2):
        raise ValueError(f"{name} must be an integer or half-integer; got {value!r}")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative; got {value!r}")
    return result


def _orbital_l(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if len(text) == 1 and text in _ORBITALS:
            return _ORBITALS.index(text)
        raise ValueError(
            f"l must be a nonnegative integer or one of {', '.join(_ORBITALS)}; "
            f"got {value!r}"
        )
    return _integer(value, "l", minimum=0)


def _generated_label(n: int, l: int, j: Fraction) -> str | None:
    if not 1 <= n <= len(_SHELLS):
        return None
    shell = _SHELLS[n - 1]
    if l == 0:
        return shell if n == 1 else f"{shell}1"
    lower = Fraction(2 * l - 1, 2)
    index = 2 * l if j == lower else 2 * l + 1
    return f"{shell}{index}"


@dataclass(frozen=True)
class Subshell:
    """Canonical one-electron ``n l j`` subshell (with implicit ``s=1/2``)."""

    n: int
    l: int
    j: Fraction
    label: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        n = _integer(self.n, "n", minimum=1)
        l = _integer(self.l, "l", minimum=0)
        j = _half_integer(self.j, "j")
        if l >= n:
            raise ValueError(f"invalid subshell: l={l} must be less than n={n}")
        permitted = {Fraction(1, 2)} if l == 0 else {
            Fraction(2 * l - 1, 2),
            Fraction(2 * l + 1, 2),
        }
        if j not in permitted:
            choices = " or ".join(_format_fraction(item) for item in sorted(permitted))
            raise ValueError(
                f"invalid one-electron j={_format_fraction(j)} for l={l}; "
                f"with s=1/2, j must be {choices}"
            )
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError(f"label must be a string or None; got {self.label!r}")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "l", l)
        object.__setattr__(self, "j", j)

    @property
    def degeneracy(self) -> int:
        """Return ``2j+1``, the number of magnetic substates."""
        return int(2 * self.j + 1)

    @property
    def name(self) -> str:
        """Return a label suitable for diagnostics and tables."""
        return self.label or _generated_label(self.n, self.l, self.j) or str(self)

    @property
    def orbital(self) -> str:
        symbol = _ORBITALS[self.l] if self.l < len(_ORBITALS) else f"l={self.l}"
        return f"{self.n}{symbol}_{_format_fraction(self.j)}"

    def __str__(self) -> str:
        generated = self.label or _generated_label(self.n, self.l, self.j)
        quantum = f"n={self.n}, l={self.l}, j={_format_fraction(self.j)}"
        return f"{generated} ({quantum})" if generated else quantum


StateInput = Subshell | str | Mapping[str, Any]


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _parse_edge_label(text: str) -> Subshell:
    normalized = text.strip().upper().replace("_", "")
    match = re.fullmatch(r"([K-Q])(\d+)?", normalized)
    if not match:
        raise ValueError(
            f"invalid edge label {text!r}; expected K or a label such as L3, M5, or N7"
        )
    shell, index_text = match.groups()
    n = _SHELLS.index(shell) + 1
    if n == 1:
        if index_text is not None:
            raise ValueError("the K shell has only the label 'K' (1s_1/2), not a numbered edge")
        return Subshell(1, 0, Fraction(1, 2), "K")
    if index_text is None:
        raise ValueError(f"shell {shell} must include a subshell index, for example {shell}1")
    index = int(index_text)
    if index < 1:
        raise ValueError(f"edge index must be positive; got {text!r}")
    if index == 1:
        l, j = 0, Fraction(1, 2)
    elif index % 2 == 0:
        l, j = index // 2, Fraction(index - 1, 2)
    else:
        l, j = (index - 1) // 2, Fraction(index, 2)
    if l >= n:
        largest = 2 * n - 1
        raise ValueError(
            f"impossible edge label {text!r}: it implies l={l} for n={n}, "
            f"but l must be below n (largest index is {largest})"
        )
    canonical = _generated_label(n, l, j)
    return Subshell(n, l, j, canonical)


def parse_subshell(state: StateInput) -> Subshell:
    """Parse a shell label, explicit mapping, or existing :class:`Subshell`.

    Examples include ``"L3"`` and ``{"n": 3, "l": "d", "j": 2.5}``.
    Explicit mappings may have an optional display ``label``.
    """
    if isinstance(state, Subshell):
        return state
    if isinstance(state, str):
        return _parse_edge_label(state)
    if not isinstance(state, Mapping):
        raise ValueError(
            "a subshell must be a Subshell, an edge label, or a mapping with n, l, and j"
        )
    missing = {"n", "l", "j"}.difference(state)
    if missing:
        raise ValueError(f"subshell mapping is missing required key(s): {', '.join(sorted(missing))}")
    unknown = set(state).difference({"n", "l", "j", "label"})
    if unknown:
        raise ValueError(f"unknown subshell mapping key(s): {', '.join(sorted(map(str, unknown)))}")
    n = _integer(state["n"], "n", minimum=1)
    l = _orbital_l(state["l"])
    j = _half_integer(state["j"], "j")
    label = state.get("label") or _generated_label(n, l, j)
    return Subshell(n, l, j, label)


def e1_allowed(initial: StateInput, final: StateInput) -> tuple[bool, str]:
    """Return whether a channel obeys E1 rules and a human-readable reason."""
    first, second = parse_subshell(initial), parse_subshell(final)
    delta_l = second.l - first.l
    delta_j = abs(second.j - first.j)
    failures: list[str] = []
    if abs(delta_l) != 1:
        failures.append(f"Delta l={delta_l}, but E1 requires Delta l=+/-1")
    if delta_j not in (Fraction(0), Fraction(1)):
        failures.append(
            f"|Delta j|={_format_fraction(delta_j)}, but E1 requires 0 or 1"
        )
    if first.j == 0 and second.j == 0:
        failures.append("E1 excludes j=0 <-> j=0")
    if failures:
        return False, "; ".join(failures)
    return True, (
        f"allowed: Delta l={delta_l} changes parity, |Delta j|="
        f"{_format_fraction(delta_j)}, and spin remains s=1/2"
    )


def _as_int(value: Fraction, context: str) -> int:
    if value.denominator != 1:
        raise ValueError(f"internal angular-momentum inconsistency in {context}: {value}")
    return value.numerator


def _triangle(a: Fraction, b: Fraction, c: Fraction) -> bool:
    return (
        abs(a - b) <= c <= a + b
        and (a + b + c).denominator == 1
    )


def _delta_squared(a: Fraction, b: Fraction, c: Fraction) -> Fraction:
    if not _triangle(a, b, c):
        return Fraction(0)
    x = _as_int(a + b - c, "triangle coefficient")
    y = _as_int(a - b + c, "triangle coefficient")
    z = _as_int(-a + b + c, "triangle coefficient")
    denominator = _as_int(a + b + c + 1, "triangle coefficient")
    return Fraction(factorial(x) * factorial(y) * factorial(z), factorial(denominator))


def wigner_3j(
    j1: Any, j2: Any, j3: Any, m1: Any, m2: Any, m3: Any
) -> float:
    """Evaluate a Wigner 3-j symbol with the Racah factorial formula.

    Arguments must be integer or half-integer quantum numbers.  A coupling
    that violates a triangle, projection bound, or projection sum returns
    zero; malformed quantum numbers raise :class:`ValueError`.
    """
    a = _half_integer(j1, "j1")
    b = _half_integer(j2, "j2")
    c = _half_integer(j3, "j3")
    x = _half_integer(m1, "m1", nonnegative=False)
    y = _half_integer(m2, "m2", nonnegative=False)
    zed = _half_integer(m3, "m3", nonnegative=False)
    for j, m, name in ((a, x, "j1,m1"), (b, y, "j2,m2"), (c, zed, "j3,m3")):
        if abs(m) > j:
            return 0.0
        if (j + m).denominator != 1 or (j - m).denominator != 1:
            raise ValueError(f"{name} have incompatible integer/half-integer parity")
    if x + y + zed != 0 or not _triangle(a, b, c):
        return 0.0

    phase = _as_int(a - b - zed, "3-j phase")
    radicand = _delta_squared(a, b, c)
    for argument in (a + x, a - x, b + y, b - y, c + zed, c - zed):
        radicand *= factorial(_as_int(argument, "3-j factorial"))

    lower = max(
        0,
        _as_int(b - c - x, "3-j sum bound"),
        _as_int(a + y - c, "3-j sum bound"),
    )
    upper = min(
        _as_int(a + b - c, "3-j sum bound"),
        _as_int(a - x, "3-j sum bound"),
        _as_int(b + y, "3-j sum bound"),
    )
    total = Fraction(0)
    for k in range(lower, upper + 1):
        arguments = (
            k,
            _as_int(a + b - c, "3-j sum") - k,
            _as_int(a - x, "3-j sum") - k,
            _as_int(b + y, "3-j sum") - k,
            _as_int(c - b + x, "3-j sum") + k,
            _as_int(c - a - y, "3-j sum") + k,
        )
        denominator = math.prod(factorial(item) for item in arguments)
        total += Fraction((-1) ** k, denominator)
    return float(((-1) ** phase) * total) * math.sqrt(float(radicand))


def wigner_6j(j1: Any, j2: Any, j3: Any, j4: Any, j5: Any, j6: Any) -> float:
    """Evaluate a Wigner 6-j symbol with Racah's single-sum formula."""
    a, b, c, d, e, f = (
        _half_integer(value, name)
        for value, name in zip(
            (j1, j2, j3, j4, j5, j6),
            ("j1", "j2", "j3", "j4", "j5", "j6"),
        )
    )
    triangles = ((a, b, c), (a, e, f), (d, b, f), (d, e, c))
    if not all(_triangle(*triangle) for triangle in triangles):
        return 0.0

    alpha = (
        _as_int(a + b + c, "6-j sum bound"),
        _as_int(a + e + f, "6-j sum bound"),
        _as_int(d + b + f, "6-j sum bound"),
        _as_int(d + e + c, "6-j sum bound"),
    )
    beta = (
        _as_int(a + b + d + e, "6-j sum bound"),
        _as_int(a + c + d + f, "6-j sum bound"),
        _as_int(b + c + e + f, "6-j sum bound"),
    )
    total = Fraction(0)
    for zed in range(max(alpha), min(beta) + 1):
        denominator = math.prod(factorial(zed - item) for item in alpha)
        denominator *= math.prod(factorial(item - zed) for item in beta)
        total += Fraction(((-1) ** zed) * factorial(zed + 1), denominator)

    delta_product = Fraction(1)
    for triangle in triangles:
        delta_product *= _delta_squared(*triangle)
    return float(total) * math.sqrt(float(delta_product))


@dataclass(frozen=True)
class CorrectionFactors:
    """Optional, explicitly selected non-angular factors for one channel.

    ``radial_matrix_element`` is squared before use.  Alternatively,
    ``radial_strength_factor`` is used directly; supplying both is an error.
    ``weight`` is an additional nonnegative multiplicative factor.
    """

    radial_matrix_element: float | None = None
    radial_strength_factor: float | None = None
    photon_energy: float | None = None
    energy_weighting: str = "angular"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.radial_matrix_element is not None and self.radial_strength_factor is not None:
            raise ValueError("supply either radial_matrix_element or radial_strength_factor, not both")
        for name in ("radial_matrix_element", "radial_strength_factor", "photon_energy", "weight"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite real number; got {value!r}")
        if self.radial_strength_factor is not None and self.radial_strength_factor < 0:
            raise ValueError("radial_strength_factor must be nonnegative")
        if self.photon_energy is not None and self.photon_energy <= 0:
            raise ValueError("photon_energy must be positive")
        if self.weight < 0:
            raise ValueError("weight must be nonnegative")
        if not isinstance(self.energy_weighting, str):
            raise ValueError("energy_weighting must be 'angular', 'absorption', or 'emission'")
        convention = self.energy_weighting.lower().replace("-", "_")
        aliases = {"none": "angular", "omega": "absorption", "omega3": "emission"}
        convention = aliases.get(convention, convention)
        if convention not in {"angular", "absorption", "emission"}:
            raise ValueError("energy_weighting must be 'angular', 'absorption', or 'emission'")
        if convention != "angular" and self.photon_energy is None:
            raise ValueError(f"energy_weighting={convention!r} requires photon_energy")
        object.__setattr__(self, "energy_weighting", convention)

    @property
    def radial_factor(self) -> float:
        if self.radial_matrix_element is not None:
            return float(self.radial_matrix_element) ** 2
        if self.radial_strength_factor is not None:
            return float(self.radial_strength_factor)
        return 1.0

    @property
    def energy_factor(self) -> float:
        if self.energy_weighting == "angular":
            return 1.0
        power = 1 if self.energy_weighting == "absorption" else 3
        assert self.photon_energy is not None
        return float(self.photon_energy) ** power

    @property
    def total_factor(self) -> float:
        return self.radial_factor * self.energy_factor * float(self.weight)


@dataclass(frozen=True)
class Transition:
    """A directed electron transition and its optional explicit corrections."""

    initial: StateInput
    final: StateInput
    corrections: CorrectionFactors = field(default_factory=CorrectionFactors)


@dataclass(frozen=True)
class LineStrengthResult:
    """Angular factors and optional corrected strength for one channel."""

    initial: Subshell
    final: Subshell
    allowed: bool
    reason: str
    wigner_6j: float
    orbital_3j: float
    orbital_reduced_squared: float
    angular_strength: float
    corrections: CorrectionFactors
    corrected_strength: float
    notes: tuple[str, ...]


def _transition_parts(
    transition: Transition | tuple[StateInput, StateInput] | StateInput,
    final: StateInput | None,
    corrections: CorrectionFactors | None,
) -> tuple[Subshell, Subshell, CorrectionFactors]:
    if isinstance(transition, Transition):
        if final is not None or corrections is not None:
            raise ValueError("a Transition already contains its final state and corrections")
        return (
            parse_subshell(transition.initial),
            parse_subshell(transition.final),
            transition.corrections,
        )
    if final is None:
        if not isinstance(transition, tuple) or len(transition) != 2:
            raise ValueError("use a (from_state, to_state) tuple or supply both states")
        initial_input, final_input = transition
    else:
        initial_input, final_input = transition, final
    return parse_subshell(initial_input), parse_subshell(final_input), corrections or CorrectionFactors()


def angular_line_strength(
    transition: Transition | tuple[StateInput, StateInput] | StateInput,
    final: StateInput | None = None,
    *,
    corrections: CorrectionFactors | None = None,
) -> LineStrengthResult:
    """Calculate a complete-multiplet one-electron angular E1 strength.

    A forbidden channel returns a result with zero strengths.  Tuple order is
    electron direction, ``from_state -> to_state``.  Optional corrections are
    explicit and never alter the reported ``angular_strength``.
    """
    initial, target, factors = _transition_parts(transition, final, corrections)
    allowed, reason = e1_allowed(initial, target)
    notes = [
        "S_ang is a complete magnetic-substate/polarization sum; it is symmetric, not a rate."
    ]
    if factors.photon_energy is not None and factors.energy_weighting == "angular":
        notes.append("photon_energy was supplied but is not used for angular weighting")
    if not allowed:
        return LineStrengthResult(
            initial, target, False, reason, 0.0, 0.0, 0.0, 0.0, factors, 0.0,
            tuple(notes),
        )

    orbital_3j = wigner_3j(target.l, 1, initial.l, 0, 0, 0)
    orbital_reduced_sq = (
        (2 * target.l + 1) * (2 * initial.l + 1) * orbital_3j**2
    )
    six_j = wigner_6j(
        target.l, target.j, Fraction(1, 2),
        initial.j, initial.l, 1,
    )
    angular = initial.degeneracy * target.degeneracy * six_j**2 * orbital_reduced_sq
    if abs(angular) < 1e-15:
        angular = 0.0
    corrected = angular * factors.total_factor
    return LineStrengthResult(
        initial, target, True, reason, six_j, orbital_3j,
        orbital_reduced_sq, angular, factors, corrected, tuple(notes),
    )


@dataclass(frozen=True)
class BranchingChannel:
    line: LineStrengthResult
    fraction: float


@dataclass(frozen=True)
class BranchingResult:
    """Normalized branching data for a set of directed channels."""

    channels: tuple[BranchingChannel, ...]
    total_strength: float
    relative_ratio: tuple[int, ...] | None
    ratio_tolerance: float
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _integer_ratio(values: Sequence[float], tolerance: float) -> tuple[int, ...] | None:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        return None
    positive = [value for value in values if value > tolerance]
    if not positive:
        return None
    scale = min(positive)
    fractions = [Fraction(value / scale).limit_denominator(10000) for value in values]
    denominator = reduce(_lcm, (item.denominator for item in fractions), 1)
    integers = [item.numerator * (denominator // item.denominator) for item in fractions]
    common = reduce(gcd, integers)
    if common:
        integers = [item // common for item in integers]
    if max(integers, default=0) > 1_000_000:
        return None
    positive_integers = [item for item in integers if item]
    if not positive_integers:
        return None
    integer_scale = min(positive_integers)
    for value, integer in zip(values, integers):
        expected = integer / integer_scale
        actual = value / scale
        if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
            return None
    return tuple(integers)


def _orbital_pair(line: LineStrengthResult) -> frozenset[tuple[int, int]]:
    return frozenset(((line.initial.n, line.initial.l), (line.final.n, line.final.l)))


def branching_ratios(
    transitions: Iterable[Transition | tuple[StateInput, StateInput]],
    *,
    ratio_tolerance: float = _RATIO_TOLERANCE,
) -> BranchingResult:
    """Calculate normalized strengths and a recognizable integer ratio.

    Fractions use ``corrected_strength``.  With default corrections this is
    exactly the angular-only branching fraction.  Per-channel factors are
    provided by wrapping a channel in :class:`Transition`.
    """
    if ratio_tolerance <= 0:
        raise ValueError("ratio_tolerance must be positive")
    lines = tuple(angular_line_strength(item) for item in transitions)
    if not lines:
        raise ValueError("at least one transition is required")
    strengths = [line.corrected_strength for line in lines]
    total = math.fsum(strengths)
    fractions = [value / total if total > 0 else 0.0 for value in strengths]
    channels = tuple(
        BranchingChannel(line, fraction) for line, fraction in zip(lines, fractions)
    )
    assumptions = (
        "one-electron subshells with unchanged s=1/2",
        "complete sum over initial/final magnetic substates and photon polarizations",
        "no population/vacancy averaging in this channel comparison",
        "radial factor and photon-energy factor are unity unless explicitly supplied",
    )
    warnings: list[str] = []
    if total == 0:
        warnings.append("all supplied channels have zero strength; normalized fractions are undefined")
    uncorrected_pairs = {
        _orbital_pair(line)
        for line in lines
        if line.allowed
        and line.corrections.radial_matrix_element is None
        and line.corrections.radial_strength_factor is None
    }
    if len(uncorrected_pairs) > 1:
        warnings.append(
            "channels span different (n,l) orbital pairs without radial factors; "
            "their ratio is angular-only and need not be quantitatively physical"
        )
    if any(line.corrections.radial_matrix_element is None and
           line.corrections.radial_strength_factor is None for line in lines):
        warnings.append(
            "radial integrals are factored out; even spin-orbit partners may have "
            "slightly different relativistic radial functions"
        )
    return BranchingResult(
        channels, total, _integer_ratio(strengths, ratio_tolerance),
        ratio_tolerance, assumptions, tuple(warnings),
    )


@dataclass(frozen=True)
class AbsorptionChannel:
    line: LineStrengthResult
    initial_electrons: float
    final_holes: float
    occupation_factor: float
    vacancy_factor: float
    weighted_strength: float


@dataclass(frozen=True)
class EdgeTotal:
    initial: Subshell
    strength: float
    fraction: float


@dataclass(frozen=True)
class AbsorptionEdgeResult:
    """Independent-particle, uniformly populated absorption-edge totals."""

    channels: tuple[AbsorptionChannel, ...]
    edges: tuple[EdgeTotal, ...]
    total_strength: float
    relative_ratio: tuple[int, ...] | None
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


def _population_values(
    states: Sequence[Subshell],
    supplied: Mapping[Any, float] | None,
    description: str,
) -> dict[Subshell, float]:
    values = {state: float(state.degeneracy) for state in states}
    if supplied is None:
        return values
    for key, raw_value in supplied.items():
        state = parse_subshell(key)
        if state not in values:
            names = ", ".join(item.name for item in states)
            raise ValueError(f"{description} key {state.name} is not among the requested states: {names}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{description} for {state.name} must be a finite number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{description} for {state.name} must be finite")
        if not 0 <= value <= state.degeneracy:
            raise ValueError(
                f"{description} for {state.name} must satisfy 0 <= value <= "
                f"2j+1={state.degeneracy}; got {raw_value}"
            )
        values[state] = value
    return values


def _correction_lookup(
    factors: Mapping[tuple[str, str], CorrectionFactors] | None,
    initial: Subshell,
    final: Subshell,
) -> CorrectionFactors:
    if factors is None:
        return CorrectionFactors()
    for key, value in factors.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("transition_factors keys must be (from_label, to_label) tuples")
        if parse_subshell(key[0]) == initial and parse_subshell(key[1]) == final:
            if not isinstance(value, CorrectionFactors):
                raise ValueError("transition_factors values must be CorrectionFactors")
            return value
    return CorrectionFactors()


def absorption_edge_ratios(
    initial_edges: Sequence[StateInput],
    final_states: Sequence[StateInput],
    *,
    initial_electrons: Mapping[Any, float] | None = None,
    final_holes: Mapping[Any, float] | None = None,
    transition_factors: Mapping[tuple[str, str], CorrectionFactors] | None = None,
    ratio_tolerance: float = _RATIO_TOLERANCE,
) -> AbsorptionEdgeResult:
    """Compare integrated absorption edges for an explicit target manifold.

    Uniform occupation/vacancy weighting is
    ``S_ang * N_i/(2j_i+1) * H_f/(2j_f+1)``.  Missing population entries
    default to a filled initial or empty final subshell.  This remains an
    independent-particle isotropic model, not a many-body prediction.
    """
    initials = tuple(parse_subshell(item) for item in initial_edges)
    finals = tuple(parse_subshell(item) for item in final_states)
    if not initials or not finals:
        raise ValueError("at least one initial edge and one final state are required")
    if len(set(initials)) != len(initials) or len(set(finals)) != len(finals):
        raise ValueError("initial_edges and final_states must not contain duplicates")
    electrons = _population_values(initials, initial_electrons, "initial electron count")
    holes = _population_values(finals, final_holes, "final hole count")

    channels: list[AbsorptionChannel] = []
    totals = {state: 0.0 for state in initials}
    for initial in initials:
        for final in finals:
            correction = _correction_lookup(transition_factors, initial, final)
            line = angular_line_strength(initial, final, corrections=correction)
            occupied = electrons[initial] / initial.degeneracy
            vacant = holes[final] / final.degeneracy
            weighted = line.corrected_strength * occupied * vacant
            totals[initial] += weighted
            channels.append(
                AbsorptionChannel(
                    line, electrons[initial], holes[final], occupied, vacant, weighted
                )
            )
    grand_total = math.fsum(totals.values())
    edge_totals = tuple(
        EdgeTotal(state, totals[state], totals[state] / grand_total if grand_total else 0.0)
        for state in initials
    )
    assumptions = (
        "one-electron, independent-particle model with unchanged s=1/2",
        "uniform occupation within each initial subshell and uniform holes within each final subshell",
        "isotropic/unpolarized complete magnetic-substate sum",
        "initial subshells default to filled and specified final subshells default to empty",
        "radial factors are common/unity and photon-energy differences are ignored unless explicitly supplied",
    )
    warnings = [
        "A real partially occupied interacting d or f shell is not determined by edge labels alone; "
        "multiplets, <L.S>, configuration interaction, crystal fields, covalency, polarization, "
        "and nonuniform holes may change the ratio."
    ]
    pairs = {
        _orbital_pair(channel.line)
        for channel in channels
        if channel.line.allowed
        and channel.line.corrections.radial_matrix_element is None
        and channel.line.corrections.radial_strength_factor is None
    }
    if len(pairs) > 1:
        warnings.append(
            "different (n,l) orbital pairs are being compared without radial factors; "
            "the displayed ratio is angular-only"
        )
    return AbsorptionEdgeResult(
        tuple(channels), edge_totals, grand_total,
        _integer_ratio([edge.strength for edge in edge_totals], ratio_tolerance),
        assumptions, tuple(warnings),
    )


def _format_number(value: float) -> str:
    if abs(value) < 5e-15:
        return "0"
    return f"{value:.12g}"


def _scope_banner() -> str:
    return (
        "Model: one-electron |n l s j m_j> subshells with s=1/2; j is one-electron "
        "total angular momentum, not spin projection. This is not a many-electron term model."
    )


def _print_assumptions(assumptions: Sequence[str], warnings: Sequence[str]) -> None:
    print("Assumptions:")
    for item in assumptions:
        print(f"  - {item}")
    for item in warnings:
        print(f"Warning: {item}")


def _print_line(line: LineStrengthResult) -> None:
    print(f"Electron direction: {line.initial.name} -> {line.final.name}")
    print(f"  from: {line.initial}")
    print(f"  to:   {line.final}")
    print(f"  E1 allowed: {'yes' if line.allowed else 'no'} ({line.reason})")
    print(f"  orbital 3-j: {_format_number(line.orbital_3j)}")
    print(f"  Wigner 6-j: {_format_number(line.wigner_6j)}")
    print(f"  |<l_f||C^1||l_i>|^2: {_format_number(line.orbital_reduced_squared)}")
    print(f"  S_ang: {_format_number(line.angular_strength)}")
    if not math.isclose(line.corrected_strength, line.angular_strength, rel_tol=1e-15, abs_tol=1e-15):
        print(f"  corrected strength: {_format_number(line.corrected_strength)}")


def _print_branching(result: BranchingResult) -> None:
    print(_scope_banner())
    for channel in result.channels:
        line = channel.line
        print(
            f"{line.initial.name} -> {line.final.name}: allowed={'yes' if line.allowed else 'no'}, "
            f"S_ang={_format_number(line.angular_strength)}, "
            f"S_used={_format_number(line.corrected_strength)}, "
            f"fraction={_format_number(channel.fraction)} ({100 * channel.fraction:.6g}%)"
        )
        print(f"  {line.reason}; 3-j={_format_number(line.orbital_3j)}, 6-j={_format_number(line.wigner_6j)}")
    if result.relative_ratio is not None:
        ratio = ":".join(str(item) for item in result.relative_ratio)
        print(f"Relative intensity/probability ratio: {ratio} (recognized at tolerance {result.ratio_tolerance:g})")
        if len(result.relative_ratio) == 2 and set(result.relative_ratio) == {1, 9}:
            print("The corresponding magnitude-amplitude ratio is 3:1; the displayed 9:1 is an intensity ratio.")
    else:
        print("Relative ratio: no simple integer ratio recognized")
    _print_assumptions(result.assumptions, result.warnings)
    print("Note: fluorescence notation such as L3-M5 names the core hole first; emission moves M5 -> L3.")


def _print_edge_result(result: AbsorptionEdgeResult) -> None:
    print(_scope_banner())
    positive = [channel.weighted_strength for channel in result.channels if channel.weighted_strength > 1e-14]
    unit = min(positive) if positive else 1.0
    print("Channels (relative units use the smallest nonzero channel as 1):")
    for channel in result.channels:
        line = channel.line
        relative = channel.weighted_strength / unit
        print(
            f"  {line.initial.name} -> {line.final.name}: "
            f"S_ang={_format_number(line.angular_strength)}, N/g={_format_number(channel.occupation_factor)}, "
            f"H/g={_format_number(channel.vacancy_factor)}, weighted={_format_number(channel.weighted_strength)}, "
            f"relative={_format_number(relative)}; {'allowed' if line.allowed else 'forbidden'}"
        )
        if not line.allowed:
            print(f"    {line.reason}")
    print("Edge totals:")
    for edge in result.edges:
        print(
            f"  {edge.initial.name}: total={_format_number(edge.strength)}, "
            f"relative={_format_number(edge.strength / unit)}, "
            f"fraction={_format_number(edge.fraction)} ({100 * edge.fraction:.6g}%)"
        )
    if result.relative_ratio is not None:
        print("Edge intensity ratio: " + ":".join(str(item) for item in result.relative_ratio))
    else:
        print("Edge intensity ratio: no simple integer ratio recognized")
    _print_assumptions(result.assumptions, result.warnings)


def _parse_channel_argument(text: str) -> tuple[str, str]:
    pieces = text.split(":")
    if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
        raise ValueError(f"channel {text!r} must have FROM:TO form, for example L3:M5")
    return pieces[0].strip(), pieces[1].strip()


def _parse_counts(items: Sequence[str] | None, description: str) -> dict[str, float] | None:
    if not items:
        return None
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{description} entry {item!r} must have LABEL=COUNT form")
        label, value = item.split("=", 1)
        label = label.strip()
        if not label or label in result:
            raise ValueError(f"invalid or duplicate {description} label in {item!r}")
        try:
            result[label] = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid numeric count in {description} entry {item!r}") from exc
    return result


class _SelfTests(unittest.TestCase):
    def assertClose(self, first: float, second: float) -> None:
        self.assertTrue(math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12), (first, second))

    def test_label_parsing(self) -> None:
        self.assertEqual(parse_subshell("L3"), Subshell(2, 1, Fraction(3, 2)))
        self.assertEqual(parse_subshell("M5"), Subshell(3, 2, Fraction(5, 2)))
        self.assertEqual(parse_subshell("N7"), Subshell(4, 3, Fraction(7, 2)))
        self.assertEqual(
            parse_subshell({"n": 3, "l": "d", "j": 2.5}),
            parse_subshell("M5"),
        )

    def test_allowed_and_forbidden(self) -> None:
        self.assertTrue(e1_allowed("L3", "M5")[0])
        allowed, reason = e1_allowed("L2", "M5")
        self.assertFalse(allowed)
        self.assertIn("|Delta j|=2", reason)
        allowed, reason = e1_allowed("L3", "L2")
        self.assertFalse(allowed)
        self.assertIn("Delta l=0", reason)

    def test_channel_ratio(self) -> None:
        result = branching_ratios([("L3", "M5"), ("L3", "M4")])
        self.assertEqual(result.relative_ratio, (9, 1))
        self.assertClose(result.channels[0].line.angular_strength /
                         result.channels[1].line.angular_strength, 9.0)

    def test_three_channel_pattern(self) -> None:
        result = branching_ratios([("L2", "M4"), ("L3", "M4"), ("L3", "M5")])
        self.assertEqual(result.relative_ratio, (5, 1, 9))

    def test_statistical_edge_ratio(self) -> None:
        result = absorption_edge_ratios(["L2", "L3"], ["M4", "M5"])
        self.assertEqual(result.relative_ratio, (1, 2))
        self.assertClose(result.edges[0].fraction, 1 / 3)
        self.assertClose(result.edges[1].fraction, 2 / 3)
        channel_relative = [
            channel.weighted_strength /
            min(item.weighted_strength for item in result.channels if item.weighted_strength > 0)
            for channel in result.channels
        ]
        self.assertEqual(_integer_ratio(channel_relative, _RATIO_TOLERANCE), (5, 0, 1, 9))

    def test_reversal_symmetry(self) -> None:
        forward = angular_line_strength(("L3", "M5"))
        reverse = angular_line_strength(("M5", "L3"))
        self.assertClose(forward.angular_strength, reverse.angular_strength)
        self.assertIn("symmetric, not a rate", forward.notes[0])

    def test_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer or half-integer"):
            parse_subshell({"n": 2, "l": 1, "j": "1/3"})
        with self.assertRaisesRegex(ValueError, "impossible edge label"):
            parse_subshell("N8")
        with self.assertRaisesRegex(ValueError, "0 <= value"):
            absorption_edge_ratios(["L2"], ["M4"], initial_electrons={"L2": -1})
        with self.assertRaisesRegex(ValueError, "0 <= value"):
            absorption_edge_ratios(["L2"], ["M4"], final_holes={"M4": 5})


def _run_self_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SelfTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        f"Self-test summary: {result.testsRun - len(result.failures) - len(result.errors)}/"
        f"{result.testsRun} passed"
    )
    return result.wasSuccessful()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-electron spin-orbit-resolved angular E1 branching calculator (standard library only)."
    )
    parser.add_argument("--self-test", action="store_true", help="run the built-in verification suite")
    subparsers = parser.add_subparsers(dest="command")

    channel = subparsers.add_parser("channel", help="calculate one directed electron channel")
    channel.add_argument("initial", help="electron's initial subshell, e.g. L3")
    channel.add_argument("final", help="electron's final subshell, e.g. M5")
    channel.add_argument("--radial-matrix", type=float)
    channel.add_argument("--radial-strength", type=float)
    channel.add_argument("--photon-energy", type=float)
    channel.add_argument(
        "--energy-weighting", choices=("angular", "absorption", "emission"), default="angular"
    )
    channel.add_argument("--weight", type=float, default=1.0)

    ratio = subparsers.add_parser("ratio", help="compare FROM:TO channels")
    ratio.add_argument("channels", nargs="+", help="channels such as L3:M5 L3:M4")

    edge = subparsers.add_parser("edge-ratio", help="compare absorption edges into a final manifold")
    edge.add_argument("--initial", nargs="+", required=True, help="initial edges, e.g. L2 L3")
    edge.add_argument("--final", nargs="+", required=True, help="final states, e.g. M4 M5")
    edge.add_argument(
        "--initial-electrons", action="append", metavar="LABEL=COUNT",
        help="uniform initial electron count; repeat as needed",
    )
    edge.add_argument(
        "--final-holes", action="append", metavar="LABEL=COUNT",
        help="uniform final hole count; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.command is not None:
            parser.error("--self-test cannot be combined with a command")
        return 0 if _run_self_tests() else 1
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "channel":
            factors = CorrectionFactors(
                radial_matrix_element=args.radial_matrix,
                radial_strength_factor=args.radial_strength,
                photon_energy=args.photon_energy,
                energy_weighting=args.energy_weighting,
                weight=args.weight,
            )
            result = angular_line_strength(args.initial, args.final, corrections=factors)
            print(_scope_banner())
            _print_line(result)
            print("Parity changes because E1 requires Delta l=+/-1; spin is unchanged.")
            print("The radial integral is factored out of S_ang unless an explicit radial correction is supplied.")
            print("S_ang is symmetric under reversal, but populations and physical rate/cross-section factors are directional.")
            print("Fluorescence notation such as L3-M5 names the core hole first; emission moves M5 -> L3.")
        elif args.command == "ratio":
            _print_branching(branching_ratios(_parse_channel_argument(item) for item in args.channels))
        elif args.command == "edge-ratio":
            result = absorption_edge_ratios(
                args.initial,
                args.final,
                initial_electrons=_parse_counts(args.initial_electrons, "initial-electrons"),
                final_holes=_parse_counts(args.final_holes, "final-holes"),
            )
            _print_edge_result(result)
        else:  # pragma: no cover - argparse constrains this
            parser.error(f"unknown command {args.command!r}")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
