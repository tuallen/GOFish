# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Outward-rounded KB4/OpenCV-fisheye Hessian bound reference."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import math
from typing import Iterable, Union


NEG_INF = -math.inf
POS_INF = math.inf
DECIMAL_TRIG_PRECISION = 110
DECIMAL_TRIG_TERMS = 61


def down(value: float) -> float:
    return math.nextafter(float(value), NEG_INF)


def up(value: float) -> float:
    return math.nextafter(float(value), POS_INF)


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo <= self.hi):
            raise ValueError(f"invalid interval {self}")

    @staticmethod
    def point(value: float) -> "Interval":
        return Interval(float(value), float(value))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(down(self.lo + other.lo), up(self.hi + other.hi))

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(down(self.lo - other.hi), up(self.hi - other.lo))

    def __mul__(self, other: "Interval") -> "Interval":
        values = (self.lo*other.lo, self.lo*other.hi,
                  self.hi*other.lo, self.hi*other.hi)
        return Interval(down(min(values)), up(max(values)))

    def reciprocal_positive(self) -> "Interval":
        if not self.lo > 0.0:
            raise ValueError("positive reciprocal required")
        return Interval(down(1.0/self.hi), up(1.0/self.lo))

    def __truediv__(self, other: "Interval") -> "Interval":
        return self * other.reciprocal_positive()

    def absolute_upper(self) -> float:
        return up(max(abs(self.lo), abs(self.hi)))


def _decimal_binary(left: Decimal, right: Decimal, *, operation: str,
                    rounding: str) -> Decimal:
    with localcontext() as context:
        context.prec = DECIMAL_TRIG_PRECISION
        context.rounding = rounding
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        if operation == "divide":
            return left / right
    raise ValueError(f"unknown Decimal operation {operation}")


def _decimal_trig_bounds(value: float, *, cosine: bool) -> tuple[float, float]:
    """Directed 110-digit Taylor interval plus explicit Lagrange remainder.

    ``Decimal.from_float`` is exact. Every finite-precision Decimal primitive is
    evaluated once toward -infinity and once toward +infinity. After 61 terms,
    the degree-p Taylor remainder is bounded by x^(p+1)/(p+1)! because every
    derivative of sin/cos has magnitude at most one. No empirical pad is used.
    """
    x = Decimal.from_float(float(value))
    if x < 0:
        positive_lo, positive_hi = _decimal_trig_bounds(-float(value), cosine=cosine)
        if cosine:
            return positive_lo, positive_hi
        return down(-positive_hi), up(-positive_lo)
    if cosine:
        term_lo = term_hi = Decimal(1)
        total_lo = total_hi = Decimal(1)
        power = 0
    else:
        term_lo = term_hi = x
        total_lo = total_hi = x
        power = 1
    x2_lo = _decimal_binary(x, x, operation="multiply", rounding=ROUND_FLOOR)
    x2_hi = _decimal_binary(x, x, operation="multiply", rounding=ROUND_CEILING)
    for index in range(1, DECIMAL_TRIG_TERMS):
        denominator = Decimal(
            (2*index-1)*(2*index) if cosine else (2*index)*(2*index+1))
        term_lo = _decimal_binary(
            _decimal_binary(term_lo, x2_lo, operation="multiply", rounding=ROUND_FLOOR),
            denominator, operation="divide", rounding=ROUND_FLOOR)
        term_hi = _decimal_binary(
            _decimal_binary(term_hi, x2_hi, operation="multiply", rounding=ROUND_CEILING),
            denominator, operation="divide", rounding=ROUND_CEILING)
        if index % 2:
            total_lo = _decimal_binary(
                total_lo, term_hi, operation="subtract", rounding=ROUND_FLOOR)
            total_hi = _decimal_binary(
                total_hi, term_lo, operation="subtract", rounding=ROUND_CEILING)
        else:
            total_lo = _decimal_binary(
                total_lo, term_lo, operation="add", rounding=ROUND_FLOOR)
            total_hi = _decimal_binary(
                total_hi, term_hi, operation="add", rounding=ROUND_CEILING)
        power += 2

    # Taylor/Lagrange: |R_p(x)| <= x^(p+1)/(p+1)!.
    remainder_hi = _decimal_binary(
        _decimal_binary(term_hi, x, operation="multiply", rounding=ROUND_CEILING),
        Decimal(power+1), operation="divide", rounding=ROUND_CEILING)
    lo_dec = _decimal_binary(
        total_lo, remainder_hi, operation="subtract", rounding=ROUND_FLOOR)
    hi_dec = _decimal_binary(
        total_hi, remainder_hi, operation="add", rounding=ROUND_CEILING)
    lo, hi = float(lo_dec), float(hi_dec)
    while Decimal.from_float(lo) > lo_dec:
        lo = math.nextafter(lo, NEG_INF)
    while Decimal.from_float(hi) < hi_dec:
        hi = math.nextafter(hi, POS_INF)
    return down(lo), up(hi)


def sin_interval(theta: Interval) -> Interval:
    lo0, hi0 = _decimal_trig_bounds(theta.lo, cosine=False)
    lo1, hi1 = _decimal_trig_bounds(theta.hi, cosine=False)
    lo, hi = min(lo0, lo1), max(hi0, hi1)
    if theta.lo <= math.pi/2 <= theta.hi:
        hi = up(1.0)
    return Interval(down(lo), up(hi))


def cos_interval(theta: Interval) -> Interval:
    lo_hi, hi_hi = _decimal_trig_bounds(theta.hi, cosine=True)
    lo_lo, hi_lo = _decimal_trig_bounds(theta.lo, cosine=True)
    return Interval(down(lo_hi), up(hi_lo))


def _as_interval(value: Union[float, Interval]) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(value)


def _scaled_coefficient(value: float, multiplier: int) -> Interval:
    return Interval.point(value) * Interval.point(float(multiplier))


def _horner(theta2: Interval, coefficients: Iterable[Union[float, Interval]]) -> Interval:
    values = list(coefficients)
    result = _as_interval(values[-1])
    for value in reversed(values[:-1]):
        result = _as_interval(value) + theta2*result
    return result


def radial_intervals(theta: Interval, coefficients: Iterable[float]):
    k1, k2, k3, k4 = (float(value) for value in coefficients)
    t2 = theta*theta
    q = theta * _horner(t2, (1.0, k1, k2, k3, k4))
    q1 = _horner(t2, (Interval.point(1.0),
                             _scaled_coefficient(k1, 3),
                             _scaled_coefficient(k2, 5),
                             _scaled_coefficient(k3, 7),
                             _scaled_coefficient(k4, 9)))
    q2 = theta * _horner(t2, (_scaled_coefficient(k1, 6),
                                     _scaled_coefficient(k2, 20),
                                     _scaled_coefficient(k3, 42),
                                     _scaled_coefficient(k4, 72)))
    q3 = _horner(t2, (_scaled_coefficient(k1, 6),
                             _scaled_coefficient(k2, 60),
                             _scaled_coefficient(k3, 210),
                             _scaled_coefficient(k4, 504)))
    return q, q1, q2, q3


def _square_interval(value: Interval) -> Interval:
    if value.lo <= 0.0 <= value.hi:
        return Interval(0.0, up(max(value.lo*value.lo, value.hi*value.hi)))
    products = (value.lo*value.lo, value.hi*value.hi)
    return Interval(down(min(products)), up(max(products)))


def _sum_squares_upper(values: Iterable[float], multiplicities: Iterable[int]) -> float:
    total = 0.0
    for value, count in zip(values, multiplicities):
        square = up(value*value)
        for _ in range(count):
            total = up(total+square)
    return up(math.sqrt(total))


def _add_up(left: float, right: float) -> float:
    return up(left + right)


def _subtract_down(left: float, right: float) -> float:
    return down(left - right)


def _multiply_up(left: float, right: float) -> float:
    return up(left * right)


def _multiply_down(left: float, right: float) -> float:
    return down(left * right)


def _divide_up(left: float, right: float) -> float:
    if not right > 0.0:
        raise ValueError("positive denominator required")
    return up(left / right)


def hessian_bound_interval(theta: Interval, coefficients: Iterable[float], *,
                           small_axis_limit: float) -> dict[str, float]:
    q, q1, q2, q3 = radial_intervals(theta, coefficients)
    s, c = sin_interval(theta), cos_interval(theta)
    s2, c2, sc = s*s, c*c, s*c
    hxx = q2*c2 - Interval.point(2.0)*q1*sc
    hxz = Interval.point(-1.0)*q2*sc + q1*(s2-c2)
    hzz = q2*s2 + Interval.point(2.0)*q1*sc
    if theta.lo < small_axis_limit:
        sc_abs = sc.absolute_upper()
        q3_term = _multiply_up(q3.absolute_upper(), sc_abs)
        c2_plus_3s2 = _add_up(
            c2.absolute_upper(), _multiply_up(3.0, s2.absolute_upper()))
        q2_term = _multiply_up(q2.absolute_upper(), c2_plus_3s2)
        q1_term = _multiply_up(
            _multiply_up(4.0, q1.absolute_upper()), sc_abs)
        n2 = _add_up(_add_up(q3_term, q2_term), q1_term)
        theta2_upper = _multiply_up(theta.hi, theta.hi)
        sinc_lower = _subtract_down(1.0, _divide_up(theta2_upper, 6.0))
        if not sinc_lower > 0.0:
            raise ValueError("small-axis sinc lower bound failed")
        sinc_squared_lower = _multiply_down(sinc_lower, sinc_lower)
        denominator_lower = _multiply_down(2.0, sinc_squared_lower)
        a_abs = _divide_up(n2, denominator_lower)
    else:
        numerator = q1*c*s-q
        a_abs = (numerator/_square_interval(s)).absolute_upper()
    q1_abs = q1.absolute_upper()
    normalized = _sum_squares_upper(
        (hxx.absolute_upper(), hzz.absolute_upper(), hxz.absolute_upper(), a_abs, q1_abs),
        (1, 1, 2, 3, 2),
    )
    return {"normalized_frobenius_upper": normalized, "A_abs_upper": a_abs,
            "q1_abs_upper": q1_abs}


def band_bound(theta_lo: float, theta_hi: float, coefficients: Iterable[float], *,
               focal_max: float, small_axis_limit: float,
               subdivisions: int = 128) -> dict[str, float]:
    if not (0.0 <= theta_lo < theta_hi and subdivisions > 0 and focal_max > 0.0):
        raise ValueError("invalid band inputs")
    normalized = a_abs = q1_abs = 0.0
    for index in range(subdivisions):
        lo = theta_lo+(theta_hi-theta_lo)*index/subdivisions
        hi = theta_lo+(theta_hi-theta_lo)*(index+1)/subdivisions
        row = hessian_bound_interval(Interval(down(lo), up(hi)), coefficients,
                                     small_axis_limit=small_axis_limit)
        normalized = max(normalized, row["normalized_frobenius_upper"])
        a_abs = max(a_abs, row["A_abs_upper"])
        q1_abs = max(q1_abs, row["q1_abs_upper"])
    normalized = up(normalized)
    return {
        "normalized_frobenius_upper": normalized,
        "pixel_frobenius_upper": up(focal_max*normalized),
        "A_abs_upper": up(a_abs),
        "q1_abs_upper": up(q1_abs),
    }


def exact_combined_norm(theta: float, coefficients: Iterable[float]) -> float:
    k1, k2, k3, k4 = (float(value) for value in coefficients)
    if theta == 0.0:
        return 2.0
    t2 = theta*theta
    q = theta*(1+t2*(k1+t2*(k2+t2*(k3+t2*k4))))
    q1 = 1+t2*(3*k1+t2*(5*k2+t2*(7*k3+t2*9*k4)))
    q2 = theta*(6*k1+t2*(20*k2+t2*(42*k3+t2*72*k4)))
    s, c = math.sin(theta), math.cos(theta)
    a = (q1*c*s-q)/(s*s)
    hxx = q2*c*c-2*q1*s*c
    hxz = -q2*s*c+q1*(s*s-c*c)
    hzz = q2*s*s+2*q1*s*c
    return math.sqrt(hxx*hxx+hzz*hzz+2*hxz*hxz+3*a*a+2*q1*q1)


__all__ = ["DECIMAL_TRIG_PRECISION", "DECIMAL_TRIG_TERMS", "Interval",
           "band_bound", "exact_combined_norm", "hessian_bound_interval",
           "radial_intervals"]
