from typing import Literal, TypedDict


class OptionLegSpec(TypedDict, total=False):
    day_to_expiry_target: int
    strike_target: float
    strike_col: Literal["strike", "moneyness", "delta"]
    call_or_put: Literal["C", "P"]
    weight: float
    leg_name: str
    rebal_week_day: list[int]
