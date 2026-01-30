from investment_lab.dataclass import OptionLegSpec

CALENDAR_SPREAD_1W_1M_ATM_C: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1,
        "leg_name": "Short ATM Call 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1 / 4,
        "leg_name": "Long ATM Call 1M",
        "rebal_week_day": [2],
    },
]
CALENDAR_SPREAD_1M_6M_ATM_C: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1 / 4,
        "leg_name": "Short ATM Call 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4 * 6,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1 / (4 * 6),
        "leg_name": "Long ATM Call 6M",
        "rebal_week_day": [2],
    },
]

CALENDAR_SPREAD_1W_1M_ATM_P: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": -1,
        "leg_name": "Short ATM Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": 1 / 4,
        "leg_name": "Long ATM Put 1M",
        "rebal_week_day": [2],
    },
]


REVERSE_CALENDAR_SPREAD_1W_1M_ATM_C: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1,
        "leg_name": "Long ATM Call 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1 / 4,
        "leg_name": "Short ATM Call 1M",
        "rebal_week_day": [2],
    },
]
REVERSE_CALENDAR_SPREAD_1M_6M_ATM_C: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1 / 4,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1,
        "leg_name": "Long ATM Call 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4 * 6,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1 / (4 * 6),
        "leg_name": "Short ATM Call 6M",
        "rebal_week_day": [2],
    },
]

REVERSE_CALENDAR_SPREAD_1W_1M_ATM_P: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": 1,
        "leg_name": "Long ATM Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": -1 / 4,
        "leg_name": "Short ATM Put 1M",
        "rebal_week_day": [2],
    },
]


SHORT_1W_STRADDLE: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.5,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1 / 2,
        "leg_name": "Short ATM Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.5,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1 / 2,
        "leg_name": "Short ATM Call 1W",
        "rebal_week_day": [2],
    },
]


SHORT_1M_STRADDLE: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": -0.5,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1 / 2,
        "leg_name": "Short ATM Put 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.5,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1 / 2,
        "leg_name": "Short ATM Call 1M",
        "rebal_week_day": [2],
    },
]

SHORT_1W_STRANGLE_95_105: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.95,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": -1 / 2 / 3,
        "leg_name": "Short K=95% Put 1W",
        "rebal_week_day": [0, 2, 4],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 1.05,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1 / 2 / 3,
        "leg_name": "Short K=105% Call 1W",
        "rebal_week_day": [0, 2, 4],
    },
]

SHORT_1W_STRANGLE_20D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.2,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1 / 2,
        "leg_name": "Short 20D Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.2,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1 / 2,
        "leg_name": "Short 20D Call 1W",
        "rebal_week_day": [2],
    },
]

RISK_REVERSAL_1W_15D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.15,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1,
        "leg_name": "Short 15D Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.15,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": 1,
        "leg_name": "Long 15D Call 1W",
        "rebal_week_day": [2],
    },
]

RISK_REVERSAL_1M_15D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": -0.15,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1 / 4,
        "leg_name": "Short 15D Put 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.25,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": 1 / 4,
        "leg_name": "Long 15D Call 1M",
        "rebal_week_day": [2],
    },
]


INVERSE_RISK_REVERSAL_1W_15D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.15,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": 1,
        "leg_name": "Long 15D Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.15,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1,
        "leg_name": "Short 15D Call 1W",
        "rebal_week_day": [2],
    },
]

INVERSE_RISK_REVERSAL_1M_15D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": -0.15,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": 1 / 4,
        "leg_name": "Long 15D Put 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.15,
        "strike_col": "delta",
        "call_or_put": "C",
        "weight": -1 / 4,
        "leg_name": "Short 15D Call 1M",
        "rebal_week_day": [2],
    },
]


LONG_CALL_SPREAD_1M_100_105: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1.0,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1 / 4,
        "leg_name": "Long K=100% Call 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1.05,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1 / 4,
        "leg_name": "Short K=105% Call 1M",
        "rebal_week_day": [2],
    },
]


LONG_CALL_SPREAD_1W_100_105: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1.0,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": 1,
        "leg_name": "Long K=100% Call 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 1.05,
        "strike_col": "moneyness",
        "call_or_put": "C",
        "weight": -1,
        "leg_name": "Short K=105% Call 1W",
        "rebal_week_day": [2],
    },
]

SHORT_PUT_SPREAD_1W_98_100: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": 1.0,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": -1,
        "leg_name": "Short K=100% Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": 0.98,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": 1,
        "leg_name": "Long K=98% Put 1W",
        "rebal_week_day": [2],
    },
]

SHORT_PUT_SPREAD_1M_95_100: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 1.0,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": -1 / 4,
        "leg_name": "Short K=100% Put 1M",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7 * 4,
        "strike_target": 0.95,
        "strike_col": "moneyness",
        "call_or_put": "P",
        "weight": 1 / 4,
        "leg_name": "Long K=95% Put 1M",
        "rebal_week_day": [2],
    },
]

SHORT_PUT_SPREAD_1W_20D_40D: list[OptionLegSpec] = [
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.3,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": -1,
        "leg_name": "Short 30D Put 1W",
        "rebal_week_day": [2],
    },
    {
        "day_to_expiry_target": 7,
        "strike_target": -0.1,
        "strike_col": "delta",
        "call_or_put": "P",
        "weight": 1,
        "leg_name": "Long 10D Put 1W",
        "rebal_week_day": [2],
    },
]
