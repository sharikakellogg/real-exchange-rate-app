"""Bilateral real exchange rate dashboard built on IMF SDMX 3.0 data.

Definitions
    E_i(t)  = units of country i currency per USD (IMF ER dataflow, XDC_USD, period average)
    S(t)    = E2(t) / E1(t)           units of Country 2 currency per unit of Country 1 currency
    RER(t)  = S(t) * P1(t) / P2(t)
    RER*    = arithmetic mean of valid RER(t) inside the selected sample
    Real(t) = 100 * RER(t) / RER*
"""

from __future__ import annotations

import calendar
import html
import inspect
import io
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="Real Exchange Rate Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
BASE_URL = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"
DATA_START = "1976-01"
DATA_START_TS = pd.Timestamp("1976-01-01")
EURO_AREA = "G163"
CONNECT_TIMEOUT = 6
READ_TIMEOUT = 40
SERIES_TTL = 6 * 3600
COUNTRY_TTL = 24 * 3600
ERROR_COOLDOWN = 45  # seconds before a failed download is retried automatically
JUMP_THRESHOLD = 0.35  # month-on-month move in S flagged as a possible break

# IMF three-letter regional aggregates that are not countries.
REGIONAL_CODES = {"AFR", "APD", "EUR", "MCD", "WHD", "SDS"}

# Euro members whose pre-adoption exchange rates may be reported in legacy currency units.
EURO_MEMBERS = {
    "AUT", "BEL", "BGR", "CYP", "DEU", "ESP", "EST", "FIN", "FRA", "GRC", "HRV",
    "IRL", "ITA", "LTU", "LUX", "LVA", "MLT", "NLD", "PRT", "SVK", "SVN",
}

# Present-day currency codes used only for labels. Euro members are deliberately absent
# because their historical currency differs before euro adoption.
CURRENCY = {
    "USA": "USD", "GBR": "GBP", EURO_AREA: "EUR", "JPN": "JPY", "CHE": "CHF", "CAN": "CAD",
    "AUS": "AUD", "NZL": "NZD", "CHN": "CNY", "IND": "INR", "BRA": "BRL", "MEX": "MXN",
    "KOR": "KRW", "SWE": "SEK", "NOR": "NOK", "DNK": "DKK", "ZAF": "ZAR", "RUS": "RUB",
    "TUR": "TRY", "SGP": "SGD", "HKG": "HKD", "ISR": "ILS", "POL": "PLN", "CZE": "CZK",
    "HUN": "HUF", "IDN": "IDR", "THA": "THB", "MYS": "MYR", "PHL": "PHP", "SAU": "SAR",
    "ARE": "AED", "EGY": "EGP", "NGA": "NGN", "KEN": "KES", "ARG": "ARS", "CHL": "CLP",
    "COL": "COP", "PER": "PEN", "ISL": "ISK", "PAK": "PKR", "LKA": "LKR", "UKR": "UAH",
    "ROU": "RON", "MAR": "MAD", "GHA": "GHS",
}

# Used only if the IMF country codelist cannot be downloaded.
FALLBACK_COUNTRIES = {
    "USA": "United States", "GBR": "United Kingdom", EURO_AREA: "Euro area", "JPN": "Japan",
    "CHE": "Switzerland", "CAN": "Canada", "AUS": "Australia", "NZL": "New Zealand",
    "CHN": "China", "IND": "India", "BRA": "Brazil", "MEX": "Mexico", "KOR": "Korea, Rep. of",
    "SWE": "Sweden", "NOR": "Norway", "DNK": "Denmark", "ZAF": "South Africa",
    "RUS": "Russian Federation", "TUR": "Turkiye", "SGP": "Singapore", "HKG": "Hong Kong SAR",
    "ISR": "Israel", "POL": "Poland", "CZE": "Czech Republic", "HUN": "Hungary",
    "IDN": "Indonesia", "THA": "Thailand", "MYS": "Malaysia", "PHL": "Philippines",
    "SAU": "Saudi Arabia", "ARE": "United Arab Emirates", "EGY": "Egypt", "NGA": "Nigeria",
    "KEN": "Kenya", "ARG": "Argentina", "CHL": "Chile", "COL": "Colombia", "PER": "Peru",
    "ISL": "Iceland", "PAK": "Pakistan", "LKA": "Sri Lanka", "UKR": "Ukraine",
    "ROU": "Romania", "MAR": "Morocco", "GHA": "Ghana", "DEU": "Germany", "FRA": "France",
    "ITA": "Italy", "ESP": "Spain", "NLD": "Netherlands", "BEL": "Belgium", "AUT": "Austria",
    "IRL": "Ireland", "PRT": "Portugal", "GRC": "Greece", "FIN": "Finland",
}

# Palette
NAVY = "#0B2545"
TEAL = "#0F9D94"
TEAL_DARK = "#0B7A73"
OFFWHITE = "#F6F4EF"
INK = "#1B2A41"
MUTED = "#5F6B7A"
GRID = "#E6E2D9"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


class IMFError(Exception):
    """Raised for download or response-structure problems that the user should see."""


# --------------------------------------------------------------------------------------
# Streamlit version helpers
# --------------------------------------------------------------------------------------
def stretch(fn) -> dict:
    """Return the full-width keyword that this Streamlit version understands."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


# --------------------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=1,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    session.headers.update({"User-Agent": "rer-dashboard-streamlit/1.0 (academic project)"})
    return session


def http_get(url: str, params: dict | None, accept: str) -> requests.Response:
    try:
        return get_session().get(
            url,
            params=params,
            headers={"Accept": accept},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.Timeout as exc:
        raise IMFError("the IMF API timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise IMFError("could not connect to the IMF API") from exc
    except requests.exceptions.RequestException as exc:
        raise IMFError(f"request failed ({type(exc).__name__})") from exc


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
KNOWN_DIMS = {
    "COUNTRY", "INDEX_TYPE", "INDICATOR", "COICOP_1999",
    "TYPE_OF_TRANSFORMATION", "FREQUENCY", "FREQ",
}
SCALE_WORDS = {
    "": 0, "unit": 0, "units": 0, "ones": 0, "thousands": 3, "millions": 6,
    "billions": 9, "trillions": 12,
}


def empty_series() -> pd.DataFrame:
    return pd.DataFrame({"month": pd.DatetimeIndex([]), "value": pd.Series([], dtype=float)})


def scale_exponent(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in SCALE_WORDS:
        return SCALE_WORDS[text]
    try:
        return int(float(text))  # numeric SDMX unit-multiplier code = power of ten
    except ValueError:
        return None


def parse_sdmx_csv(text: str, expect_country: str) -> tuple[pd.DataFrame, dict]:
    info: dict = {"rows_raw": 0, "dropped_blank": 0, "dropped_nonnumeric": 0,
                  "dropped_period": 0, "dropped_pre_start": 0, "scale_note": ""}
    if not text.strip():
        return empty_series(), info
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:  # malformed CSV
        raise IMFError(f"response was not readable CSV ({type(exc).__name__})") from exc
    raw.columns = [re.split(r"\s*:\s*", str(c).strip(), maxsplit=1)[0].upper() for c in raw.columns]
    info["rows_raw"] = int(len(raw))
    if raw.empty:
        return empty_series(), info

    missing = {"TIME_PERIOD", "OBS_VALUE"} - set(raw.columns)
    if missing:
        raise IMFError(
            "unexpected response structure (missing "
            f"{', '.join(sorted(missing))}; columns seen: {', '.join(raw.columns[:10])})"
        )
    dims = [c for c in raw.columns if c in KNOWN_DIMS]
    if dims and len(raw[dims].drop_duplicates()) > 1:
        raise IMFError("response held more than one series for a single-series request")
    if "COUNTRY" in raw.columns and not (raw["COUNTRY"].str.strip().str.upper() == expect_country).all():
        raise IMFError("response country code did not match the request")

    # Periods such as 2024-M01 (or 2024-01) become month-start timestamps.
    parts = raw["TIME_PERIOD"].str.strip().str.extract(r"^(\d{4})-(?:M)?(\d{1,2})(?:-\d{1,2})?$")
    good_period = parts[0].notna()
    info["dropped_period"] = int((~good_period).sum())
    work = raw.loc[good_period].copy()
    p = parts.loc[good_period]
    work["month"] = pd.to_datetime(
        {"year": p[0].astype(int), "month": p[1].astype(int), "day": 1}, errors="coerce"
    )

    obs = work["OBS_VALUE"].str.strip()
    blank = obs == ""
    info["dropped_blank"] = int(blank.sum())
    values = pd.to_numeric(obs.where(~blank), errors="coerce")
    info["dropped_nonnumeric"] = int((values.isna() & ~blank).sum())

    if "SCALE" in work.columns:
        exps = work["SCALE"].map(scale_exponent)
        if exps.isna().any():
            bad = sorted(set(work.loc[exps.isna(), "SCALE"].astype(str)))[:3]
            info["scale_note"] = f"unrecognised SCALE value(s) {bad} were not applied"
            exps = exps.fillna(0)
        elif (exps != 0).any():
            info["scale_note"] = "non-unit SCALE values were applied"
        values = values * np.power(10.0, exps.astype(float))

    out = pd.DataFrame({"month": work["month"], "value": values})
    out = out.dropna(subset=["month", "value"])
    out = out[np.isfinite(out["value"])]

    pre = out["month"] < DATA_START_TS
    info["dropped_pre_start"] = int(pre.sum())
    out = out.loc[~pre]

    dup = out["month"].duplicated(keep=False)
    if dup.any():
        spread = out.loc[dup].groupby("month")["value"].nunique()
        if (spread > 1).any():
            raise IMFError("duplicate months with different values in the response")
        out = out.drop_duplicates("month")
    return out.sort_values("month").reset_index(drop=True), info


# --------------------------------------------------------------------------------------
# Series download (cached)
# --------------------------------------------------------------------------------------
def fetch_series(flow: str, key: str, country: str) -> dict:
    """Download one monthly series. Tries the version tokens and filter styles seen in the wild."""
    statuses: list[int] = []
    resp = None
    for version in ("+", "~"):
        for use_filter in (True, False):
            url = f"{BASE_URL}/data/dataflow/{AGENCY}/{flow}/{version}/{key}"
            params = {"c[TIME_PERIOD]": f"ge:{DATA_START}"} if use_filter else None
            r = http_get(url, params, "text/csv")
            if r.status_code == 200:
                resp = r
                break
            statuses.append(r.status_code)
        if resp is not None:
            break

    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    info = {"flow": flow, "key": key, "retrieved_utc": retrieved, "http_status": None,
            "url": f"{BASE_URL}/data/dataflow/{AGENCY}/{flow}/+/{key}", "note": ""}

    if resp is None:
        if statuses and all(s == 404 for s in statuses):
            info["http_status"] = 404
            info["note"] = "HTTP 404 on every attempt, treated as no matching series"
            return {"data": empty_series(), "info": {**info, "rows_raw": 0}}
        raise IMFError(f"IMF API answered HTTP {statuses[-1] if statuses else 'unknown'} for {key}")

    info["http_status"] = 200
    info["url"] = resp.url
    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" in ctype:
        raise IMFError("received an HTML page instead of data (possible gateway or access page)")
    text = resp.content.decode("utf-8-sig", errors="replace")
    data, parse_info = parse_sdmx_csv(text, country)
    info.update(parse_info)
    return {"data": data, "info": info}


@st.cache_data(ttl=SERIES_TTL, show_spinner=False)
def fetch_cpi(code: str) -> dict:
    res = fetch_series("CPI", f"{code}.CPI._T.IX.M", code)
    res["info"]["series_label"] = "CPI, all items, index"
    if res["data"].empty and code == EURO_AREA:
        alt = fetch_series("CPI", f"{code}.HICP._T.IX.M", code)
        if not alt["data"].empty:
            alt["info"]["series_label"] = "HICP, all items, index"
            alt["info"]["note"] = (
                "The IMF has no CPI-type series for the euro area; its harmonised index (HICP) is used and labelled."
            )
            return alt
    return res


@st.cache_data(ttl=SERIES_TTL, show_spinner=False)
def fetch_fx(code: str) -> dict:
    res = fetch_series("ER", f"{code}.XDC_USD.PA_RT.M", code)
    res["info"]["series_label"] = "Exchange rate, local currency per USD, period average"
    if res["data"].empty and code == "USA" and res["info"].get("http_status") == 200:
        months = pd.date_range(DATA_START_TS, pd.Timestamp.today().to_period("M").to_timestamp(), freq="MS")
        res["data"] = pd.DataFrame({"month": months, "value": 1.0})
        res["info"]["note"] = "No USD per USD series returned; set to 1 by definition."
    return res


@st.cache_data(ttl=COUNTRY_TTL, show_spinner=False)
def fetch_country_list() -> dict:
    """Return {code: name} from the IMF CL_COUNTRY codelist."""
    url = f"{BASE_URL}/structure/codelist/IMF/CL_COUNTRY/+"
    last = "no response"
    for accept in (
        "application/vnd.sdmx.structure+json;version=2.0.0, application/json;q=0.9",
        "application/json",
        "*/*",
    ):
        r = http_get(url, None, accept)
        if r.status_code != 200:
            last = f"HTTP {r.status_code}"
            continue
        try:
            codes = extract_codes(r.json())
        except ValueError:
            last = "response was not JSON"
            continue
        if codes:
            return codes
        last = "no codes found in response"
    raise IMFError(f"country list unavailable ({last})")


def extract_codes(payload) -> dict:
    """Pull {id: name} out of an SDMX-JSON structure message, tolerating layout differences."""

    def name_of(item: dict) -> str:
        name = item.get("name")
        if isinstance(name, dict):
            name = name.get("en") or next(iter(name.values()), None)
        if not name and isinstance(item.get("names"), dict):
            names = item["names"]
            name = names.get("en") or next(iter(names.values()), None)
        return str(name or item.get("id", "")).strip()

    def walk(node):
        if isinstance(node, dict):
            if node.get("id") == "CL_COUNTRY" and isinstance(node.get("codes"), list):
                yield node["codes"]
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)

    for codes in walk(payload):
        out = {str(c["id"]).strip(): name_of(c) for c in codes if isinstance(c, dict) and c.get("id")}
        if out:
            return out
    return {}


def tidy_country_list(raw: dict) -> dict:
    out = {}
    for code, name in raw.items():
        if code == EURO_AREA:
            out[code] = "Euro area"
        elif re.fullmatch(r"[A-Z]{3}", code) and code not in REGIONAL_CODES:
            out[code] = name or code
    out.setdefault(EURO_AREA, "Euro area")
    return dict(sorted(out.items(), key=lambda kv: kv[1].lower()))


# --------------------------------------------------------------------------------------
# Economics (pure functions, no Streamlit)
# --------------------------------------------------------------------------------------
def build_panel(fx1, fx2, cpi1, cpi2) -> pd.DataFrame:
    """Align the four input series on a complete monthly grid starting in 1976-01."""
    series = {"fx1": fx1, "fx2": fx2, "cpi1": cpi1, "cpi2": cpi2}
    lasts = [s["month"].max() for s in series.values() if len(s)]
    if not lasts:
        return pd.DataFrame(columns=list(series), index=pd.DatetimeIndex([], freq="MS"), dtype=float)
    idx = pd.date_range(DATA_START_TS, max(lasts), freq="MS")
    cols = {}
    for name, s in series.items():
        cols[name] = s.set_index("month")["value"].reindex(idx) if len(s) else pd.Series(np.nan, index=idx)
    return pd.DataFrame(cols, index=idx)


def add_rer(panel: pd.DataFrame) -> pd.DataFrame:
    """S(t) and RER(t) for every month. Non-positive or missing inputs give NaN."""
    d = panel.copy()
    fx1, fx2 = d["fx1"].where(d["fx1"] > 0), d["fx2"].where(d["fx2"] > 0)
    p1, p2 = d["cpi1"].where(d["cpi1"] > 0), d["cpi2"].where(d["cpi2"] > 0)
    d["S"] = fx2 / fx1  # units of Country 2 currency per unit of Country 1 currency
    d["rer"] = d["S"] * p1 / p2
    d["valid"] = np.isfinite(d["rer"])
    return d


def benchmark(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """RER* = mean of valid RER in [start, end]; Real Index = 100 * RER / RER*."""
    window = d.loc[start:end].copy()
    valid = window.loc[window["valid"], "rer"]
    star = float(valid.mean()) if len(valid) else float("nan")
    window["rer_star"] = star
    window["real_index"] = 100.0 * window["rer"] / star if np.isfinite(star) else np.nan
    return window, star


# --------------------------------------------------------------------------------------
# Labels and formatting
# --------------------------------------------------------------------------------------
def unit_label(c1: str, c2: str, names: dict) -> str:
    a, b = CURRENCY.get(c1), CURRENCY.get(c2)
    if a and b:
        return f"{b} per {a}"
    return f"units of {names.get(c2, c2)} currency per unit of {names.get(c1, c1)} currency"


def short_unit(c1: str, c2: str) -> str:
    a, b = CURRENCY.get(c1), CURRENCY.get(c2)
    return f"{b} per {a}" if a and b else f"{c2} currency per {c1} currency"


def fmt_num(x, digits=4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{digits}f}"


def fmt_month(ts) -> str:
    return "n/a" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%b %Y")


def month_ts(year, month) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=1)


# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');
:root {{ color-scheme: light; }}
.stApp {{ background: {OFFWHITE}; color: {INK}; font-family: {FONT}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: #ECE9E1; border-right: 1px solid {GRID}; }}
[data-testid="stSidebar"] * {{ color: {INK}; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ color: {NAVY}; font-family: 'Source Serif 4', Georgia, serif; letter-spacing: -0.01em; }}
.hero {{
  background: linear-gradient(120deg, {NAVY} 0%, #12365F 62%, {TEAL_DARK} 140%);
  border-radius: 16px; padding: 26px 30px; margin-bottom: 18px; color: #fff;
}}
.hero h1 {{ color: #fff; margin: 0 0 4px 0; font-size: 1.85rem; }}
.hero p {{ color: #D5E3EE; margin: 0; font-size: 0.95rem; }}
.hero .pair {{ margin-top: 12px; font-size: 0.9rem; color: #BFE9E5; font-weight: 500; }}
.kpi {{
  background: #fff; border: 1px solid {GRID}; border-top: 4px solid {TEAL};
  border-radius: 12px; padding: 16px 18px 14px 18px; height: 100%;
  box-shadow: 0 1px 2px rgba(11,37,69,0.05);
}}
.kpi.navy {{ border-top-color: {NAVY}; }}
.kpi-label {{ font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; font-weight: 600; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {NAVY}; line-height: 1.25; margin: 4px 0 2px 0; }}
.kpi-sub {{ font-size: 0.82rem; color: {MUTED}; }}
.section-note {{ color: {MUTED}; font-size: 0.88rem; margin: -4px 0 10px 0; }}
button[data-baseweb="tab"] p {{ font-weight: 600; color: {NAVY}; }}
[data-testid="stExpander"] {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; }}
[data-baseweb="select"] > div {{ background-color: #fff; color: {INK}; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px; border: 1px solid {TEAL_DARK}; color: {TEAL_DARK}; background: #fff; font-weight: 600;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {TEAL}; color: #fff; border-color: {TEAL}; }}
</style>
""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, navy: bool = False) -> str:
    cls = "kpi navy" if navy else "kpi"
    return (
        f'<div class="{cls}"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
    )


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def style_figure(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=12, r=12, t=56, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=GRID,
        tickformatstops=[
            dict(dtickrange=[None, "M1"], value="%d %b %Y"),
            dict(dtickrange=["M1", "M12"], value="%b %Y"),
            dict(dtickrange=["M12", None], value="%Y"),
        ],
        rangeselector=dict(
            buttons=[
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(count=10, label="10y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=OFFWHITE,
            activecolor="#CBE8E5",
            font=dict(size=12),
        ),
    )
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def main_chart(df: pd.DataFrame, view: str, unit: str, c1: str, c2: str) -> go.Figure:
    fig = go.Figure()
    show_real = view in ("Real", "Both")
    show_nom = view in ("Nominal", "Both")
    if show_real:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["real_index"], name="Real index (sample average = 100)",
            mode="lines", line=dict(color=TEAL, width=2.6), connectgaps=False,
            customdata=df["rer"],
            hovertemplate="%{y:.2f}  (RER %{customdata:.4f})<extra></extra>",
        ))
        fig.add_hline(y=100, line_dash="dot", line_color=TEAL_DARK, line_width=1,
                      annotation_text="sample average = 100", annotation_position="bottom right",
                      annotation_font=dict(size=11, color=TEAL_DARK))
    if show_nom:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["S"], name=f"Nominal exchange rate S ({unit})",
            mode="lines", line=dict(color=NAVY, width=2.2), connectgaps=False,
            yaxis="y2" if view == "Both" else "y",
            hovertemplate="%{y:,.4f} " + unit + "<extra></extra>",
        ))
    if view == "Real":
        fig.update_yaxes(title_text="Real index (sample average = 100)")
    elif view == "Nominal":
        fig.update_yaxes(title_text=f"Nominal S ({unit})")
    else:
        fig.update_layout(
            yaxis=dict(title=dict(text="Real index (sample average = 100)", font=dict(color=TEAL_DARK)),
                       tickfont=dict(color=TEAL_DARK)),
            yaxis2=dict(title=dict(text=f"Nominal S ({unit})", font=dict(color=NAVY)),
                        tickfont=dict(color=NAVY), overlaying="y", side="right", showgrid=False),
        )
    return style_figure(fig)


def yoy_chart(real_yoy: pd.Series, nom_yoy: pd.Series, view: str) -> go.Figure:
    fig = go.Figure()
    if view in ("Real", "Both"):
        fig.add_trace(go.Scatter(x=real_yoy.index, y=real_yoy * 100, name="Real index, annual change",
                                 mode="lines", line=dict(color=TEAL, width=2.4), connectgaps=False,
                                 hovertemplate="%{y:+.1f}%<extra></extra>"))
    if view in ("Nominal", "Both"):
        fig.add_trace(go.Scatter(x=nom_yoy.index, y=nom_yoy * 100, name="Nominal S, annual change",
                                 mode="lines", line=dict(color=NAVY, width=2.0), connectgaps=False,
                                 hovertemplate="%{y:+.1f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.update_yaxes(title_text="% change vs same month one year earlier", ticksuffix="%")
    return style_figure(fig, 420)


def long_window_chart(sample: pd.DataFrame, longer: pd.DataFrame, label_long: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample.index, y=sample["real_index"], name="Selected-sample benchmark (headline)",
                             mode="lines", line=dict(color=TEAL, width=2.6), connectgaps=False,
                             hovertemplate="%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=longer.index, y=longer["real_index"], name=label_long,
                             mode="lines", line=dict(color=NAVY, width=2.0, dash="dash"), connectgaps=False,
                             hovertemplate="%{y:.2f}<extra></extra>"))
    fig.add_hline(y=100, line_dash="dot", line_color=MUTED, line_width=1)
    fig.update_yaxes(title_text="Real index (100 = that window's average)")
    return style_figure(fig, 420)


# --------------------------------------------------------------------------------------
# Data loading with graceful failure
# --------------------------------------------------------------------------------------
def safe_fetch(kind: str, code: str) -> tuple[dict | None, str | None]:
    errors = st.session_state.setdefault("_errors", {})
    key = (kind, code)
    if key in errors and time.time() - errors[key][0] < ERROR_COOLDOWN:
        return None, errors[key][1]
    try:
        result = (fetch_cpi if kind == "cpi" else fetch_fx)(code)
        errors.pop(key, None)
        return result, None
    except IMFError as exc:
        message = str(exc)
    except Exception as exc:  # unexpected, still shown rather than crashing
        message = f"unexpected error ({type(exc).__name__}: {exc})"
    errors[key] = (time.time(), message)
    return None, message


def load_inputs(c1: str, c2: str, names: dict) -> dict:
    plan = [
        ("cpi1", "cpi", c1, f"{names[c1]} CPI"),
        ("cpi2", "cpi", c2, f"{names[c2]} CPI"),
        ("fx1", "fx", c1, f"{names[c1]} exchange rate"),
        ("fx2", "fx", c2, f"{names[c2]} exchange rate"),
    ]
    out = {}
    with st.status("Loading IMF data", expanded=False) as status:
        for tag, kind, code, label in plan:
            status.update(label=f"Loading {label} ({code})")
            res, err = safe_fetch(kind, code)
            out[tag] = {"label": label, "code": code, "kind": kind, "result": res, "error": err}
        failed = [v["label"] for v in out.values() if v["error"]]
        status.update(
            label="Some IMF downloads failed" if failed else "IMF data loaded",
            state="error" if failed else "complete",
        )
    return out


def series_of(entry: dict) -> pd.DataFrame:
    return entry["result"]["data"] if entry["result"] else empty_series()


def coverage_table(inputs: dict) -> pd.DataFrame:
    rows = []
    for tag in ("cpi1", "cpi2", "fx1", "fx2"):
        e = inputs[tag]
        data = series_of(e)
        info = e["result"]["info"] if e["result"] else {}
        if e["error"]:
            status = f"Download failed: {e['error']}"
        elif data.empty:
            status = "No observations"
        else:
            status = "OK"
        first = data["month"].min() if len(data) else pd.NaT
        last = data["month"].max() if len(data) else pd.NaT
        expected = ((last.year - first.year) * 12 + last.month - first.month + 1) if len(data) else 0
        notes = [info.get("note", ""), info.get("scale_note", "")]
        dropped = sum(info.get(k, 0) for k in ("dropped_blank", "dropped_nonnumeric", "dropped_period"))
        if dropped:
            notes.append(f"{dropped} unusable rows dropped")
        rows.append({
            "Input": e["label"],
            "Series": info.get("series_label", "n/a"),
            "Status": status,
            "First": first,
            "Last": last,
            "Observations": int(len(data)),
            "Missing months inside range": int(expected - len(data)) if len(data) else 0,
            "Notes": "; ".join(n for n in notes if n),
            "Retrieved": info.get("retrieved_utc", ""),
            "Request": info.get("url", ""),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------------------
def missing_reasons(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    labels = {"fx1": f"{c1} FX", "fx2": f"{c2} FX", "cpi1": f"{c1} CPI", "cpi2": f"{c2} CPI"}
    reasons = []
    for _, row in df.iterrows():
        parts = []
        for col, lab in labels.items():
            v = row[col]
            if pd.isna(v):
                parts.append(f"{lab} missing")
            elif v <= 0:
                parts.append(f"{lab} non-positive")
        reasons.append("; ".join(parts))
    return pd.Series(reasons, index=df.index)


def export_frame(sample: pd.DataFrame, c1: str, c2: str, names: dict, unit: str,
                 star: float, inputs: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=sample.index)
    out["month"] = sample.index.strftime("%Y-%m")
    out["country1_code"], out["country1_name"] = c1, names[c1]
    out["country2_code"], out["country2_name"] = c2, names[c2]
    out["sample_start"] = sample.index.min().strftime("%Y-%m")
    out["sample_end"] = sample.index.max().strftime("%Y-%m")
    out["fx1_country1_lcu_per_usd"] = sample["fx1"]
    out["fx2_country2_lcu_per_usd"] = sample["fx2"]
    out["cpi1"] = sample["cpi1"]
    out["cpi2"] = sample["cpi2"]
    out["S_country2_per_country1"] = sample["S"]
    out["S_unit"] = unit
    out["RER"] = sample["rer"]
    out["RER_star_sample_average"] = star
    out["real_index"] = sample["real_index"]
    out["valid_rer_observation"] = sample["valid"]
    out["missing_or_invalid_inputs"] = missing_reasons(sample, c1, c2)
    for tag in ("cpi1", "cpi2", "fx1", "fx2"):
        info = inputs[tag]["result"]["info"] if inputs[tag]["result"] else {}
        out[f"source_{tag}"] = f"{info.get('flow', '')}:{info.get('key', '')} [{info.get('series_label', '')}]"
    out["retrieved_utc"] = max(
        (inputs[t]["result"]["info"].get("retrieved_utc", "") for t in inputs if inputs[t]["result"]),
        default="",
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Sidebar sample selection (session-state driven so presets and selectors stay in sync)
# --------------------------------------------------------------------------------------
def set_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    st.session_state.update(sy=start.year, sm=start.month, ey=end.year, em=end.month)


def apply_preset(years: int | None) -> None:
    first, last = st.session_state["_shared_first"], st.session_state["_shared_last"]
    start = first if years is None else max(first, last - pd.DateOffset(months=12 * years - 1))
    set_range(start, last)


# --------------------------------------------------------------------------------------
# Notices
# --------------------------------------------------------------------------------------
def collect_notices(inputs, cov, sample, c1, c2, names, full, start, end) -> tuple[list, list, list]:
    errors, warnings, infos = [], [], []
    for tag in ("cpi1", "cpi2", "fx1", "fx2"):
        e = inputs[tag]
        if e["error"]:
            errors.append(f"Could not download {e['label']}: {e['error']}. Anything that depends on it is left blank.")
    for _, r in cov.iterrows():
        if r["Status"] == "No observations":
            warnings.append(f"The IMF returned no observations for {r['Input']}.")
        elif r["Status"] == "OK":
            if r["First"] > DATA_START_TS:
                infos.append(f"{r['Input']} starts in {fmt_month(r['First'])}, not January 1976.")
            if r["Missing months inside range"] > 0:
                warnings.append(f"{r['Input']} has {r['Missing months inside range']} missing months between "
                                f"{fmt_month(r['First'])} and {fmt_month(r['Last'])}. They are left blank, never filled.")
            if r["Notes"]:
                infos.append(f"{r['Input']}: {r['Notes']}")
    n_months = len(sample)
    n_valid = int(sample["valid"].sum())
    if n_months and n_valid < n_months:
        warnings.append(f"Only {n_valid} of {n_months} months in the selected sample have all four inputs, "
                        "so RER and the real index are blank for the rest.")
    bad = int(((sample[["fx1", "fx2", "cpi1", "cpi2"]] <= 0).any(axis=1)).sum())
    if bad:
        warnings.append(f"{bad} month(s) contain non-positive input values and are excluded from RER.")
    for code in (c1, c2):
        if code in EURO_MEMBERS:
            warnings.append(f"{names[code]} adopted the euro. Its exchange rate before adoption may be quoted in the "
                            "legacy currency, which creates a level break. Check the nominal series before interpreting.")
    ratio = full["S"].loc[start:end]
    jumps = (ratio / ratio.shift(1) - 1).abs()
    flagged = jumps[jumps > JUMP_THRESHOLD].index
    if len(flagged):
        shown = ", ".join(fmt_month(m) for m in flagged[:6]) + (" ..." if len(flagged) > 6 else "")
        warnings.append(f"S moves by more than {int(JUMP_THRESHOLD * 100)}% in a single month ({shown}). "
                        "This can be a genuine devaluation or a redenomination or series break.")
    return errors, warnings, infos


# --------------------------------------------------------------------------------------
# Methodology text
# --------------------------------------------------------------------------------------
def render_methodology(c1: str, c2: str, unit: str) -> None:
    with st.expander("Methodology, formulas and a worked UK to US example"):
        st.markdown("**Definitions**")
        st.latex(r"S(t)=\frac{E_2(t)}{E_1(t)}\qquad RER(t)=S(t)\,\frac{P_1(t)}{P_2(t)}")
        st.latex(r"RER^{*}=\frac{1}{N}\sum_{t\in T}RER(t)\qquad \text{Real Index}(t)=100\cdot\frac{RER(t)}{RER^{*}}")
        st.markdown(
            """
- **E_i(t)** is the IMF period-average rate, in units of country *i* currency per US dollar.
- **S(t)** is units of Country 2 currency per one unit of Country 1 currency. One unit of Country 1 currency buys
  1/E1 dollars, and each dollar buys E2 units of Country 2 currency, so S = E2 / E1. The US series is 1 by definition.
- **P1, P2** are monthly price indices (all-items CPI, or HICP for the euro area).
- **T** is the set of months in the selected sample where S, P1 and P2 are all present and positive. **N** is its size.
  RER* is recomputed whenever the countries or the sample dates change, so the real index always averages 100 over T.
- A rising RER means Country 1 has become more expensive relative to Country 2 in common-currency terms (real appreciation
  of Country 1 against Country 2). RER* is a sample average. It is not an estimate of equilibrium or purchasing power parity.
- CPI base years differ across countries. A constant rebasing of either index multiplies RER and RER* by the same factor,
  so the real index does not change. The raw RER level does depend on the base years, which is why only the index is charted.
- The nominal exchange rate S is always shown in actual currency units and is never normalised.
"""
        )
        st.markdown("**Worked example: UK as Country 1, US as Country 2** (illustrative numbers, not IMF data)")
        example = pd.DataFrame(
            {
                "Quantity": ["E_UK (GBP per USD)", "E_US (USD per USD)", "S = E_US / E_UK (USD per GBP)",
                             "P_UK", "P_US", "RER = S x P_UK / P_US", "RER* (sample average, assumed)",
                             "Real Index = 100 x RER / RER*"],
                "Value": ["0.80", "1.00", "1.25", "130", "125", "1.25 x 130 / 125 = 1.30", "1.25",
                          "100 x 1.30 / 1.25 = 104.0"],
            }
        )
        st.dataframe(example, hide_index=True, **stretch(st.dataframe))
        st.caption("A real index of 104 means the real exchange rate is 4% above its average over the selected sample.")
        st.markdown(f"**Current selection:** S is shown in *{unit}*.")


# --------------------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------------------
def main() -> None:
    inject_css()

    # Country list
    list_error = None
    try:
        countries = tidy_country_list(fetch_country_list())
    except IMFError as exc:
        list_error = str(exc)
        countries = dict(sorted(FALLBACK_COUNTRIES.items(), key=lambda kv: kv[1].lower()))
    names = countries
    codes = list(countries)

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Controls")
        if st.button("Refresh IMF data", **stretch(st.button)):
            st.cache_data.clear()
            st.session_state["_errors"] = {}
            st.toast("Cache cleared. Reloading from the IMF.")
        default1 = codes.index("GBR") if "GBR" in codes else 0
        default2 = codes.index("USA") if "USA" in codes else min(1, len(codes) - 1)
        c1 = st.selectbox("Country 1", codes, index=default1, format_func=lambda c: f"{names[c]} ({c})")
        c2 = st.selectbox("Country 2", codes, index=default2, format_func=lambda c: f"{names[c]} ({c})")
        if list_error:
            st.warning(f"IMF country list unavailable ({list_error}). Showing a built-in list of major economies.")

    if c1 == c2:
        inject_hero(c1, c2, names, None)
        st.warning("Choose two different countries. A country against itself has S = 1 and a constant real index.")
        st.stop()

    unit = unit_label(c1, c2, names)
    inputs = load_inputs(c1, c2, names)

    panel = build_panel(series_of(inputs["fx1"]), series_of(inputs["fx2"]),
                        series_of(inputs["cpi1"]), series_of(inputs["cpi2"]))
    cov = coverage_table(inputs)
    inject_hero(c1, c2, names, unit)

    for tag in ("cpi1", "cpi2", "fx1", "fx2"):
        if inputs[tag]["error"]:
            st.error(f"Could not download {inputs[tag]['label']}: {inputs[tag]['error']}. "
                     "Use Refresh IMF data to retry.")
    if panel.empty or panel.dropna(how="all").empty:
        st.error("No usable IMF data were returned for this pair, so nothing can be plotted.")
        render_methodology(c1, c2, unit)
        st.stop()

    full = add_rer(panel)
    valid_months = full.index[full["valid"]]
    nominal_months = full.index[full["S"].notna()]
    any_months = full.index[full[["fx1", "fx2", "cpi1", "cpi2"]].notna().any(axis=1)]
    if len(valid_months):
        shared_first, shared_last, basis = valid_months.min(), valid_months.max(), "real"
    elif len(nominal_months):
        shared_first, shared_last, basis = nominal_months.min(), nominal_months.max(), "nominal"
    else:
        shared_first, shared_last, basis = any_months.min(), any_months.max(), "none"
    st.session_state["_shared_first"], st.session_state["_shared_last"] = shared_first, shared_last

    bound_min, bound_max = full.index.min(), full.index.max()
    years = list(range(bound_min.year, bound_max.year + 1))

    # Reset the sample whenever the pair changes or stored values fall out of range.
    if st.session_state.get("_pair") != (c1, c2) or any(
        k not in st.session_state for k in ("sy", "sm", "ey", "em")
    ):
        set_range(shared_first, shared_last)
        st.session_state["_pair"] = (c1, c2)
    for k in ("sy", "ey"):
        if st.session_state[k] not in years:
            st.session_state[k] = min(max(st.session_state[k], years[0]), years[-1])

    with st.sidebar:
        st.markdown("**Sample period**")
        p1, p2, p3, p4 = st.columns(4)
        p1.button("All", on_click=apply_preset, args=(None,), key="p_all", **stretch(st.button))
        p2.button("30y", on_click=apply_preset, args=(30,), key="p_30", **stretch(st.button))
        p3.button("10y", on_click=apply_preset, args=(10,), key="p_10", **stretch(st.button))
        p4.button("5y", on_click=apply_preset, args=(5,), key="p_5", **stretch(st.button))
        a, b = st.columns([1.2, 1])
        a.selectbox("Start year", years, key="sy")
        b.selectbox("Start month", list(range(1, 13)), key="sm", format_func=lambda m: calendar.month_abbr[m])
        a, b = st.columns([1.2, 1])
        a.selectbox("End year", years, key="ey")
        b.selectbox("End month", list(range(1, 13)), key="em", format_func=lambda m: calendar.month_abbr[m])
        st.caption(
            "Presets use the longest span where all four inputs exist"
            if basis == "real" else "No month has all four inputs, so presets fall back to partial coverage."
        )

    start = max(month_ts(st.session_state["sy"], st.session_state["sm"]), bound_min)
    end = min(month_ts(st.session_state["ey"], st.session_state["em"]), bound_max)
    if start > end:
        st.error("The start month is after the end month. Adjust the sample period in the sidebar.")
        st.stop()

    sample, rer_star = benchmark(full, start, end)
    n_months = len(sample)
    n_valid = int(sample["valid"].sum())
    _errors, warnings, infos = collect_notices(inputs, cov, sample, c1, c2, names, full, start, end)

    if basis != "real":
        st.warning("No month has CPI and exchange-rate data for both countries, so the real index cannot be "
                   "computed. Nominal rates are shown where available.")
    if basis == "real" and n_valid == 0:
        st.warning("The selected sample contains no month with all four inputs, so RER* and the real index are "
                   "blank. Widen the sample or use the All preset.")

    # ---- KPI row ----
    last_valid = sample.index[sample["valid"]].max() if n_valid else None
    last_nom = sample.index[sample["S"].notna()].max() if sample["S"].notna().any() else None
    latest_real = sample.loc[last_valid, "real_index"] if last_valid is not None else np.nan
    latest_nom = sample.loc[last_nom, "S"] if last_nom is not None else np.nan
    delta = (latest_real - 100.0) if np.isfinite(latest_real) else np.nan

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card(
        "Sample-average RER (RER*)", fmt_num(rer_star, 4),
        f"Mean of {n_valid} valid months, {fmt_month(start)} to {fmt_month(end)}"), unsafe_allow_html=True)
    k2.markdown(kpi_card(
        "Latest real index", fmt_num(latest_real, 1),
        (f"{abs(delta):.1f}% {'above' if delta >= 0 else 'below'} sample average, {fmt_month(last_valid)}"
         if np.isfinite(delta) else "No valid month in sample"), navy=True), unsafe_allow_html=True)
    k3.markdown(kpi_card(
        "Latest nominal exchange rate", fmt_num(latest_nom, 4),
        f"{short_unit(c1, c2)}, {fmt_month(last_nom)}"), unsafe_allow_html=True)
    k4.markdown(kpi_card(
        "Valid observations", f"{n_valid:,}",
        f"of {n_months:,} months in sample ({n_valid / n_months:.0%})" if n_months else "No months", navy=True),
        unsafe_allow_html=True)
    st.write("")

    # ---- Notices (download failures were already shown above) ----
    if warnings:
        st.warning("**Data notes**\n\n" + "\n".join(f"- {w}" for w in warnings))
    if infos:
        with st.expander(f"Coverage notes ({len(infos)})"):
            for msg in infos:
                st.markdown(f"- {msg}")

    # ---- Tabs ----
    tab_chart, tab_yoy, tab_long, tab_table, tab_cov = st.tabs(
        ["Chart", "Annual changes", "Longer-window comparison", "Monthly data", "Coverage and checks"]
    )

    with tab_chart:
        view = st.radio("Series shown", ["Real", "Nominal", "Both"], horizontal=True, key="view")
        st.markdown(
            f'<div class="section-note">Real index: 100 equals the average over {fmt_month(start)} to {fmt_month(end)}. '
            f"Nominal S is in {html.escape(unit)} and is not normalised. Gaps are real gaps in the IMF data.</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(main_chart(sample, view, short_unit(c1, c2), c1, c2), config=PLOTLY_CONFIG,
                        **stretch(st.plotly_chart))

    with tab_yoy:
        st.markdown("**Annual changes** (optional view)")
        st.markdown(
            '<div class="section-note">Percentage change against the same month a year earlier. The 12-month lag may '
            "use months before the sample start. Annual changes of the real index do not depend on RER*, and nothing "
            "here alters the headline benchmark.</div>", unsafe_allow_html=True)
        yview = st.radio("Series shown", ["Real", "Nominal", "Both"], horizontal=True, key="yoy_view", index=2)
        real_yoy = (full["rer"] / full["rer"].shift(12) - 1).reindex(sample.index)
        nom_yoy = (full["S"] / full["S"].shift(12) - 1).reindex(sample.index)
        st.plotly_chart(yoy_chart(real_yoy, nom_yoy, yview), config=PLOTLY_CONFIG, **stretch(st.plotly_chart))
        m1, m2 = st.columns(2)
        lr = real_yoy.dropna()
        ln = nom_yoy.dropna()
        m1.metric("Latest annual change, real index", f"{lr.iloc[-1] * 100:+.1f}%" if len(lr) else "n/a",
                  help=fmt_month(lr.index[-1]) if len(lr) else None)
        m2.metric("Latest annual change, nominal S", f"{ln.iloc[-1] * 100:+.1f}%" if len(ln) else "n/a",
                  help=fmt_month(ln.index[-1]) if len(ln) else None)

    with tab_long:
        st.markdown("**Comparison with a longer historical window** (optional view)")
        st.markdown(
            '<div class="section-note">Re-expresses the same RER against the average of a longer window that ends at '
            "the sample end. This is supplementary. The KPIs, the main chart and the CSV always use the selected "
            "sample.</div>", unsafe_allow_html=True)
        choice = st.selectbox(
            "Longer window", ["All available history from 1976", "Sample start minus 10 years",
                              "Sample start minus 20 years"], key="long_choice")
        back = {"All available history from 1976": None, "Sample start minus 10 years": 10,
                "Sample start minus 20 years": 20}[choice]
        long_start = DATA_START_TS if back is None else max(DATA_START_TS, start - pd.DateOffset(years=back))
        longer, long_star = benchmark(full, long_start, end)
        n_long = int(longer["valid"].sum())
        if n_long == 0:
            st.info("No valid months in the longer window, so there is nothing to compare.")
        else:
            long_view = longer.reindex(sample.index)
            if long_start >= start:
                st.info("The sample already begins at the start of the available history, so both benchmarks coincide.")
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("RER*, selected sample", fmt_num(rer_star, 4), help=f"{n_valid} valid months")
            c_b.metric("RER*, longer window", fmt_num(long_star, 4), help=f"{n_long} valid months from {fmt_month(long_start)}")
            c_c.metric("Sample average vs longer average",
                       f"{(rer_star / long_star - 1) * 100:+.1f}%" if np.isfinite(rer_star) else "n/a")
            label = f"Longer window ({fmt_month(long_start)} to {fmt_month(end)}, {n_long} months)"
            st.plotly_chart(long_window_chart(sample, long_view, label), config=PLOTLY_CONFIG,
                            **stretch(st.plotly_chart))
            st.caption("Both lines are the same RER rescaled, so they differ only by a constant factor. "
                       "Reading the dashed line shows where today sits relative to the longer history.")

    table = pd.DataFrame({
        "Month": sample.index,
        f"S ({short_unit(c1, c2)})": sample["S"],
        f"{c1} FX (per USD)": sample["fx1"],
        f"{c2} FX (per USD)": sample["fx2"],
        f"{c1} CPI": sample["cpi1"],
        f"{c2} CPI": sample["cpi2"],
        "RER": sample["rer"],
        "Real index": sample["real_index"],
        "Valid": sample["valid"],
    }).reset_index(drop=True)
    csv_bytes = export_frame(sample, c1, c2, names, unit, rer_star, inputs).to_csv(index=False).encode("utf-8")
    csv_name = f"rer_{c1}_{c2}_{start:%Y-%m}_{end:%Y-%m}.csv"

    with tab_table:
        st.download_button("Download selected sample as CSV", csv_bytes, file_name=csv_name, mime="text/csv",
                           key="dl_table", **stretch(st.download_button))
        st.caption("The CSV holds the source inputs, S, RER, RER*, the real index, validity flags and the reason "
                   "any month is blank, at full precision.")
        cfg = {"Month": st.column_config.DatetimeColumn("Month", format="MMM YYYY"),
               "RER": st.column_config.NumberColumn(format="%.4f"),
               "Real index": st.column_config.NumberColumn(format="%.2f")}
        for col in table.columns[1:6]:
            cfg[col] = st.column_config.NumberColumn(format="%.4f" if col.startswith("S ") else "%.3f")
        st.dataframe(table.iloc[::-1], hide_index=True, column_config=cfg, height=480, **stretch(st.dataframe))

    with tab_cov:
        st.markdown("**Input coverage**")
        cov_show = cov.copy()
        cov_show["First"] = cov_show["First"].map(fmt_month)
        cov_show["Last"] = cov_show["Last"].map(fmt_month)
        st.dataframe(cov_show.drop(columns=["Request"]), hide_index=True, **stretch(st.dataframe))
        with st.expander("Request URLs"):
            for _, r in cov.iterrows():
                st.code(r["Request"] or "n/a", language=None)
        st.markdown("**Selected sample**")
        s1, s2, s3 = st.columns(3)
        s1.metric("Months in sample", f"{n_months:,}")
        s2.metric("Months with nominal S", f"{int(sample['S'].notna().sum()):,}")
        s3.metric("Months with valid RER", f"{n_valid:,}")
        if n_valid:
            check = sample.loc[sample["valid"], "real_index"].mean()
            st.success(f"Check: the real index averages {check:.6f} across the {n_valid} valid months used for RER*.")
        st.caption("Latest observation dates differ by series. The sample end defaults to the last month in which "
                   "all four inputs are available.")

    with st.sidebar:
        st.download_button("Download CSV", csv_bytes, file_name=csv_name, mime="text/csv", key="dl_side",
                           **stretch(st.download_button))
        st.caption("Source: International Monetary Fund, CPI and ER datasets, SDMX 3.0 API.")

    render_methodology(c1, c2, unit)


def inject_hero(c1: str, c2: str, names: dict, unit: str | None) -> None:
    pair = (f"{html.escape(names[c1])} vs {html.escape(names[c2])}"
            + (f" &nbsp;|&nbsp; S in {html.escape(unit)}" if unit else ""))
    st.markdown(
        f"""<div class="hero"><h1>Real Exchange Rate Dashboard</h1>
<p>Bilateral CPI-based real exchange rates from monthly IMF data, indexed to the sample average.</p>
<div class="pair">{pair}</div></div>""",
        unsafe_allow_html=True,
    )


main()