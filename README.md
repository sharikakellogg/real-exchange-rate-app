# Real Exchange Rate Dashboard

A Streamlit dashboard that builds the bilateral CPI-based real exchange rate (RER) for any two IMF economies, or the
euro area, from monthly IMF data. It plots a real index normalised to the selected sample, and the nominal exchange rate
in actual currency units.

## Economics

| Symbol | Definition |
| --- | --- |
| E_i(t) | Units of country i currency per USD (IMF period-average rate) |
| S(t) | Units of Country 2 currency per one unit of Country 1 currency = E2(t) / E1(t) |
| P1(t), P2(t) | Monthly price index of Country 1 and Country 2 |
| RER(t) | S(t) x P1(t) / P2(t) |
| RER* | Arithmetic mean of valid RER(t) within the selected sample |
| Real Index(t) | 100 x RER(t) / RER* |

**Why S = E2 / E1.** One unit of Country 1 currency buys 1/E1 dollars. Each dollar buys E2 units of Country 2 currency.
For the UK (Country 1) and the US (Country 2), E_UK is GBP per USD and E_US is 1, so S = 1 / E_UK, which is USD per GBP.

**Valid observation.** A month is valid when E1, E2, P1 and P2 are all present and strictly positive. RER* is the mean of
RER over valid months inside the selected sample only. It is recomputed whenever the countries or the sample dates
change, so the real index averages exactly 100 over the months used. The app shows this check on the Coverage tab.

**Interpretation.** A rising RER is a real appreciation of Country 1 against Country 2. RER* is a sample average, not an
equilibrium or purchasing power parity estimate, and the app never labels a currency over- or undervalued. CPI base years
differ by country. A constant rebasing scales RER and RER* equally, so the real index is unaffected.

**Optional features** (neither changes RER* or the headline numbers):

- *Annual changes:* percentage change of the real index and of S against the same month a year earlier. The 12-month lag
  may use months before the sample start.
- *Longer-window comparison:* re-expresses the same RER against the average of a longer window (all history from 1976,
  or the sample start minus 10 or 20 years). Shown only on its own tab.

## Data source and API assumptions

Base URL: `https://api.imf.org/external/sdmx/3.0`

| Item | Request |
| --- | --- |
| CPI | `/data/dataflow/IMF.STA/CPI/+/{ISO3}.CPI._T.IX.M` |
| Exchange rate | `/data/dataflow/IMF.STA/ER/+/{ISO3}.XDC_USD.PA_RT.M` |
| Euro area prices | `/data/dataflow/IMF.STA/CPI/+/G163.HICP._T.IX.M` |
| Country list | `/structure/codelist/IMF/CL_COUNTRY/+` |

Assumptions the code relies on (taken from IMF-facing documentation and open-source clients, not from an IMF
specification I could read end to end):

- Data requests send `Accept: text/csv` and the filter `c[TIME_PERIOD]=ge:1976-01`. If that fails, the app retries
  without the filter and trims to 1976 locally. It tries version token `+` first and `~` second.
- The CSV has dimension columns, `TIME_PERIOD` (for example `2024-M01`) and `OBS_VALUE`. Header labels of the form
  `COUNTRY: Country` are also accepted. A `SCALE` column is applied if present.
- Euro area code is `G163`. The IMF returns no CPI-type series for it, so the app uses the IMF HICP series and labels it.
  Euro area exchange-rate data begin in January 1999 and HICP in December 1999, so pairs involving the euro area start
  in December 1999. Nothing earlier is constructed.
- No API key is used. If the IMF starts requiring one, requests will fail with an HTTP error shown in the app.

The IMF terms of use apply to the data.

## Missing and invalid data

- Data are never interpolated, carried forward or invented. Missing months stay blank and appear as gaps in charts.
- Series that start after January 1976, have internal gaps, or end earlier than others are reported on the Coverage tab
  and in the data notes.
- Blank, non-numeric and non-monthly rows are dropped and counted. Duplicate months with different values, a wrong
  country code, or more than one series in a response are rejected with an explicit message.
- Zero and negative inputs are excluded from RER.
- Timeouts (6 s connect, 40 s read), connection errors, HTTP errors and HTML gateway pages show a specific error. A failed
  download is distinct from "no observations".
- Euro members before adoption can report legacy-currency rates. The app warns about this, and also flags any month where
  S moves by more than 35% as a possible devaluation or series break. It does not alter the data.
- The IMF country list is filtered to three-letter codes plus G163. Some listed economies have no data, in which case the
  app says so. If the codelist cannot be downloaded, a built-in list of major economies is shown with a warning.
- If the IMF has no US exchange-rate series, USD per USD is set to 1 and labelled. A failed download never triggers this.

Caching: series for 6 hours, country list for 24 hours. **Refresh IMF data** clears the cache.

## Run locally

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Put `app.py`, `requirements.txt` and `README.md` at the root of a public GitHub repository.
2. Sign in at https://share.streamlit.io with GitHub and choose **Create app**.
3. Select the repository, branch and `app.py` as the main file.
4. Under **Advanced settings**, choose Python 3.12. No secrets are needed.
5. Deploy, then open the app and check that the IMF country list loads and the UK to US pair returns data.

Optional: to fix the light theme regardless of a visitor's system setting, add `.streamlit/config.toml` with
`[theme]`, `base = "light"`, `primaryColor = "#0F9D94"`, `backgroundColor = "#F6F4EF"`,
`secondaryBackgroundColor = "#ECE9E1"` and `textColor = "#1B2A41"`. The app works without it.

## Manual verification

1. Select UK as Country 1 and US as Country 2 and confirm S is USD per GBP.
2. Download the CSV. For one row, check S = fx2 / fx1 and RER = S x cpi1 / cpi2.
3. Average the valid RER rows to get RER*, then check 100 x RER / RER* matches the real index.
4. Confirm the real index averages 100 across valid rows, and that the nominal S is not normalised.
5. Change the sample dates and repeat. RER* and the real index should change, S should not.

## Limitations

- CPI-based real rates depend on index quality and series breaks. Long samples for high-inflation countries can contain
  large discontinuities.
- Currency labels exist for about 45 economies. Others show a generic label.
- Latest observations differ by series, so the default sample ends at the last month where all four inputs exist.