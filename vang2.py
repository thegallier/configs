#!/usr/bin/env python3
import argparse
import io
import json
import re
import time
from datetime import date, datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome Safari"
)
PROFILE_URL_TPL = "https://investor.vanguard.com/investment-products/etfs/profile/{ticker}"
BASE_ADVISORS = "https://advisors.vanguard.com"

# Heuristics to detect the header row in weird CSVs
KEYWORDS = ["cusip", "ticker", "security", "name", "weight", "market value", "issuer", "par value"]


# ────────────────────────────────────────────────────────────────────────────────
# Date helpers
# ────────────────────────────────────────────────────────────────────────────────

def last_day_previous_month() -> date:
    first_day_this_month = date.today().replace(day=1)
    return first_day_this_month - timedelta(days=1)

def last_day_of_month(d: date) -> date:
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)

def shift_months(d: date, n: int) -> date:
    y, m = d.year, d.month + n
    while m < 1:
        y -= 1
        m += 12
    while m > 12:
        y += 1
        m -= 12
    day = min(d.day, last_day_of_month(date(y, m, 1)).day)
    return date(y, m, day)


# ────────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ────────────────────────────────────────────────────────────────────────────────

def http_get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    return requests.get(url, headers=headers, timeout=45, **kwargs)

def get_html_and_soup(ticker: str):
    url = PROFILE_URL_TPL.format(ticker=ticker.lower())
    resp = http_get(url)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    return url, html, soup

def set_url_param(url: str, **params) -> str:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    for k, v in params.items():
        q[k] = [v]
    new_query = urlencode({k: v[0] for k, v in q.items()})
    return urlunparse(parsed._replace(query=new_query))


# ────────────────────────────────────────────────────────────────────────────────
# CSV parsing
# ────────────────────────────────────────────────────────────────────────────────

def smart_read_csv_from_text(text: str) -> pd.DataFrame:
    # Try fast path
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        if not df.empty and len(df.columns) > 1:
            return df
    except Exception:
        pass

    # Try to locate a likely header row
    lines = text.splitlines()
    header_line_idx = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(k in lower for k in KEYWORDS) and ("," in line or ";" in line or "\t" in line):
            header_line_idx = i
            break

    if header_line_idx is not None:
        try:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python", skiprows=header_line_idx)
        except Exception:
            pass

    return pd.DataFrame()


def parse_json_to_df(raw_text: str) -> pd.DataFrame:
    try:
        data = json.loads(raw_text)
    except Exception:
        return pd.DataFrame()

    # Heuristics: flatten dict/list of dicts
    if isinstance(data, dict):
        # find first list of dicts inside
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return pd.json_normalize(v)
        # or treat dict itself as single-row
        return pd.json_normalize(data)
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            return pd.json_normalize(data)
        else:
            return pd.DataFrame(data)
    return pd.DataFrame()


def find_csv_link_in_html(html: str) -> str | None:
    # Look for an obvious "Holdings_details" CSV link
    m = re.search(r'https?://[^\s"\'<>]*Holdings_details[^\s"\'<>]*\.csv', html, re.IGNORECASE)
    if m:
        return m.group(0)

    # Fallback: any advisors.vanguard.com CSV
    m = re.search(r'https?://advisors\.vanguard\.com[^\s"\'<>]*\.csv', html, re.IGNORECASE)
    if m:
        return m.group(0)

    # Relative CSV?
    m = re.search(r'/[^\s"\'<>]*\.csv', html, re.IGNORECASE)
    if m:
        return urljoin(BASE_ADVISORS, m.group(0))

    return None


def download_and_parse_any(url: str, debug_prefix: str = "debug") -> tuple[pd.DataFrame, str]:
    """
    Download URL, auto-detect type (CSV/HTML/JSON/etc). If HTML, search inside for CSV link and retry.
    Returns (df, final_url_used).
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/csv,application/json;q=0.9,text/html;q=0.8,*/*;q=0.5",
        "Referer": "https://investor.vanguard.com/",
    }
    resp = http_get(url, headers=headers)
    resp.raise_for_status()

    raw_bytes = resp.content
    try:
        raw_text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        raw_text = ""

    content_type = resp.headers.get("Content-Type", "").lower()
    ts = int(time.time())

    # Always save for debugging
    dbg_file = f"{debug_prefix}_{ts}.txt"
    try:
        with open(dbg_file, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n")
            f.write(f"Content-Type: {content_type}\n\n")
            f.write(raw_text[:200000])  # cap
        print(f"📝 Saved raw response to {dbg_file}")
    except Exception:
        pass

    # JSON?
    if "application/json" in content_type or raw_text.strip().startswith(("{", "[")):
        df = parse_json_to_df(raw_text)
        return df, url

    # CSV?
    if "text/csv" in content_type or raw_text.count(",") > 2:
        df = smart_read_csv_from_text(raw_text)
        if not df.empty:
            return df, url

    # Excel?
    if "application/vnd.ms-excel" in content_type or "application/vnd.openxml" in content_type:
        try:
            return pd.read_excel(io.BytesIO(raw_bytes)), url
        except Exception:
            pass

    # HTML? → try to grab the real CSV link
    if "text/html" in content_type or "<html" in raw_text.lower():
        csv_link = find_csv_link_in_html(raw_text)
        if csv_link and csv_link != url:
            print(f"🔁 Found CSV link inside HTML, retrying: {csv_link}")
            return download_and_parse_any(csv_link, debug_prefix=debug_prefix)

    # Give up
    return pd.DataFrame(), url


# ────────────────────────────────────────────────────────────────────────────────
# Method 4 (direct holdings URL extraction – preferred)
# ────────────────────────────────────────────────────────────────────────────────

def get_holdings_url_method4(ticker: str) -> str | None:
    """
    Improved: scan <a>, <link>, <script> href/src, and inline JS for full or relative
    advisor holdings URLs.
    """
    _, html, soup = get_html_and_soup(ticker)

    # 1) href/src attributes
    for tag in soup.find_all(["a", "link", "script"]):
        href = tag.get("href") or tag.get("src")
        if href and "holdings/latest" in href:
            return urljoin(BASE_ADVISORS, href)

    # 2) Inline scripts
    for script in soup.find_all("script"):
        text = script.string or ""
        m = re.search(
            r'https?://advisors\.vanguard\.com/investments/products/holdings/latest/\d+[^\s"\'<>]*',
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(0)

        m = re.search(
            r'/investments/products/holdings/latest/\d+[^\s"\'<>]*',
            text,
            re.IGNORECASE,
        )
        if m:
            return urljoin(BASE_ADVISORS, m.group(0))

    # 3) Entire HTML
    m = re.search(
        r'https?://advisors\.vanguard\.com/investments/products/holdings/latest/\d+[^\s"\'<>]*',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(0)
    return None


# ────────────────────────────────────────────────────────────────────────────────
# Methods 1–3 (fundId discovery – robust)
# ────────────────────────────────────────────────────────────────────────────────

def _extract_fundid_from_json_text(text: str) -> str | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "fundId" in data:
            return str(data["fundId"])
    except Exception:
        pass
    m = re.search(r'"fundId"\s*:\s*"(\d+)"', text)
    if m:
        return m.group(1)
    return None

def get_fund_id_method1(ticker: str) -> str | None:
    """Parse JSON (or JSON-like) in script tags looking for fundId."""
    _, _, soup = get_html_and_soup(ticker)
    for script in soup.find_all("script"):
        text = script.string
        if not text or "fundId" not in text:
            continue
        fid = _extract_fundid_from_json_text(text)
        if fid:
            return fid
    return None

def get_fund_id_method2(ticker: str) -> str | None:
    """Search HTML & all script tags for /holdings/latest/<id>."""
    _, html, soup = get_html_and_soup(ticker)
    all_text = [html]
    for script in soup.find_all("script"):
        if script.string:
            all_text.append(script.string)

    for block in all_text:
        match = re.search(r'/holdings/latest/(\d+)', block)
        if match:
            return match.group(1)
    return None

def get_fund_id_method3(ticker: str) -> str | None:
    """Search script tags for either holdings/latest/<id> or fundId in JSON."""
    _, _, soup = get_html_and_soup(ticker)
    for script in soup.find_all("script"):
        text = script.string
        if not text:
            continue

        m = re.search(r'holdings/latest/(\d+)', text)
        if m:
            return m.group(1)

        if "fundId" in text:
            fid = _extract_fundid_from_json_text(text)
            if fid:
                return fid
    return None

def build_holdings_url_from_fund_id(fund_id: str) -> str:
    return f"{BASE_ADVISORS}/investments/products/holdings/latest/{fund_id}"


# ────────────────────────────────────────────────────────────────────────────────
# Method 5 (Holdings_details.csv or asOfDate=<prev-month-last-day>&format=csv)
# ────────────────────────────────────────────────────────────────────────────────

def get_holdings_url_method5(ticker: str, fund_id: str | None) -> str | None:
    profile_url, html, soup = get_html_and_soup(ticker)

    # Direct CSV link (e.g., Holdings_details...)
    csv_match = re.search(
        r'https?://[^\s"\'<>]*Holdings_details[^\s"\'<>]*\.csv',
        html,
        re.IGNORECASE,
    )
    if csv_match:
        return csv_match.group(0)

    # Check JSON/script blobs
    for script in soup.find_all("script"):
        text = script.string
        if not text:
            continue
        if "Holdings_details" in text:
            csv_match = re.search(
                r'https?://[^\s"\'<>]*Holdings_details[^\s"\'<>]*\.csv',
                text,
                re.IGNORECASE,
            )
            if csv_match:
                return csv_match.group(0)

    # Fallback to fund_id + asOfDate + format=csv
    if fund_id:
        last_day = last_day_previous_month().strftime("%Y-%m-%d")
        return (
            f"{BASE_ADVISORS}/investments/products/holdings/latest/"
            f"{fund_id}?asOfDate={last_day}&format=csv"
        )
    return None


# ────────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────────────────

def discover_holdings_url(ticker: str):
    """
    Try all methods and return:
      - final_url: chosen URL to download
      - fund_id: if discovered
      - report: dict of method_name -> success/fail/error + value
    """
    fund_id: str | None = None
    final_url: str | None = None
    report = {}

    methods = [
        ("method4", get_holdings_url_method4, True),
        ("method1", get_fund_id_method1, False),
        ("method2", get_fund_id_method2, False),
        ("method3", get_fund_id_method3, False),
    ]

    for name, func, is_direct in methods:
        try:
            val = func(ticker)
            if val:
                if is_direct:
                    final_url = val
                    report[name] = ("success", final_url)
                else:
                    fund_id = val
                    final_url = build_holdings_url_from_fund_id(fund_id)
                    report[name] = ("success", final_url)
            else:
                report[name] = ("fail", None)
        except Exception as e:
            report[name] = ("error", str(e))

    # Method 5 (may override final_url)
    try:
        m5_url = get_holdings_url_method5(ticker, fund_id)
        if m5_url:
            final_url = m5_url
            report["method5"] = ("success", final_url)
        else:
            report["method5"] = ("fail", None)
    except Exception as e:
        report["method5"] = ("error", str(e))

    if not final_url:
        raise RuntimeError(f"Could not determine holdings URL for {ticker}")

    return final_url, fund_id, report


# ────────────────────────────────────────────────────────────────────────────────
# Download logic with multi-month fallback
# ────────────────────────────────────────────────────────────────────────────────

def try_download_with_fallback(initial_url: str, fund_id: str | None, months_back: int, debug_prefix: str):
    """
    Try downloading initial_url. If empty and we have a fund_id, try last N months
    (including last-day-of-previous-month going backwards) with format=csv.
    Returns (df, final_url_used)
    """
    df, used_url = download_and_parse_any(initial_url, debug_prefix=debug_prefix)
    if not df.empty:
        return df, used_url

    if fund_id is None:
        return df, used_url

    # If there is an asOfDate in URL, strip it and rebuild ourselves
    base_url = build_holdings_url_from_fund_id(fund_id)
    parsed = urlparse(initial_url)
    if "holdings/latest" in initial_url:
        base_url = urlunparse(parsed._replace(query=""))

    last_day_prev = last_day_previous_month()
    for i in range(months_back):
        target_month_last = last_day_of_month(shift_months(last_day_prev, -i))
        candidate = set_url_param(base_url, asOfDate=target_month_last.strftime("%Y-%m-%d"), format="csv")
        print(f"🔁 Trying previous month: {candidate}")
        df_try, final_url_try = download_and_parse_any(candidate, debug_prefix=debug_prefix)
        if not df_try.empty:
            return df_try, final_url_try

    return pd.DataFrame(), used_url


# ────────────────────────────────────────────────────────────────────────────────
# Date extraction
# ────────────────────────────────────────────────────────────────────────────────

def extract_data_date(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in col.lower():
            try:
                date_val = pd.to_datetime(df[col].dropna().iloc[0])
                return date_val.strftime("%Y-%m-%d")
            except Exception:
                continue
    return last_day_previous_month().strftime("%Y-%m-%d")


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Vanguard ETF holdings via multiple resilient methods."
    )
    parser.add_argument("--ticker", help="ETF ticker, e.g. VGSH (default VGSH)")
    parser.add_argument("--outdir", default=".", help="Directory to save CSV")
    parser.add_argument(
        "--months-back", type=int, default=3,
        help="If empty data, try up to this many previous months with asOfDate (default 3)"
    )
    parser.add_argument("--debug-prefix", default="debug_vanguard_response",
                        help="Prefix for raw response dump files")
    args = parser.parse_args()

    ticker = (args.ticker or "VGSH").upper()

    print(f"🔍 Discovering holdings URL for {ticker}...")
    final_url, fund_id, report = discover_holdings_url(ticker)

    for name, (status, val) in report.items():
        if status == "success":
            print(f"✅ {name} succeeded: {val}")
        elif status == "fail":
            print(f"❌ {name} did not find a match.")
        else:
            print(f"❌ {name} error: {val}")

    print(f"➡ Using initial URL: {final_url}")

    print("⬇ Downloading holdings CSV / JSON (auto-detect)…")
    df, used_url = try_download_with_fallback(
        final_url, fund_id, args.months_back, debug_prefix=args.debug_prefix
    )

    if df.empty:
        print("⚠️ Warning: Downloaded data is still empty after fallbacks.")
    else:
        print(f"✅ Downloaded {len(df)} rows from {used_url}")

    data_date = extract_data_date(df)
    allowed_date = last_day_previous_month()
    if datetime.strptime(data_date, "%Y-%m-%d").date() > allowed_date:
        print(f"⚠️ Adjusting data date from {data_date} to {allowed_date}")
        data_date = allowed_date.strftime("%Y-%m-%d")

    out_path = f"{args.outdir.rstrip('/')}/{ticker}_{data_date}.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to {out_path}")


if __name__ == "__main__":
    main()
