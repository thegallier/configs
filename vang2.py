#!/usr/bin/env python3
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io
from datetime import datetime

def get_previous_month_last_day():
    today = datetime.today()
    first_day_this_month = today.replace(day=1)
    last_day_prev_month = first_day_this_month - pd.Timedelta(days=1)
    return last_day_prev_month.strftime("%Y-%m-%d")

def get_fund_id_method1(ticker):
    """Method 1: Find 'fundId' in HTML."""
    url = f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    match = re.search(r'"fundId"\s*:\s*"(\d+)"', html)
    return match.group(1) if match else None

def get_fund_id_method2(ticker):
    """Method 2: Look for holdings/latest/<id> in HTML."""
    url = f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    matches = re.findall(r'/holdings/latest/(\d+)', html)
    return matches[0] if matches else None

def get_fund_id_method3(ticker):
    """Method 3: Parse script tags."""
    url = f"https://investor.vanguard.com/investment-products/etfs/profile/{ticker.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        if script.string and "holdings/latest/" in script.string:
            match = re.search(r'holdings/latest/(\d+)', script.string)
            if match:
                return match.group(1)
    return None

def get_fund_id(ticker):
    for method in (get_fund_id_method1, get_fund_id_method2, get_fund_id_method3):
        try:
            fund_id = method(ticker)
            if fund_id:
                print(f"Found fund ID: {fund_id} using {method.__name__}")
                return fund_id
        except Exception as e:
            print(f"{method.__name__} failed: {e}")
    raise RuntimeError(f"Could not determine fund ID for {ticker}")

def download_holdings(ticker):
    fund_id = get_fund_id(ticker)
    url = f"https://advisors.vanguard.com/investments/products/holdings/latest/{fund_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    last_day = get_previous_month_last_day()
    filename = f"{ticker.upper()}_{last_day}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Holdings downloaded to {filename}")

if __name__ == "__main__":
    download_holdings("VGSH")
