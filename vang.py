import requests
import pandas as pd
import io

url = "https://advisors.vanguard.com/investments/products/holdings/latest/3142"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

# Attempt to load as CSV
try:
    df = pd.read_csv(io.StringIO(response.text))
    print(df.head())
    df.to_csv("vgsh_holdings.csv", index=False)
    print("Holdings saved to vgsh_holdings.csv")
except Exception:
    print("Response is not CSV, raw text:")
    print(response.text[:500])
