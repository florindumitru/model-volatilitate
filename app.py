import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st

# Funcție pentru analiză
def analyze_volatility_after_similar_periods(ticker, strike, dte, option_type="call", recent_window=30, vol_tolerance=0.2):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        if len(hist) < dte + recent_window:
            raise ValueError("Date insuficiente pentru analiza istorică.")

        hist['Log Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
        recent_vol = hist['Log Returns'][-recent_window:].std() * np.sqrt(252)
        if np.isnan(recent_vol):
            raise ValueError("Volatilitatea recentă nu poate fi calculată.")

        vol_min = recent_vol * (1 - vol_tolerance)
        vol_max = recent_vol * (1 + vol_tolerance)

        vol_data = []
        for i in range(len(hist) - recent_window - dte):
            window = hist['Log Returns'].iloc[i:i+recent_window]
            if len(window.dropna()) == recent_window:
                window_vol = window.std() * np.sqrt(252)
                if not np.isnan(window_vol):
                    vol_data.append({"index": i, "vol": window_vol})

        selected_periods = [p for p in vol_data if vol_min <= p["vol"] <= vol_max]
        if not selected_periods:
            raise ValueError("Nu s-au găsit perioade potrivite în intervalul de volatilitate.")

        dte_vols = []
        price_changes = []
        current_price = hist['Close'][-1]
        strike_ratio = strike / current_price
        itm_count = 0

        for period in selected_periods:
            i = period["index"]
            dte_window = hist['Log Returns'].iloc[i+recent_window:i+recent_window+dte]
            if len(dte_window.dropna()) == dte:
                dte_vol = dte_window.std() * np.sqrt(252)
                dte_vols.append(dte_vol)

            start_price = hist['Close'].iloc[i+recent_window-1]
            end_price = hist['Close'].iloc[i+recent_window+dte-1]
            price_change = end_price / start_price
            price_changes.append(price_change)

            historical_strike = start_price * strike_ratio
            if option_type == "call" and end_price > historical_strike:
                itm_count += 1
            elif option_type == "put" and end_price < historical_strike:
                itm_count += 1

        avg_dte_vol = np.mean(dte_vols) if dte_vols else recent_vol
        prob_itm_historical = itm_count / len(selected_periods) if selected_periods else 0

        r = 0.044
        t = dte / 365
        sigma = avg_dte_vol
        d1 = (np.log(1 / strike_ratio) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        prob_itm_theoretical = norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)

        return {
            "Current Price": current_price,
            "Recent Volatility": recent_vol,
            "Average DTE Volatility": avg_dte_vol,
            "Number of Matches": len(selected_periods),
            "Volatility Range": (vol_min, vol_max),
            "Probability ITM (Historical)": prob_itm_historical,
            "Probability ITM (Theoretical)": prob_itm_theoretical,
            "Price Changes": price_changes,
            "Strike Ratio": strike_ratio
        }

    except Exception as e:
        return {"Error": str(e)}


# Interfață Streamlit
st.set_page_config(page_title="Volatility ITM Analyzer", layout="centered")
st.title("🔍 Analiză Probabilitate ITM")

with st.form("input_form"):
    ticker = st.text_input("Ticker (ex: AAPL)", value="AAPL").upper()
    strike = st.number_input("Strike Price", value=150.0)
    dte = st.number_input("Zile până la Expirare (DTE)", value=30)
    option_type = st.selectbox("Tip Opțiune", ["call", "put"])
    vol_tolerance = st.slider("Toleranță Volatilitate (%)", 5, 50, 10) / 100
    submitted = st.form_submit_button("🔍 Analizează")

if submitted:
    with st.spinner("Se analizează datele..."):
        result = analyze_volatility_after_similar_periods(ticker, strike, dte, option_type, 30, vol_tolerance)

    if "Error" in result:
        st.error(result["Error"])
    else:
        st.subheader(f"📈 Rezultate pentru {ticker}")
        st.write(f"**Preț curent:** ${result['Current Price']:.2f}")
        st.write(f"**Volatilitate recentă:** {result['Recent Volatility']*100:.2f}%")
        st.write(f"**Interval de toleranță:** {result['Volatility Range'][0]*100:.2f}% - {result['Volatility Range'][1]*100:.2f}%")
        st.write(f"**Volatilitate medie în {dte} zile:** {result['Average DTE Volatility']*100:.2f}%")
        st.write(f"**Număr de potriviri istorice:** {result['Number of Matches']}")
        st.write(f"**Probabilitate ITM (Istoric):** {result['Probability ITM (Historical)']*100:.2f}%")
        st.write(f"**Probabilitate ITM (Teoretic):** {result['Probability ITM (Theoretical)']*100:.2f}%")

        # Histogramă
        fig, ax = plt.subplots()
        ax.hist(result["Price Changes"], bins=30, density=True, alpha=0.7, label="Distribuție istorică")
        ax.axvline(x=result["Strike Ratio"], color='red', linestyle='--', label=f'Strike/Current ({result["Strike Ratio"]:.2f})')
        ax.set_title(f"Distribuția schimbărilor de preț ({dte} zile)")
        ax.set_xlabel("Schimbare de preț")
        ax.set_ylabel("Densitate")
        ax.legend()
        st.pyplot(fig)
