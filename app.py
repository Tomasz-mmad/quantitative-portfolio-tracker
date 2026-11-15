import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import uuid
import json
from pathlib import Path


# Plik z trwałym zapisem portfela (obok skryptu)
PORTFOLIO_FILE = Path(__file__).resolve().parent / "portfolio_data.json"


def load_portfolio_data():
    """Wczytaj zapisany portfel z dysku."""
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_portfolio_data():
    """Zapisz aktualny stan portfela na dysk."""
    data = {
        "transactions_general": st.session_state.get("transactions_general", []),
        "transactions_ike": st.session_state.get("transactions_ike", []),
        "gold_holdings": st.session_state.get("gold_holdings", []),
        "silver_holdings": st.session_state.get("silver_holdings", []),
        "dividends": st.session_state.get("dividends", []),
        "goals": st.session_state.get("goals", {}),
    }
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Nie udało się zapisać portfela: {e}")


@st.cache_data(ttl=1800)
def get_fx_rate(pair="USDPLN=X"):
    """Aktualny kurs FX z yfinance (np. USDPLN=X, EURPLN=X)."""
    try:
        s = get_data(pair, "5d")
        if s.empty:
            s = get_data(pair, "1mo")
        if s.empty:
            return None
        return float(s.iloc[-1])
    except Exception:
        return None


def asset_currency(asset_name):
    """Przybliżona waluta notowania instrumentu."""
    if asset_name in ("MSCI World ETF", "Airbus"):
        return "EUR"
    if asset_name in ("Złoto fizyczne", "Srebro fizyczne"):
        return "USD"
    t = TICKERS.get(asset_name, "")
    if t.endswith(".DE") or t.endswith(".PA") or t.endswith(".AS"):
        return "EUR"
    return "USD"


def to_pln(amount, currency, usdpln, eurpln):
    if amount is None:
        return None
    if currency == "PLN":
        return amount
    if currency == "EUR" and eurpln:
        return amount * eurpln
    if currency == "USD" and usdpln:
        return amount * usdpln
    return amount  # fallback bez konwersji


# ============================================================
# KONFIGURACJA
# ============================================================
st.set_page_config(page_title="Panel Inwestycyjny", layout="wide", page_icon="📈")

TICKERS = {
    'Nvidia': 'NVDA',
    'Apple': 'AAPL',
    'Amazon': 'AMZN',
    'MSCI World ETF': 'EUNL.DE',
    'Coca Cola': 'KO',
    'Alphabet (Google)': 'GOOGL',
    'Palantir': 'PLTR',
    'Airbus': 'AIR.PA',
    'Visa': 'V',
    'PayPal': 'PYPL',
    'Johnson & Johnson': 'JNJ',
    'Microsoft': 'MSFT',
    'AMD': 'AMD',
    'S&P 500': 'SPY'
}

METAL_TICKERS = {
    'Złoto fizyczne': 'GC=F',
    'Srebro fizyczne': 'SI=F'
}

PERIODS = {
    '1 Miesiąc': '1mo',
    '3 Miesiące': '3mo',
    '6 Miesięcy': '6mo',
    '1 Rok': '1y',
    '2 Lata': '2y',
    '3 Lata': '3y',
    '5 Lat': '5y'
}

GOLD_BARS = {
    '1 g': 1.0, '5 g': 5.0, '10 g': 10.0, '20 g': 20.0,
    '1 oz (31,1 g)': 31.1035, '50 g': 50.0, '100 g': 100.0,
    '250 g': 250.0, '500 g': 500.0, '1 kg': 1000.0
}

SILVER_BARS = {
    '1 oz (31,1 g)': 31.1035, '10 oz (~311 g)': 311.035,
    '100 g': 100.0, '250 g': 250.0, '500 g': 500.0, '1 kg': 1000.0
}

TROY_OUNCE_GRAMS = 31.1034768
RISK_FREE_RATE = 0.045

# ============================================================
# FUNKCJE DANYCH
# ============================================================
@st.cache_data(ttl=1800)
def get_data(ticker_symbol, period):
    try:
        data = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        return data['Close'].dropna() if 'Close' in data.columns else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=1800)
def get_multiple_data(tickers_dict, period):
    symbols = list(tickers_dict.values())
    try:
        data = yf.download(symbols, period=period, progress=False, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        inv_map = {v: k for k, v in tickers_dict.items()}
        data = data.rename(columns=inv_map)
        return data.dropna(how='all')
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_metal_price_per_gram(metal='gold'):
    ticker = 'GC=F' if metal == 'gold' else 'SI=F'
    series = get_data(ticker, '5d')
    if series.empty:
        return None
    return float(series.iloc[-1]) / TROY_OUNCE_GRAMS

@st.cache_data(ttl=1800)
def get_metal_history_per_gram(metal='gold', period='3y'):
    ticker = 'GC=F' if metal == 'gold' else 'SI=F'
    series = get_data(ticker, period)
    if series.empty:
        return pd.Series(dtype=float)
    return series / TROY_OUNCE_GRAMS

# ============================================================
# FUNKCJE TRANSAKCJI I PORTFELA
# ============================================================
def calculate_holdings_from_transactions(transactions):
    """Zwraca dict: asset -> {'qty': float, 'avg_cost': float, 'total_cost': float}
    Prowizja przy kupnie zwiększa koszt nabycia; przy sprzedaży też doliczana do kosztu (zmniejsza zysk).
    """
    holdings = {}
    for t in transactions:
        asset = t['asset']
        fee = float(t.get('fee', 0) or 0)
        if asset not in holdings:
            holdings[asset] = {'qty': 0.0, 'total_cost': 0.0}
        
        if t['type'] == 'buy':
            holdings[asset]['qty'] += t['qty']
            holdings[asset]['total_cost'] += t['qty'] * t['price'] + fee
        else:  # sell
            if holdings[asset]['qty'] > 0:
                avg = holdings[asset]['total_cost'] / holdings[asset]['qty']
                holdings[asset]['qty'] -= t['qty']
                holdings[asset]['total_cost'] -= t['qty'] * avg
                holdings[asset]['total_cost'] += fee  # prowizja od sprzedaży
                if holdings[asset]['qty'] < 0:
                    holdings[asset]['qty'] = 0
                    holdings[asset]['total_cost'] = 0
    
    result = {}
    for asset, h in holdings.items():
        if h['qty'] > 0.0001:
            result[asset] = {
                'qty': h['qty'],
                'avg_cost': h['total_cost'] / h['qty'] if h['qty'] > 0 else 0,
                'total_cost': h['total_cost']
            }
    return result

def max_drawdown(series):
    """Oblicza Maximum Drawdown"""
    if series.empty or len(series) < 2:
        return 0.0
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return float(dd.min())

# ============================================================
# MENU
# ============================================================
st.sidebar.title("Nawigacja")
menu = st.sidebar.radio("Wybierz moduł:", [
    "Analiza i Wykresy",
    "Macierz Korelacji",
    "Symulacja Monte Carlo",
    "Mój Portfel",
    "Projekcja IKE",
    "Raport podatkowy",
    "Cele i FIRE",
    "Porównanie strategii",
    "Ryzyko portfela",
    "Eksport danych",
])

# ============================================================
# MODUŁ 1: ANALIZA I WYKRESY
# ============================================================
if menu == "Analiza i Wykresy":
    st.title("📈 Przegląd rynku i wykresy historyczne")
    
    all_assets = list(TICKERS.keys()) + list(METAL_TICKERS.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_asset = st.selectbox("Wybierz aktywo:", all_assets)
    with col2:
        selected_period = st.selectbox("Wybierz okres:", list(PERIODS.keys()), index=4)
        
    period_code = PERIODS[selected_period]
    
    with st.spinner('Pobieranie danych...'):
        if selected_asset in METAL_TICKERS:
            data = get_metal_history_per_gram('gold' if 'Złoto' in selected_asset else 'silver', period_code)
            unit = "USD / gram"
        else:
            data = get_data(TICKERS[selected_asset], period_code)
            unit = "USD"
        
    if not data.empty:
        current_price = float(data.iloc[-1])
        start_price = float(data.iloc[0])
        pct_change = ((current_price - start_price) / start_price) * 100
        
        st.metric(
            label=f"Ostatnia cena: {selected_asset}", 
            value=f"{current_price:.4f} {unit}" if selected_asset in METAL_TICKERS else f"{current_price:.2f} {unit}",
            delta=f"{pct_change:.2f}% (w wybranym okresie)"
        )
        
        fig = px.line(data, title=f"Wykres ceny {selected_asset} (Okres: {selected_period})",
                      labels={'value': unit, 'index': 'Data'})
        fig.update_layout(showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        if selected_asset in METAL_TICKERS:
            st.caption("Cena na podstawie futures (GC=F / SI=F) ≈ spot. 1 troy ounce = 31,1035 g.")
    else:
        st.error("Nie udało się pobrać danych.")

# ============================================================
# MODUŁ 2: MACIERZ KORELACJI
# ============================================================
elif menu == "Macierz Korelacji":
    st.title("🔗 Macierz Korelacji")
    
    selected_period = st.selectbox("Wybierz okres badawczy:", list(PERIODS.keys()), index=4)
    
    with st.spinner('Pobieranie i przeliczanie danych...'):
        df_stocks = get_multiple_data(TICKERS, PERIODS[selected_period])
        gold = get_metal_history_per_gram('gold', PERIODS[selected_period])
        silver = get_metal_history_per_gram('silver', PERIODS[selected_period])
        
        df_all = df_stocks.copy()
        if not gold.empty:
            df_all['Złoto fizyczne'] = gold
        if not silver.empty:
            df_all['Srebro fizyczne'] = silver
            
        df_all = df_all.dropna(axis=1, thresh=int(len(df_all) * 0.6))
        returns = df_all.pct_change().dropna()
        
        if returns.empty or returns.shape[1] < 2:
            st.error("Za mało danych dla wybranego okresu.")
        else:
            corr_matrix = returns.corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            stacked = corr_matrix.where(mask).stack().dropna()
            
            if len(stacked) > 0:
                col1, col2, col3 = st.columns(3)
                col1.metric("Średnia korelacja", f"{stacked.mean():.3f}")
                strongest = stacked.idxmax()
                col2.metric("Najsilniejsza para", f"{strongest[0]} ↔ {strongest[1]}", f"{stacked.max():.3f}")
                weakest = stacked.idxmin()
                col3.metric("Najsłabsza para", f"{weakest[0]} ↔ {weakest[1]}", f"{stacked.min():.3f}")
            
            st.markdown("---")
            sort_by_corr = st.checkbox("Posortuj według siły korelacji", value=False)
            if sort_by_corr:
                order = corr_matrix.mean().sort_values(ascending=False).index
                corr_matrix = corr_matrix.loc[order, order]
            
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title=f"Macierz Korelacji ({selected_period})")
            fig.update_layout(height=780)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODUŁ 3: MONTE CARLO
# ============================================================
elif menu == "Symulacja Monte Carlo":
    st.title("🎲 Symulacja Monte Carlo")
    
    all_assets_mc = list(TICKERS.keys()) + list(METAL_TICKERS.keys())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_asset = st.selectbox("Wybierz pozycję:", all_assets_mc)
    with col2:
        years_forward = st.slider("Okres (lata):", 1, 5, 2)
    with col3:
        initial_investment = st.number_input("Kwota (USD):", min_value=10.0, value=10000.0, step=500.0)
        
    col_a, col_b = st.columns(2)
    with col_a:
        hist_years = st.selectbox("Dane historyczne (lata):", [1, 2, 3, 4, 5], index=2)
    with col_b:
        use_custom = st.checkbox("Własna prognoza wzrostu")
        custom_return = st.number_input("Roczna stopa (%):", value=8.0, step=0.5) if use_custom else None

    num_simulations = st.number_input("Liczba ścieżek (max 1 000 000):", 1000, 1000000, 500000, 50000)
    
    if st.button("🚀 Uruchom Symulację", type="primary"):
        progress = st.progress(0)
        status = st.empty()
        
        status.text("Pobieranie danych...")
        progress.progress(15)
        
        if selected_asset in METAL_TICKERS:
            hist_data = get_metal_history_per_gram('gold' if 'Złoto' in selected_asset else 'silver', f"{hist_years}y")
        else:
            hist_data = get_data(TICKERS[selected_asset], f"{hist_years}y")
            
        if not hist_data.empty and len(hist_data) > 30:
            prices = hist_data.dropna()
            status.text("Generowanie scenariuszy...")
            progress.progress(40)
            
            log_ret = np.log(prices / prices.shift(1)).dropna()
            u = (custom_return / 100 / 252) if use_custom and custom_return else float(log_ret.mean())
            var = float(log_ret.var())
            drift = u - 0.5 * var
            stdev = float(log_ret.std())
            
            days = int(years_forward * 252)
            daily_returns = np.exp(np.random.normal(drift, stdev, size=(days, num_simulations)).astype(np.float32))
            
            S0 = float(prices.iloc[-1])
            paths = np.zeros((days, num_simulations), dtype=np.float32)
            paths[0] = S0
            for t in range(1, days):
                paths[t] = paths[t-1] * daily_returns[t]
            
            port = paths * (initial_investment / S0)
            end = port[-1]
            
            min_v, max_v = float(end.min()), float(end.max())
            med_v, mean_v = float(np.median(end)), float(end.mean())
            var95 = float(np.percentile(end, 5))
            cvar95 = float(end[end <= var95].mean())
            
            def pct(v): return ((v / initial_investment) - 1) * 100
            
            progress.progress(100)
            status.empty()
            progress.empty()
            
            st.subheader(f"Wyniki dla {initial_investment:,.0f} USD")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Najmniej", f"{min_v:,.0f} USD", f"{pct(min_v):+.1f}%")
            c2.metric(f"Mediana ({years_forward}r)", f"{med_v:,.0f} USD", f"{pct(med_v):+.1f}%")
            c3.metric("Najwięcej", f"{max_v:,.0f} USD", f"{pct(max_v):+.1f}%")
            
            c4, c5, c6 = st.columns(3)
            c4.metric("Średnia", f"{mean_v:,.0f} USD", f"{pct(mean_v):+.1f}%")
            c5.metric("VaR 95%", f"{var95:,.0f} USD", f"{pct(var95):+.1f}%")
            c6.metric("CVaR 95%", f"{cvar95:,.0f} USD", f"{pct(cvar95):+.1f}%")
            
            fig = go.Figure()
            time_idx = pd.date_range(start=datetime.now().date(), periods=days, freq='B')
            for i in range(min(num_simulations, 100)):
                fig.add_trace(go.Scatter(x=time_idx, y=port[:, i], mode='lines',
                                         line=dict(width=0.4), opacity=0.1, showlegend=False))
            
            med_path = np.median(port, axis=1)
            fig.add_trace(go.Scatter(x=time_idx, y=med_path, mode='lines',
                                     name='Mediana', line=dict(color='lime', width=3)))
            fig.add_hline(y=initial_investment, line_dash="dash", line_color="white")
            fig.update_layout(title=f"Prognoza – {selected_asset}", height=520,
                              xaxis_title="Data", yaxis_title="Wartość (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Za mało danych historycznych.")

# ============================================================
# ============================================================
# MODUŁ 4: MÓJ PORTFEL (rozdzielony: ogólny + IKE)
# ============================================================
# ============================================================
# MODUŁ 4: MÓJ PORTFEL (rozdzielony: ogólny + IKE)
# ============================================================
elif menu == "Mój Portfel":
    st.title("💼 Mój Portfel Inwestycyjny")

    # --- Trwały zapis: wczytaj z dysku przy pierwszym starcie sesji ---
    if 'portfolio_loaded' not in st.session_state:
        saved = load_portfolio_data()
        st.session_state.transactions_general = saved.get('transactions_general', [])
        st.session_state.transactions_ike = saved.get('transactions_ike', [])
        st.session_state.gold_holdings = saved.get('gold_holdings', [])
        st.session_state.silver_holdings = saved.get('silver_holdings', [])
        st.session_state.dividends = saved.get('dividends', [])
        st.session_state.goals = saved.get('goals', {})
        st.session_state.portfolio_loaded = True
    if 'dividends' not in st.session_state:
        st.session_state.dividends = []
    if 'goals' not in st.session_state:
        st.session_state.goals = {}

    if 'transactions_general' not in st.session_state:
        st.session_state.transactions_general = []
    if 'transactions_ike' not in st.session_state:
        st.session_state.transactions_ike = []
    if 'gold_holdings' not in st.session_state:
        st.session_state.gold_holdings = []
    if 'silver_holdings' not in st.session_state:
        st.session_state.silver_holdings = []

    if 'transactions' in st.session_state and st.session_state.transactions:
        if not st.session_state.transactions_general:
            st.session_state.transactions_general = st.session_state.transactions.copy()
        del st.session_state.transactions
        save_portfolio_data()

    portfolio_type = st.radio(
        "Wybierz konto:",
        ["Portfel ogólny (maklerski + metale)", "Konto IKE"],
        horizontal=True,
        key="portfolio_type"
    )
    is_ike = portfolio_type == "Konto IKE"
    active_key = 'transactions_ike' if is_ike else 'transactions_general'
    transactions = st.session_state[active_key]

    st.caption(
        "IKE: tylko instrumenty notowane (akcje/ETF). Złoto i srebro fizyczne doliczane wyłącznie do portfela ogólnego."
        if is_ike else
        "Portfel ogólny: akcje, ETF + metale fizyczne. Transakcje IKE są osobno."
    )

    with st.sidebar:
        st.markdown("---")
        st.write("**Czyszczenie danych**")
        if st.button("🗑️ Wyczyść portfel ogólny"):
            st.session_state.transactions_general = []
            st.session_state.gold_holdings = []
            st.session_state.silver_holdings = []
            save_portfolio_data()
            st.rerun()
        if st.button("🗑️ Wyczyść konto IKE"):
            st.session_state.transactions_ike = []
            save_portfolio_data()
            st.rerun()
        if st.button("🗑️ Wyczyść wszystko"):
            st.session_state.transactions_general = []
            st.session_state.transactions_ike = []
            st.session_state.gold_holdings = []
            st.session_state.silver_holdings = []
            save_portfolio_data()
            st.rerun()

    if is_ike:
        tab1, tab2, tab3, tab5 = st.tabs([
            "📋 Transakcje & Skład",
            "📊 Performance & Drawdown",
            "⚖️ Sharpe & Rebalancing",
            "🎲 Monte Carlo portfela"
        ])
        tab4 = None
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Transakcje & Skład",
            "📊 Performance & Drawdown",
            "⚖️ Sharpe & Rebalancing",
            "🥇 Metale fizyczne",
            "🎲 Monte Carlo portfela"
        ])

    # ---------- TAB 1: Transakcje + pełny skład ----------
    with tab1:
        st.subheader(f"Dodaj transakcję ({'IKE' if is_ike else 'portfel ogólny'})")

        show_pln = st.toggle("Pokaż wartości w PLN", value=True, key=f"show_pln_{active_key}")
        usdpln = get_fx_rate("USDPLN=X")
        eurpln = get_fx_rate("EURPLN=X")
        if show_pln:
            fx_bits = []
            if usdpln: fx_bits.append(f"USD/PLN {usdpln:.4f}")
            if eurpln: fx_bits.append(f"EUR/PLN {eurpln:.4f}")
            st.caption("Kursy: " + " | ".join(fx_bits) if fx_bits else "Nie udało się pobrać kursów FX")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            t_asset = st.selectbox("Aktywo", list(TICKERS.keys()), key=f"t_asset_{active_key}")
        with col2:
            t_type = st.selectbox("Typ", ["buy", "sell"], key=f"t_type_{active_key}")
        with col3:
            t_qty = st.number_input("Ilość", min_value=0.01, value=1.0, step=0.1, key=f"t_qty_{active_key}")
        with col4:
            t_price = st.number_input(f"Cena ({asset_currency(t_asset)})", min_value=0.01, value=100.0, step=0.1, key=f"t_price_{active_key}")

        colf1, colf2 = st.columns(2)
        with colf1:
            t_fee = st.number_input("Prowizja / opłata (w walucie transakcji)", min_value=0.0, value=0.0, step=0.1, key=f"t_fee_{active_key}")
        with colf2:
            t_date = st.date_input("Data transakcji", value=datetime.now().date(), key=f"t_date_{active_key}")

        if st.button("➕ Dodaj transakcję", key=f"add_tx_{active_key}"):
            st.session_state[active_key].append({
                'id': str(uuid.uuid4())[:8],
                'date': t_date.strftime("%Y-%m-%d"),
                'asset': t_asset,
                'type': t_type,
                'qty': t_qty,
                'price': t_price,
                'fee': float(t_fee),
                'currency': asset_currency(t_asset),
            })
            save_portfolio_data()
            st.success("Transakcja dodana i zapisana")
            st.rerun()

        # --- Import CSV ---
        with st.expander("📥 Import transakcji z CSV"):
            st.caption("Kolumny: date, asset, type (buy/sell), qty, price, fee (opcjonalnie). Nazwa asset jak na liście (np. Nvidia, MSCI World ETF).")
            up = st.file_uploader("Plik CSV", type=["csv"], key=f"csv_up_{active_key}")
            if up is not None:
                try:
                    df_imp = pd.read_csv(up)
                    df_imp.columns = [c.strip().lower() for c in df_imp.columns]
                    required = {"date", "asset", "type", "qty", "price"}
                    if not required.issubset(set(df_imp.columns)):
                        st.error(f"Brakuje kolumn. Wymagane: {required}")
                    else:
                        st.dataframe(df_imp.head(20), use_container_width=True)
                        if st.button("Zaimportuj do tego konta", key=f"csv_btn_{active_key}"):
                            added = 0
                            for _, row in df_imp.iterrows():
                                asset = str(row["asset"]).strip()
                                if asset not in TICKERS:
                                    continue
                                fee = float(row["fee"]) if "fee" in df_imp.columns and pd.notna(row.get("fee")) else 0.0
                                st.session_state[active_key].append({
                                    'id': str(uuid.uuid4())[:8],
                                    'date': str(row["date"])[:10],
                                    'asset': asset,
                                    'type': str(row["type"]).strip().lower(),
                                    'qty': float(row["qty"]),
                                    'price': float(row["price"]),
                                    'fee': fee,
                                    'currency': asset_currency(asset),
                                })
                                added += 1
                            save_portfolio_data()
                            st.success(f"Zaimportowano {added} transakcji")
                            st.rerun()
                except Exception as e:
                    st.error(f"Błąd odczytu CSV: {e}")

        holdings = calculate_holdings_from_transactions(transactions)

        st.markdown("---")
        st.subheader("Aktualny skład portfela")

        rows = []
        total_market = 0.0
        total_cost = 0.0
        allocation = {}  # do wykresu kołowego

        # --- Akcje / ETF ---
        if holdings:
            active = {k: TICKERS[k] for k in holdings.keys() if k in TICKERS}
            prices_now = get_multiple_data(active, '5d') if active else pd.DataFrame()
            if prices_now.empty or prices_now.dropna(how='all').empty:
                prices_now = get_multiple_data(active, '1mo') if active else pd.DataFrame()

            for asset, h in holdings.items():
                price = None
                if asset in prices_now.columns:
                    series = prices_now[asset].dropna()
                    if not series.empty:
                        price = float(series.iloc[-1])
                if price is None:
                    s = get_data(TICKERS.get(asset, ''), '1mo')
                    if not s.empty:
                        price = float(s.iloc[-1])

                if price is not None:
                    market_val = price * h['qty']
                    pnl = market_val - h['total_cost']
                    pnl_pct = (pnl / h['total_cost'] * 100) if h['total_cost'] > 0 else 0
                    total_market += market_val
                    total_cost += h['total_cost']
                    allocation[asset] = market_val
                    rows.append({
                        'Aktywo': asset,
                        'Typ': 'Akcje/ETF',
                        'Ilość': h['qty'],
                        'Śr. cena nabycia': h['avg_cost'],
                        'Aktualna cena': price,
                        'Wartość rynkowa': market_val,
                        'Koszt nabycia': h['total_cost'],
                        'P&L': pnl,
                        'P&L %': pnl_pct
                    })
                else:
                    total_cost += h['total_cost']
                    rows.append({
                        'Aktywo': asset,
                        'Typ': 'Akcje/ETF',
                        'Ilość': h['qty'],
                        'Śr. cena nabycia': h['avg_cost'],
                        'Aktualna cena': None,
                        'Wartość rynkowa': None,
                        'Koszt nabycia': h['total_cost'],
                        'P&L': None,
                        'P&L %': None
                    })

        # --- Metale (tylko portfel ogólny) ---
        if not is_ike:
            gold_p = get_metal_price_per_gram('gold')
            silver_p = get_metal_price_per_gram('silver')

            # Agregacja złota (z kosztem nabycia)
            total_g = 0.0
            cost_g = 0.0
            for h in st.session_state.gold_holdings:
                g = h['grams'] * h['qty']
                total_g += g
                cost_g += h.get('total_cost', 0.0) or 0.0

            total_s = 0.0
            cost_s = 0.0
            for h in st.session_state.silver_holdings:
                g = h['grams'] * h['qty']
                total_s += g
                cost_s += h.get('total_cost', 0.0) or 0.0

            if total_g > 0 and gold_p:
                val_g = total_g * gold_p
                total_market += val_g
                total_cost += cost_g
                allocation['Złoto fizyczne'] = val_g
                avg_g = (cost_g / total_g) if total_g > 0 and cost_g > 0 else None
                pnl_g = (val_g - cost_g) if cost_g > 0 else None
                pnl_g_pct = (pnl_g / cost_g * 100) if cost_g > 0 else None
                rows.append({
                    'Aktywo': 'Złoto fizyczne',
                    'Typ': 'Metal',
                    'Ilość': total_g,
                    'Śr. cena nabycia': avg_g,
                    'Aktualna cena': gold_p,
                    'Wartość rynkowa': val_g,
                    'Koszt nabycia': cost_g if cost_g > 0 else None,
                    'P&L': pnl_g,
                    'P&L %': pnl_g_pct
                })
            if total_s > 0 and silver_p:
                val_s = total_s * silver_p
                total_market += val_s
                total_cost += cost_s
                allocation['Srebro fizyczne'] = val_s
                avg_s = (cost_s / total_s) if total_s > 0 and cost_s > 0 else None
                pnl_s = (val_s - cost_s) if cost_s > 0 else None
                pnl_s_pct = (pnl_s / cost_s * 100) if cost_s > 0 else None
                rows.append({
                    'Aktywo': 'Srebro fizyczne',
                    'Typ': 'Metal',
                    'Ilość': total_s,
                    'Śr. cena nabycia': avg_s,
                    'Aktualna cena': silver_p,
                    'Wartość rynkowa': val_s,
                    'Koszt nabycia': cost_s if cost_s > 0 else None,
                    'P&L': pnl_s,
                    'P&L %': pnl_s_pct
                })

        if rows:
            df = pd.DataFrame(rows)

            # Konwersja do PLN (przybliżona – każda pozycja w swojej walucie notowania)
            if show_pln and (usdpln or eurpln):
                pln_market = 0.0
                pln_cost = 0.0
                for i, r in enumerate(rows):
                    cur = asset_currency(r['Aktywo'])
                    mv = r.get('Wartość rynkowa')
                    cs = r.get('Koszt nabycia')
                    mv_pln = to_pln(mv, cur, usdpln, eurpln) if mv is not None else None
                    cs_pln = to_pln(cs, cur, usdpln, eurpln) if cs is not None else None
                    rows[i]['Wartość PLN'] = mv_pln
                    rows[i]['Koszt PLN'] = cs_pln
                    if mv_pln is not None:
                        pln_market += mv_pln
                    if cs_pln is not None:
                        pln_cost += cs_pln
                df = pd.DataFrame(rows)
                st.dataframe(df.style.format({
                    'Ilość': '{:.4f}',
                    'Śr. cena nabycia': '{:.2f}',
                    'Aktualna cena': '{:.4f}',
                    'Wartość rynkowa': '{:,.2f}',
                    'Koszt nabycia': '{:,.2f}',
                    'Wartość PLN': '{:,.2f}',
                    'Koszt PLN': '{:,.2f}',
                    'P&L': '{:+,.2f}',
                    'P&L %': '{:+.1f}%'
                }, na_rep="—"), use_container_width=True)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Wartość rynkowa (PLN)", f"{pln_market:,.2f} PLN")
                col_b.metric("Koszt nabycia (PLN)", f"{pln_cost:,.2f} PLN" if pln_cost else "—")
                if pln_cost > 0:
                    col_c.metric("Niezrealizowany P&L (PLN)", f"{pln_market - pln_cost:+,.2f} PLN",
                                 f"{((pln_market/pln_cost)-1)*100:+.1f}%")
                else:
                    col_c.metric("Niezrealizowany P&L (PLN)", "—")
            else:
                st.dataframe(df.style.format({
                    'Ilość': '{:.4f}',
                    'Śr. cena nabycia': '{:.2f}',
                    'Aktualna cena': '{:.4f}',
                    'Wartość rynkowa': '{:,.2f}',
                    'Koszt nabycia': '{:,.2f}',
                    'P&L': '{:+,.2f}',
                    'P&L %': '{:+.1f}%'
                }, na_rep="—"), use_container_width=True)
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Wartość rynkowa (cały portfel)", f"{total_market:,.2f}")
                col_b.metric("Całkowity koszt nabycia", f"{total_cost:,.2f}" if total_cost else "—")
                if total_cost > 0 and total_market > 0:
                    col_c.metric("Niezrealizowany P&L", f"{total_market - total_cost:+,.2f}",
                                 f"{((total_market/total_cost)-1)*100:+.1f}%")
                else:
                    col_c.metric("Niezrealizowany P&L", "—")

            # --- Wykres kołowy ---
            if allocation and total_market > 0:
                st.markdown("---")
                st.subheader("Alokacja portfela")
                pie_df = pd.DataFrame({
                    'Aktywo': list(allocation.keys()),
                    'Wartość': list(allocation.values())
                })
                pie_df['Udział %'] = pie_df['Wartość'] / pie_df['Wartość'].sum() * 100
                fig_pie = px.pie(
                    pie_df, values='Wartość', names='Aktywo',
                    title=f"Skład portfela (wartość: {total_market:,.0f})",
                    hole=0.35
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                st.dataframe(
                    pie_df.style.format({'Wartość': '{:,.2f}', 'Udział %': '{:.1f}%'}),
                    use_container_width=True
                )
        else:
            st.info("Brak pozycji. Dodaj transakcje lub metale.")

        # --- Dywidendy ---
        st.markdown("---")
        st.subheader("💸 Dywidendy")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            d_asset = st.selectbox("Aktywo (dywidenda)", list(TICKERS.keys()), key=f"d_asset_{active_key}")
        with dcol2:
            d_amount = st.number_input("Kwota brutto", min_value=0.0, value=0.0, step=1.0, key=f"d_amt_{active_key}")
        with dcol3:
            d_currency = st.selectbox("Waluta", ["USD", "EUR", "PLN"], key=f"d_cur_{active_key}")
        with dcol4:
            d_date = st.date_input("Data wypłaty", value=datetime.now().date(), key=f"d_date_{active_key}")
        if st.button("➕ Dodaj dywidendę", key=f"add_div_{active_key}"):
            if d_amount > 0:
                st.session_state.dividends.append({
                    'id': str(uuid.uuid4())[:8],
                    'date': d_date.strftime("%Y-%m-%d"),
                    'asset': d_asset,
                    'amount': float(d_amount),
                    'currency': d_currency,
                    'account': 'ike' if is_ike else 'general',
                })
                save_portfolio_data()
                st.success("Dywidenda zapisana")
                st.rerun()
        divs_acc = [d for d in st.session_state.dividends if d.get('account', 'general') == ('ike' if is_ike else 'general')]
        if divs_acc:
            st.dataframe(pd.DataFrame(divs_acc), use_container_width=True)
            del_d = st.selectbox("Usuń dywidendę (ID)", [d['id'] for d in divs_acc], key=f"del_div_{active_key}")
            if st.button("Usuń dywidendę", key=f"btn_del_div_{active_key}"):
                st.session_state.dividends = [d for d in st.session_state.dividends if d['id'] != del_d]
                save_portfolio_data()
                st.rerun()

        if transactions:
            st.markdown("---")
            st.subheader("Historia transakcji")
            df_trans = pd.DataFrame(transactions)
            st.dataframe(df_trans.sort_values('date', ascending=False), use_container_width=True)
            to_delete = st.selectbox(
                "Usuń transakcję (ID)",
                [t['id'] for t in transactions],
                key=f"del_tx_{active_key}"
            )
            if st.button("Usuń wybraną transakcję", key=f"btn_del_{active_key}"):
                st.session_state[active_key] = [t for t in transactions if t['id'] != to_delete]
                save_portfolio_data()
                st.rerun()

    # ---------- TAB 2: Performance & Drawdown ----------
    with tab2:
        st.subheader(f"Performance, Maximum Drawdown i porównanie z benchmarkiem")

        period_perf = st.selectbox("Okres analizy", ['6mo', '1y', '2y', '3y', '5y'], index=2, key=f"period_perf_{active_key}")

        if st.button("Oblicz performance", key=f"calc_perf_{active_key}"):
            holdings = calculate_holdings_from_transactions(transactions)

            # dołącz metale do „holdings” jako wartości w czasie
            series_dict = {}
            for asset, h in holdings.items():
                if asset in TICKERS:
                    s = get_data(TICKERS[asset], period_perf)
                    if not s.empty:
                        series_dict[asset] = s * h['qty']

            if not is_ike:
                total_gold = sum(h['grams'] * h['qty'] for h in st.session_state.gold_holdings)
                total_silver = sum(h['grams'] * h['qty'] for h in st.session_state.silver_holdings)
                if total_gold > 0:
                    s = get_metal_history_per_gram('gold', period_perf)
                    if not s.empty:
                        series_dict['Złoto fizyczne'] = s * total_gold
                if total_silver > 0:
                    s = get_metal_history_per_gram('silver', period_perf)
                    if not s.empty:
                        series_dict['Srebro fizyczne'] = s * total_silver

            if not series_dict:
                st.warning("Brak pozycji w portfelu.")
            else:
                with st.spinner("Budowanie historii..."):
                    df_val = pd.DataFrame(series_dict).dropna(how='all').ffill().dropna()
                    if len(df_val) < 10:
                        st.error("Za mało danych historycznych.")
                    else:
                        port_value = df_val.sum(axis=1)
                        spy = get_data('SPY', period_perf)
                        eunl = get_data('EUNL.DE', period_perf)

                        port_norm = port_value / port_value.iloc[0] * 100
                        spy_norm = spy / spy.iloc[0] * 100 if not spy.empty else None
                        eunl_norm = eunl / eunl.iloc[0] * 100 if not eunl.empty else None

                        total_return = (port_value.iloc[-1] / port_value.iloc[0] - 1) * 100
                        mdd = max_drawdown(port_value) * 100

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Zwrot w okresie", f"{total_return:+.1f}%")
                        c2.metric("Maximum Drawdown", f"{mdd:.1f}%")
                        c3.metric("Wartość końcowa", f"{port_value.iloc[-1]:,.0f}")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=port_norm.index, y=port_norm, name='Twój portfel',
                                                 line=dict(width=3, color='lime')))
                        if spy_norm is not None:
                            fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name='S&P 500 (SPY)',
                                                     line=dict(width=2, dash='dot')))
                        if eunl_norm is not None:
                            fig.add_trace(go.Scatter(x=eunl_norm.index, y=eunl_norm, name='MSCI World (EUNL.DE)',
                                                     line=dict(width=2, dash='dash')))
                        fig.update_layout(
                            title="Porównanie z benchmarkami (znormalizowane do 100)",
                            yaxis_title="Wartość (start = 100)", height=500,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Wykres składu w czasie (stacked area)
                        if len(df_val.columns) >= 1:
                            fig2 = go.Figure()
                            for col in df_val.columns:
                                fig2.add_trace(go.Scatter(
                                    x=df_val.index, y=df_val[col], name=col,
                                    stackgroup='one', mode='lines'
                                ))
                            fig2.update_layout(
                                title="Wartość pozycji w czasie",
                                yaxis_title="Wartość", height=420,
                                hovermode="x unified"
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        st.caption("Buy & hold obecnego składu (w tym metale) w wybranym okresie.")

    # ---------- TAB 3: Sharpe & Rebalancing ----------
    with tab3:
        st.subheader(f"Sharpe Ratio i propozycja rebalancingu")

        period_sh = st.selectbox("Okres", ['1y', '2y', '3y', '5y'], index=1, key=f"sh_period_{active_key}")

        if st.button("Oblicz Sharpe + rebalancing", type="primary", key=f"calc_sharpe_{active_key}"):
            holdings = calculate_holdings_from_transactions(transactions)

            if not is_ike:
                total_gold = sum(h['grams'] * h['qty'] for h in st.session_state.gold_holdings)
                total_silver = sum(h['grams'] * h['qty'] for h in st.session_state.silver_holdings)
                if total_gold > 0:
                    holdings['Złoto fizyczne'] = {'qty': total_gold, 'avg_cost': 0, 'total_cost': 0}
                if total_silver > 0:
                    holdings['Srebro fizyczne'] = {'qty': total_silver, 'avg_cost': 0, 'total_cost': 0}

            if len(holdings) < 1:
                st.warning("Brak aktywów.")
            else:
                with st.spinner("Obliczanie..."):
                    series_dict = {}
                    missing_hist = []
                    for asset, h in holdings.items():
                        if asset in TICKERS:
                            s = get_data(TICKERS[asset], period_sh)
                        elif 'Złoto' in asset:
                            s = get_metal_history_per_gram('gold', period_sh)
                        elif 'Srebro' in asset:
                            s = get_metal_history_per_gram('silver', period_sh)
                        else:
                            s = pd.Series(dtype=float)
                        if not s.empty and h['qty'] > 0:
                            series_dict[asset] = s.astype(float) * float(h['qty'])
                        else:
                            missing_hist.append(asset)

                    if missing_hist:
                        st.warning("Brak historii cen dla: " + ", ".join(missing_hist) + " — pominięte w Sharpe.")

                    if not series_dict:
                        st.error("Brak danych.")
                    else:
                        df_val = pd.DataFrame(series_dict).dropna(how='all').ffill().dropna()
                        port = df_val.sum(axis=1)
                        rets = port.pct_change().dropna()

                        excess = rets - RISK_FREE_RATE / 252
                        sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0
                        ann_ret = float(rets.mean() * 252)
                        ann_vol = float(rets.std() * np.sqrt(252))

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        c2.metric("Roczny zwrot", f"{ann_ret*100:.1f}%")
                        c3.metric("Roczna zmienność", f"{ann_vol*100:.1f}%")

                        last = df_val.iloc[-1]
                        total = last.sum()
                        st.write("**Aktualne wagi:**", {k: f"{v/total*100:.1f}%" for k, v in last.items()})

                        if len(holdings) >= 2:
                            rets_df = df_val.pct_change().dropna()
                            mean_r = rets_df.mean() * 252
                            cov = rets_df.cov() * 252
                            n = len(mean_r)

                            def neg_sharpe(w):
                                r = np.dot(w, mean_r)
                                v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                                return -(r - RISK_FREE_RATE) / v if v > 0 else 0

                            res = minimize(
                                neg_sharpe, np.array([1 / n] * n), method='SLSQP',
                                bounds=tuple((0, 1) for _ in range(n)),
                                constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
                            )

                            if res.success:
                                target = dict(zip(mean_r.index, res.x))
                                st.write("**Docelowe wagi (max Sharpe):**", {k: f"{v*100:.1f}%" for k, v in target.items()})

                                rows = []
                                for asset in mean_r.index:
                                    curr = last[asset]
                                    tgt = target[asset] * total
                                    diff = tgt - curr
                                    rows.append({
                                        'Aktywo': asset,
                                        'Obecna': curr,
                                        'Docelowa': tgt,
                                        'Różnica': diff,
                                        'Akcja': f"{'Kup' if diff > 0 else 'Sprzedaj'} {abs(diff):,.0f}"
                                    })
                                st.dataframe(pd.DataFrame(rows).style.format({
                                    'Obecna': '{:,.0f}', 'Docelowa': '{:,.0f}', 'Różnica': '{:+,.0f}'
                                }), use_container_width=True)

    # ---------- TAB 4: Metale ----------
    if not is_ike and tab4 is not None:
        with tab4:
            st.subheader("Metale fizyczne (tylko portfel ogólny)")
            st.caption("Podaj formę, wagę, ilość, cenę zakupu i datę — liczy się do P&L całego portfela.")

            col1, col2 = st.columns(2)

            with col1:
                st.write("### 🥇 Złoto")
                g_form = st.radio("Forma", ["Sztabka", "Moneta"], horizontal=True, key="g_form")
                g_custom = st.checkbox("Własna waga (g)", key="g_custom")
                if g_custom:
                    g_grams = st.number_input("Waga jednej sztuki (g)", min_value=0.1, value=1.0, step=0.1, key="g_grams_custom")
                    g_label = f"{g_form} {g_grams:g} g"
                else:
                    g_preset = st.selectbox("Waga / nominał", list(GOLD_BARS.keys()), key="g_bar")
                    g_grams = GOLD_BARS[g_preset]
                    g_label = f"{g_form} {g_preset}"
                g_qty = st.number_input("Ilość sztuk", min_value=0.0, value=1.0, step=1.0, key="g_qty")
                g_price_mode = st.radio("Cena zakupu jako", ["za 1 gram (USD)", "łącznie za całość (USD)"], key="g_price_mode")
                g_price = st.number_input("Cena (USD)", min_value=0.0, value=0.0, step=0.1, key="g_price")
                g_date = st.date_input("Data zakupu", value=datetime.now().date(), key="g_date")
                if st.button("➕ Dodaj złoto", key="btn_add_gold"):
                    if g_qty > 0 and g_grams > 0:
                        total_grams = float(g_grams) * float(g_qty)
                        if g_price_mode.startswith("za 1 gram"):
                            price_per_g = float(g_price)
                            total_cost = price_per_g * total_grams
                        else:
                            total_cost = float(g_price)
                            price_per_g = (total_cost / total_grams) if total_grams > 0 else 0.0
                        st.session_state.gold_holdings.append({
                            'form': g_form,
                            'bar': g_label,
                            'grams': float(g_grams),
                            'qty': float(g_qty),
                            'price_per_gram': price_per_g,
                            'total_cost': total_cost,
                            'date': g_date.strftime("%Y-%m-%d"),
                        })
                        save_portfolio_data()
                        st.success(f"Dodano: {g_qty:g} × {g_label} | koszt {total_cost:,.2f} USD")
                        st.rerun()

                if st.session_state.gold_holdings:
                    st.markdown("**Posiadane złoto:**")
                    for i, h in enumerate(st.session_state.gold_holdings):
                        form = h.get('form', 'Sztabka')
                        cost = h.get('total_cost')
                        date = h.get('date', '—')
                        cost_txt = f"{cost:,.2f} USD" if cost else "brak ceny"
                        st.write(
                            f"• {h['qty']:g} × {form} — {h['bar']} "
                            f"({h['grams']*h['qty']:.2f} g) | zakup: {cost_txt} | {date}"
                        )
                        if st.button(f"Usuń złoto #{i+1}", key=f"dg{i}"):
                            st.session_state.gold_holdings.pop(i)
                            save_portfolio_data()
                            st.rerun()

            with col2:
                st.write("### 🥈 Srebro")
                s_form = st.radio("Forma", ["Sztabka", "Moneta"], horizontal=True, key="s_form")
                s_custom = st.checkbox("Własna waga (g)", key="s_custom")
                if s_custom:
                    s_grams = st.number_input("Waga jednej sztuki (g)", min_value=0.1, value=31.1, step=0.1, key="s_grams_custom")
                    s_label = f"{s_form} {s_grams:g} g"
                else:
                    s_preset = st.selectbox("Waga / nominał", list(SILVER_BARS.keys()), key="s_bar")
                    s_grams = SILVER_BARS[s_preset]
                    s_label = f"{s_form} {s_preset}"
                s_qty = st.number_input("Ilość sztuk", min_value=0.0, value=1.0, step=1.0, key="s_qty")
                s_price_mode = st.radio("Cena zakupu jako", ["za 1 gram (USD)", "łącznie za całość (USD)"], key="s_price_mode")
                s_price = st.number_input("Cena (USD)", min_value=0.0, value=0.0, step=0.01, key="s_price")
                s_date = st.date_input("Data zakupu", value=datetime.now().date(), key="s_date")
                if st.button("➕ Dodaj srebro", key="btn_add_silver"):
                    if s_qty > 0 and s_grams > 0:
                        total_grams = float(s_grams) * float(s_qty)
                        if s_price_mode.startswith("za 1 gram"):
                            price_per_g = float(s_price)
                            total_cost = price_per_g * total_grams
                        else:
                            total_cost = float(s_price)
                            price_per_g = (total_cost / total_grams) if total_grams > 0 else 0.0
                        st.session_state.silver_holdings.append({
                            'form': s_form,
                            'bar': s_label,
                            'grams': float(s_grams),
                            'qty': float(s_qty),
                            'price_per_gram': price_per_g,
                            'total_cost': total_cost,
                            'date': s_date.strftime("%Y-%m-%d"),
                        })
                        save_portfolio_data()
                        st.success(f"Dodano: {s_qty:g} × {s_label} | koszt {total_cost:,.2f} USD")
                        st.rerun()

                if st.session_state.silver_holdings:
                    st.markdown("**Posiadane srebro:**")
                    for i, h in enumerate(st.session_state.silver_holdings):
                        form = h.get('form', 'Sztabka')
                        cost = h.get('total_cost')
                        date = h.get('date', '—')
                        cost_txt = f"{cost:,.2f} USD" if cost else "brak ceny"
                        st.write(
                            f"• {h['qty']:g} × {form} — {h['bar']} "
                            f"({h['grams']*h['qty']:.2f} g) | zakup: {cost_txt} | {date}"
                        )
                        if st.button(f"Usuń srebro #{i+1}", key=f"ds{i}"):
                            st.session_state.silver_holdings.pop(i)
                            save_portfolio_data()
                            st.rerun()

            gold_p = get_metal_price_per_gram('gold')
            silver_p = get_metal_price_per_gram('silver')
            total_g = sum(h['grams'] * h['qty'] for h in st.session_state.gold_holdings)
            total_s = sum(h['grams'] * h['qty'] for h in st.session_state.silver_holdings)
            cost_g = sum(h.get('total_cost', 0) or 0 for h in st.session_state.gold_holdings)
            cost_s = sum(h.get('total_cost', 0) or 0 for h in st.session_state.silver_holdings)

            if total_g > 0 or total_s > 0:
                st.markdown("---")
                st.subheader("Wycena metali")
                c1, c2, c3 = st.columns(3)
                if gold_p and total_g > 0:
                    val_g = total_g * gold_p
                    delta_g = f"{val_g - cost_g:+,.2f} USD" if cost_g > 0 else f"{total_g:.2f} g × {gold_p:.2f}"
                    c1.metric("Wartość złota", f"{val_g:,.2f} USD", delta_g)
                if silver_p and total_s > 0:
                    val_s = total_s * silver_p
                    delta_s = f"{val_s - cost_s:+,.2f} USD" if cost_s > 0 else f"{total_s:.2f} g × {silver_p:.4f}"
                    c2.metric("Wartość srebra", f"{val_s:,.2f} USD", delta_s)
                if (gold_p or silver_p) and (total_g > 0 or total_s > 0):
                    total_metals = (total_g * (gold_p or 0)) + (total_s * (silver_p or 0))
                    total_cost_m = cost_g + cost_s
                    c3.metric(
                        "Metale łącznie",
                        f"{total_metals:,.2f} USD",
                        f"{total_metals - total_cost_m:+,.2f} USD" if total_cost_m > 0 else None
                    )

    # ---------- TAB 5: Monte Carlo całego portfela ----------
    with tab5:
        st.subheader("🎲 Monte Carlo – cały portfel")
        st.caption("Symulacja przyszłej wartości obecnego składu (akcje/ETF + metale) na podstawie historycznych stóp zwrotu.")

        col1, col2, col3 = st.columns(3)
        with col1:
            years_fwd = st.slider("Horyzont (lata)", 1, 10, 3, key="mc_port_years")
        with col2:
            hist_y = st.selectbox("Dane historyczne (lata)", [1, 2, 3, 4, 5], index=2, key="mc_port_hist")
        with col3:
            n_sims = st.number_input("Liczba ścieżek (max 1 000 000)", 1000, 1000000, 100000, 10000, key="mc_port_n")

        if st.button("🚀 Uruchom Monte Carlo portfela", type="primary", key="mc_port_run"):
            holdings = calculate_holdings_from_transactions(transactions)
            series_dict = {}
            weights_info = {}

            for asset, h in holdings.items():
                if asset in TICKERS:
                    s = get_data(TICKERS[asset], f"{hist_y}y")
                    if not s.empty:
                        series_dict[asset] = s
                        weights_info[asset] = h['qty'] * float(s.iloc[-1])

            if not is_ike:
                tg = sum(h['grams'] * h['qty'] for h in st.session_state.gold_holdings)
                ts = sum(h['grams'] * h['qty'] for h in st.session_state.silver_holdings)
                if tg > 0:
                    s = get_metal_history_per_gram('gold', f"{hist_y}y")
                    if not s.empty:
                        series_dict['Złoto fizyczne'] = s
                        weights_info['Złoto fizyczne'] = tg * float(s.iloc[-1])
                if ts > 0:
                    s = get_metal_history_per_gram('silver', f"{hist_y}y")
                    if not s.empty:
                        series_dict['Srebro fizyczne'] = s
                        weights_info['Srebro fizyczne'] = ts * float(s.iloc[-1])

            if not series_dict:
                st.warning("Brak pozycji do symulacji.")
            else:
                with st.spinner("Symulacja..."):
                    df_p = pd.DataFrame(series_dict).dropna(how='all').ffill().dropna()
                    rets = df_p.pct_change().dropna()
                    if len(rets) < 30:
                        st.error("Za mało danych historycznych.")
                    else:
                        # wagi rynkowe
                        total_w = sum(weights_info.values())
                        w = np.array([weights_info[c] / total_w for c in df_p.columns])

                        # dzienne parametry portfela
                        port_rets = (rets * w).sum(axis=1)
                        mu = float(port_rets.mean())
                        sigma = float(port_rets.std())
                        drift = mu - 0.5 * sigma ** 2

                        days = int(years_fwd * 252)
                        S0 = total_w
                        shocks = np.random.normal(drift, sigma, size=(days, int(n_sims))).astype(np.float32)
                        paths = np.zeros((days, int(n_sims)), dtype=np.float32)
                        paths[0] = S0
                        for t in range(1, days):
                            paths[t] = paths[t - 1] * np.exp(shocks[t])

                        end = paths[-1]
                        min_v, max_v = float(end.min()), float(end.max())
                        med_v, mean_v = float(np.median(end)), float(end.mean())
                        var95 = float(np.percentile(end, 5))
                        cvar95 = float(end[end <= var95].mean())

                        def pct(v):
                            return ((v / S0) - 1) * 100

                        st.subheader(f"Wyniki (start: {S0:,.0f})")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Najmniej", f"{min_v:,.0f}", f"{pct(min_v):+.1f}%")
                        c2.metric(f"Mediana ({years_fwd} lat)", f"{med_v:,.0f}", f"{pct(med_v):+.1f}%")
                        c3.metric("Najwięcej", f"{max_v:,.0f}", f"{pct(max_v):+.1f}%")
                        c4, c5, c6 = st.columns(3)
                        c4.metric("Średnia", f"{mean_v:,.0f}", f"{pct(mean_v):+.1f}%")
                        c5.metric("VaR 95%", f"{var95:,.0f}", f"{pct(var95):+.1f}%")
                        c6.metric("CVaR 95%", f"{cvar95:,.0f}", f"{pct(cvar95):+.1f}%")

                        fig = go.Figure()
                        time_idx = pd.date_range(start=datetime.now().date(), periods=days, freq='B')
                        for i in range(min(int(n_sims), 80)):
                            fig.add_trace(go.Scatter(
                                x=time_idx, y=paths[:, i], mode='lines',
                                line=dict(width=0.4), opacity=0.12, showlegend=False
                            ))
                        med_path = np.median(paths, axis=1)
                        fig.add_trace(go.Scatter(
                            x=time_idx, y=med_path, mode='lines',
                            name='Mediana', line=dict(color='lime', width=3)
                        ))
                        fig.add_hline(y=S0, line_dash="dash", line_color="white")
                        fig.update_layout(
                            title="Prognoza wartości całego portfela",
                            xaxis_title="Data", yaxis_title="Wartość", height=520
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Wagi startowe: " + ", ".join(f"{k}: {weights_info[k]/total_w*100:.1f}%" for k in weights_info))

# ============================================================
# MODUŁ 5: PROJEKCJA IKE / IKZE (rozbudowany)
# ============================================================
elif menu == "Projekcja IKE":
    st.title("🏦 Projekcja IKE / IKZE")
    st.write(
        "Rozbudowany kalkulator przyszłej wartości konta emerytalnego przy systematycznych wpłatach. "
        "Uwzględnia limity, inflację, podatek Belki oraz różnice między IKE a IKZE."
    )

    account_type = st.radio("Typ konta", ["IKE", "IKZE"], horizontal=True, key="ike_type")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parametry")
        current_value = st.number_input("Obecna wartość konta (PLN)", min_value=0.0, value=0.0, step=1000.0)
        monthly_contrib = st.number_input("Miesięczna wpłata (PLN)", min_value=0.0, value=500.0, step=50.0)
        years = st.slider("Horyzont (lata)", 5, 40, 25)
        expected_return = st.slider("Oczekiwana roczna stopa zwrotu (%)", 2.0, 12.0, 7.0, 0.5)
        inflation = st.slider("Średnia inflacja (%)", 0.0, 6.0, 2.5, 0.1)

        if account_type == "IKE":
            default_limit = 26019.0
            limit_help = "Limit IKE na 2026 r. (przybliżony) – sprawdź aktualny na stronie KNF / ZUS"
        else:
            default_limit = 13009.5
            limit_help = "Limit IKZE 2026 (przybliżony). Dla samozatrudnionych limit jest wyższy."

        annual_limit = st.number_input(
            f"Roczny limit wpłat {account_type} (PLN)",
            value=default_limit,
            help=limit_help
        )

        st.markdown("---")
        st.subheader("Scenariusze i opcje")
        show_scenarios = st.checkbox("Pokaż scenariusze (pesymistyczny / bazowy / optymistyczny)", value=True)
        contrib_growth = st.slider(
            "Roczny wzrost wpłat (% nominalny)", 0.0, 5.0, 0.0, 0.5,
            help="Np. 2% = wpłaty rosną co roku o 2% (np. z pensji)"
        )
        fee_annual = st.slider("Roczna opłata za zarządzanie (%)", 0.0, 1.5, 0.0, 0.1)

    with col2:
        st.subheader("Wyniki")

        net_return = expected_return - fee_annual
        r_month = net_return / 100 / 12
        months = years * 12

        annual_contrib = monthly_contrib * 12
        over_limit = annual_contrib > annual_limit
        if over_limit:
            st.warning(
                f"⚠️ Twoja planowana wpłata roczna ({annual_contrib:,.0f} PLN) przekracza limit "
                f"{account_type} ({annual_limit:,.0f} PLN). Nadwyżka nie wejdzie na konto emerytalne."
            )
            effective_monthly = annual_limit / 12
        else:
            effective_monthly = monthly_contrib

        values = []
        contrib_cum = []
        val = current_value
        total_paid = current_value
        monthly = effective_monthly

        for m in range(months + 1):
            values.append(val)
            contrib_cum.append(total_paid)
            if m < months:
                if m > 0 and m % 12 == 0 and contrib_growth > 0:
                    monthly *= (1 + contrib_growth / 100)
                    monthly = min(monthly, annual_limit / 12)
                val = val * (1 + r_month) + monthly
                total_paid += monthly

        total_fv = values[-1]
        total_contrib = contrib_cum[-1]
        gain = total_fv - total_contrib
        real_fv = total_fv / (1 + inflation / 100) ** years
        tax_belka = gain * 0.19
        fv_after_belka = total_fv - tax_belka

        if account_type == "IKZE":
            tax_ikze = total_fv * 0.10
            fv_after_ikze = total_fv - tax_ikze
            tax_saving_contrib = (total_contrib - current_value) * 0.17
        else:
            fv_after_ikze = total_fv
            tax_saving_contrib = 0

        st.metric("Wartość nominalna na koniec", f"{total_fv:,.0f} PLN")
        st.metric("Wartość realna (dzisiejsze PLN)", f"{real_fv:,.0f} PLN")
        st.metric("Suma wpłat (własne środki)", f"{total_contrib:,.0f} PLN")
        st.metric("Zysk nominalny", f"{gain:,.0f} PLN")

        st.markdown("---")
        if account_type == "IKE":
            st.write("**Porównanie z kontem zwykłym (podatek Belki 19%):**")
            st.metric("Wartość po podatku Belki", f"{fv_after_belka:,.0f} PLN")
            st.metric("Korzyść z IKE (oszczędność podatku)", f"{tax_belka:,.0f} PLN", delta="brak podatku Belki")
        else:
            st.write("**IKZE – uproszczone rozliczenie:**")
            st.metric("Wartość po ryczałcie 10%", f"{fv_after_ikze:,.0f} PLN")
            st.metric("Szacunkowa korzyść z odliczenia wpłat (17%)", f"{tax_saving_contrib:,.0f} PLN")
            st.caption("Przy IKZE odliczasz wpłaty od dochodu, a przy wypłacie płacisz 10% ryczałtu.")

        if show_scenarios:
            st.markdown("---")
            st.subheader("Scenariusze stopy zwrotu")
            scenarios = {
                "Pesymistyczny": max(1.0, expected_return - 3.0),
                "Bazowy": expected_return,
                "Optymistyczny": expected_return + 3.0
            }
            scen_rows = []
            for name, ret in scenarios.items():
                r_m = (ret - fee_annual) / 100 / 12
                v = current_value
                mth = effective_monthly
                paid = current_value
                for m in range(months):
                    if m > 0 and m % 12 == 0 and contrib_growth > 0:
                        mth *= (1 + contrib_growth / 100)
                        mth = min(mth, annual_limit / 12)
                    v = v * (1 + r_m) + mth
                    paid += mth
                real = v / (1 + inflation / 100) ** years
                scen_rows.append({
                    "Scenariusz": name,
                    "Stopa netto": f"{ret - fee_annual:.1f}%",
                    "Wartość nominalna": v,
                    "Wartość realna": real,
                    "Zysk": v - paid
                })
            df_scen = pd.DataFrame(scen_rows)
            st.dataframe(
                df_scen.style.format({
                    "Wartość nominalna": "{:,.0f} PLN",
                    "Wartość realna": "{:,.0f} PLN",
                    "Zysk": "{:+,.0f} PLN"
                }),
                use_container_width=True
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=values, mode='lines', name=f'Wartość {account_type}',
            line=dict(color='lime', width=3)
        ))
        fig.add_trace(go.Scatter(
            y=contrib_cum, mode='lines', name='Suma wpłat',
            line=dict(color='gray', width=2, dash='dot')
        ))
        fig.update_layout(
            title=f"Wzrost wartości {account_type} przez {years} lat (netto opłat)",
            xaxis_title="Miesiąc", yaxis_title="Wartość (PLN)", height=420,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Uwaga: model uproszczony. Nie uwzględnia zmienności stóp zwrotu, zmian limitów w przyszłości, "
            "dokładnych zasad podatkowych przy wcześniejszej wypłacie ani opłat transakcyjnych. "
            "Limity IKE/IKZE aktualizuj co roku."
        )

# ============================================================
# MODUŁ 6: RAPORT PODATKOWY (Belka / uproszczony)
# ============================================================
elif menu == "Raport podatkowy":
    st.title("📑 Raport podatkowy (uproszczony)")
    st.write(
        "Szacunek podatku Belki (19%) od **zrealizowanych** zysków ze sprzedaży oraz dywidend. "
        "To narzędzie poglądowe — nie zastępuje księgowego ani oficjalnego PIT."
    )

    year = st.number_input("Rok podatkowy", min_value=2020, max_value=2035, value=datetime.now().year, step=1)
    account_filter = st.radio("Konto", ["Portfel ogólny", "IKE", "Oba"], horizontal=True)

    txs = []
    if account_filter in ("Portfel ogólny", "Oba"):
        txs += [{**t, 'account': 'general'} for t in st.session_state.get('transactions_general', [])]
    if account_filter in ("IKE", "Oba"):
        txs += [{**t, 'account': 'ike'} for t in st.session_state.get('transactions_ike', [])]

    # FIFO realized gains
    lots = {}  # asset -> list of {qty, price, fee_per_unit, date, account}
    realized_rows = []

    def push_lot(asset, qty, price, fee, date, account):
        lots.setdefault(asset, []).append({
            'qty': qty, 'price': price,
            'fee_pu': (fee / qty) if qty else 0,
            'date': date, 'account': account
        })

    txs_sorted = sorted(txs, key=lambda x: x.get('date', ''))
    for t in txs_sorted:
        y = int(str(t.get('date', '9999'))[:4]) if t.get('date') else 9999
        asset = t['asset']
        qty = float(t['qty'])
        price = float(t['price'])
        fee = float(t.get('fee', 0) or 0)
        account = t.get('account', 'general')
        if t['type'] == 'buy':
            push_lot(asset, qty, price, fee, t.get('date'), account)
        else:
            remaining = qty
            sell_fee_left = fee
            while remaining > 1e-9 and lots.get(asset):
                lot = lots[asset][0]
                take = min(remaining, lot['qty'])
                # proporcjonalna prowizja sprzedaży
                sell_fee_part = sell_fee_left * (take / qty) if qty else 0
                cost = take * (lot['price'] + lot['fee_pu'])
                proceeds = take * price - sell_fee_part
                gain = proceeds - cost
                sell_year = y
                if sell_year == year:
                    # IKE: zasadniczo bez Belki przy spełnieniu warunków – oznaczamy osobno
                    tax_flag = (account != 'ike')
                    realized_rows.append({
                        'Data': t.get('date'),
                        'Aktywo': asset,
                        'Konto': 'IKE' if account == 'ike' else 'Ogólny',
                        'Ilość': take,
                        'Przychód': proceeds,
                        'Koszt': cost,
                        'Zysk/Strata': gain,
                        'Podlega Belce': 'Tak' if tax_flag else 'Nie (IKE)',
                    })
                lot['qty'] -= take
                remaining -= take
                if lot['qty'] <= 1e-9:
                    lots[asset].pop(0)

    df_r = pd.DataFrame(realized_rows)
    st.subheader(f"Zrealizowane transakcje sprzedaży – {year}")
    if df_r.empty:
        st.info("Brak zrealizowanych sprzedaży w wybranym roku.")
        total_gain_taxable = 0.0
    else:
        st.dataframe(df_r.style.format({
            'Ilość': '{:.4f}', 'Przychód': '{:,.2f}', 'Koszt': '{:,.2f}', 'Zysk/Strata': '{:+,.2f}'
        }), use_container_width=True)
        taxable = df_r[df_r['Podlega Belce'] == 'Tak']['Zysk/Strata'].sum()
        total_gain_taxable = float(taxable)
        st.metric("Zysk/strata podlegająca Belce (sprzedaże)", f"{total_gain_taxable:+,.2f}")

    # Dywidendy
    st.subheader(f"Dywidendy – {year}")
    divs = st.session_state.get('dividends', [])
    divs_y = []
    for d in divs:
        if str(d.get('date', ''))[:4] == str(year):
            if account_filter == "Portfel ogólny" and d.get('account') == 'ike':
                continue
            if account_filter == "IKE" and d.get('account') != 'ike':
                continue
            divs_y.append(d)
    div_total = 0.0
    if divs_y:
        st.dataframe(pd.DataFrame(divs_y), use_container_width=True)
        # Uproszczenie: dywidendy zagraniczne – pokazujemy sumę; Belka często pobrana u źródła / do rozliczenia
        for d in divs_y:
            if d.get('account') != 'ike':
                div_total += float(d.get('amount', 0))
        st.metric("Suma dywidend (konta poza IKE)", f"{div_total:,.2f}")
    else:
        st.info("Brak dywidend w tym roku.")

    st.markdown("---")
    st.subheader("Szacunek podatku Belki 19%")
    # Uproszczenie: 19% od dodatniego zysku ze sprzedaży + 19% od dywidend (jeśli nie pobrano u źródła)
    tax_base_sales = max(total_gain_taxable, 0)
    tax_sales = tax_base_sales * 0.19
    tax_div = max(div_total, 0) * 0.19
    st.write(f"**Podstawa (sprzedaże, tylko zysk):** {tax_base_sales:,.2f}")
    st.write(f"**Szacunek Belki od sprzedaży:** {tax_sales:,.2f}")
    st.write(f"**Szacunek Belki od dywidend (jeśli w pełni opodatkowane 19%):** {tax_div:,.2f}")
    st.metric("Razem szacunek Belki", f"{tax_sales + tax_div:,.2f}")
    st.caption(
        "Uwagi: metoda FIFO; waluty nie są przeliczane na PLN w tym raporcie (użyj kwot już w jednej walucie "
        "albo uzupełnij później). IKE przy spełnieniu warunków jest zwolnione z Belki. "
        "Dywidendy zagraniczne mają własne reguły (np. podatek u źródła, umowa o unikaniu podwójnego opodatkowania)."
    )

# ============================================================
# MODUŁ 7: CELE I FIRE
# ============================================================
elif menu == "Cele i FIRE":
    st.title("🎯 Cele inwestycyjne i ścieżka FIRE")
    st.write("Policz, ile odkładać, żeby dojść do docelowej kwoty — z uwzględnieniem inflacji i IKE.")

    goals = st.session_state.get('goals', {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parametry celu")
        target = st.number_input("Docelowa kwota (PLN, dzisiejsza siła nabywcza)", min_value=0.0,
                                 value=float(goals.get('target', 1_000_000)), step=10000.0)
        years = st.slider("Horyzont (lata)", 1, 50, int(goals.get('years', 20)))
        monthly = st.number_input("Ile już odkładasz miesięcznie (PLN)", min_value=0.0,
                                  value=float(goals.get('monthly', 1000)), step=100.0)
        current = st.number_input("Obecny kapitał (PLN)", min_value=0.0,
                                  value=float(goals.get('current', 0)), step=1000.0)
        exp_ret = st.slider("Oczekiwana realna stopa zwrotu (% rocznie)", 0.0, 12.0,
                            float(goals.get('exp_ret', 5.0)), 0.5)
        inflation = st.slider("Inflacja nominalna do projekcji (% )", 0.0, 8.0,
                              float(goals.get('inflation', 2.5)), 0.1)
        # nominal return approx
        nom_ret = (1 + exp_ret/100) * (1 + inflation/100) - 1

    with col2:
        st.subheader("Wynik")
        months = years * 12
        r = nom_ret / 12

        # Future value of current + monthly
        if r > 0:
            fv_current = current * (1 + r) ** months
            fv_contrib = monthly * (((1 + r) ** months - 1) / r)
        else:
            fv_current = current
            fv_contrib = monthly * months
        fv = fv_current + fv_contrib
        # real FV
        real_fv = fv / ((1 + inflation/100) ** years)

        # Required monthly to hit target in today's PLN (solve for PMT on real terms)
        # Work in real return
        rr = exp_ret / 100 / 12
        target_future_nominal = target * ((1 + inflation/100) ** years)
        if r > 0:
            # PMT = (FV - PV*(1+r)^n) * r / ((1+r)^n - 1)
            need_from_contrib = target_future_nominal - current * (1 + r) ** months
            if need_from_contrib <= 0:
                req_monthly = 0.0
            else:
                req_monthly = need_from_contrib * r / ((1 + r) ** months - 1)
        else:
            need_from_contrib = target_future_nominal - current
            req_monthly = max(need_from_contrib / months, 0)

        st.metric("Prognozowana wartość nominalna", f"{fv:,.0f} PLN")
        st.metric("Prognozowana wartość realna (dziisiejsze PLN)", f"{real_fv:,.0f} PLN")
        st.metric("Wymagana miesięczna wpłata do celu", f"{req_monthly:,.0f} PLN")
        gap = req_monthly - monthly
        if gap > 1:
            st.warning(f"Brakuje ok. **{gap:,.0f} PLN/mies.** przy obecnych założeniach.")
        elif real_fv >= target:
            st.success("Przy obecnych wpłatach cel jest w zasięgu (w ujęciu realnym).")
        else:
            st.info("Jesteś blisko — drobna korekta wpłat lub stopy zwrotu domknie cel.")

        if st.button("💾 Zapisz parametry celu"):
            st.session_state.goals = {
                'target': target, 'years': years, 'monthly': monthly,
                'current': current, 'exp_ret': exp_ret, 'inflation': inflation,
            }
            save_portfolio_data()
            st.success("Zapisano")

    # Chart
    values = []
    val = current
    r_m = nom_ret / 12
    for m in range(months + 1):
        values.append(val)
        if m < months:
            val = val * (1 + r_m) + monthly
    target_line = [target * ((1 + inflation/100) ** (m/12)) for m in range(months + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=values, mode='lines', name='Twój kapitał (nominalnie)',
                             line=dict(color='lime', width=3)))
    fig.add_trace(go.Scatter(y=target_line, mode='lines', name='Cel (rosnący z inflacją)',
                             line=dict(color='orange', width=2, dash='dash')))
    fig.update_layout(title="Ścieżka do celu", xaxis_title="Miesiąc", yaxis_title="PLN",
                      height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Szybkie scenariusze stopy zwrotu")
    scen_rows = []
    for name, er in [("Pesymistyczny", max(0.0, exp_ret - 2)), ("Bazowy", exp_ret), ("Optymistyczny", exp_ret + 2)]:
        nr = (1 + er/100) * (1 + inflation/100) - 1
        rm = nr / 12
        v = current
        for m in range(months):
            v = v * (1 + rm) + monthly
        real = v / ((1 + inflation/100) ** years)
        scen_rows.append({"Scenariusz": name, "Realna stopa": f"{er:.1f}%",
                          "Wartość nominalna": v, "Wartość realna": real})
    st.dataframe(pd.DataFrame(scen_rows).style.format({
        "Wartość nominalna": "{:,.0f} PLN", "Wartość realna": "{:,.0f} PLN"
    }), use_container_width=True)

    st.caption("Model uproszczony (stała stopa, brak podatków w ścieżce FIRE poza tym, co masz w IKE). "
               "Cel podajesz w dzisiejszych PLN; linia celu rośnie z inflacją.")

# ============================================================
# MODUŁ 8: PORÓWNANIE STRATEGII
# ============================================================
elif menu == "Porównanie strategii":
    st.title("⚔️ Porównanie strategii")
    st.write(
        "Porównaj swój obecny portfel z prostymi benchmarkami: 100% MSCI World, 100% S&P 500, "
        "klasyczne 60/40 (akcje/obligacje proxy) oraz opcjonalnie własną mieszankę."
    )

    period = st.selectbox("Okres porównania", ["1y", "2y", "3y", "5y"], index=2, key="strat_period")
    use_current = st.checkbox("Uwzględnij mój obecny portfel (ogólny)", value=True)

    # Opcjonalna własna alokacja
    st.subheader("Własna strategia (opcjonalnie)")
    custom_on = st.checkbox("Dodaj własną mieszankę wag")
    custom_weights = {}
    if custom_on:
        cols = st.columns(3)
        assets_pick = st.multiselect("Aktywa do własnej strategii", list(TICKERS.keys()), default=["MSCI World ETF", "Palantir"])
        if assets_pick:
            weights_in = []
            for i, a in enumerate(assets_pick):
                w = st.number_input(f"Waga % – {a}", 0.0, 100.0, value=round(100/len(assets_pick), 1), key=f"cw_{a}")
                weights_in.append((a, w))
            s = sum(w for _, w in weights_in)
            if s > 0:
                custom_weights = {a: w/s for a, w in weights_in}
            st.caption(f"Suma wag: {s:.1f}% (zostanie znormalizowana do 100%)")

    if st.button("Uruchom porównanie", type="primary"):
        with st.spinner("Pobieranie danych i liczenie..."):
            strategies = {}

            # Benchmarki
            spy = get_data("SPY", period)
            eunl = get_data("EUNL.DE", period)
            # 60/40: SPY + proxy obligacji (TLT)
            tlt = get_data("TLT", period)

            def norm_series(s):
                s = s.dropna()
                if s.empty:
                    return s
                return s / s.iloc[0] * 100

            if not spy.empty:
                strategies["100% S&P 500"] = norm_series(spy)
            if not eunl.empty:
                strategies["100% MSCI World"] = norm_series(eunl)
            if not spy.empty and not tlt.empty:
                df6040 = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
                if not df6040.empty:
                    port6040 = 0.6 * df6040["spy"] + 0.4 * df6040["tlt"]
                    strategies["60/40 (SPY/TLT)"] = norm_series(port6040)

            # Obecny portfel
            if use_current:
                holdings = calculate_holdings_from_transactions(st.session_state.get("transactions_general", []))
                series_dict = {}
                for asset, h in holdings.items():
                    if asset in TICKERS:
                        s = get_data(TICKERS[asset], period)
                        if not s.empty:
                            series_dict[asset] = s * h["qty"]
                tg = sum(h["grams"] * h["qty"] for h in st.session_state.get("gold_holdings", []))
                ts = sum(h["grams"] * h["qty"] for h in st.session_state.get("silver_holdings", []))
                if tg > 0:
                    s = get_metal_history_per_gram("gold", period)
                    if not s.empty:
                        series_dict["Złoto"] = s * tg
                if ts > 0:
                    s = get_metal_history_per_gram("silver", period)
                    if not s.empty:
                        series_dict["Srebro"] = s * ts
                if series_dict:
                    dfp = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()
                    if not dfp.empty:
                        strategies["Mój portfel"] = norm_series(dfp.sum(axis=1))

            # Własna
            if custom_weights:
                parts = {}
                for a, w in custom_weights.items():
                    s = get_data(TICKERS[a], period)
                    if not s.empty:
                        parts[a] = s
                if parts:
                    dfc = pd.DataFrame(parts).dropna(how="all").ffill().dropna()
                    if not dfc.empty:
                        # równe startowe wagi wartości → stosujemy wagi do znormalizowanych cen
                        normed = dfc / dfc.iloc[0]
                        port_c = sum(normed[a] * custom_weights[a] for a in normed.columns if a in custom_weights)
                        strategies["Własna strategia"] = port_c * 100

            if len(strategies) < 2:
                st.error("Za mało danych do porównania.")
            else:
                # Wyrównaj indeksy
                df_all = pd.DataFrame(strategies).dropna(how="all").ffill().dropna()

                fig = go.Figure()
                for col in df_all.columns:
                    fig.add_trace(go.Scatter(x=df_all.index, y=df_all[col], mode="lines", name=col, line=dict(width=2)))
                fig.update_layout(
                    title=f"Porównanie strategii ({period})",
                    yaxis_title="Wartość (start = 100)",
                    height=500,
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metryki
                rows = []
                for col in df_all.columns:
                    s = df_all[col].dropna()
                    rets = s.pct_change().dropna()
                    total_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
                    ann_ret = rets.mean() * 252 * 100
                    ann_vol = rets.std() * np.sqrt(252) * 100
                    mdd = max_drawdown(s) * 100
                    excess = rets - RISK_FREE_RATE / 252
                    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0
                    rows.append({
                        "Strategia": col,
                        "Zwrot całkowity %": total_ret,
                        "Zwrot roczny %": ann_ret,
                        "Zmienność %": ann_vol,
                        "Max DD %": mdd,
                        "Sharpe": sharpe,
                    })
                st.dataframe(
                    pd.DataFrame(rows).style.format({
                        "Zwrot całkowity %": "{:+.1f}",
                        "Zwrot roczny %": "{:+.1f}",
                        "Zmienność %": "{:.1f}",
                        "Max DD %": "{:.1f}",
                        "Sharpe": "{:.2f}",
                    }),
                    use_container_width=True,
                )

# ============================================================
# MODUŁ 9: RYZYKO PORTFELA
# ============================================================
elif menu == "Ryzyko portfela":
    st.title("📉 Ryzyko portfela")
    st.write(
        "Beta względem benchmarku, tracking error oraz wkład poszczególnych pozycji w ryzyko portfela "
        "(contribution to risk)."
    )

    period = st.selectbox("Okres", ["1y", "2y", "3y", "5y"], index=1, key="risk_period")
    benchmark = st.selectbox("Benchmark", ["MSCI World (EUNL.DE)", "S&P 500 (SPY)"], key="risk_bench")
    bench_ticker = "EUNL.DE" if "MSCI" in benchmark else "SPY"

    account = st.radio("Konto", ["Portfel ogólny", "IKE"], horizontal=True, key="risk_acc")
    is_ike_r = account == "IKE"
    tx_key = "transactions_ike" if is_ike_r else "transactions_general"

    if st.button("Oblicz ryzyko", type="primary", key="risk_btn"):
        holdings = calculate_holdings_from_transactions(st.session_state.get(tx_key, []))
        if not is_ike_r:
            tg = sum(h["grams"] * h["qty"] for h in st.session_state.get("gold_holdings", []))
            ts = sum(h["grams"] * h["qty"] for h in st.session_state.get("silver_holdings", []))
            if tg > 0:
                holdings["Złoto fizyczne"] = {"qty": tg, "avg_cost": 0, "total_cost": 0}
            if ts > 0:
                holdings["Srebro fizyczne"] = {"qty": ts, "avg_cost": 0, "total_cost": 0}

        if not holdings:
            st.warning("Brak pozycji w wybranym koncie.")
        else:
            with st.spinner("Liczenie..."):
                series_dict = {}
                for asset, h in holdings.items():
                    if asset in TICKERS:
                        s = get_data(TICKERS[asset], period)
                    elif "Złoto" in asset:
                        s = get_metal_history_per_gram("gold", period)
                    elif "Srebro" in asset:
                        s = get_metal_history_per_gram("silver", period)
                    else:
                        s = pd.Series(dtype=float)
                    if not s.empty and h["qty"] > 0:
                        series_dict[asset] = s.astype(float) * float(h["qty"])

                if not series_dict:
                    st.error("Brak danych cenowych.")
                else:
                    df_val = pd.DataFrame(series_dict).dropna(how="all").ffill().dropna()
                    port = df_val.sum(axis=1)
                    port_rets = port.pct_change().dropna()

                    bench = get_data(bench_ticker, period)
                    # wyrównanie
                    aligned = pd.concat([port_rets.rename("port"), bench.pct_change().rename("bench")], axis=1).dropna()

                    if len(aligned) < 30:
                        st.error("Za mało wspólnych obserwacji z benchmarkiem.")
                    else:
                        # Beta i tracking error
                        cov = np.cov(aligned["port"], aligned["bench"])[0, 1]
                        var_b = np.var(aligned["bench"])
                        beta = cov / var_b if var_b > 0 else 0
                        active = aligned["port"] - aligned["bench"]
                        te = float(active.std() * np.sqrt(252))  # annualized tracking error
                        corr = float(aligned["port"].corr(aligned["bench"]))

                        ann_vol = float(port_rets.std() * np.sqrt(252))
                        ann_ret = float(port_rets.mean() * 252)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Beta", f"{beta:.2f}")
                        c2.metric("Tracking error (roczny)", f"{te*100:.1f}%")
                        c3.metric("Korelacja z benchmarkiem", f"{corr:.2f}")
                        c4.metric("Zmienność portfela", f"{ann_vol*100:.1f}%")

                        st.caption(
                            f"Beta > 1 = portfel bardziej agresywny niż {benchmark}. "
                            "Tracking error = jak bardzo wyniki odbiegają od benchmarku."
                        )

                        # Contribution to risk (marginal risk contribution ≈ w_i * (cov w)_i / sigma
                        rets_df = df_val.pct_change().dropna()
                        # użyj wag z ostatniego dnia
                        last = df_val.iloc[-1]
                        w = (last / last.sum()).values
                        cov_m = rets_df.cov().values * 252  # annualized
                        port_var = float(w @ cov_m @ w)
                        port_sigma = np.sqrt(port_var) if port_var > 0 else 0
                        if port_sigma > 0:
                            mrc = cov_m @ w  # marginal
                            ctr = w * mrc  # contribution to variance
                            ctr_pct = ctr / port_var * 100  # % of variance

                            risk_rows = []
                            for i, asset in enumerate(rets_df.columns):
                                risk_rows.append({
                                    "Aktywo": asset,
                                    "Waga %": w[i] * 100,
                                    "Wkład w ryzyko %": ctr_pct[i],
                                    "Vol. własna %": float(rets_df[asset].std() * np.sqrt(252) * 100),
                                })
                            rdf = pd.DataFrame(risk_rows).sort_values("Wkład w ryzyko %", ascending=False)
                            st.subheader("Wkład w ryzyko portfela")
                            st.dataframe(
                                rdf.style.format({
                                    "Waga %": "{:.1f}",
                                    "Wkład w ryzyko %": "{:.1f}",
                                    "Vol. własna %": "{:.1f}",
                                }),
                                use_container_width=True,
                            )
                            fig = px.bar(
                                rdf, x="Aktywo", y="Wkład w ryzyko %",
                                title="Która pozycja najbardziej „huśta” portfel",
                                color="Wkład w ryzyko %",
                                color_continuous_scale="Reds",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Wykres excess return
                        cum_port = (1 + aligned["port"]).cumprod() * 100
                        cum_bench = (1 + aligned["bench"]).cumprod() * 100
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="Portfel", line=dict(width=3)))
                        fig2.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench, name=benchmark, line=dict(width=2, dash="dot")))
                        fig2.update_layout(title="Portfel vs benchmark (znormalizowane)", height=420, hovermode="x unified")
                        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# MODUŁ 10: EKSPORT DANYCH
# ============================================================
elif menu == "Eksport danych":
    st.title("📦 Eksport danych")
    st.write("Pobierz skład portfela, transakcje, dywidendy i podsumowanie do CSV / Excel.")

    account = st.radio("Które konto eksportować?", ["Portfel ogólny", "IKE", "Oba"], horizontal=True, key="exp_acc")

    def holdings_table(tx_list, include_metals=False):
        holdings = calculate_holdings_from_transactions(tx_list)
        rows = []
        for asset, h in holdings.items():
            price = None
            if asset in TICKERS:
                s = get_data(TICKERS[asset], "1mo")
                if not s.empty:
                    price = float(s.iloc[-1])
            mv = price * h["qty"] if price is not None else None
            rows.append({
                "Aktywo": asset,
                "Ilość": h["qty"],
                "Śr. koszt": h["avg_cost"],
                "Koszt nabycia": h["total_cost"],
                "Cena bieżąca": price,
                "Wartość rynkowa": mv,
                "P&L": (mv - h["total_cost"]) if mv is not None else None,
            })
        if include_metals:
            gp = get_metal_price_per_gram("gold")
            sp = get_metal_price_per_gram("silver")
            tg = sum(h["grams"] * h["qty"] for h in st.session_state.get("gold_holdings", []))
            ts = sum(h["grams"] * h["qty"] for h in st.session_state.get("silver_holdings", []))
            cg = sum(h.get("total_cost", 0) or 0 for h in st.session_state.get("gold_holdings", []))
            cs = sum(h.get("total_cost", 0) or 0 for h in st.session_state.get("silver_holdings", []))
            if tg > 0 and gp:
                rows.append({
                    "Aktywo": "Złoto fizyczne", "Ilość": tg, "Śr. koszt": (cg/tg if tg else None),
                    "Koszt nabycia": cg, "Cena bieżąca": gp, "Wartość rynkowa": tg*gp,
                    "P&L": tg*gp - cg if cg else None,
                })
            if ts > 0 and sp:
                rows.append({
                    "Aktywo": "Srebro fizyczne", "Ilość": ts, "Śr. koszt": (cs/ts if ts else None),
                    "Koszt nabycia": cs, "Cena bieżąca": sp, "Wartość rynkowa": ts*sp,
                    "P&L": ts*sp - cs if cs else None,
                })
        return pd.DataFrame(rows)

    frames = {}
    if account in ("Portfel ogólny", "Oba"):
        frames["sklad_ogolny"] = holdings_table(st.session_state.get("transactions_general", []), include_metals=True)
        frames["transakcje_ogolny"] = pd.DataFrame(st.session_state.get("transactions_general", []))
    if account in ("IKE", "Oba"):
        frames["sklad_ike"] = holdings_table(st.session_state.get("transactions_ike", []), include_metals=False)
        frames["transakcje_ike"] = pd.DataFrame(st.session_state.get("transactions_ike", []))

    frames["dywidendy"] = pd.DataFrame(st.session_state.get("dividends", []))
    frames["metale_zloto"] = pd.DataFrame(st.session_state.get("gold_holdings", []))
    frames["metale_srebro"] = pd.DataFrame(st.session_state.get("silver_holdings", []))

    st.subheader("Podgląd")
    for name, df in frames.items():
        if df is not None and not df.empty:
            st.write(f"**{name}**")
            st.dataframe(df, use_container_width=True)

    # CSV zip-like: multiple download buttons
    st.subheader("Pobierz CSV")
    for name, df in frames.items():
        if df is not None and not df.empty:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                f"⬇️ {name}.csv",
                data=csv,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )

    st.subheader("Pobierz Excel (wszystkie arkusze)")
    try:
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, df in frames.items():
                if df is not None and not df.empty:
                    sheet = name[:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
        buffer.seek(0)
        st.download_button(
            "⬇️ portfel_eksport.xlsx",
            data=buffer,
            file_name=f"portfel_eksport_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
        )
    except Exception as e:
        st.info(
            "Eksport Excel wymaga biblioteki openpyxl. Zainstaluj: `pip install openpyxl` "
            f"albo użyj CSV powyżej. ({e})"
        )

    st.caption("Eksport odzwierciedla stan zapisany w aplikacji (w tym portfolio_data.json).")
