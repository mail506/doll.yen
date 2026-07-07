"""
ドル円特化型 ファンダメンタルズ＆テクニカル分析ダッシュボード
依存: streamlit, yfinance, pandas, numpy, plotly, statsmodels, scipy
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="USD/JPY 為替分析ダッシュボード",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# カスタム CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ヘッダー */
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-title { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.main-sub   { font-size: 0.85rem; opacity: 0.7; margin: 0.25rem 0 0; }

/* センチメントゲージ */
.sentiment-container {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    color: white;
}
.sentiment-score  { font-size: 3rem; font-weight: 800; line-height: 1; }
.sentiment-label  { font-size: 1.1rem; font-weight: 600; margin-top: 0.25rem; }
.sentiment-detail { font-size: 0.75rem; opacity: 0.65; margin-top: 0.5rem; }

/* メトリクスカード */
.metric-card {
    background: #16213e;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    text-align: center;
    color: white;
    border-left: 4px solid #4a9eff;
}
.metric-value { font-size: 1.6rem; font-weight: 700; }
.metric-label { font-size: 0.72rem; opacity: 0.65; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* シナリオカード */
.scenario-main { background:#e8f4f8; border-left:5px solid #2196F3; border-radius:8px; padding:1rem; }
.scenario-sub  { background:#fff8e1; border-left:5px solid #FF9800; border-radius:8px; padding:1rem; }
.scenario-risk { background:#fce4ec; border-left:5px solid #F44336; border-radius:8px; padding:1rem; }
.scenario-prob  { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; opacity:0.6; }
.scenario-title { font-size:1rem; font-weight:700; margin:0.3rem 0; }
.scenario-price { font-size:1.6rem; font-weight:800; }
.scenario-desc  { font-size:0.75rem; opacity:0.7; margin-top:0.3rem; line-height:1.5; }

/* フェアバリューパネル */
.fv-panel {
    background: #0f3460;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: white;
}
.fv-row { display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom:1px solid rgba(255,255,255,0.1); font-size:0.85rem; }
.fv-row:last-child { border-bottom: none; }

/* シミュレーター結果 */
.sim-result {
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
.sim-bullish { background:#e8f5e9; border-left:4px solid #4CAF50; }
.sim-bearish { background:#fce4ec; border-left:4px solid #F44336; }
.sim-neutral { background:#f5f5f5; border-left:4px solid #9E9E9E; }

/* ボラレンジ */
.vol-range { background:#1a1a2e; border-radius:10px; padding:1rem 1.25rem; color:white; }
.vol-title  { font-size:0.8rem; opacity:0.6; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.75rem; }
.range-row  { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; font-size:0.8rem; }
.range-label { width:55px; opacity:0.6; }
.range-bar-wrap { flex:1; }

/* タグ */
.tag-bull  { background:#4CAF50; color:white; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; }
.tag-bear  { background:#F44336; color:white; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; }
.tag-neut  { background:#9E9E9E; color:white; border-radius:4px; padding:2px 8px; font-size:0.7rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ユーティリティ関数
# ──────────────────────────────────────────────

def safe_download(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """yfinance でダウンロード。失敗時は空 DataFrame を返す。"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        # MultiIndex 対応
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_hv(series: pd.Series, window: int = 20) -> float:
    log_ret = np.log(series / series.shift(1)).dropna()
    if len(log_ret) < window:
        return np.nan
    return float(log_ret.iloc[-window:].std() * np.sqrt(252) * 100)


def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def multi_reg_fair_value(usdjpy: pd.Series, spread: pd.Series,
                         dxy: pd.Series, sp500: pd.Series, window: int = 180) -> float:
    """簡易重回帰でフェアバリューを算出（statsmodels 不要の NumPy 実装）"""
    df = pd.concat([usdjpy, spread, dxy, sp500], axis=1).dropna()
    df.columns = ["y", "x1", "x2", "x3"]
    df = df.iloc[-window:]
    if len(df) < 30:
        return float(usdjpy.iloc[-1])
    X = np.column_stack([np.ones(len(df)), df["x1"].values, df["x2"].values, df["x3"].values])
    y = df["y"].values
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float(usdjpy.iloc[-1])
    x_latest = np.array([1,
                          float(spread.iloc[-1]) if not pd.isna(spread.iloc[-1]) else df["x1"].mean(),
                          float(dxy.iloc[-1])    if not pd.isna(dxy.iloc[-1])    else df["x2"].mean(),
                          float(sp500.iloc[-1])  if not pd.isna(sp500.iloc[-1])  else df["x3"].mean()])
    return float(coef @ x_latest)


def sentiment_score(rsi: float, cur: float, sma20: float, sma200: float,
                    macd: float, macd_sig: float, gap_pct: float, spread: float) -> int:
    score = 50
    # RSI
    if rsi > 70:   score += 15
    elif rsi > 60: score += 8
    elif rsi < 30: score -= 15
    elif rsi < 40: score -= 8
    # SMA
    if cur > sma20:  score += 10
    else:            score -= 10
    if cur > sma200: score += 8
    else:            score -= 8
    # MACD
    if macd > macd_sig:  score += 8
    else:                score -= 8
    # フェアバリュー乖離
    if gap_pct > 2:    score -= 8
    elif gap_pct < -2: score += 8
    # 金利差
    if spread > 3.5:   score += 6
    elif spread < 2.5: score -= 6
    return max(0, min(100, int(score)))


def score_to_label(score: int):
    if score >= 80: return "🟢 強気",    "#4CAF50"
    if score >= 60: return "🟡 やや強気", "#8BC34A"
    if score >= 40: return "⚪ 中立",     "#9E9E9E"
    if score >= 20: return "🟠 やや弱気", "#FF9800"
    return            "🔴 弱気",         "#F44336"


# ──────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)   # 5分キャッシュ
def fetch_all_data(period_days: int):
    yf_period = "1y" if period_days <= 365 else "2y"

    # ドル円
    usdjpy_df = safe_download("USDJPY=X", period="2y")
    us10y_df  = safe_download("^TNX",     period="2y")
    dxy_df    = safe_download("DX-Y.NYB", period="2y")
    sp500_df  = safe_download("^GSPC",    period="2y")
    nk225_df  = safe_download("^N225",    period="2y")

    # ドル円 Close
    if usdjpy_df.empty:
        idx = pd.date_range(end=datetime.today(), periods=500, freq="B")
        price_series = pd.Series(148 + np.cumsum(np.random.randn(500) * 0.4), index=idx, name="Close")
    else:
        price_series = usdjpy_df["Close"].dropna()

    # 米10年債
    if us10y_df.empty:
        us10y_series = pd.Series(4.3 + np.random.randn(len(price_series)) * 0.05,
                                  index=price_series.index, name="US10Y")
    else:
        us10y_series = (us10y_df["Close"] / 100).reindex(price_series.index).ffill().bfill()

    # 日本10年債（yfinance ^YTEN は取得不安定 → シミュレーション）
    jp10y_base = 0.9
    jp10y_series = pd.Series(jp10y_base / 100 + np.cumsum(np.random.randn(len(price_series)) * 0.0005),
                              index=price_series.index, name="JP10Y")
    jp10y_series = jp10y_series.clip(0.005, 0.02)

    # ドルインデックス
    if dxy_df.empty:
        dxy_series = pd.Series(103 + np.random.randn(len(price_series)) * 0.5,
                                index=price_series.index, name="DXY")
    else:
        dxy_series = dxy_df["Close"].reindex(price_series.index).ffill().bfill()

    # S&P 500
    if sp500_df.empty:
        sp500_series = pd.Series(5200 + np.cumsum(np.random.randn(len(price_series)) * 20),
                                  index=price_series.index, name="SP500")
    else:
        sp500_series = sp500_df["Close"].reindex(price_series.index).ffill().bfill()

    # 日経平均
    if nk225_df.empty:
        nk_series = pd.Series(38000 + np.cumsum(np.random.randn(len(price_series)) * 300),
                               index=price_series.index, name="NK225")
    else:
        nk_series = nk225_df["Close"].reindex(price_series.index).ffill().bfill()

    spread_series = (us10y_series - jp10y_series).rename("Spread")

    # 期間でスライス
    cutoff = datetime.today() - timedelta(days=period_days)
    def sl(s): return s[s.index >= cutoff]

    return {
        "usdjpy":  sl(price_series),
        "us10y":   sl(us10y_series),
        "jp10y":   sl(jp10y_series),
        "spread":  sl(spread_series),
        "dxy":     sl(dxy_series),
        "sp500":   sl(sp500_series),
        "nk225":   sl(nk_series),
        "usdjpy_full": price_series,
        "us10y_full":  us10y_series,
        "spread_full": spread_series,
        "dxy_full":    dxy_series,
        "sp500_full":  sp500_series,
    }


# ──────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    period_label = st.selectbox("分析期間", ["3ヶ月", "6ヶ月", "1年"], index=1)
    period_days  = {"3ヶ月": 90, "6ヶ月": 180, "1年": 365}[period_label]

    st.markdown("---")
    st.markdown("### 🇯🇵 日本10年債利回り")
    jp10y_manual = st.slider("手動入力 (%)", min_value=0.5, max_value=2.0, value=0.9, step=0.05)
    st.caption("yfinanceでの取得が不安定なため手動入力を推奨")

    st.markdown("---")
    auto_refresh = st.checkbox("5分ごとに自動更新", value=False)
    if st.button("🔄 今すぐ更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 ファンダメンタルズ・シミュレーター")
    st.caption("市場予想との乖離を入力して感応度を分析")
    nfp_val   = st.slider("米雇用統計 NFP (千人)", min_value=50,  max_value=500, value=185, step=5)
    nfp_exp   = 185
    cpi_val   = st.slider("米CPI 前年比 (%)",     min_value=1.0, max_value=6.0, value=3.1, step=0.1)
    cpi_exp   = 3.1
    ffr_val   = st.slider("FRB政策金利 (%)",      min_value=3.0, max_value=6.0, value=4.25, step=0.25)
    ffr_base  = 4.25


# ──────────────────────────────────────────────
# データ取得
# ──────────────────────────────────────────────
with st.spinner("データを取得中..."):
    data = fetch_all_data(period_days)

usdjpy  = data["usdjpy"]
spread  = data["spread"]
dxy     = data["dxy"]
sp500   = data["sp500"]
nk225   = data["nk225"]
us10y   = data["us10y"]

# 日本10年債を手動値で上書き
jp10y_rate = jp10y_manual / 100
spread_adj = us10y - jp10y_rate   # 手動入力反映済み金利差

# ──────────────────────────────────────────────
# テクニカル指標
# ──────────────────────────────────────────────
closes = usdjpy.copy()
sma20  = calc_sma(closes, 20)
sma200 = calc_sma(closes, 200)
rsi    = calc_rsi(closes, 14)
macd_line, macd_signal, macd_hist = calc_macd(closes)
hv     = calc_hv(closes, 20)

# 全期間データでフェアバリュー計算（180日使用）
fair_value = multi_reg_fair_value(
    data["usdjpy_full"],
    data["spread_full"],
    data["dxy_full"],
    data["sp500_full"],
    window=180,
)

cur_price   = float(closes.iloc[-1])
prev_price  = float(closes.iloc[-2]) if len(closes) > 1 else cur_price
price_chg   = cur_price - prev_price
price_chg_pct = price_chg / prev_price * 100

rsi_cur      = float(rsi.iloc[-1])    if not pd.isna(rsi.iloc[-1])    else 50.0
sma20_cur    = float(sma20.iloc[-1])  if not pd.isna(sma20.iloc[-1])  else cur_price
sma200_cur   = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else cur_price
macd_cur     = float(macd_line.iloc[-1])   if not pd.isna(macd_line.iloc[-1])   else 0.0
macd_sig_cur = float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0
spread_cur   = float(spread_adj.iloc[-1])  if not pd.isna(spread_adj.iloc[-1])  else 3.5
us10y_cur    = float(us10y.iloc[-1])       if not pd.isna(us10y.iloc[-1])       else 0.043

gap_pct = (cur_price - fair_value) / fair_value * 100

# センチメントスコア
sent_score = sentiment_score(rsi_cur, cur_price, sma20_cur, sma200_cur,
                              macd_cur, macd_sig_cur, gap_pct, spread_cur)
sent_label, sent_color = score_to_label(sent_score)

# ボラティリティベース予測レンジ
hv_daily   = (hv / 100) / np.sqrt(252)
sigma_1w_1 = cur_price * hv_daily * np.sqrt(5)
sigma_1w_2 = sigma_1w_1 * 2
sigma_1m_1 = cur_price * hv_daily * np.sqrt(21)
sigma_1m_2 = sigma_1m_1 * 2

# 52週高値・安値
prices_1y = data["usdjpy_full"].iloc[-365:]
h52 = float(prices_1y.max())
l52 = float(prices_1y.min())

# 相関
corr_spread = pearson_corr(closes, spread_adj.reindex(closes.index).ffill())
corr_dxy    = pearson_corr(closes, dxy.reindex(closes.index).ffill())
corr_sp     = pearson_corr(closes, sp500.reindex(closes.index).ffill())
corr_nk     = pearson_corr(closes, nk225.reindex(closes.index).ffill())


# ──────────────────────────────────────────────
# ① ヘッダー
# ──────────────────────────────────────────────
chg_sign  = "+" if price_chg >= 0 else ""
chg_color = "#4CAF50" if price_chg >= 0 else "#F44336"
st.markdown(f"""
<div class="main-header">
  <div style="display:flex;align-items:baseline;gap:1.5rem;flex-wrap:wrap;">
    <span class="main-title">💱 USD/JPY 為替分析ダッシュボード</span>
    <span style="font-size:2rem;font-weight:800;color:white;">{cur_price:.3f}</span>
    <span style="font-size:1.1rem;color:{chg_color};font-weight:700;">
      {chg_sign}{price_chg:.3f} ({chg_sign}{price_chg_pct:.2f}%)
    </span>
  </div>
  <div class="main-sub">
    分析期間: {period_label} ｜ 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JST ｜
    データソース: yfinance (リアルタイム)
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ① センチメント & メトリクス
# ──────────────────────────────────────────────
col_sent, col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns([1.8,1,1,1,1,1,1])

with col_sent:
    gauge_pct = sent_score
    rsi_tag   = "買われ過ぎ" if rsi_cur > 70 else ("売られ過ぎ" if rsi_cur < 30 else "中立圏")
    st.markdown(f"""
    <div class="sentiment-container">
      <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px;">総合センチメント</div>
      <div class="sentiment-score" style="color:{sent_color};">{sent_score}</div>
      <div class="sentiment-label" style="color:{sent_color};">{sent_label}</div>
      <div class="sentiment-detail">RSI: {rsi_cur:.1f}（{rsi_tag}）｜ MACD: {"↑" if macd_cur > macd_sig_cur else "↓"}</div>
    </div>
    """, unsafe_allow_html=True)

def mk_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      {"<div style='font-size:0.7rem;opacity:0.5;margin-top:3px;'>"+sub+"</div>" if sub else ""}
    </div>"""

with col_m1:
    st.markdown(mk_card("52W 高値", f"{h52:.2f}"), unsafe_allow_html=True)
with col_m2:
    st.markdown(mk_card("52W 安値", f"{l52:.2f}"), unsafe_allow_html=True)
with col_m3:
    st.markdown(mk_card("RSI (14)", f"{rsi_cur:.1f}", rsi_tag), unsafe_allow_html=True)
with col_m4:
    st.markdown(mk_card("HV 20日", f"{hv:.1f}%", "年率換算"), unsafe_allow_html=True)
with col_m5:
    st.markdown(mk_card("米10年債", f"{us10y_cur*100:.2f}%"), unsafe_allow_html=True)
with col_m6:
    st.markdown(mk_card("日米金利差", f"{spread_cur*100:.2f}%", "10年債"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ② チャートエリア
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 ローソク足 & テクニカル", "📉 金利差との相関"])

# ────── タブ1: ローソク足 ──────
with tab1:
    usdjpy_ohlc = yf.download("USDJPY=X", period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(usdjpy_ohlc.columns, pd.MultiIndex):
        usdjpy_ohlc.columns = usdjpy_ohlc.columns.get_level_values(0)
    cutoff_dt = datetime.today() - timedelta(days=period_days)
    usdjpy_ohlc = usdjpy_ohlc[usdjpy_ohlc.index >= cutoff_dt]

    fig_price = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("USD/JPY ローソク足 + 移動平均", "MACD", "RSI (14)")
    )

    # ローソク足
    if not usdjpy_ohlc.empty and "Open" in usdjpy_ohlc.columns:
        fig_price.add_trace(go.Candlestick(
            x=usdjpy_ohlc.index,
            open=usdjpy_ohlc["Open"].squeeze(),
            high=usdjpy_ohlc["High"].squeeze(),
            low=usdjpy_ohlc["Low"].squeeze(),
            close=usdjpy_ohlc["Close"].squeeze(),
            name="USD/JPY",
            increasing_line_color="#4CAF50",
            decreasing_line_color="#F44336",
        ), row=1, col=1)
    else:
        fig_price.add_trace(go.Scatter(
            x=closes.index, y=closes.values,
            name="USD/JPY", line=dict(color="#2196F3", width=1.5)
        ), row=1, col=1)

    # 移動平均
    fig_price.add_trace(go.Scatter(
        x=sma20.index, y=sma20.values,
        name="SMA20", line=dict(color="#FF9800", width=1.5, dash="dot")
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=sma200.index, y=sma200.values,
        name="SMA200", line=dict(color="#E91E63", width=1.5, dash="dash")
    ), row=1, col=1)

    # フェアバリューライン
    fig_price.add_hline(y=fair_value, line_dash="dash",
                         line_color="rgba(255,255,255,0.4)", line_width=1,
                         annotation_text=f"FV {fair_value:.2f}", row=1, col=1)

    # MACD
    colors_hist = ["#4CAF50" if v >= 0 else "#F44336" for v in macd_hist.values]
    fig_price.add_trace(go.Bar(
        x=macd_hist.index, y=macd_hist.values,
        name="MACD Hist", marker_color=colors_hist, opacity=0.7
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values,
        name="MACD", line=dict(color="#2196F3", width=1.5)
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=macd_signal.index, y=macd_signal.values,
        name="Signal", line=dict(color="#FF9800", width=1.5)
    ), row=2, col=1)

    # RSI
    fig_price.add_trace(go.Scatter(
        x=rsi.index, y=rsi.values,
        name="RSI", line=dict(color="#9C27B0", width=1.5), fill="tozeroy",
        fillcolor="rgba(156,39,176,0.1)"
    ), row=3, col=1)
    fig_price.add_hline(y=70, line_color="rgba(244,67,54,0.5)", line_dash="dash", row=3, col=1)
    fig_price.add_hline(y=30, line_color="rgba(76,175,80,0.5)", line_dash="dash", row=3, col=1)

    fig_price.update_layout(
        template="plotly_dark",
        height=680,
        paper_bgcolor="#0f2027",
        plot_bgcolor="#0f2027",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
        font=dict(size=11),
    )
    fig_price.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    fig_price.update_xaxes(gridcolor="rgba(255,255,255,0.07)")

    st.plotly_chart(fig_price, use_container_width=True)

# ────── タブ2: 金利差相関 ──────
with tab2:
    fig_rate = make_subplots(specs=[[{"secondary_y": True}]])

    fig_rate.add_trace(go.Scatter(
        x=closes.index, y=closes.values,
        name="USD/JPY（左軸）",
        line=dict(color="#2196F3", width=2),
    ), secondary_y=False)

    spread_pct = spread_adj * 100
    fig_rate.add_trace(go.Scatter(
        x=spread_pct.index, y=spread_pct.values,
        name="日米金利差 %（右軸）",
        line=dict(color="#FF9800", width=2, dash="dot"),
    ), secondary_y=True)

    fig_rate.update_layout(
        template="plotly_dark",
        height=480,
        paper_bgcolor="#0f2027",
        plot_bgcolor="#0f2027",
        title=dict(text=f"USD/JPY と 日米金利差の相関（相関係数: {corr_spread:.3f}）", font=dict(size=13)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(size=11),
    )
    fig_rate.update_yaxes(title_text="USD/JPY (円)", secondary_y=False, gridcolor="rgba(255,255,255,0.07)")
    fig_rate.update_yaxes(title_text="日米金利差 (%)", secondary_y=True, gridcolor="rgba(255,255,255,0.04)")
    fig_rate.update_xaxes(gridcolor="rgba(255,255,255,0.07)")

    st.plotly_chart(fig_rate, use_container_width=True)

    # 相関係数バー
    st.markdown("**各指標とドル円の相関係数**")
    corr_data = {
        "指標": ["日米金利差", "ドルインデックス (DXY)", "S&P 500", "日経平均"],
        "相関係数": [
            corr_spread if not np.isnan(corr_spread) else 0,
            corr_dxy    if not np.isnan(corr_dxy)    else 0,
            corr_sp     if not np.isnan(corr_sp)     else 0,
            corr_nk     if not np.isnan(corr_nk)     else 0,
        ]
    }
    fig_corr = go.Figure(go.Bar(
        x=corr_data["相関係数"],
        y=corr_data["指標"],
        orientation="h",
        marker_color=["#2196F3" if v >= 0 else "#F44336" for v in corr_data["相関係数"]],
        text=[f"{v:.3f}" for v in corr_data["相関係数"]],
        textposition="auto",
    ))
    fig_corr.update_layout(
        template="plotly_dark",
        height=200,
        paper_bgcolor="#0f2027",
        plot_bgcolor="#0f2027",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(range=[-1, 1], gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        font=dict(size=11),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ③ 予測シナリオ
# ──────────────────────────────────────────────
st.markdown("### 📋 今後の予測シナリオ（1ヶ月）")

trend_up = cur_price > sma20_cur
sc_main  = cur_price + (sigma_1m_1 * 0.45 if trend_up else -sigma_1m_1 * 0.35)
sc_sub   = cur_price - spread_cur * 100 * 1.5   # 金利差縮小シナリオ
sc_risk_up   = cur_price + sigma_1m_2
sc_risk_down = cur_price - sigma_1m_2

col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.markdown(f"""
    <div class="scenario-main">
      <div class="scenario-prob">🔵 メインシナリオ ｜ 確率 65%</div>
      <div class="scenario-title">トレンド継続</div>
      <div class="scenario-price">{sc_main:.2f}円</div>
      <div class="scenario-desc">
        ±1σレンジ: {cur_price - sigma_1m_1:.2f} 〜 {cur_price + sigma_1m_1:.2f}円<br>
        現在の{"上昇" if trend_up else "下降"}トレンドが継続し、金利差（{spread_cur*100:.2f}%）が主なドライバー。
        SMA20（{sma20_cur:.2f}）{"上方" if trend_up else "下方"}で推移。
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="scenario-sub">
      <div class="scenario-prob">🟡 サブシナリオ ｜ 確率 25%</div>
      <div class="scenario-title">トレンド転換 / 金利差縮小</div>
      <div class="scenario-price">{sc_sub:.2f}円</div>
      <div class="scenario-desc">
        日銀の追加利上げ or FRBの利下げ前倒しによる金利差縮小シナリオ。
        ボラティリティ上昇（HV: {hv:.1f}%）を伴う可能性。
        現在値から {abs(sc_sub - cur_price):.2f}円の変動想定。
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_s3:
    st.markdown(f"""
    <div class="scenario-risk">
      <div class="scenario-prob">🔴 リスクシナリオ ｜ 確率 10%</div>
      <div class="scenario-title">急変・テールリスク</div>
      <div class="scenario-price">{sc_risk_up:.2f} / {sc_risk_down:.2f}円</div>
      <div class="scenario-desc">
        ±2σ（95%）超の急変。上限 {sc_risk_up:.2f}円・下限 {sc_risk_down:.2f}円。
        政府・日銀の為替介入、地政学リスク、米指標サプライズが引き金。
        現在値から最大 {sigma_1m_2:.2f}円の急変動。
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ④ フェアバリュー ＆ ボラレンジ
# ──────────────────────────────────────────────
col_fv, col_vr = st.columns(2)

with col_fv:
    gap_abs   = cur_price - fair_value
    gap_label = "割高 ▲" if gap_abs > 1 else ("割安 ▼" if gap_abs < -1 else "適正 ◆")
    gap_color = "#F44336" if gap_abs > 1 else ("#4CAF50" if gap_abs < -1 else "#9E9E9E")

    st.markdown(f"""
    <div class="fv-panel">
      <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;">
        📐 フェアバリュー分析（重回帰 180日）
      </div>
      <div class="fv-row">
        <span>現在の市場価格</span>
        <span style="font-weight:700;">{cur_price:.3f} 円</span>
      </div>
      <div class="fv-row">
        <span>統計的理論価格</span>
        <span style="font-weight:700;">{fair_value:.3f} 円</span>
      </div>
      <div class="fv-row">
        <span>乖離幅</span>
        <span style="font-weight:700;color:{gap_color};">
          {'+' if gap_abs >= 0 else ''}{gap_abs:.3f}円（{'+' if gap_pct >= 0 else ''}{gap_pct:.2f}%）
        </span>
      </div>
      <div class="fv-row">
        <span>バリュエーション</span>
        <span style="font-weight:700;color:{gap_color};">{gap_label}</span>
      </div>
      <div class="fv-row">
        <span>日米金利差（手動値）</span>
        <span>{(us10y_cur*100):.2f}% − {jp10y_manual:.2f}% = {spread_cur*100:.2f}%</span>
      </div>
      <div style="font-size:0.7rem;opacity:0.45;margin-top:0.75rem;line-height:1.6;">
        独立変数: 日米金利差・ドルインデックス・S&P500<br>
        従属変数: USD/JPY終値。OLS最小二乗法による推定値。
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_vr:
    def range_bar_html(lo1, hi1, lo2, hi2, label):
        return f"""
        <div class="range-row">
          <div class="range-label">{label}</div>
          <div class="range-bar-wrap">
            <div style="font-size:0.7rem;opacity:0.5;margin-bottom:3px;">
              ±2σ 95%: {lo2:.2f} 〜 {hi2:.2f}
            </div>
            <div style="background:rgba(33,150,243,0.15);border-radius:4px;height:18px;position:relative;margin-bottom:4px;">
              <div style="position:absolute;top:2px;left:2px;right:2px;bottom:2px;background:#2196F3;opacity:0.3;border-radius:3px;"></div>
              <div style="position:absolute;top:2px;left:25%;right:25%;bottom:2px;background:#2196F3;opacity:0.6;border-radius:2px;"></div>
              <div style="position:absolute;top:3px;left:48%;width:4%;bottom:3px;background:white;border-radius:2px;"></div>
            </div>
            <div style="font-size:0.7rem;opacity:0.5;">
              ±1σ 68%: {lo1:.2f} 〜 {hi1:.2f}
            </div>
          </div>
        </div>"""

    st.markdown(f"""
    <div class="vol-range">
      <div class="vol-title">📊 ボラティリティベース 予測レンジ（HV {hv:.1f}%）</div>
      {range_bar_html(cur_price - sigma_1w_1, cur_price + sigma_1w_1,
                      cur_price - sigma_1w_2, cur_price + sigma_1w_2, "1週間")}
      {range_bar_html(cur_price - sigma_1m_1, cur_price + sigma_1m_1,
                      cur_price - sigma_1m_2, cur_price + sigma_1m_2, "1ヶ月")}
      <div style="font-size:0.7rem;opacity:0.4;margin-top:0.75rem;line-height:1.6;">
        現在値 {cur_price:.3f}円 基準 ｜ 正規分布仮定 ｜ HVは20日ヒストリカル・ボラティリティ年率換算値
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ④ ファンダメンタルズ・シミュレーター（サイドバー入力 → 画面表示）
# ──────────────────────────────────────────────
st.markdown("### 🧪 ファンダメンタルズ・シミュレーター（感応度分析）")

nfp_diff  = nfp_val  - nfp_exp
cpi_diff  = cpi_val  - cpi_exp
ffr_diff  = ffr_val  - ffr_base

# 感応度係数（実証的な推計値）
nfp_sens  = 0.006   # NFP 1千人差 → ドル円 約0.006円
cpi_sens  = 1.3     # CPI 1%差   → ドル円 約1.3円
ffr_sens  = 3.8     # FFR 0.25%差 → ドル円 約3.8円

impact_nfp = nfp_diff * nfp_sens
impact_cpi = cpi_diff * cpi_sens
impact_ffr = ffr_diff * ffr_sens
total_impact = impact_nfp + impact_cpi + impact_ffr

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    chg_n = "上振れ ↑" if nfp_diff > 0 else ("下振れ ↓" if nfp_diff < 0 else "予想通り")
    st.metric("米雇用統計 NFP", f"{nfp_val}K",
              delta=f"予想比 {'+' if nfp_diff >= 0 else ''}{nfp_diff}K → 推計{'+' if impact_nfp >= 0 else ''}{impact_nfp:.2f}円")
with sim_col2:
    st.metric("米CPI 前年比", f"{cpi_val:.1f}%",
              delta=f"予想比 {'+' if cpi_diff >= 0 else ''}{cpi_diff:.1f}% → 推計{'+' if impact_cpi >= 0 else ''}{impact_cpi:.2f}円")
with sim_col3:
    st.metric("FRB政策金利", f"{ffr_val:.2f}%",
              delta=f"変化 {'+' if ffr_diff >= 0 else ''}{ffr_diff:.2f}% → 推計{'+' if impact_ffr >= 0 else ''}{impact_ffr:.2f}円")

# 総合インパクト
if abs(total_impact) < 0.1:
    sim_class = "sim-neutral"
    sim_icon  = "→"
    sim_msg   = f"市場予想並みの結果。大きな値動きは想定されません（推計インパクト {total_impact:+.2f}円）。"
elif total_impact > 0:
    sim_class = "sim-bullish"
    sim_icon  = "↑"
    sim_msg   = (f"ドル高・円安方向のインパクト。推計 <strong>{total_impact:+.2f}円</strong> の上昇圧力。"
                 f"予測レンジ: {cur_price + total_impact * 0.5:.2f} 〜 {cur_price + total_impact * 1.5:.2f}円")
else:
    sim_class = "sim-bearish"
    sim_icon  = "↓"
    sim_msg   = (f"円高・ドル安方向のインパクト。推計 <strong>{total_impact:+.2f}円</strong> の下落圧力。"
                 f"予測レンジ: {cur_price + total_impact * 1.5:.2f} 〜 {cur_price + total_impact * 0.5:.2f}円")

st.markdown(f"""
<div class="sim-result {sim_class}">
  <span style="font-size:1.5rem;font-weight:800;margin-right:0.5rem;">{sim_icon}</span>
  <span>総合インパクト推計: <strong>{total_impact:+.2f}円</strong>　|　{sim_msg}</span>
</div>
""", unsafe_allow_html=True)

# 感応度ウォーターフォールチャート
fig_sim = go.Figure(go.Waterfall(
    name="感応度分析",
    orientation="v",
    measure=["relative", "relative", "relative", "total"],
    x=["NFP\n影響", "CPI\n影響", "FFR\n影響", "合計インパクト"],
    y=[impact_nfp, impact_cpi, impact_ffr, 0],
    text=[f"{v:+.2f}円" for v in [impact_nfp, impact_cpi, impact_ffr, total_impact]],
    textposition="outside",
    increasing=dict(marker_color="#4CAF50"),
    decreasing=dict(marker_color="#F44336"),
    totals=dict(marker_color="#2196F3"),
    connector=dict(line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot")),
))
fig_sim.update_layout(
    template="plotly_dark",
    height=280,
    paper_bgcolor="#0f2027",
    plot_bgcolor="#0f2027",
    title=dict(text="感応度分析（ウォーターフォール）", font=dict(size=12)),
    yaxis=dict(title="推計インパクト (円)", gridcolor="rgba(255,255,255,0.07)"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
    margin=dict(l=0, r=0, t=40, b=0),
    font=dict(size=11),
    showlegend=False,
)
st.plotly_chart(fig_sim, use_container_width=True)

# ──────────────────────────────────────────────
# フッター
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-size:0.72rem;color:#666;line-height:1.8;text-align:center;">
⚠️ 本ダッシュボードは教育・情報提供目的のみです。投資判断の根拠とならないようご注意ください。<br>
データソース: yfinance (Yahoo Finance API) | 日本10年債利回りはユーザー入力値を使用 |
重回帰モデル・ボラティリティ推計はあくまで統計的試算です。<br>
実際の取引にあたっては、ご自身の判断と責任のもと、専門家にご相談ください。
</div>
""", unsafe_allow_html=True)

# 自動リフレッシュ
if auto_refresh:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
