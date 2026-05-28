import streamlit as st
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier, XGBRegressor
import plotly.graph_objects as go
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI 量化指揮中心", page_icon="📈", layout="wide")

# ==========================================
# 🔒 密碼登入系統
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "8888":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 系統已鎖定")
        st.info("請輸入專屬通關密碼，以啟用 AI 量化分析系統。")
        st.text_input("輸入密碼：", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 系統已鎖定")
        st.text_input("輸入密碼：", type="password", on_change=password_entered, key="password")
        st.error("❌ 密碼錯誤，請重新輸入！")
        return False
    else:
        return True

if not check_password():
    st.stop()


# ==========================================
# 📈 核心資料處理
# ==========================================
@st.cache_data(ttl=900) # 快取時間縮短為 15 分鐘，確保夜盤抓到最新報價
def get_data(symbol, days):
    ticker = f"{symbol}.TW"
    stock = yf.Ticker(ticker)
    try:
        name = stock.info.get('longName', stock.info.get('shortName', '未知'))
        price = stock.info.get('regularMarketPrice', stock.info.get('currentPrice', None))
    except:
        name, price = "未知名稱", None

    df = stock.history(period="250d").reset_index()
    if df.empty: return df, [], name, price
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.floor('D')
    if price is None: price = df['Close'].iloc[-1]
    
    # 1. 大盤連動 (日盤)
    market = yf.Ticker("^TWII").history(period="250d").reset_index()
    if not market.empty:
        market['Date'] = pd.to_datetime(market['Date']).dt.tz_localize(None).dt.floor('D')
        market = market[['Date', 'Close']].rename(columns={'Close': 'Market_Close'})
        df = df.merge(market, on='Date', how='left').ffill()
        df['Market_Return'] = df['Market_Close'].pct_change()
        df['Market_MA5'] = df['Market_Close'].rolling(window=5).mean()
        df['Market_Trend'] = (df['Market_Close'] > df['Market_MA5']).astype(int)
        df['Stock_Return'] = df['Close'].pct_change()
        df['Relative_Strength'] = df['Stock_Return'] - df['Market_Return']
    else:
        df['Market_Return'], df['Market_Trend'], df['Relative_Strength'] = 0, 0, 0

    # 🌟 2. 加入美股與夜盤動能 (TSMC ADR & 那斯達克)
    # 抓取台積電 ADR
    adr = yf.Ticker("TSM").history(period="250d").reset_index()
    if not adr.empty:
        adr['Date'] = pd.to_datetime(adr['Date']).dt.tz_localize(None).dt.floor('D')
        adr = adr[['Date', 'Close']].rename(columns={'Close': 'ADR_Close'})
        df = df.merge(adr, on='Date', how='left').ffill() # 美股休市時沿用前一天
        df['ADR_Return'] = df['ADR_Close'].pct_change()
    else:
        df['ADR_Return'] = 0

    # 抓取那斯達克指數
    nasdaq = yf.Ticker("^IXIC").history(period="250d").reset_index()
    if not nasdaq.empty:
        nasdaq['Date'] = pd.to_datetime(nasdaq['Date']).dt.tz_localize(None).dt.floor('D')
        nasdaq = nasdaq[['Date', 'Close']].rename(columns={'Close': 'NDX_Close'})
        df = df.merge(nasdaq, on='Date', how='left').ffill()
        df['NDX_Return'] = df['NDX_Close'].pct_change()
    else:
        df['NDX_Return'] = 0

    # 3. 籌碼大戶
    start_date_str = (pd.Timestamp.today() - pd.Timedelta(days=250)).strftime('%Y-%m-%d')
    url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={symbol}&start_date={start_date_str}"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('status') == 200 and len(res.get('data', [])) > 0:
            inst_df = pd.DataFrame(res['data'])
            inst_df['Date'] = pd.to_datetime(inst_df['date']).dt.floor('D')
            inst_df['Net_Buy'] = inst_df['buy'] - inst_df['sell']
            inst_grouped = inst_df.groupby('Date')['Net_Buy'].sum().reset_index()
            inst_grouped.rename(columns={'Net_Buy': 'Inst_Net_Buy'}, inplace=True)
            df = df.merge(inst_grouped, on='Date', how='left').fillna(0)
        else:
            df['Inst_Net_Buy'] = 0
    except:
        df['Inst_Net_Buy'] = 0 

    # 4. 技術指標
    for d in [5, 10, 20]: df[f'MA{d}'] = df['Close'].rolling(window=d).mean()
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    df['K'] = ((df['Close'] - low_min) / (high_max - low_min + 1e-5) * 100).ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['ATR'] = df['High'] - df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Inst_Buy_Ratio'] = df['Inst_Net_Buy'] / (df['Volume'] + 1)
    
    # 目標值設定 (隔日漲跌幅)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) 
    df['Target_Price'] = df['Close'].shift(-1)                       
    df['Target_Return'] = (df['Close'].shift(-1) - df['Close']) / df['Close']
    df['Next_Date'] = df['Date'].shift(-1)                           
    
    # 🌟 將夜盤特徵加入訓練清單
    features = [
        'Close', 'Volume', 'MA5', 'MA10', 'MA20', 'K', 'D', 'ATR', 'Volume_Change', 
        'Market_Return', 'Market_Trend', 'Relative_Strength', 
        'Inst_Net_Buy', 'Inst_Buy_Ratio', 
        'ADR_Return', 'NDX_Return'  # <--- 新增這兩個夜盤/美股特徵
    ]
    df = df.dropna(subset=features).reset_index(drop=True)
    
    return df.tail(days).reset_index(drop=True), features, name, round(price, 2)

# ==========================================
# 介面排版與模型運算
# ==========================================
st.sidebar.title("⚙️ AI 指揮中心")
st.sidebar.markdown("結合 **技術面** + **籌碼** + **🌙 夜盤動能**")
st.sidebar.divider()

if st.sidebar.button("🚪 登出系統"):
    st.session_state["password_correct"] = False
    st.rerun()

symbol = st.sidebar.text_input("🔍 輸入台股代號 (例: 2330, 2313)", value="2313")
timeframe = st.sidebar.radio("⏳ 選擇 AI 訓練區間", ("一個月 (30天)", "一季 (90天)", "半年 (180天)"), index=2)
days_dict = {"一個月 (30天)": 30, "一季 (90天)": 90, "半年 (180天)": 180}
selected_days = days_dict[timeframe]

if symbol:
    with st.spinner(f'📡 正在同步台股日盤與美股夜盤數據...'):
        df, features, name, price = get_data(symbol, selected_days)
        
        if len(df) < 10:
            st.error("❌ 擷取到的有效交易日太少，無法訓練模型。")
            st.stop()

        train_df, test_df, latest_df = df.iloc[:-6], df.iloc[-6:-1], df.iloc[-1:]
        
        model_cls = XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        model_cls.fit(train_df[features], train_df['Target'])

        model_reg = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model_reg.fit(train_df[features], train_df['Target_Return'])

        prob = model_cls.predict_proba(latest_df[features])[0]
        pred_trend = 1 if prob[1] > 0.5 else 0
        confidence = max(prob) * 100
        
# 1. 先算出原始的預估漲跌幅
        raw_return = model_reg.predict(latest_df[features])[0]
        
        # 2. 強制同步：保留迴歸器的「振幅大小 (abs)」，但方向無條件服從分類器的「趨勢 (pred_trend)」
        aligned_return = abs(raw_return) if pred_trend == 1 else -abs(raw_return)
        
        # 3. 算出最終價格
        pred_price = latest_df['Close'].values[0] * (1 + aligned_return)

        st.title(f"📊 {name} ({symbol})")
        st.divider()

        # 數據看板區
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("即時收盤價", f"NT$ {price}")
        col2.metric("明日趨勢", "📈 看漲" if pred_trend == 1 else "📉 看跌", delta=f"信心 {confidence:.1f}%", delta_color="normal" if pred_trend==1 else "inverse")
        
        price_diff = pred_price - price
        col3.metric("AI 明日估值", f"NT$ {pred_price:.1f}", delta=f"預估差價: {price_diff:+.1f}", delta_color="normal" if price_diff > 0 else "inverse")
        
        high, low, close = latest_df['High'].values[0], latest_df['Low'].values[0], latest_df['Close'].values[0]
        pivot = (high + low + close) / 3
        res = (2 * pivot) - low
        sup = (2 * pivot) - high
        col4.metric("關鍵轉折價 (Pivot)", f"{pivot:.1f}")

        # 夜盤與籌碼狀態列
        st.subheader(f"🌙 跨時區動能分析 (美股夜盤連動中)")
        
        adr_val = latest_df['ADR_Return'].values[0] * 100
        ndx_val = latest_df['NDX_Return'].values[0] * 100
        
        c1, c2, c3 = st.columns(3)
        c1.info(f"**台積電 ADR (夜盤風向球)**\n\n👉 當前漲跌: **{adr_val:+.2f}%**")
        c2.info(f"**那斯達克指數 (科技股氛圍)**\n\n👉 當前漲跌: **{ndx_val:+.2f}%**")
        
        if adr_val > 0.5 and ndx_val > 0.5:
            night_status = "🔥 夜盤火熱，明日開高機率大"
        elif adr_val < -0.5 and ndx_val < -0.5:
            night_status = "❄️ 夜盤承壓，明日需留意回檔"
        else:
            night_status = "⚖️ 夜盤平穩，依循技術面與籌碼"
            
        c3.success(f"**AI 綜合夜盤判定**\n\n👉 {night_status}")

        st.divider()

        # K線圖
        st.subheader("📉 近期走勢與均線")
        plot_df = df.tail(60)
        fig = go.Figure(data=[go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='K線')])
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA5'], line=dict(color='orange', width=1.5), name='5日均線'))
        fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['MA20'], line=dict(color='blue', width=1.5), name='20日均線'))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # 回測與權重
        with st.expander("📝 展開查看：近 5 日回測對帳單與 AI 決策權重"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**近 5 日回測結果**")
                correct = 0
                for i in range(len(test_df)):
                    r = test_df.iloc[i:i+1]
                    actual_trend = r['Target'].values[0]
                    actual_price = r['Target_Price'].values[0]
                    t_date = pd.to_datetime(r['Next_Date'].values[0]).strftime('%m/%d')
                    
                    p_prob = model_cls.predict_proba(r[features])[0]
                    p_trend = 1 if p_prob[1] > 0.5 else 0
                    p_return = model_reg.predict(r[features])[0]
                    p_price = r['Close'].values[0] * (1 + p_return)
                    
                    if p_trend == actual_trend: correct += 1
                    icon = "✅" if p_trend == actual_trend else "❌"
                    
                    st.write(f"📅 **{t_date}** | 預測: {'漲' if p_trend==1 else '跌'} (估 {p_price:.1f}) | 實際: {'漲' if actual_trend==1 else '跌'} ({actual_price:.1f}) | {icon}")
                
                st.progress(correct/5, text=f"近 5 日趨勢勝率: {(correct/5)*100:.0f}%")
            
            with col_b:
                st.markdown("**AI 決策權重 TOP 5 (檢視夜盤影響力)**")
                imp = pd.DataFrame({'特徵': features, '重要性': model_cls.feature_importances_}).sort_values(by='重要性', ascending=False).head(5)
                st.bar_chart(imp.set_index('特徵'))

st.divider()
st.caption("⚠️ **免責聲明 (Disclaimer)**：本系統提供之所有數據與 AI 估值，均由歷史資料與機器學習演算法自動運算而得，僅供學術研究與投資參考，絕不構成任何買賣邀約或投資建議。投資人應審慎評估風險並自負盈虧。")