import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import requests
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- 페이지 설정 ---
st.set_page_config(page_title="주도주 추세 & 퀀트 지표 총망라 분석기", layout="wide")

with st.sidebar:
    if st.button("🔄 종목 데이터 리셋 (오류시 클릭)"):
        st.cache_data.clear()
        st.success("캐시가 초기화되었습니다. 다시 분석을 실행해 주세요!")

# --- 데이터 캐싱 및 로드 ---
@st.cache_data(ttl=86400, show_spinner="데이터를 불러오는 중입니다...")
def load_stock_listings():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        krx = fdr.StockListing('KRX-DESC') 
    except:
        try:
            url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
            res = requests.get(url, headers=headers, timeout=10)
            krx = pd.read_html(res.text, header=0)[0]
            krx = krx.rename(columns={'회사명': 'Name', '종목코드': 'Code'})
            krx['Code'] = krx['Code'].astype(str).str.zfill(6)
        except:
            krx = pd.DataFrame(columns=['Name', 'Code'])

    try:
        us_df = pd.concat([fdr.StockListing('NASDAQ'), fdr.StockListing('NYSE')])
    except:
        us_df = pd.DataFrame(columns=['Name', 'Symbol'])
    
    krx_dict = dict(zip(krx['Name'], krx['Code'])) if not krx.empty else {}
    krx_code_dict = dict(zip(krx['Code'], krx['Name'])) if not krx.empty else {}
    us_dict = dict(zip(us_df['Name'], us_df['Symbol'])) if not us_df.empty else {}
    us_code_dict = dict(zip(us_df['Symbol'], us_df['Name'])) if not us_df.empty else {}
    
    return krx_dict, krx_code_dict, us_dict, us_code_dict

krx_dict, krx_code_dict, us_dict, us_code_dict = load_stock_listings()

def search_naver_ticker(name):
    try:
        res = requests.get(f"https://ac.finance.naver.com/ac?q={name}&q_enc=utf-8&st=111&se=1&tx=0", headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        data = res.json()
        if data.get('items') and len(data['items'][0]) > 0:
            return data['items'][0][0][1]
    except: pass
    return None

def parse_tickers(input_text, market):
    raw_list = [x for x in re.split(r'[\n,\t\s]+', input_text.strip()) if x]
    parsed = []
    for item in raw_list:
        ticker, name = item, item
        if market == '한국 (KRX)':
            if item in krx_dict: ticker, name = krx_dict[item], item
            elif item in krx_code_dict: name, ticker = krx_code_dict[item], item
            else:
                live_code = search_naver_ticker(item)
                if live_code and live_code.isdigit(): ticker, name = live_code, item
                else: name = "국내종목"
        else:
            item = item.upper()
            if item in us_code_dict: name, ticker = us_code_dict[item], item
            elif item in us_dict: ticker, name = us_dict[item], item
            else: name = "미국종목"
        parsed.append({'name': name, 'ticker': ticker})
    return parsed

def get_market_index(market, start_date, end_date):
    if market == '한국 (KRX)': df = fdr.DataReader('KS11', start_date, end_date); name = "KOSPI"
    else:
        df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        name = "S&P 500"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df is not None and not df.empty and df.index.tz is not None: df.index = df.index.tz_localize(None)
    return name, df

def get_vix_full(start_date, end_date):
    vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.get_level_values(0)
    if not vix_df.empty and vix_df.index.tz is not None: vix_df.index = vix_df.index.tz_localize(None)
    return vix_df

def calculate_fear_greed_score(vix, market_rsi, market_20ma_disparity):
    v_score = max(0, min(100, (40 - vix) / 30 * 100))
    r_score = max(0, min(100, market_rsi))
    d_score = max(0, min(100, (market_20ma_disparity + 0.05) / 0.10 * 100))
    return (v_score + r_score + d_score) / 3

def calculate_streak_averages(df):
    if df.empty or 'Close' not in df.columns or len(df) < 2:
        return 0.0, 0.0, 0.0, 0
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    if df.empty: return 0.0, 0.0, 0.0, 0
    
    df['Sign'] = np.sign(df['Return'].round(4))
    df['Streak_ID'] = (df['Sign'] != df['Sign'].shift(1)).cumsum()
    streak_returns = df.groupby('Streak_ID').apply(lambda x: float((1 + x['Return']).prod() - 1), include_groups=False).to_frame(name='Cumulative_Return')
    streak_returns['Sign'] = df.groupby('Streak_ID')['Sign'].first()
    
    avg_up = float(streak_returns[streak_returns['Sign'] == 1]['Cumulative_Return'].mean()) if not streak_returns[streak_returns['Sign'] == 1].empty else 0.0
    avg_down = float(streak_returns[streak_returns['Sign'] == -1]['Cumulative_Return'].mean()) if not streak_returns[streak_returns['Sign'] == -1].empty else 0.0
    curr_ret = float(streak_returns.iloc[-1]['Cumulative_Return']) if not streak_returns.empty else 0.0
    curr_sign = int(streak_returns.iloc[-1]['Sign']) if not streak_returns.empty else 0
    return avg_up, avg_down, curr_ret, curr_sign

# --- UI 레이아웃 ---
st.title("🔥 대세 주도주 & 퀀트 지표 총망라 분석기")
st.caption("과열 지표(RSI/이격률/볼린저밴드), 연속 상승률, 수급(OBV), 그리고 장전 갭(Gap) 패턴을 모두 종합하여 최적의 타점을 분석합니다.")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    default_input = "005930, 000660" if market_choice == '한국 (KRX)' else "MU, NVDA, AAPL"
    stock_input = st.text_area("종목 입력", value=default_input)
    
    st.subheader("기간 설정")
    st.caption("200일선 계산을 위해 전체 기간은 1년 이상 유지하세요.")
    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("전체 시작일", datetime.today() - timedelta(days=500))
    with col2: end_date = st.date_input("전체 종료일", datetime.today())
        
    st.divider()
    sub_start = st.date_input("부분 분석 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("📌 기준일(종료일)", datetime.today())
    run_btn = st.button("🚀 전체 지표 분석 실행", type='primary', use_container_width=True)

# --- 분석 로직 ---
if run_btn:
    if not stock_input:
        st.warning("종목을 입력해 주세요.")
        st.stop()
        
    parsed_stocks = parse_tickers(stock_input, market_choice)
    market_name, market_df_full = get_market_index(market_choice, start_date, end_date)
    vix_df_full = get_vix_full(start_date, end_date)
    
    sub_start_dt = pd.to_datetime(sub_start)
    sub_end_dt = pd.to_datetime(sub_end)
    
    st.subheader(f"🌐 시장 매크로 공포탐욕 지수 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
    if market_df_full is not None and not market_df_full.empty:
        m_df_as_of = market_df_full[market_df_full.index <= sub_end_dt].copy()
        vix_as_of = vix_df_full[vix_df_full.index <= sub_end_dt].copy() if not vix_df_full.empty else None
        
        if not m_df_as_of.empty:
            m_df_as_of['20MA'] = m_df_as_of['Close'].rolling(window=20).mean()
            m_disp_20 = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['20MA'].iloc[-1] - 1) if not pd.isna(m_df_as_of['20MA'].iloc[-1]) else 0.0
            
            m_delta = m_df_as_of['Close'].diff()
            m_rs = (m_delta.where(m_delta > 0, 0)).rolling(window=14).mean() / (-m_delta.where(m_delta < 0, 0)).rolling(window=14).mean()
            current_m_rsi = float(100 - (100 / (1 + m_rs)).iloc[-1]) if len(m_df_as_of) >= 15 else 50.0
            current_vix = float(vix_as_of['Close'].iloc[-1]) if vix_as_of is not None and not vix_as_of.empty else 20.0
            
            fg_score = calculate_fear_greed_score(current_vix, current_m_rsi, m_disp_20)
            fg_status = "중립 (추세 지속 가능)"
            if fg_score <= 25: fg_status = "😨 극단적 공포 (단기 낙폭 과대 / 줍줍)"
            elif fg_score <= 45: fg_status = "📉 공포 (반등 모색 구간)"
            elif fg_score >= 80: fg_status = "🚨 단기 과열 극심 (조정 임박)"
            elif fg_score >= 60: fg_status = "📈 탐욕 (추가 상승 여력 있으나 주의)"
            
            st.progress(int(fg_score), text=f"🔥 탐욕(100) ↔ 🧊 공포(0) | 현재 스코어: {fg_score:.1f}점 ({fg_status})")
            
    st.divider()
    st.subheader(f"🎯 개별 종목 지표 총망라 및 전략 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
    results = []
    progress_bar = st.progress(0)
    failed_stocks = []
    
    for i, stock in enumerate(parsed_stocks):
        t_name, t_code = stock['name'], stock['ticker']
        
        try:
            if market_choice == '한국 (KRX)': df_full = fdr.DataReader(t_code, start_date, end_date)
            else:
                df_full = yf.download(t_code, start=start_date, end=end_date, progress=False)
                if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
            
            if df_full.empty:
                failed_stocks.append(f"{t_name}({t_code})")
                continue
                
            if df_full.index.tz is not None: df_full.index = df_full.index.tz_localize(None)
            df_as_of = df_full[df_full.index <= sub_end_dt].copy()
            if len(df_as_of) < 20: continue # 최소 데이터 보호
            
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)' and t_code in krx_code_dict:
                t_name = krx_code_dict[t_code]
            
            display_name = f"{t_name} ({t_code})"
            
            # --- 1. OHLC 및 갭(Gap) 분석 ---
            today_open = float(df_as_of['Open'].iloc[-1])
            today_close = float(df_as_of['Close'].iloc[-1])
            prev_close = float(df_as_of['Close'].iloc[-2]) if len(df_as_of) > 1 else today_open
            
            gap_pct = (today_open - prev_close) / prev_close
            intraday_pct = (today_close - today_open) / today_open
            is_up_day = today_close > prev_close 
            is_yin_candle = today_close < today_open 
            
            # --- 2. 이동평균선 및 이격률 분석 ---
            df_as_of['5MA'] = df_as_of['Close'].rolling(window=5).mean()
            df_as_of['20MA'] = df_as_of['Close'].rolling(window=20).mean()
            df_as_of['120MA'] = df_as_of['Close'].rolling(window=120).mean()
            df_as_of['200MA'] = df_as_of['Close'].rolling(window=200).mean()
            
            is_above_5ma = today_close > df_as_of['5MA'].iloc[-1]
            long_term_bull = len(df_as_of) >= 120 and (today_close > df_as_of['120MA'].iloc[-1]) and (df_as_of['120MA'].iloc[-1] > df_as_of['120MA'].iloc[-20])
            
            disparity_20 = float(today_close / df_as_of['20MA'].iloc[-1] - 1) if not pd.isna(df_as_of['20MA'].iloc[-1]) else 0.0
            disparity_200 = float(today_close / df_as_of['200MA'].iloc[-1] - 1) if len(df_as_of) >= 200 and not pd.isna(df_as_of['200MA'].iloc[-1]) else 0.0
            
            # --- 3. 볼린저 밴드, 거래량, OBV, RSI ---
            df_as_of['20STD'] = df_as_of['Close'].rolling(window=20).std()
            df_as_of['BB_Upper'] = df_as_of['20MA'] + (df_as_of['20STD'] * 2)
            df_as_of['BB_Lower'] = df_as_of['20MA'] - (df_as_of['20STD'] * 2)
            
            bb_upper = df_as_of['BB_Upper'].iloc[-1]
