import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import requests
import re
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="주도주 포착 및 차익실현 분석기 (Quant Ver.)", layout="wide")

with st.sidebar:
    if st.button("🔄 종목 데이터 리셋 (오류시 클릭)", help="네트워크 오류나 캐시 꼬임 발생 시 눌러주세요."):
        st.cache_data.clear()
        st.success("캐시가 초기화되었습니다. 다시 분석을 실행해 주세요!")

# --- 데이터 캐싱 및 로드 ---
@st.cache_data(ttl=86400, show_spinner="증권사 서버에서 최신 종목 목록을 가져오는 중입니다...")
def load_stock_listings():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        krx = fdr.StockListing('KRX-DESC') 
    except Exception:
        try:
            url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download"
            res = requests.get(url, headers=headers, timeout=10)
            krx = pd.read_html(res.text, header=0)[0]
            krx = krx.rename(columns={'회사명': 'Name', '종목코드': 'Code'})
            krx['Code'] = krx['Code'].astype(str).str.zfill(6)
        except Exception:
            krx = pd.DataFrame(columns=['Name', 'Code'])

    try:
        nasdaq = fdr.StockListing('NASDAQ')
        nyse = fdr.StockListing('NYSE')
        us_df = pd.concat([nasdaq, nyse])
    except Exception:
        us_df = pd.DataFrame(columns=['Name', 'Symbol'])
    
    krx_dict = dict(zip(krx['Name'], krx['Code'])) if not krx.empty else {}
    krx_code_dict = dict(zip(krx['Code'], krx['Name'])) if not krx.empty else {}
    us_dict = dict(zip(us_df['Name'], us_df['Symbol'])) if not us_df.empty else {}
    us_code_dict = dict(zip(us_df['Symbol'], us_df['Name'])) if not us_df.empty else {}
    
    return krx_dict, krx_code_dict, us_dict, us_code_dict

krx_dict, krx_code_dict, us_dict, us_code_dict = load_stock_listings()

def search_naver_ticker(name):
    try:
        url = f"https://ac.finance.naver.com/ac?q={name}&q_enc=utf-8&st=111&se=1&tx=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=3)
        data = res.json()
        if data.get('items') and len(data['items'][0]) > 0:
            return data['items'][0][0][1]
    except Exception:
        pass
    return None

def parse_tickers(input_text, market):
    raw_list = re.split(r'[\n,\t\s]+', input_text.strip())
    raw_list = [x for x in raw_list if x]
    parsed = []
    for item in raw_list:
        ticker, name = item, item
        if market == '한국 (KRX)':
            if item in krx_dict:
                ticker, name = krx_dict[item], item
            elif item in krx_code_dict:
                name, ticker = krx_code_dict[item], item
            else:
                live_code = search_naver_ticker(item)
                if live_code and live_code.isdigit():
                    ticker, name = live_code, item
                else:
                    name = "국내종목"
        else:
            item = item.upper()
            if item in us_code_dict:
                name, ticker = us_code_dict[item], item
            elif item in us_dict:
                ticker, name = us_dict[item], item
            else:
                name = "미국종목"
        parsed.append({'name': name, 'ticker': ticker})
    return parsed

def calculate_streak_averages(df):
    if df.empty or 'Close' not in df.columns:
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

def get_market_index(market, start_date, end_date):
    if market == '한국 (KRX)':
        df = fdr.DataReader('KS11', start_date, end_date)
        name = "KOSPI"
    else:
        df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        name = "S&P 500"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    if df is not None and not df.empty and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return name, df

def get_vix(start_date, end_date):
    vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    if not vix_df.empty:
        return float(vix_df['Close'].iloc[-1])
    return 20.0 # 기본값

# --- UI 레이아웃 ---
st.title("📊 퀀트 기반 시장 주도주 및 차익실현 분석기")
st.caption("단순 추세를 넘어 RSI, 200일 이격률, 거래량 및 매크로 공포지수(VIX)를 융합하여 조정 확률을 진단합니다.")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    
    default_input = "네이버, 005930, 카카오\nSK하이닉스" if market_choice == '한국 (KRX)' else "TEAM, AAPL, MSFT"
    stock_input = st.text_area("종목 입력 (이름, 티커, 종목코드 혼용)", value=default_input)
    
    st.subheader("기간 설정")
    st.caption("⚠️ 200일 이격률 계산을 위해 시작일은 최소 1년 이전으로 설정하세요.")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("전체 시작일", datetime.today() - timedelta(days=730)) # 2년으로 확장
    with col2:
        end_date = st.date_input("전체 종료일", datetime.today())
        
    st.subheader("부분 범위 확인 (서브 레인지)")
    sub_start = st.date_input("부분 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("부분 종료일", datetime.today())
    
    run_btn = st.button("🚀 정밀 분석 실행", type='primary', use_container_width=True)

# --- 분석 로직 ---
if run_btn:
    if not stock_input:
        st.warning("종목을 입력해 주세요.")
        st.stop()
        
    parsed_stocks = parse_tickers(stock_input, market_choice)
    market_name, market_df = get_market_index(market_choice, start_date, end_date)
    current_vix = get_vix(start_date, end_date)
    
    sub_start_dt = pd.to_datetime(sub_start)
    sub_end_dt = pd.to_datetime(sub_end)
    
    st.subheader("🌐 매크로 시장 환경 진단")
    if market_df is not None and not market_df.empty:
        m_avg_up, m_avg_down, m_curr_ret, m_curr_sign = calculate_streak_averages(market_df)
        
        # 시장 200일 이격률 계산
        market_df['200MA'] = market_df['Close'].rolling(window=200).mean()
        m_disparity = float(market_df['Close'].iloc[-1] / market_df['200MA'].iloc[-1] - 1) if not pd.isna(market_df['200MA'].iloc[-1]) else 0.0
        
        m_sub_df = market_df[(market_df.index >= sub_start_dt) & (market_df.index <= sub_end_dt)]
        m_sub_ret = float(m_sub_df['Close'].iloc[-1] / m_sub_df['Close'].iloc[0] - 1) if len(m_sub_df) > 1 else 0.0
        
        # 시장 과열도 스코어링 (VIX 및 이격률 기반)
        market_status = "🟢 안정적 상승장"
        if current_vix < 15 and m_disparity > 0.10:
            market_status = "🚨 극단적 탐욕 (조정 임박)"
        elif current_vix > 30 and m_disparity < -0.10:
            market_status = "💡 극단적 공포 (반등 임박)"
        elif m_curr_sign == -1:
            market_status = "⚠️ 하락 추세 진행중"
            
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(f"{market_name} 200일 이격률", f"{m_disparity*100:+.2f}%", help="+10% 이상 과열, -10% 이하 침체")
        col_m2.metric("글로벌 공포지수 (VIX)", f"{current_vix:.2f}", help="25 이상 공포(매수기회), 15 이하 탐욕(차익실현)")
        sign_text = "상승중" if m_curr_sign == 1 else ("하락중" if m_curr_sign == -1 else "보합")
        col_m3.metric(f"부분 구간 수익률", f"{m_sub_ret*100:+.2f}%")
        col_m4.metric("현재 시장 상태", market_status)
    else:
        st.warning("시장 지수 데이터를 불러올 수 없습니다.")

    st.divider()
    st.subheader("🎯 개별 종목 차익실현 및 주도주 분석")
    
    results = []
    progress_bar = st.progress(0)
    failed_stocks = []
    
    for i, stock in enumerate(parsed_stocks):
        t_name = stock['name']
        t_code = stock['ticker']
        
        try:
            if market_choice == '한국 (KRX)':
                df = fdr.DataReader(t_code, start_date, end_date)
            else:
                df = yf.download(t_code, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            
            if df.empty:
                failed_stocks.append(f"{t_name}({t_code})")
                progress_bar.progress((i + 1) / len(parsed_stocks))
                continue
                
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)' and t_code in krx_code_dict:
                t_name = krx_code_dict[t_code]
            
            display_name = f"{t_name} ({t_code})"
                
            s_avg_up, s_avg_down, s_curr_ret, s_curr_sign = calculate_streak_averages(df)
            
            # 1. 200일 이격률 계산 (+10% 이상 고평가, -10% 이하 저평가)
            df['200MA'] = df['Close'].rolling(window=200).mean()
            disparity_200 = float(df['Close'].iloc[-1] / df['200MA'].iloc[-1] - 1) if not pd.isna(df['200MA'].iloc[-1]) else 0.0
            
            # 2. RSI (14일) 계산 (70 이상 고평가, 30 이하 저평가)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
            
            # 3. 거래량 배수 및 수익률
            df['Vol_20MA'] = df['Volume'].rolling(window=20).mean()
            vol_ratio = float(df['Volume'].iloc[-1] / df['Vol_20MA'].iloc[-1]) if df['Vol_20MA'].iloc[-1] > 0 else 1.0
            
            sub_df = df[(df.index >= sub_start_dt) & (df.index <= sub_end_dt)]
            sub_ret = float(sub_df['Close'].iloc[-1] / sub_df['Close'].iloc[0] - 1) if len(sub_df) > 1 else 0.0
            
            stock_20d_ret = float(df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0.0
            market_20d_ret = float(market_df['Close'].iloc[-1] / market_df['Close'].iloc[-20] - 1) if (market_df is not None and len(market_df) >= 20) else 0.0
            relative_strength = stock_20d_ret - market_20d_ret
            
            # 🔴 종합 퀀트 진단 로직 (사진 지표 반영)
            score = 0
            # 과열 시그널
            if current_rsi >= 70: score -= 1
            if disparity_200 >= 0.10: score -= 1
            if s_curr_sign == 1 and s_curr_ret >= s_avg_up * 0.9: score -= 1
            # 바닥 시그널
            if current_rsi <= 30: score += 1
            if disparity_200 <= -0.10: score += 1
            if s_curr_sign == -1 and s_curr_ret <= s_avg_down * 0.9: score += 1
            
            action = "관망 (중립)"
            if score <= -2:
                action = "🚨 강력 차익실현 (고평가/과열)"
            elif score == -1:
                action = "⚠️ 분할 매도 고려"
            elif score >= 2:
                action = "🟢 적극 매수 (저평가/바닥)"
            elif score == 1:
                action = "💡 저점 매수 모니터링"
                
            is_leader = "🔥 주도주" if relative_strength > 0.05 and vol_ratio > 1.5 and disparity_200 > 0 else "-"
            
            results.append({
                "종목명(티커)": display_name,
                "퀀트 진단결과": action,
                "주도주 여부": is_leader,
                "200일 이격률": f"{disparity_200*100:+.1f}%",
                "RSI (14)": f"{current_rsi:.1f}",
                "현재 연속 수익률": f"{s_curr_ret*100:+.2f}%",
                "부분 구간 수익률": f"{sub_ret*100:+.2f}%",
                "거래량(20MA)": f"{vol_ratio:.1f}x",
                "시장대비 상대수익": f"{relative_strength*100:+.1f}%"
            })
        except Exception as e:
            failed_stocks.append(f"{t_name}({t_code})")
            
        progress_bar.progress((i + 1) / len(parsed_stocks))
        
    if failed_stocks:
        st.warning(f"⚠️ 다음 종목들은 서버 오류 또는 티커 인식 실패로 제외되었습니다: {', '.join(failed_stocks)}")
        
    if results:
        res_df = pd.DataFrame(results)
        # 진단 결과가 잘 보이도록 스타일링 적용
        st.dataframe(res_df, use_container_width=True)
    else:
        st.error("조회 가능한 주식 데이터가 없습니다. 좌측 메뉴의 [🔄 종목 데이터 리셋] 버튼을 눌러 캐시를 초기화한 후 다시 시도해 주세요.")
