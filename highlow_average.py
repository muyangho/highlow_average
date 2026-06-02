import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import re
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="주도주 포착 및 차익실현 분석기", layout="wide")

# --- 데이터 캐싱 및 로드 ---
@st.cache_data(ttl=86400)
def load_stock_listings():
    # 한국 및 미국 주식 리스트 로드
    krx = fdr.StockListing('KRX')
    # US 로드 시 시간이 걸릴 수 있으므로 대표적인 시장만 로드하거나 yfinance에 의존
    nasdaq = fdr.StockListing('NASDAQ')
    nyse = fdr.StockListing('NYSE')
    us_df = pd.concat([nasdaq, nyse])
    
    krx_dict = dict(zip(krx['Name'], krx['Code']))
    krx_code_dict = dict(zip(krx['Code'], krx['Name']))
    
    us_dict = dict(zip(us_df['Name'], us_df['Symbol']))
    us_code_dict = dict(zip(us_df['Symbol'], us_df['Name']))
    
    return krx_dict, krx_code_dict, us_dict, us_code_dict

krx_dict, krx_code_dict, us_dict, us_code_dict = load_stock_listings()

# --- 헬퍼 함수 ---
def parse_tickers(input_text, market):
    # 줄바꿈, 쉼표, 탭, 여러 공백을 단일 공백으로 치환 후 분리
    raw_list = re.split(r'[\n,\t\s]+', input_text.strip())
    raw_list = [x for x in raw_list if x]
    
    parsed = []
    for item in raw_list:
        ticker = item
        name = item
        if market == '한국 (KRX)':
            if item in krx_dict: # 이름으로 입력한 경우
                ticker = krx_dict[item]
            elif item in krx_code_dict: # 코드로 입력한 경우
                name = krx_code_dict[item]
            else:
                ticker = item # 매칭 안되면 그대로
        else: # 미국 주식
            # 티커 대문자 변환
            item = item.upper()
            if item in us_code_dict:
                name = us_code_dict[item]
                ticker = item
            elif item in us_dict:
                ticker = us_dict[item]
            else:
                ticker = item
        parsed.append({'name': name, 'ticker': ticker})
    return parsed

def calculate_streak_averages(df):
    if df.empty or 'Close' not in df.columns:
        return 0, 0, 0, 0
    
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    
    # 방향 설정 (1: 상승, -1: 하락, 0: 보합)
    df['Sign'] = np.sign(df['Return'].round(4))
    # 연속 구간 ID 부여
    df['Streak_ID'] = (df['Sign'] != df['Sign'].shift(1)).cumsum()
    
    # 각 구간별 누적 수익률 계산
    streak_returns = df.groupby('Streak_ID').apply(
        lambda x: (1 + x['Return']).prod() - 1, include_groups=False
    ).to_frame(name='Cumulative_Return')
    
    # Sign 매핑
    streak_signs = df.groupby('Streak_ID')['Sign'].first()
    streak_returns['Sign'] = streak_signs
    
    # 평균 계산
    up_streaks = streak_returns[streak_returns['Sign'] == 1]['Cumulative_Return']
    down_streaks = streak_returns[streak_returns['Sign'] == -1]['Cumulative_Return']
    
    avg_up = up_streaks.mean() if not up_streaks.empty else 0
    avg_down = down_streaks.mean() if not down_streaks.empty else 0
    
    # 현재 진행중인 Streak 계산
    current_streak_return = streak_returns.iloc[-1]['Cumulative_Return'] if not streak_returns.empty else 0
    current_sign = streak_returns.iloc[-1]['Sign'] if not streak_returns.empty else 0
    
    return avg_up, avg_down, current_streak_return, current_sign

def get_market_index(market, start_date, end_date):
    if market == '한국 (KRX)':
        # KOSPI
        df = fdr.DataReader('KS11', start_date, end_date)
        name = "KOSPI 지수"
    else:
        # S&P 500
        df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        name = "S&P 500 지수"
    return name, df

# --- UI 레이아웃 ---
st.title("📊 시장 주도주 포착 및 조정/반등 타이밍 분석기")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    
    # 삼성전자, TEAM 등 종목들을 자연스럽게 기본값으로 배치
    default_input = "삼성전자, SK하이닉스\n카카오" if market_choice == '한국 (KRX)' else "TEAM, AAPL, MSFT"
    stock_input = st.text_area("종목 입력 (이름 또는 티커/종목코드)", value=default_input, help="띄어쓰기, 줄바꿈, 쉼표로 구분 가능합니다.")
    
    st.subheader("기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("전체 시작일", datetime.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("전체 종료일", datetime.today())
        
    st.subheader("부분 범위 확인 (서브 레인지)")
    st.caption("선택한 범위 내에서 특정 구간의 수익률을 확인합니다.")
    sub_start = st.date_input("부분 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("부분 종료일", datetime.today())
    
    run_btn = st.button("분석 실행", type='primary')

# --- 분석 로직 ---
if run_btn:
    if not stock_input:
        st.warning("종목을 입력해 주세요.")
        st.stop()
        
    parsed_stocks = parse_tickers(stock_input, market_choice)
    
    # 1. 시장 지수 분석
    st.subheader("🌐 시장 환경 분석")
    market_name, market_df = get_market_index(market_choice, start_date, end_date)
    
    if market_df is not None and not market_df.empty:
        m_avg_up, m_avg_down, m_curr_ret, m_curr_sign = calculate_streak_averages(market_df)
        
        # 시장의 최신 누적 수익률 계산 (서브 레인지)
        m_sub_df = market_df.loc[sub_start:sub_end]
        m_sub_ret = (m_sub_df['Close'].iloc[-1] / m_sub_df['Close'].iloc[0] - 1) if len(m_sub_df) > 1 else 0
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(f"{market_name} 평균 연속 상승률", f"{m_avg_up*100:.2f}%")
        col_m2.metric("평균 연속 하락률 (조정치)", f"{m_avg_down*100:.2f}%")
        
        sign_text = "상승중" if m_curr_sign == 1 else ("하락중" if m_curr_sign == -1 else "보합")
        col_m3.metric(f"현재 추세 누적률 ({sign_text})", f"{m_curr_ret*100:.2f}%")
        
        # 조정 위험도 평가
        warning_level = "안전/초입"
        if m_curr_sign == 1 and m_curr_ret >= m_avg_up * 0.8:
            warning_level = "🚨 조정 임박 (과매수)"
        elif m_curr_sign == -1 and m_curr_ret <= m_avg_down * 0.8: # 음수 비교이므로 작거나 같을 때
            warning_level = "💡 반등 임박 (과매도)"
            
        col_m4.metric("현재 시장 상태", warning_level)
    else:
        st.warning("시장 지수 데이터를 불러올 수 없습니다.")

    st.divider()

    # 2. 개별 종목 분석
    st.subheader("🎯 개별 종목 차익실현 및 주도주 분석")
    
    results = []
    
    progress_bar = st.progress(0)
    for i, stock in enumerate(parsed_stocks):
        t_name = stock['name']
        t_code = stock['ticker']
        
        display_name = f"{t_name} ({t_code})"
        
        try:
            if market_choice == '한국 (KRX)':
                df = fdr.DataReader(t_code, start_date, end_date)
            else:
                df = yf.download(t_code, start=start_date, end=end_date, progress=False)
                # yfinance 멀티인덱스 정리 (최신 버전 yfinance 대응)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
            
            if df.empty:
                continue
                
            # 전체 평균 누적 상승/하락 계산
            s_avg_up, s_avg_down, s_curr_ret, s_curr_sign = calculate_streak_averages(df)
            
            # 서브 레인지 수익률
            sub_df = df.loc[sub_start:sub_end]
            sub_ret = (sub_df['Close'].iloc[-1] / sub_df['Close'].iloc[0] - 1) if len(sub_df) > 1 else 0
            
            # 기술적 지표 (최근 데이터 기준)
            # 1. 거래량 급증 (Volume Spike)
            df['Vol_20MA'] = df['Volume'].rolling(window=20).mean()
            vol_ratio = (df['Volume'].iloc[-1] / df['Vol_20MA'].iloc[-1]) if df['Vol_20MA'].iloc[-1] > 0 else 1
            
            # 2. RSI (14일)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 3. 시장 대비 상대 수익률 (최근 20일)
            stock_20d_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0
            market_20d_ret = (market_df['Close'].iloc[-1] / market_df['Close'].iloc[-20] - 1) if (market_df is not None and len(market_df) >= 20) else 0
            relative_strength = stock_20d_ret - market_20d_ret
            
            # 진단 코멘트
            action = "관망"
            if s_curr_sign == 1 and s_curr_ret >= s_avg_up * 0.85:
                action = "⚠️ 차익실현 고려 (평균 상승치 근접)"
            elif s_curr_sign == -1 and s_curr_ret <= s_avg_down * 0.85:
                action = "🟢 저점매수 고려 (평균 하락치 근접)"
                
            is_leader = "🔥 주도주" if relative_strength > 0.05 and vol_ratio > 1.5 else "-"
            
            results.append({
                "종목명(티커)": display_name,
                "평균 연속 상승률": f"{s_avg_up*100:.2f}%",
                "평균 연속 하락률": f"{s_avg_down*100:.2f}%",
                "현재 연속 수익률": f"{s_curr_ret*100:.2f}%",
                "부분 구간 수익률": f"{sub_ret*100:.2f}%",
                "RSI (14)": f"{current_rsi:.1f}",
                "거래량 배수(20MA)": f"{vol_ratio:.1f}x",
                "시장대비 상대수익": f"{relative_strength*100:.1f}%",
                "조정/반등 진단": action,
                "주도주 여부": is_leader
            })
            
        except Exception as e:
            st.error(f"{display_name} 데이터 처리 중 오류 발생: {e}")
            
        progress_bar.progress((i + 1) / len(parsed_stocks))
        
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
    else:
        st.info("조건에 맞는 종목 데이터가 없습니다.")