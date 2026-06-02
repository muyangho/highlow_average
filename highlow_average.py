import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import re
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="주도주 포착 및 차익실현 분석기", layout="wide")

# --- 서버 차단 대비 주요 종목 백업 맵 (Fallback Data) ---
# KRX 서버가 클라우드 IP를 차단하더라도 자주 쓰이는 주요 종목은 이름/코드 매핑이 가능하게 합니다.
BACKUP_KRX_DICT = {
    "삼성전자": "005930", "SK하이닉스": "000660", "LG에너지솔루션": "373220",
    "삼성바이오로직스": "207940", "현대차": "005380", "기아": "000270",
    "셀트리온": "068270", "KB금융": "105560", "네이버": "035420",
    "NAVER": "035420", "신한지주": "055550", "카카오": "035720",
    "현대모비스": "012330", "포스코홀딩스": "005490", "POSCO홀딩스": "005490"
}
BACKUP_KRX_CODE_DICT = {v: k for k, v in BACKUP_KRX_DICT.items()}

# --- 데이터 캐싱 및 로드 ---
@st.cache_data(ttl=86400)
def load_stock_listings():
    # 한국 주식 로드 시도
    try:
        krx = fdr.StockListing('KRX-DESC') 
    except Exception:
        try:
            kospi = fdr.StockListing('KOSPI')
            kosdaq = fdr.StockListing('KOSDAQ')
            krx = pd.concat([kospi, kosdaq])
        except Exception:
            krx = pd.DataFrame(columns=['Name', 'Code'])

    # 미국 주식 로드 시도
    try:
        nasdaq = fdr.StockListing('NASDAQ')
        nyse = fdr.StockListing('NYSE')
        us_df = pd.concat([nasdaq, nyse])
    except Exception:
        us_df = pd.DataFrame(columns=['Name', 'Symbol'])
    
    # 기본 딕셔너리 생성
    krx_dict = dict(zip(krx['Name'], krx['Code'])) if not krx.empty else {}
    krx_code_dict = dict(zip(krx['Code'], krx['Name'])) if not krx.empty else {}
    
    # 백업 데이터 병합 (서버 차단 시에도 최소한의 작동 보장)
    for k, v in BACKUP_KRX_DICT.items():
        if k not in krx_dict: krx_dict[k] = v
    for k, v in BACKUP_KRX_CODE_DICT.items():
        if k not in krx_code_dict: krx_code_dict[k] = v
        
    us_dict = dict(zip(us_df['Name'], us_df['Symbol'])) if not us_df.empty else {}
    us_code_dict = dict(zip(us_df['Symbol'], us_df['Name'])) if not us_df.empty else {}
    
    return krx_dict, krx_code_dict, us_dict, us_code_dict

krx_dict, krx_code_dict, us_dict, us_code_dict = load_stock_listings()

# --- 헬퍼 함수: 입력값 분석 및 이름/티커 동시 추출 ---
def parse_tickers(input_text, market):
    # 띄어쓰기, 줄바꿈, 콤마, 탭 등을 기준으로 분리
    raw_list = re.split(r'[\n,\t\s]+', input_text.strip())
    raw_list = [x for x in raw_list if x]
    
    parsed = []
    for item in raw_list:
        ticker = item
        name = item
        
        if market == '한국 (KRX)':
            if item in krx_dict:           # 사용자가 '이름'을 입력한 경우
                ticker = krx_dict[item]
                name = item
            elif item in krx_code_dict:    # 사용자가 '6자리 코드'를 입력한 경우
                name = krx_code_dict[item]
                ticker = item
            else:                          # 매핑 데이터에 없는 경우 (코드라 가정)
                ticker = item
                name = f"국내종목"
        else:
            item = item.upper()
            if item in us_code_dict:       # 사용자가 '티커'를 입력한 경우
                name = us_code_dict[item]
                ticker = item
            elif item in us_dict:          # 사용자가 '풀네임'을 입력한 경우
                ticker = us_dict[item]
                name = item
            else:                          # 매핑 데이터에 없는 경우 (티커라 가정)
                ticker = item
                name = "미국종목"
                
        parsed.append({'name': name, 'ticker': ticker})
    return parsed

def calculate_streak_averages(df):
    if df.empty or 'Close' not in df.columns:
        return 0, 0, 0, 0
    
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df = df.dropna()
    
    df['Sign'] = np.sign(df['Return'].round(4))
    df['Streak_ID'] = (df['Sign'] != df['Sign'].shift(1)).cumsum()
    
    streak_returns = df.groupby('Streak_ID').apply(
        lambda x: (1 + x['Return']).prod() - 1, include_groups=False
    ).to_frame(name='Cumulative_Return')
    
    streak_signs = df.groupby('Streak_ID')['Sign'].first()
    streak_returns['Sign'] = streak_signs
    
    up_streaks = streak_returns[streak_returns['Sign'] == 1]['Cumulative_Return']
    down_streaks = streak_returns[streak_returns['Sign'] == -1]['Cumulative_Return']
    
    avg_up = up_streaks.mean() if not up_streaks.empty else 0
    avg_down = down_streaks.mean() if not down_streaks.empty else 0
    
    current_streak_return = streak_returns.iloc[-1]['Cumulative_Return'] if not streak_returns.empty else 0
    current_sign = streak_returns.iloc[-1]['Sign'] if not streak_returns.empty else 0
    
    return avg_up, avg_down, current_streak_return, current_sign

def get_market_index(market, start_date, end_date):
    if market == '한국 (KRX)':
        df = fdr.DataReader('KS11', start_date, end_date)
        name = "KOSPI 지수"
    else:
        df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        name = "S&P 500 지수"
    return name, df

# --- UI 레이아웃 ---
st.title("📊 시장 주도주 포착 및 조정/반등 타이밍 분석기")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    
    # 기본 입력 예시 구성
    default_input = "네이버, 005930, 카카오\nSK하이닉스" if market_choice == '한국 (KRX)' else "TEAM, AAPL, Microsoft"
    stock_input = st.text_area("종목 입력 (이름, 티커, 종목코드 혼용 가능)", value=default_input, help="띄어쓰기, 줄바꿈, 쉼표로 자유롭게 구분하세요.")
    
    st.subheader("기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("전체 시작일", datetime.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("전체 종료일", datetime.today())
        
    st.subheader("부분 범위 확인 (서브 레인지)")
    sub_start = st.date_input("부분 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("부분 종료일", datetime.today())
    
    run_btn = st.button("분석 실행", type='primary')

# --- 분석 로직 ---
if run_btn:
    if not stock_input:
        st.warning("종목을 입력해 주세요.")
        st.stop()
        
    parsed_stocks = parse_tickers(stock_input, market_choice)
    market_name, market_df = get_market_index(market_choice, start_date, end_date)
    
    st.subheader("🌐 시장 환경 분석")
    if market_df is not None and not market_df.empty:
        m_avg_up, m_avg_down, m_curr_ret, m_curr_sign = calculate_streak_averages(market_df)
        m_sub_df = market_df.loc[sub_start:sub_end]
        m_sub_ret = (m_sub_df['Close'].iloc[-1] / m_sub_df['Close'].iloc[0] - 1) if len(m_sub_df) > 1 else 0
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(f"{market_name} 평균 연속 상승률", f"{m_avg_up*100:.2f}%")
        col_m2.metric("평균 연속 하락률 (조정치)", f"{m_avg_down*100:.2f}%")
        
        sign_text = "상승중" if m_curr_sign == 1 else ("하락중" if m_curr_sign == -1 else "보합")
        col_m3.metric(f"현재 추세 누적률 ({sign_text})", f"{m_curr_ret*100:.2f}%")
        
        warning_level = "안전/초입"
        if m_curr_sign == 1 and m_curr_ret >= m_avg_up * 0.8:
            warning_level = "🚨 조정 임박 (과매수)"
        elif m_curr_sign == -1 and m_curr_ret <= m_avg_down * 0.8:
            warning_level = "💡 반등 임박 (과매도)"
        col_m4.metric("현재 시장 상태", warning_level)
    else:
        st.warning("시장 지수 데이터를 불러올 수 없습니다.")

    st.divider()
    st.subheader("🎯 개별 종목 차익실현 및 주도주 분석")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, stock in enumerate(parsed_stocks):
        t_name = stock['name']
        t_code = stock['ticker']
        
        try:
            if market_choice == '한국 (KRX)':
                df = fdr.DataReader(t_code, start_date, end_date)
            else:
                df = yf.download(t_code, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
            
            if df.empty:
                # 데이터가 비어있고 매핑에 실패한 항목에 대한 예외 알림
                if market_choice == '한국 (KRX)' and not t_code.isdigit():
                    st.error(f"❌ '{t_name}' 인식 실패: 서버 차단으로 인해 검색되지 않는 종목명입니다. '6자리 숫자 코드'로 직접 입력해 주세요.")
                continue
            
            # 주식 데이터를 정상적으로 가져왔다면 야후/FDR 기반으로 실제 이름 복원 시도
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)':
                if t_code in krx_code_dict:
                    t_name = krx_code_dict[t_code]
            
            # 요구사항 반영: 최종 출력 포맷 정의 -> 종목명(티커or종목번호)
            display_name = f"{t_name} ({t_code})"
                
            s_avg_up, s_avg_down, s_curr_ret, s_curr_sign = calculate_streak_averages(df)
            sub_df = df.loc[sub_start:sub_end]
            sub_ret = (sub_df['Close'].iloc[-1] / sub_df['Close'].iloc[0] - 1) if len(sub_df) > 1 else 0
            
            df['Vol_20MA'] = df['Volume'].rolling(window=20).mean()
            vol_ratio = (df['Volume'].iloc[-1] / df['Vol_20MA'].iloc[-1]) if df['Vol_20MA'].iloc[-1] > 0 else 1
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            stock_20d_ret = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0
            market_20d_ret = (market_df['Close'].iloc[-1] / market_df['Close'].iloc[-20] - 1) if (market_df is not None and len(market_df) >= 20) else 0
            relative_strength = stock_20d_ret - market_20d_ret
            
            action = "관망"
            if s_curr_sign == 1 and s_curr_ret >= s_avg_up * 0.85:
                action = "⚠️ 차익실현 고려"
            elif s_curr_sign == -1 and s_curr_ret <= s_avg_down * 0.85:
                action = "🟢 저점매수 고려"
                
            is_leader = "🔥 주도주" if relative_strength > 0.05 and vol_ratio > 1.5 else "-"
            
            results.append({
                "종목명(티커or종목번호)": display_name,
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
            pass # 에러 발생 시 앱이 깨지지 않고 부드럽게 넘어가도록 처리
            
        progress_bar.progress((i + 1) / len(parsed_stocks))
        
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
    else:
        st.info("조회 가능한 주식 데이터가 없습니다. 입력 값을 다시 확인해 주세요.")
