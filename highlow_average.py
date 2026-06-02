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
st.set_page_config(page_title="주도주 단기/스윙 타점 분석기 (Quant Ver.)", layout="wide")

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

def get_vix_full(start_date, end_date):
    vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    if vix_df is not None and not vix_df.empty and vix_df.index.tz is not None:
        vix_df.index = vix_df.index.tz_localize(None)
    return vix_df

# --- 단기 스윙 맞춤형 공포탐욕지수 ---
def calculate_fear_greed_score(vix, market_rsi, market_20ma_disparity):
    v_score = max(0, min(100, (40 - vix) / 30 * 100))
    r_score = max(0, min(100, market_rsi))
    d_score = max(0, min(100, (market_20ma_disparity + 0.05) / 0.10 * 100))
    return (v_score + r_score + d_score) / 3

# --- UI 레이아웃 ---
st.title("⏱️ 추세/스윙 타점 및 차익실현 백테스터")
st.caption("거래량 폭발, 시장 대비 상대수익, 20일선 및 볼린저밴드를 종합하여 단기 조정(차익실현) 시그널을 잡아냅니다.")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    
    default_input = "네이버, 005930, 카카오\nSK하이닉스" if market_choice == '한국 (KRX)' else "TEAM, AAPL, MSFT"
    stock_input = st.text_area("종목 입력 (이름, 티커, 종목코드 혼용)", value=default_input)
    
    st.subheader("기간 설정")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("전체 데이터 시작일", datetime.today() - timedelta(days=365))
    with col2:
        end_date = st.date_input("전체 데이터 종료일", datetime.today())
        
    st.divider()
    st.subheader("⏱️ 과거 복기 (타임머신 설정)")
    sub_start = st.date_input("구간 수익률 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("📌 분석 기준일 (종료일)", datetime.today())
    
    run_btn = st.button("🚀 정밀 분석 실행", type='primary', use_container_width=True)

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
    
    st.subheader(f"🌐 현재 상승 파동 모멘텀 진단 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
    if market_df_full is not None and not market_df_full.empty:
        m_df_as_of = market_df_full[market_df_full.index <= sub_end_dt].copy()
        vix_as_of = vix_df_full[vix_df_full.index <= sub_end_dt].copy() if not vix_df_full.empty else None
        
        if not m_df_as_of.empty:
            m_avg_up, m_avg_down, m_curr_ret, m_curr_sign = calculate_streak_averages(m_df_as_of)
            
            m_df_as_of['20MA'] = m_df_as_of['Close'].rolling(window=20).mean()
            m_disp_20 = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['20MA'].iloc[-1] - 1) if not pd.isna(m_df_as_of['20MA'].iloc[-1]) else 0.0
            
            m_delta = m_df_as_of['Close'].diff()
            m_gain = (m_delta.where(m_delta > 0, 0)).rolling(window=14).mean()
            m_loss = (-m_delta.where(m_delta < 0, 0)).rolling(window=14).mean()
            m_rs = m_gain / m_loss
            m_rsi = 100 - (100 / (1 + m_rs))
            current_m_rsi = float(m_rsi.iloc[-1]) if len(m_df_as_of) >= 15 else 50.0
            
            m_sub_df = m_df_as_of[m_df_as_of.index >= sub_start_dt]
            m_sub_ret = float(m_sub_df['Close'].iloc[-1] / m_sub_df['Close'].iloc[0] - 1) if len(m_sub_df) > 1 else 0.0
            current_vix = float(vix_as_of['Close'].iloc[-1]) if vix_as_of is not None and not vix_as_of.empty else 20.0
            
            fg_score = calculate_fear_greed_score(current_vix, current_m_rsi, m_disp_20)
            
            fg_status = "중립 (추세 지속 가능)"
            if fg_score <= 25: fg_status = "😨 극단적 공포 (단기 낙폭 과대 / 줍줍)"
            elif fg_score <= 45: fg_status = "📉 공포 (반등 모색 구간)"
            elif fg_score >= 80: fg_status = "🚨 단기 과열 극심 (조정 임박)"
            elif fg_score >= 60: fg_status = "📈 탐욕 (추가 상승 여력 있으나 주의)"
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(f"단기 스윙 공포탐욕지수", f"{fg_score:.1f}점", fg_status)
            col_m2.metric("글로벌 공포지수 (VIX)", f"{current_vix:.2f}")
            col_m3.metric(f"{market_name} 20일(단기) 이격률", f"{m_disp_20*100:+.2f}%")
            col_m4.metric(f"지정 구간 수익률", f"{m_sub_ret*100:+.2f}%")
            
            st.progress(int(fg_score), text=f"🔥 단기 과열(100) ↔ 🧊 단기 침체(0) | 현재 스코어: {fg_score:.1f}점")
            
        else:
            st.warning("설정한 기준일 이전의 시장 데이터가 부족합니다.")
    else:
        st.warning("시장 지수 데이터를 불러올 수 없습니다.")

    st.divider()
    st.subheader(f"🎯 개별 종목 정밀 타점 및 차익실현 분석 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
    results = []
    progress_bar = st.progress(0)
    failed_stocks = []
    
    for i, stock in enumerate(parsed_stocks):
        t_name = stock['name']
        t_code = stock['ticker']
        
        try:
            if market_choice == '한국 (KRX)':
                df_full = fdr.DataReader(t_code, start_date, end_date)
            else:
                df_full = yf.download(t_code, start=start_date, end=end_date, progress=False)
                if isinstance(df_full.columns, pd.MultiIndex):
                    df_full.columns = df_full.columns.get_level_values(0)
            
            if df_full.empty:
                failed_stocks.append(f"{t_name}({t_code})")
                progress_bar.progress((i + 1) / len(parsed_stocks))
                continue
                
            if df_full.index.tz is not None:
                df_full.index = df_full.index.tz_localize(None)
            
            df_as_of = df_full[df_full.index <= sub_end_dt].copy()
            if df_as_of.empty:
                continue
            
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)' and t_code in krx_code_dict:
                t_name = krx_code_dict[t_code]
            
            display_name = f"{t_name} ({t_code})"
            
            s_avg_up, s_avg_down, s_curr_ret, s_curr_sign = calculate_streak_averages(df_as_of)
            
            # 1. 20일 이격률 및 볼린저 밴드
            df_as_of['20MA'] = df_as_of['Close'].rolling(window=20).mean()
            df_as_of['20STD'] = df_as_of['Close'].rolling(window=20).std()
            df_as_of['BB_Upper'] = df_as_of['20MA'] + (df_as_of['20STD'] * 2)
            df_as_of['BB_Lower'] = df_as_of['20MA'] - (df_as_of['20STD'] * 2)
            
            current_close = df_as_of['Close'].iloc[-1]
            bb_upper = df_as_of['BB_Upper'].iloc[-1]
            bb_lower = df_as_of['BB_Lower'].iloc[-1]
            bb_position = float((current_close - bb_lower) / (bb_upper - bb_lower)) if not pd.isna(bb_upper) and bb_upper != bb_lower else 0.5
            
            disparity_20 = float(current_close / df_as_of['20MA'].iloc[-1] - 1) if not pd.isna(df_as_of['20MA'].iloc[-1]) else 0.0
            
            # 2. RSI (14)
            delta = df_as_of['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if len(df_as_of) >= 15 else 50.0
            
            # 3. 거래량 배수 (20MA 대비)
            df_as_of['Vol_20MA'] = df_as_of['Volume'].rolling(window=20).mean()
            vol_ratio = float(df_as_of['Volume'].iloc[-1] / df_as_of['Vol_20MA'].iloc[-1]) if len(df_as_of) >= 20 and df_as_of['Vol_20MA'].iloc[-1] > 0 else 1.0
            
            # 4. 상대 수익률 (최근 20일)
            stock_20d_ret = float(df_as_of['Close'].iloc[-1] / df_as_of['Close'].iloc[-20] - 1) if len(df_as_of) >= 20 else 0.0
            if 'm_df_as_of' in locals() and len(m_df_as_of) >= 20:
                market_20d_ret = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['Close'].iloc[-20] - 1)
            else:
                market_20d_ret = 0.0
            relative_strength = stock_20d_ret - market_20d_ret
            
            # 5. 지정 구간 수익률
            sub_df = df_as_of[df_as_of.index >= sub_start_dt]
            sub_ret = float(sub_df['Close'].iloc[-1] / sub_df['Close'].iloc[0] - 1) if len(sub_df) > 1 else 0.0
            
            # 🔴 단기 스윙 최적화 채점 로직 (거래량 및 상대수익률 개입 강화)
            score = 0
            
            # 차익실현(고점) 시그널 (-)
            if current_rsi >= 70: score -= 1
            if bb_position >= 0.95: score -= 1 
            if s_curr_sign == 1 and s_curr_ret >= s_avg_up * 0.8 and s_avg_up > 0: score -= 1
            if vol_ratio >= 2.5 and s_curr_sign == 1: score -= 1 # 고점에서 거래량 폭발 (Climax)
            if relative_strength < -0.05 and s_curr_sign == 1: score -= 1 # 시장은 가는데 종목은 못 갈 때
            
            # 저점매수(바닥) 시그널 (+)
            if current_rsi <= 30: score += 1
            if bb_position <= 0.05: score += 1
            if s_curr_sign == -1 and s_curr_ret <= s_avg_down * 0.8 and s_avg_down < 0: score += 1
            if vol_ratio >= 2.5 and s_curr_sign == -1: score += 1 # 저점에서 투매 거래량 발생 (Capitulation)
            
            action = "🟢 추세 양호 (홀딩)"
            if score <= -3: action = "🚨 고점 거래량 폭발 (전량 매도)"
            elif score == -2: action = "⚠️ 밴드 이탈 과열 (분할 매도)"
            elif score == -1: action = "💡 단기 저항대 (주의)"
            elif score >= 3: action = "🔥 패닉셀 투매 (적극 줍줍)"
            elif score == 2: action = "🟢 단기 낙폭 과대 (매수 고려)"
            elif s_curr_sign == -1: action = "📉 단기 하락 추세"
                
            is_leader = "🚀 주도주" if relative_strength > 0.05 and vol_ratio > 1.2 and disparity_20 > 0 else "-"
            
            if bb_position >= 1.0: bb_text = "상단 돌파 (초과열)"
            elif bb_position <= 0.0: bb_text = "하단 이탈 (초침체)"
            else: bb_text = f"밴드 내 ({bb_position*100:.0f}%)"

            # 누락된 지표 없이 모두 결과에 포함
            results.append({
                "종목명(티커)": display_name,
                "현재 추세 진단": action,
                "주도주 여부": is_leader,
                "RSI(14)": f"{current_rsi:.1f}",
                "단기 20일 이격률": f"{disparity_20*100:+.1f}%",
                "볼린저밴드 위치": bb_text,
                "거래량(20MA 대비)": f"{vol_ratio:.1f}x",
                "시장대비 상대수익": f"{relative_strength*100:+.1f}%",
                "평균 연속 상승률": f"{s_avg_up*100:+.2f}%", 
                "평균 연속 하락률": f"{s_avg_down*100:+.2f}%", 
                "현재 연속 수익률": f"{s_curr_ret*100:+.2f}%",
                "지정 구간 수익률": f"{sub_ret*100:+.2f}%"
            })
        except Exception as e:
            failed_stocks.append(f"{t_name}({t_code})")
            
        progress_bar.progress((i + 1) / len(parsed_stocks))
        
    if failed_stocks:
        st.warning(f"⚠️ 다음 종목들은 상장 폐지되었거나 데이터가 없어 제외되었습니다: {', '.join(failed_stocks)}")
        
    if results:
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True)
    else:
        st.error("조건에 맞는 결과가 없습니다. 시작일과 종료일(기준일)을 확인해 주세요.")
