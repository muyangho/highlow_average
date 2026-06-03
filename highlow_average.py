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
st.set_page_config(page_title="주도주 정밀 타점 & 하락장 생존 분석기", layout="wide")

with st.sidebar:
    if st.button("🔄 종목 데이터 리셋 (오류시 클릭)", help="데이터 꼬임 발생 시 눌러주세요."):
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
    if market == '한국 (KRX)':
        df = fdr.DataReader('KS11', start_date, end_date)
        name = "KOSPI"
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

# --- UI 레이아웃 ---
st.title("🔥 시장 붕괴 방어주 포착 & 정밀 타점 분석기")
st.caption("시장 전체가 무너질 때 나홀로 버티는 진짜 주도주를 찾고, 과열 종목은 정확히 익절하도록 가이드합니다.")

with st.sidebar:
    st.header("설정 (Settings)")
    market_choice = st.radio("시장 선택", ['한국 (KRX)', '미국 (US)'])
    default_input = "005930, 000660" if market_choice == '한국 (KRX)' else "MU, NVDA, AAPL"
    stock_input = st.text_area("종목 입력", value=default_input)
    
    st.subheader("기간 설정")
    col1, col2 = st.columns(2)
    with col1: start_date = st.date_input("전체 시작일", datetime.today() - timedelta(days=365))
    with col2: end_date = st.date_input("전체 종료일", datetime.today())
        
    st.divider()
    sub_start = st.date_input("부분 분석 시작일", datetime.today() - timedelta(days=30))
    sub_end = st.date_input("📌 기준일(종료일)", datetime.today())
    run_btn = st.button("🚀 정밀 타점 분석", type='primary', use_container_width=True)

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
    
    st.subheader(f"🌐 시장 매크로 & 공포탐욕 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
    if market_df_full is not None and not market_df_full.empty:
        m_df_as_of = market_df_full[market_df_full.index <= sub_end_dt].copy()
        vix_as_of = vix_df_full[vix_df_full.index <= sub_end_dt].copy() if not vix_df_full.empty else None
        
        if not m_df_as_of.empty:
            m_df_as_of['20MA'] = m_df_as_of['Close'].rolling(window=20).mean()
            m_disp_20 = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['20MA'].iloc[-1] - 1) if not pd.isna(m_df_as_of['20MA'].iloc[-1]) else 0.0
            
            m_delta = m_df_as_of['Close'].diff()
            m_rs = (m_delta.where(m_delta > 0, 0)).rolling(window=14).mean() / (-m_delta.where(m_delta < 0, 0)).rolling(window=14).mean()
            current_m_rsi = float(100 - (100 / (1 + m_rs)).iloc[-1]) if len(m_df_as_of) >= 15 else 50.0
            
            m_20d_ret = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['Close'].iloc[-20] - 1) if len(m_df_as_of) >= 20 else 0.0
            
            current_vix = float(vix_as_of['Close'].iloc[-1]) if vix_as_of is not None and not vix_as_of.empty else 20.0
            fg_score = calculate_fear_greed_score(current_vix, current_m_rsi, m_disp_20)
            
            fg_status = "중립 (추세 지속 가능)"
            if fg_score <= 25: fg_status = "😨 극단적 공포 (시장 전반 줍줍 찬스)"
            elif fg_score <= 45: fg_status = "📉 공포 (반등 모색 구간)"
            elif fg_score >= 80: fg_status = "🚨 단기 과열 극심 (조정 임박, 현금화)"
            elif fg_score >= 60: fg_status = "📈 탐욕 (추가 상승 여력 있으나 주의)"
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric(f"공포탐욕지수", f"{fg_score:.1f}점", fg_status)
            col_m2.metric("VIX (변동성 지수)", f"{current_vix:.2f}")
            col_m3.metric(f"{market_name} 최근 20일 수익률", f"{m_20d_ret*100:+.2f}%")
            col_m4.metric(f"{market_name} RSI", f"{current_m_rsi:.1f}")
            st.progress(int(fg_score), text=f"🔥 단기 과열(100) ↔ 🧊 단기 침체(0) | 현재 스코어: {fg_score:.1f}점")
            
    st.divider()
    st.subheader(f"🎯 개별 종목 수급 및 타점 진단 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
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
            if len(df_as_of) < 20: continue
            
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)' and t_code in krx_code_dict:
                t_name = krx_code_dict[t_code]
            
            display_name = f"{t_name} ({t_code})"
            
            # --- 기본 차트 지표 ---
            current_close = float(df_as_of['Close'].iloc[-1])
            prev_close = float(df_as_of['Close'].iloc[-2])
            is_up_day = current_close > prev_close
            
            df_as_of['5MA'] = df_as_of['Close'].rolling(window=5).mean()
            df_as_of['20MA'] = df_as_of['Close'].rolling(window=20).mean()
            is_above_5ma = current_close > df_as_of['5MA'].iloc[-1]
            
            df_as_of['20STD'] = df_as_of['Close'].rolling(window=20).std()
            df_as_of['BB_Upper'] = df_as_of['20MA'] + (df_as_of['20STD'] * 2)
            df_as_of['BB_Lower'] = df_as_of['20MA'] - (df_as_of['20STD'] * 2)
            bb_upper, bb_lower = df_as_of['BB_Upper'].iloc[-1], df_as_of['BB_Lower'].iloc[-1]
            bb_pos = float((current_close - bb_lower) / (bb_upper - bb_lower)) if not pd.isna(bb_upper) and bb_upper != bb_lower else 0.5
            
            delta = df_as_of['Close'].diff()
            rs = (delta.where(delta > 0, 0)).rolling(window=14).mean() / (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            current_rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if len(df_as_of) >= 15 else 50.0
            
            # --- 🔴 거래량 정밀 해석 ---
            df_as_of['Vol_20MA'] = df_as_of['Volume'].rolling(window=20).mean()
            vol_ratio = float(df_as_of['Volume'].iloc[-1] / df_as_of['Vol_20MA'].iloc[-1]) if df_as_of['Vol_20MA'].iloc[-1] > 0 else 1.0
            
            if is_up_day and vol_ratio >= 1.5: vol_sig = "🔴 강한 매수세 (돌파/매집)"
            elif not is_up_day and vol_ratio >= 1.5: vol_sig = "🚨 강한 매도세 (세력/투매)"
            elif is_up_day and vol_ratio < 0.8: vol_sig = "⚠️ 상승 동력 약화 (페이크 의심)"
            elif not is_up_day and vol_ratio < 0.8: vol_sig = "💡 긍정적: 거래량 마른 조정"
            else: vol_sig = "평이함 (특이사항 없음)"

            # --- 🔴 OBV 수급 및 다이버전스 ---
            df_as_of['OBV'] = (np.sign(df_as_of['Close'].diff()) * df_as_of['Volume']).fillna(0).cumsum()
            price_trend = "상승" if current_close >= df_as_of['Close'].iloc[-20] else "하락"
            obv_trend = "상승" if df_as_of['OBV'].iloc[-1] >= df_as_of['OBV'].iloc[-20] else "하락"
            
            if price_trend == "하락" and obv_trend == "상승": obv_sig = "🔥 숨은 매수세 (세력 매집중)"
            elif price_trend == "상승" and obv_trend == "하락": obv_sig = "⚠️ 숨은 매도세 (개미 꼬시기)"
            elif price_trend == "상승" and obv_trend == "상승": obv_sig = "🚀 수급 동반 탄탄한 상승"
            else: obv_sig = "📉 수급 이탈 (하락 추세)"

            # --- 🔴 하락장 방어주 판별 로직 ---
            stock_20d_ret = float(current_close / df_as_of['Close'].iloc[-20] - 1) if len(df_as_of) >= 20 else 0.0
            relative_strength = stock_20d_ret - m_20d_ret
            
            is_leader = "-"
            # 1. 시장이 상승장일 때 일반적인 주도주
            if m_20d_ret >= 0 and relative_strength > 0.05 and obv_trend == "상승" and is_above_5ma:
                is_leader = "🚀 상승장 주도주"
            # 2. 시장이 하락장일 때 살아남는 방어 대장주 (핵심)
            elif m_20d_ret < -0.02 and stock_20d_ret > 0 and is_above_5ma and obv_trend == "상승":
                is_leader = "🛡️ 폭락장 생존주 (방어 대장)"

            # --- 🔴 프로 타점 채점 시스템 (익절/재매수 명문화) ---
            score = 0
            
            # 매수 가점
            if current_rsi < 35: score += 1
            if bb_pos <= 0.05: score += 1
            if "숨은 매수세" in obv_sig or "수급 동반" in obv_sig: score += 1
            if "거래량 마른 조정" in vol_sig or "강한 매수세" in vol_sig: score += 1
            
            # 매도 감점
            if not is_above_5ma and current_rsi > 60: score -= 2 
            if current_rsi > 70 and not is_above_5ma: score -= 1 
            if bb_pos >= 0.95: score -= 1
            if "강한 매도세" in vol_sig: score -= 2
            if "숨은 매도세" in obv_sig: score -= 2

            # 최종 액션 판별 (사용자 요청 반영)
            if score >= 3: action = "🔥 강력 매수 (눌림목/바닥 수급확인)"
            elif score in [1, 2]: action = "🟢 분할 매수 (조정 후 반등 노림)"
            elif score == 0: 
                action = "🟢 추세 홀딩 (보유자 영역)" if is_above_5ma else "👀 관망 (방향성 탐색중)"
            elif score in [-1, -2]: action = "⚠️ 분할 익절 (저항/단기꺾임)"
            else: action = "🚨 초과열 (전량 익절, 조정 시 재매수)" # 확실한 익절 및 재매수 지침

            results.append({
                "종목명(티커)": display_name,
                "주도주/방어주 상태": is_leader,
                "최종 타점 가이드": action,
                "거래량 해석 (Quality)": vol_sig,
                "OBV 수급 (Divergence)": obv_sig,
                "RSI(14)": f"{current_rsi:.0f}",
                "5일선 유지": "✅ 지지중" if is_above_5ma else "❌ 이탈",
                "당일 거래량": f"평소의 {vol_ratio:.1f}배",
                "볼린저밴드 위치": f"{bb_pos*100:.0f}% 위치"
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
        st.error("조건에 맞는 결과가 없습니다.")
