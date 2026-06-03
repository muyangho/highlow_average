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
st.set_page_config(page_title="주도주 추세 추종 & 스마트 수급 분석기", layout="wide")

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
st.title("🔥 주도주 모멘텀 & 스마트 수급 분석기")
st.caption("AI가 5일선 지지 여부, 캔들의 방향(음봉/양봉), 그리고 거래량의 폭발/축소를 종합하여 구체적인 코멘트를 달아줍니다.")

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
            col_m1.metric(f"공포탐욕지수", f"{fg_score:.1f}점", fg_status)
            col_m2.metric("VIX (변동성 지수)", f"{current_vix:.2f}")
            col_m3.metric(f"{market_name} 단기 이격률", f"{m_disp_20*100:+.2f}%")
            col_m4.metric(f"지정 구간 시장수익률", f"{m_sub_ret*100:+.2f}%")
            st.progress(int(fg_score), text=f"🔥 단기 과열(100) ↔ 🧊 단기 침체(0) | 현재 스코어: {fg_score:.1f}점")
            
    st.divider()
    st.subheader(f"🎯 개별 종목 스마트 수급 상세 진단 (기준일: {sub_end_dt.strftime('%Y-%m-%d')})")
    
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
            if df_as_of.empty: continue
            
            if t_name in ["국내종목", "미국종목"] and market_choice == '한국 (KRX)' and t_code in krx_code_dict:
                t_name = krx_code_dict[t_code]
            
            display_name = f"{t_name} ({t_code})"
            
            # --- 지표 계산 ---
            current_close = float(df_as_of['Close'].iloc[-1])
            prev_close = float(df_as_of['Close'].iloc[-2]) if len(df_as_of) > 1 else current_close
            is_up_day = current_close > prev_close # 양봉(상승) 여부
            
            df_as_of['5MA'] = df_as_of['Close'].rolling(window=5).mean()
            df_as_of['20MA'] = df_as_of['Close'].rolling(window=20).mean()
            is_above_5ma = current_close > df_as_of['5MA'].iloc[-1]
            
            df_as_of['20STD'] = df_as_of['Close'].rolling(window=20).std()
            df_as_of['BB_Upper'] = df_as_of['20MA'] + (df_as_of['20STD'] * 2)
            df_as_of['BB_Lower'] = df_as_of['20MA'] - (df_as_of['20STD'] * 2)
            bb_upper = df_as_of['BB_Upper'].iloc[-1]
            bb_lower = df_as_of['BB_Lower'].iloc[-1]
            bb_pos = float((current_close - bb_lower) / (bb_upper - bb_lower)) if not pd.isna(bb_upper) and bb_upper != bb_lower else 0.5
            
            delta = df_as_of['Close'].diff()
            rs = (delta.where(delta > 0, 0)).rolling(window=14).mean() / (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            current_rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if len(df_as_of) >= 15 else 50.0
            
            # 거래량 및 OBV (수급)
            df_as_of['Vol_20MA'] = df_as_of['Volume'].rolling(window=20).mean()
            vol_ratio = float(df_as_of['Volume'].iloc[-1] / df_as_of['Vol_20MA'].iloc[-1]) if len(df_as_of) >= 20 and df_as_of['Vol_20MA'].iloc[-1] > 0 else 1.0
            
            df_as_of['OBV'] = (np.sign(df_as_of['Close'].diff()) * df_as_of['Volume']).fillna(0).cumsum()
            obv_trend = "수급유입 (상승)" if len(df_as_of) >= 20 and df_as_of['OBV'].iloc[-1] > df_as_of['OBV'].iloc[-20] else "수급이탈 (하락)"
            
            stock_20d_ret = float(current_close / df_as_of['Close'].iloc[-20] - 1) if len(df_as_of) >= 20 else 0.0
            market_20d_ret = float(m_df_as_of['Close'].iloc[-1] / m_df_as_of['Close'].iloc[-20] - 1) if 'm_df_as_of' in locals() and len(m_df_as_of) >= 20 else 0.0
            relative_strength = stock_20d_ret - market_20d_ret
            
            # 🔴 사용자 맞춤형 텍스트 생성 로직 (상세 코멘트)
            comment = ""
            action = "🟢 추세 양호 (홀딩)"
            
            if is_above_5ma:
                if is_up_day and vol_ratio >= 1.2:
                    comment = f"5일선 지지 중 거래량이 평소 대비 {vol_ratio:.1f}배 터진 양봉 발생. 추가 상승 랠리가 기대됩니다."
                elif not is_up_day and vol_ratio <= 0.8:
                    comment = f"5일선 위에서 거래량이 뚝 끊긴({vol_ratio:.1f}배) 음봉 조정. 세력 이탈 없는 전형적인 '건강한 눌림목' 타점입니다."
                elif not is_up_day and vol_ratio > 1.2:
                    comment = f"5일선 위지만 거래량이 {vol_ratio:.1f}배 터진 음봉 마감. 단기 저항(매도세)이 강하므로 내일 5일선 이탈 여부를 예의주시하세요."
                else:
                    comment = "5일선을 타고 무난하게 상승 추세를 이어가고 있습니다."
                
                # RSI 70 이상이어도 5일선 위에 있으면 안 팔게 함
                if current_rsi >= 70:
                    action = "🔥 대세 상승 (강력 홀딩)"
                    comment = f"[RSI {current_rsi:.0f} 단기 과열] 그러나 " + comment
                    
            else: # 5일선 이탈 시
                if not is_up_day and vol_ratio >= 1.5:
                    action = "🚨 세력 차익실현 (강력 매도)"
                    comment = f"5일선이 깨지면서 거래량이 {vol_ratio:.1f}배 폭발한 대형 음봉! 전형적인 세력 차익실현(설거지) 징후이므로 즉각 매도를 권장합니다."
                elif is_up_day and vol_ratio >= 1.5:
                    action = "💡 강력 반등 (매수 검토)"
                    comment = f"5일선 아래에서 거래량이 {vol_ratio:.1f}배 터진 양봉 발생. 바닥에서 강력한 저가 매수세가 유입되었습니다."
                elif not is_up_day and vol_ratio < 1.0:
                    action = "⚠️ 단기 하락 추세"
                    comment = "5일선을 이탈하여 흘러내리는 중입니다. 의미 있는 거래량 반등이 나올 때까지 섣불리 줍지 마세요."
                else:
                    action = "⚠️ 단기 하락 추세"
                    comment = "단기 추세(5일선)가 꺾였습니다. 리스크 관리가 필요합니다."

            if current_rsi <= 30 and action != "💡 강력 반등 (매수 검토)":
                action = "🟢 바닥 확인 중 (관망)"
                comment = f"[RSI {current_rsi:.0f} 투매 구간] " + comment

            is_leader = "🚀 주도주" if relative_strength > 0.05 and obv_trend == "수급유입 (상승)" and is_above_5ma else "-"
            
            if bb_pos >= 1.0: bb_text = "상단 돌파"
            elif bb_pos <= 0.0: bb_text = "하단 이탈"
            else: bb_text = f"밴드 내 ({bb_pos*100:.0f}%)"

            results.append({
                "종목명(티커)": display_name,
                "현재 추세 진단": action,
                "상세 분석 코멘트": comment,  # 🔴 핵심 추가 사항
                "주도주 여부": is_leader,
                "RSI(14)": f"{current_rsi:.0f}",
                "OBV 수급(20일)": obv_trend,
                "5일선 유지": "✅ 지지중" if is_above_5ma else "❌ 이탈",
                "당일 거래량": f"{vol_ratio:.1f}배 {'(상승)' if is_up_day else '(하락)'}",
                "시장대비 상대수익": f"{relative_strength*100:+.1f}%"
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
