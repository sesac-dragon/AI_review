import os
import re
import sys
import subprocess
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_echarts import st_echarts
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- 페이지 설정 및 공통 함수 ---

# Streamlit 페이지의 기본 환경을 설정합니다
st.set_page_config(page_title="AI 리뷰 분석 대시보드", layout="wide")

# Seaborn/Matplotlib 그래프의 한글 폰트 깨짐 방지를 위한 전역 설정입니다
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    st.error("한글 폰트를 찾을 수 없습니다. '맑은 고딕' 폰트가 설치되어 있는지 확인해주세요.")

# 데이터베이스 커넥션을 생성하고 캐싱하여 재사용합니다
@st.cache_resource
def get_conn():
    conn = psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "kurlydb"),
        user=os.getenv("PGUSER", "kurlyuser"),
        password=os.getenv("PGPASSWORD", "kurlypassword"),
        client_encoding='utf8'
    )
    return conn

# SQL 쿼리를 실행하여 결과를 DataFrame으로 반환하고 캐싱합니다
@st.cache_data(ttl=600)
def fetch_df(sql: str, params=None) -> pd.DataFrame:
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or [])
            rows = cur.fetchall()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()

# --- Database Initialization ---
@st.cache_resource
def initialize_database():
    """
    Ensures all necessary tables are created in the database.
    Reads schema.sql and executes it.
    """
    try:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS reviews (
          id BIGSERIAL PRIMARY KEY,
          product_name VARCHAR(255),
          content TEXT NOT NULL,
          review_date DATE,
          content_hash CHAR(64) NOT NULL UNIQUE,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS review_analysis (
          id SERIAL PRIMARY KEY,
          review_id BIGINT NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
          model_name VARCHAR(50) NOT NULL,
          is_actual_review BOOLEAN NOT NULL,
          keywords TEXT[],
          categories TEXT[],
          sentiment_label VARCHAR(20),
          sentiment_score REAL,
          star_rating REAL,
          is_recipe_like BOOLEAN,
          summary TEXT,
          user_persona TEXT,
          improvement_suggestion TEXT,
          sentiment_reason TEXT,
          has_question BOOLEAN,
          raw_json JSONB,
          analyzed_at TIMESTAMPTZ DEFAULT now(),
          UNIQUE (review_id, model_name)
        );
        CREATE TABLE IF NOT EXISTS review_sentence_analysis (
          id SERIAL PRIMARY KEY,
          review_id BIGINT NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
          sent_index INTEGER NOT NULL,
          sentence TEXT,
          sentiment_label VARCHAR(20),
          sentiment_score REAL,
          model_name VARCHAR(50) NOT NULL,
          categories TEXT[],
          keywords TEXT[],
          analyzed_at TIMESTAMPTZ DEFAULT now(),
          UNIQUE (review_id, sent_index, model_name)
        );
        """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
                # Also run alterations to ensure schema is up-to-date
                cur.execute("""
                    ALTER TABLE review_analysis
                    ADD COLUMN IF NOT EXISTS summary TEXT,
                    ADD COLUMN IF NOT EXISTS user_persona TEXT,
                    ADD COLUMN IF NOT EXISTS improvement_suggestion TEXT,
                    ADD COLUMN IF NOT EXISTS sentiment_reason TEXT,
                    ADD COLUMN IF NOT EXISTS has_question BOOLEAN;
                """
                )
                cur.execute("ALTER TABLE review_analysis ALTER COLUMN user_persona TYPE TEXT;")
                cur.execute("ALTER TABLE review_analysis DROP COLUMN IF EXISTS aspects;")
                cur.execute("""
                    ALTER TABLE review_sentence_analysis
                    ADD COLUMN IF NOT EXISTS categories TEXT[],
                    ADD COLUMN IF NOT EXISTS keywords TEXT[];
                """
                )
                conn.commit()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        st.stop()

# Call the function at the start of the app
initialize_database()

st.title("✨ AI 리뷰 분석 대시보드")

# SQL 쿼리를 실행하여 결과를 DataFrame으로 반환하고 캐싱합니다
@st.cache_data(ttl=600)
def fetch_df(sql: str, params=None) -> pd.DataFrame:
    try:
        with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or [])
            rows = cur.fetchall()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()

# AI 총평 생성 함수
@st.cache_data(ttl=3600, show_spinner="AI가 리뷰들을 종합하여 총평을 생성하고 있습니다...")
def generate_overall_summary(_summaries, _product_name):
    if not _summaries:
        return "요약할 리뷰가 없습니다."

    # LangChain을 사용하여 LLM 모델과 프롬프트를 설정합니다
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.5, max_retries=2, request_timeout=120)
    except Exception as e:
        return f"AI 모델 초기화 중 오류: {e}"

    prompt_text = (
        "당신은 고객 리뷰 분석 전문가입니다. 다음은 '{product_name}' 상품에 대한 고객 리뷰 핵심 요약 모음입니다. "
        "이 요약들을 바탕으로, 고객들의 전반적인 반응과 핵심 의견을 종합하여 2~3 문장의 완성된 한글 총평을 작성해주세요. "
        "긍정적인 부분과 부정적인 부분을 균형있게 언급하되, 홍보 문구가 아닌 객관적인 분석가의 시선으로 작성해야 합니다."
        "만약 요약문의 양이 너무 적거나 내용이 부족하다면, 주어진 정보만으로 간단히 정리해주세요.\n\n"
        "--- 리뷰 핵심 요약 모음 ---\n"
        "{review_summaries}\n"
        "---"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful market analysis assistant who summarizes customer feedback."),
        ("human", prompt_text)
    ])

    # 리뷰 요약문들을 하나의 문자열로 합칩니다
    formatted_summaries = "\n- ".join(_summaries)
    
    # LLM 체인을 실행하여 총평을 생성합니다
    chain = prompt | llm
    try:
        response = chain.invoke({
            "product_name": _product_name,
            "review_summaries": formatted_summaries
        })
        return response.content
    except Exception as e:
        st.error(f"AI 총평 생성 중 오류가 발생했습니다: {e}")
        return "AI 총평을 생성하는 데 실패했습니다. 잠시 후 다시 시도해주세요."

# --- 메인 네비게이션 탭 ---

# 앱의 메인 화면을 구성하는 3개의 최상위 탭을 생성합니다
main_tab1, main_tab2, main_tab3 = st.tabs(["📊 통합 대시보드", "📥 데이터 수집", "🧠 AI 분석"])

# --- 탭 1: 통합 대시보드 ---
with main_tab1:
    # --- 사이드바 필터 ---
    with st.sidebar:
        st.header("⚙️ 필터")
        # DB에서 상품 목록을 가져와 선택 옵션을 만듭니다
        df_products = fetch_df('''
            SELECT DISTINCT product_name 
            FROM reviews 
            WHERE product_name IS NOT NULL 
            ORDER BY product_name;
        ''')
        product_choices = ["전체"] + (df_products["product_name"].tolist() if not df_products.empty else [])
        selected_product = st.selectbox("상품 선택", product_choices)

        # 날짜 범위 선택 필터를 생성합니다
        today = date.today()
        start_date = st.date_input("시작일", today - timedelta(days=365))
        end_date = st.date_input("종료일", today)

        # 키워드 검색 및 기타 필터를 생성합니다
        keyword = st.text_input("리뷰 키워드 검색 (선택 사항)")
        exclude_recipe = st.checkbox("레시피성 후기 제외", value=True)

        st.divider()
        if st.button("🔄 캐시 지우고 새로고침"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # --- 필터링 SQL 조건 생성 ---
    # 선택된 필터 값들을 기반으로 SQL의 WHERE 절을 동적으로 구성합니다
    params = [start_date, end_date]
    where_clauses = [
        "a.is_actual_review = true",
        "(r.review_date BETWEEN %s AND %s)"
    ]
    if selected_product != "전체":
        where_clauses.append("r.product_name = %s")
        params.append(selected_product)
    if keyword:
        where_clauses.append("r.content ILIKE %s")
        params.append(f"%{keyword}%")
    if exclude_recipe:
        where_clauses.append("a.is_recipe_like = false")
    WHERE_SQL = " AND ".join(where_clauses)

    # --- 대시보드 상단 요약 정보 ---
    # 펼침/접기 가능한 메뉴로 전체 데이터 현황을 보여줍니다
    with st.expander("📊 대시보드 전체 현황 보기", expanded=True):
        # DB의 각 테이블별 데이터 총 개수를 보여주는 메트릭입니다
        c1, c2, c3 = st.columns(3)
        try:
            reviews_count = fetch_df("SELECT COUNT(*) as count FROM reviews").iloc[0]['count']
            c1.metric("수집된 리뷰 수", f"{reviews_count} 건")
        except Exception as e:
            c1.metric("수집된 리뷰 수", "오류", help=str(e))
        try:
            analysis_count = fetch_df("SELECT COUNT(*) as count FROM review_analysis").iloc[0]['count']
            c2.metric("분석된 리뷰 수", f"{analysis_count} 건")
        except Exception as e:
            c2.metric("분석된 리뷰 수", "오류", help=str(e))
        try:
            sentence_count = fetch_df("SELECT COUNT(*) as count FROM review_sentence_analysis").iloc[0]['count']
            c3.metric("분석된 문장 수", f"{sentence_count} 건")
        except Exception as e:
            c3.metric("분석된 문장 수", "오류", help=str(e))
        
        st.divider()
        # 리뷰가 많은 상위 5개 상품의 핵심 지표를 보여주는 리더보드입니다
        st.markdown("##### ⭐ 주요 상품 현황 (리뷰 수 기준 TOP 5)")
        sql_leaderboard = f'''
            SELECT r.product_name, COUNT(r.id) as review_count,
                   COALESCE(ROUND(AVG(a.star_rating)::numeric, 2), 0) as avg_rating,
                   COALESCE(SUM(CASE WHEN a.sentiment_label = 'positive' THEN 1 ELSE 0 END), 0) as positive_reviews,
                   COALESCE(SUM(CASE WHEN a.sentiment_label = 'negative' THEN 1 ELSE 0 END), 0) as negative_reviews
            FROM reviews r JOIN review_analysis a ON r.id = a.review_id
            WHERE r.product_name IS NOT NULL AND a.is_actual_review = true
            GROUP BY r.product_name ORDER BY review_count DESC LIMIT 5
        '''
        df_leaderboard = fetch_df(sql_leaderboard)
        if not df_leaderboard.empty:
            df_leaderboard["sentiment_dist"] = df_leaderboard.apply(lambda row: [row['positive_reviews'], row['negative_reviews']], axis=1)
            # st.dataframe의 column_config를 사용하여 표 내부를 시각화합니다
            st.dataframe(df_leaderboard, 
                column_config={
                    "product_name": st.column_config.TextColumn("상품명", width="large"),
                    "review_count": st.column_config.NumberColumn("리뷰 수", format="%d 건"),
                    "avg_rating": st.column_config.ProgressColumn("평균 별점", format="%.2f", min_value=1.0, max_value=5.0),
                    "sentiment_dist": st.column_config.BarChartColumn("긍정/부정 분포", y_min=0),
                    "positive_reviews": None, "negative_reviews": None
                },
                hide_index=True, use_container_width=True, column_order=["product_name", "review_count", "avg_rating", "sentiment_dist"]
            )
    st.divider()

    # --- 대시보드 상세 탭 ---
    # 선택된 필터 조건에 대한 상세 분석을 보여주는 탭 메뉴입니다
    d_tab1, d_tab2, d_tab3, d_tab4, d_tab5 = st.tabs(["📊 요약", "🗂️ 카테고리 분석", "🔑 키워드 분석", "✍️ 리뷰 원문", "❓ 도움말"])
    
    # 요약 탭: 선택된 데이터의 핵심 지표와 분포를 보여줍니다
    with d_tab1:
        st.subheader(f"📈 '{selected_product}' 상품 핵심 지표")
        sql_kpi = f"SELECT COALESCE(ROUND(AVG(a.star_rating)::numeric, 2), 0) as avg_rating, COUNT(r.id) as total_reviews, COALESCE(SUM(CASE WHEN a.sentiment_label = 'positive' THEN 1 ELSE 0 END), 0) as positive_reviews, COALESCE(SUM(CASE WHEN a.sentiment_label = 'negative' THEN 1 ELSE 0 END), 0) as negative_reviews FROM reviews r JOIN review_analysis a ON r.id = a.review_id WHERE {WHERE_SQL}"
        df_kpi = fetch_df(sql_kpi, params)
        if df_kpi.empty or df_kpi.iloc[0]["total_reviews"] == 0:
            st.warning("선택하신 조건에 해당하는 리뷰 데이터가 없습니다.")
        else:
            kpi = df_kpi.iloc[0]
            total_sentiments = kpi["positive_reviews"] + kpi["negative_reviews"]
            pos_ratio = (kpi["positive_reviews"] / total_sentiments) * 100 if total_sentiments > 0 else 0
            avg_rating_value = float(kpi['avg_rating'])
            
            # KPI를 카드 형태로 시각화합니다
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True, height=150):
                    st.markdown("<p style='font-size: 18px; text-align: center; font-weight: bold;'>AI 예측 평균 평점</p>", unsafe_allow_html=True)
                    def rating_to_stars_html(rating, font_size=28):
                        rating = float(rating)
                        full_star = f'<span style="font-size: {font_size}px; color: #ffc107;">★</span>'
                        empty_star = f'<span style="font-size: {font_size}px; color: #e0e0e0;">☆</span>'
                        full_stars = int(rating)
                        half_star = 1 if rating - full_stars >= 0.5 else 0
                        empty_stars = 5 - full_stars - half_star
                        return f"{full_star * (full_stars + half_star)}{empty_star * empty_stars}"
                    stars_html = rating_to_stars_html(avg_rating_value)
                    st.markdown(f"<div style='text-align: center;'>{stars_html}</div>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; font-size: 24px; font-weight: bold; margin-top: -5px;'>{avg_rating_value:.2f}</p>", unsafe_allow_html=True)
            with col2:
                with st.container(border=True, height=150):
                    st.markdown("<p style='font-size: 18px; text-align: center; font-weight: bold;'>총 리뷰 수</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 48px; text-align: center; font-weight: bold; line-height: 1.6;'>✍️ {kpi['total_reviews']}</p>", unsafe_allow_html=True)
            with col3:
                with st.container(border=True, height=150):
                    st.markdown("<p style='font-size: 18px; text-align: center; font-weight: bold;'>긍정 리뷰 비율</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 48px; text-align: center; font-weight: bold; line-height: 1.6;'>🙂 {pos_ratio:.1f}%</p>", unsafe_allow_html=True)
            
            st.divider()

            # --- AI 종합 평가 & 주요 구매자 유형 ---
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📝 AI 종합 평가")
                with st.container(border=True, height=250):
                    # 필터링된 리뷰들의 개별 요약문을 가져옵니다
                    sql_summaries = f"SELECT a.summary FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.summary IS NOT NULL"
                    df_summaries = fetch_df(sql_summaries, params)
                    
                    if not df_summaries.empty:
                        summaries_list = df_summaries['summary'].tolist()
                        # AI를 호출하여 종합 요약을 생성합니다
                        overall_summary = generate_overall_summary(summaries_list, selected_product)
                        st.markdown(overall_summary)
                    else:
                        st.info("종합 평가를 생성하기에 충분한 리뷰 요약 데이터가 없습니다.")
            
            with col2:
                st.subheader("👥 주요 구매자 유형")
                with st.container(border=True, height=250):
                    sql_personas = f"SELECT user_persona, COUNT(*) as count FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.user_persona IS NOT NULL GROUP BY user_persona ORDER BY count DESC LIMIT 5"
                    df_personas = fetch_df(sql_personas, params)
                    if not df_personas.empty:
                        df_personas.set_index('user_persona', inplace=True)
                        st.bar_chart(df_personas['count'])
                    else:
                        st.info("구매자 유형을 분석하기에 데이터가 부족합니다.")

            st.divider()

            # --- AI가 분석한 주요 강점, 약점 및 개선 제안 ---
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 주요 강점 및 약점")
                sql_cat_summary = f"SELECT cat, sentiment_label, COUNT(*) as review_count FROM review_analysis a JOIN reviews r ON a.review_id = r.id CROSS JOIN unnest(a.categories) as cat WHERE {WHERE_SQL} AND sentiment_label IN ('positive', 'negative') GROUP BY cat, sentiment_label"
                df_cat_summary = fetch_df(sql_cat_summary, params)
                if not df_cat_summary.empty:
                    with st.container(border=True, height=280):
                        col1_sub, col2_sub = st.columns(2)
                        with col1_sub:
                            st.markdown("<h5>👍 강점</h5>", unsafe_allow_html=True)
                            pos_cats = df_cat_summary[df_cat_summary['sentiment_label'] == 'positive'].nlargest(3, 'review_count')
                            if not pos_cats.empty:
                                for index, row in pos_cats.iterrows():
                                    sql_keywords = f"SELECT keyword, COUNT(*) as count FROM (SELECT unnest(keywords) as keyword FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.sentiment_label = 'positive' AND %s = ANY(a.categories)) as kw WHERE keyword IS NOT NULL GROUP BY keyword ORDER BY count DESC LIMIT 3"
                                    df_keywords = fetch_df(sql_keywords, params + [row['cat']])
                                    keywords_str = ", ".join(df_keywords['keyword'].tolist()) if not df_keywords.empty else ""
                                    st.markdown(f"- **{row['cat']}** ({row['review_count']} 건) <br> &nbsp; *<span style='color: #666;'>{keywords_str}</span>*", unsafe_allow_html=True)
                            else:
                                st.info("분석된 강점이 없습니다.")
                        with col2_sub:
                            st.markdown("<h5>👎 약점</h5>", unsafe_allow_html=True)
                            neg_cats = df_cat_summary[df_cat_summary['sentiment_label'] == 'negative'].nlargest(3, 'review_count')
                            if not neg_cats.empty:
                                for index, row in neg_cats.iterrows():
                                    sql_keywords = f"SELECT keyword, COUNT(*) as count FROM (SELECT unnest(keywords) as keyword FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.sentiment_label = 'negative' AND %s = ANY(a.categories)) as kw WHERE keyword IS NOT NULL GROUP BY keyword ORDER BY count DESC LIMIT 3"
                                    df_keywords = fetch_df(sql_keywords, params + [row['cat']])
                                    keywords_str = ", ".join(df_keywords['keyword'].tolist()) if not df_keywords.empty else ""
                                    st.markdown(f"- **{row['cat']}** ({row['review_count']} 건) <br> &nbsp; *<span style='color: #666;'>{keywords_str}</span>*", unsafe_allow_html=True)
                            else:
                                st.info("분석된 약점이 없습니다.")
                else:
                    st.info("강점/약점 분석을 위한 데이터가 부족합니다.")
            
            with c2:
                st.subheader("💡 AI의 개선 제안")
                sql_suggestions = f"SELECT DISTINCT improvement_suggestion FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.improvement_suggestion IS NOT NULL LIMIT 5"
                df_suggestions = fetch_df(sql_suggestions, params)
                with st.container(border=True, height=280):
                    if not df_suggestions.empty:
                        for suggestion in df_suggestions['improvement_suggestion']:
                            st.info(f"💡 {suggestion}")
                    else:
                        st.info("AI가 제안한 개선점이 없습니다.")

            st.divider()
            # 긍/부정 비율과 평점 분포를 그래프로 보여줍니다
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**🙂 긍/부정 리뷰 비율**")
                if total_sentiments > 0:
                    fig_pie = go.Figure(data=[go.Pie(labels=['긍정', '부정'], values=[kpi["positive_reviews"], kpi["negative_reviews"]], hole=.6, marker_colors=['#3A7DFF', '#E9ECEF'], hoverinfo='label+percent', textinfo='none')])
                    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), annotations=[dict(text=f"<b>{pos_ratio:.1f}%</b>", x=0.5, y=0.5, font_size=28, showarrow=False, font_family="Arial")])
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("데이터가 부족하여 차트를 표시할 수 없습니다.")
            with c2:
                st.markdown("**⭐ AI 예측 평점 분포**")
                sql_rating_dist = f"SELECT star_rating, COUNT(*) as count FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.star_rating IS NOT NULL GROUP BY star_rating ORDER BY star_rating"
                df_rating_dist = fetch_df(sql_rating_dist, params)
                if not df_rating_dist.empty:
                    def rating_to_stars_label(rating):
                        rating = float(rating)
                        full_star = '★'; empty_star = '☆'
                        full_stars = int(rating)
                        half_star = 1 if rating - full_stars >= 0.5 else 0
                        empty_stars = 5 - full_stars - half_star
                        return f"{full_star * full_stars}{full_star * half_star}{empty_star * empty_stars}"
                    df_rating_dist['star_label'] = df_rating_dist['star_rating'].apply(lambda r: f"{rating_to_stars_label(r)} ({r})")
                    fig = px.bar(df_rating_dist, x='star_label', y='count', color='star_rating', color_continuous_scale=px.colors.sequential.YlOrRd, text='count')
                    fig.update_layout(xaxis_title="AI Predicted Rating", yaxis_title="Review Count", showlegend=False, coloraxis_showscale=False, uniformtext_minsize=8, uniformtext_mode='hide')
                    fig.update_traces(textposition='outside', hovertemplate='<b>Rating: %{customdata[0]}</b><br>Count: %{y}<extra></extra>', customdata=df_rating_dist[['star_rating']])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("분포를 그리기에 데이터가 부족합니다.")

    # 카테고리 분석 탭: 카테고리별로 데이터를 심층 분석합니다
    with d_tab2:
        st.subheader("🗂️ 카테고리별 상세 분석")
        st.markdown("**카테고리별 긍/부정 리뷰 비율**")
        sql_cat_sentiment = f"SELECT cat, SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive, SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative FROM review_analysis a JOIN reviews r ON a.review_id = r.id CROSS JOIN unnest(a.categories) as cat WHERE {WHERE_SQL} AND a.categories IS NOT NULL GROUP BY cat HAVING COUNT(*) > 2"
        df_cat_sentiment = fetch_df(sql_cat_sentiment, params)
        if not df_cat_sentiment.empty:
            df_cat_sentiment['total'] = df_cat_sentiment['positive'] + df_cat_sentiment['negative']
            df_cat_sentiment['pos_ratio'] = (df_cat_sentiment['positive'] / df_cat_sentiment['total']) * 100
            df_cat_sentiment['neg_ratio'] = (df_cat_sentiment['negative'] / df_cat_sentiment['total']) * 100
            fig_cat_bar = go.Figure()
            fig_cat_bar.add_trace(go.Bar(y=df_cat_sentiment['cat'], x=df_cat_sentiment['pos_ratio'], name='긍정', orientation='h', marker_color='#4CAF50'))
            fig_cat_bar.add_trace(go.Bar(y=df_cat_sentiment['cat'], x=df_cat_sentiment['neg_ratio'], name='부정', orientation='h', marker_color='#F44336'))
            fig_cat_bar.update_layout(barmode='stack', yaxis_title="카테고리", xaxis_title="비율 (%)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cat_bar, use_container_width=True)
        else:
            st.info("카테고리별 긍/부정 비율을 분석하기에 데이터가 부족합니다.")
        st.divider()
        st.markdown("**카테고리별 AI 예측 평점 분포**")
        sql_cat_dist = f"SELECT unnest(a.categories) as category, a.star_rating, COUNT(*) as count FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.star_rating IS NOT NULL AND a.categories IS NOT NULL AND array_length(a.categories, 1) > 0 GROUP BY category, a.star_rating"
        df_cat_dist = fetch_df(sql_cat_dist, params)
        if not df_cat_dist.empty:
            pivot_df_counts = df_cat_dist.pivot_table(index='category', columns='star_rating', values='count', fill_value=0)
            row_sums = pivot_df_counts.sum(axis=1)
            safe_row_sums = row_sums.replace(0, 1)
            pivot_df_perc = pivot_df_counts.div(safe_row_sums, axis=0) * 100
            categories = pivot_df_perc.index.tolist()
            star_ratings = sorted(pivot_df_perc.columns.tolist())
            colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
            star_color_map = {1.0: colors[0], 1.5: colors[0], 2.0: colors[1], 2.5: colors[1], 3.0: colors[2], 3.5: colors[2], 4.0: colors[3], 4.5: colors[3], 5.0: colors[4]}
            series = []
            for rating in star_ratings:
                series.append({"name": f'{rating}점', "type": 'bar', "stack": 'total', "label": {"show": False}, "emphasis": {"focus": 'series'}, "data": [round(x, 1) for x in pivot_df_perc[rating].tolist()], "itemStyle": {"color": star_color_map.get(rating, '#ccc')}})
            options = {"tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}}, "legend": {"data": [s['name'] for s in series]}, "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True}, "xAxis": {"type": "value", "max": 100}, "yAxis": {"type": "category", "data": categories}, "series": series}
            st_echarts(options=options, height=f"{len(categories) * 50 + 100}px")
        else:
            st.info("카테고리별 평점 분포를 분석하기에 데이터가 부족합니다.")
        st.divider()
        st.markdown("**카테고리별 대표 키워드 및 리뷰**")
        if not df_cat_sentiment.empty:
            selected_cat = st.selectbox("분석할 카테고리 선택", df_cat_sentiment['cat'].unique(), key="category_selector")
            if selected_cat:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**긍정/부정 대표 키워드**")
                    sql_cat_kw = f"SELECT sentiment_label, keyword, COUNT(*) as count FROM (SELECT unnest(keywords) as keyword, sentiment_label FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND %s = ANY(a.categories) AND a.keywords IS NOT NULL) as kw_sent WHERE sentiment_label IN ('positive', 'negative') GROUP BY sentiment_label, keyword ORDER BY count DESC LIMIT 10"
                    df_cat_kw = fetch_df(sql_cat_kw, params + [selected_cat])
                    st.dataframe(df_cat_kw, use_container_width=True)
                with c2:
                    st.markdown(f"**대표 긍정/부정 리뷰**")
                    sql_cat_rev = f"SELECT content, sentiment_label, star_rating FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND %s = ANY(a.categories) ORDER BY sentiment_score DESC LIMIT 1"
                    df_pos_rev = fetch_df(sql_cat_rev, params + [selected_cat])
                    if not df_pos_rev.empty: st.success(f"**[긍정]** {df_pos_rev.iloc[0]['content']}")
                    sql_cat_rev_neg = sql_cat_rev.replace("DESC", "ASC")
                    df_neg_rev = fetch_df(sql_cat_rev_neg, params + [selected_cat])
                    if not df_neg_rev.empty: st.error(f"**[부정]** {df_neg_rev.iloc[0]['content']}")
        else:
            st.info("먼저 위에서 카테고리 분석을 진행해주세요.")

    with d_tab3:
        st.subheader("🔑 키워드 상세 분석")
        st.markdown("**긍정 vs 부정 리뷰 핵심 키워드**")
        sql_keywords = f"SELECT unnest(keywords) as keyword, sentiment_label FROM review_analysis a JOIN reviews r ON a.review_id = r.id WHERE {WHERE_SQL} AND a.keywords IS NOT NULL AND sentiment_label IN ('positive', 'negative')"
        df_keywords = fetch_df(sql_keywords, params)
        if not df_keywords.empty:
            c1, c2 = st.columns(2)
            try:
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
                pos_text = " ".join(df_keywords[df_keywords['sentiment_label'] == 'positive']['keyword'])
                neg_text = " ".join(df_keywords[df_keywords['sentiment_label'] == 'negative']['keyword'])
                with c1:
                    st.markdown("<p style='text-align: center;'>🙂 긍정 키워드</p>", unsafe_allow_html=True)
                    if pos_text.strip():
                        wc_pos = WordCloud(font_path=font_path, background_color="white", width=400, height=300).generate(pos_text)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("긍정 키워드가 없습니다.")
                with c2:
                    st.markdown("<p style='text-align: center;'>🙁 부정 키워드</p>", unsafe_allow_html=True)
                    if neg_text.strip():
                        wc_neg = WordCloud(font_path=font_path, background_color="white", width=400, height=300).generate(neg_text)
                        fig, ax = plt.subplots()
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("부정 키워드가 없습니다.")
            except Exception as e:
                st.error(f"워드클라우드 생성 중 오류: {e}. 폰트 경로를 확인하세요.")
        else:
            st.info("키워드 분석을 위한 데이터가 부족합니다.")

    with d_tab4:
        st.subheader("✍️ 리뷰 원문 보기")
        sql_raw = f"SELECT r.review_date as \"날짜\", r.product_name as \"상품명\", r.content as \"내용\", a.star_rating as \"AI평점\", a.sentiment_label as \"AI감성\", a.keywords as \"AI키워드\" FROM reviews r JOIN review_analysis a ON r.id = a.review_id WHERE {WHERE_SQL} ORDER BY r.review_date DESC, r.id DESC LIMIT 500"
        df_raw = fetch_df(sql_raw, params)
        if not df_raw.empty:
            st.dataframe(df_raw, use_container_width=True)
            csv = df_raw.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSV 파일로 다운로드", data=csv, file_name="review_analysis.csv", mime="text/csv")
        else:
            st.warning("표시할 리뷰 데이터가 없습니다.")

    with d_tab5:
        st.subheader("❓ 도움말")
        with st.expander("용어 및 데이터 정보", expanded=True):
            st.markdown('''
            #### 용어 정의
            - **AI 예측 평점**: 리뷰 텍스트의 긍정/부정 뉘앙스, 사용된 어휘 등을 AI가 종합적으로 분석하여 1.0 ~ 5.0 사이의 점수로 변환한 값입니다.
            - **긍정/부정 리뷰**: AI가 리뷰의 전체적인 어조를 'positive', 'negative', 'neutral'로 분류한 결과입니다.
            - **카테고리**: AI가 리뷰 내용을 바탕으로 '맛', '가격' 등 미리 정의된 주제 중 어떤 것과 관련있는지 분석한 결과입니다.
            
            #### 데이터 원본
            - 마켓컬리의 특정 상품 페이지들에서 수집된 공개 사용자 리뷰를 기반으로 합니다.
            ''')
        st.markdown("### AI 리뷰 분석 대시보드 사용법")

# --- 탭 2: 데이터 수집 ---
with main_tab2:
    st.header("📥 데이터 수집 (웹 크롤링)")
    st.markdown("이 페이지에서 마켓컬리 웹사이트로부터 최신 리뷰 데이터를 수집(크롤링)할 수 있습니다.")
    if st.button("크롤링 시작하기", type="primary"):
        script_path = "kurly.py"
        if not os.path.exists(script_path):
            st.error(f"`{script_path}` 파일을 찾을 수 없습니다.")
        else:
            st.info("크롤링을 시작합니다...")
            log_area = st.empty()
            log_text = ""
            try:
                process = subprocess.Popen([sys.executable, "-u", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in iter(process.stdout.readline, ''):
                    log_text += line
                    log_area.code(log_text, language='log')
                process.stdout.close()
                process.wait()
                if process.returncode == 0:
                    st.success("✅ 크롤링이 성공적으로 완료되었습니다!")
                    st.balloons()
                else:
                    st.error("❌ 크롤링 중 오류가 발생했습니다.")
            except Exception as e:
                st.error(f"스크립트 실행 중 예외가 발생했습니다: {e}")

# --- 탭 3: AI 분석 ---
with main_tab3:
    st.header("🧠 AI 리뷰 분석")
    st.markdown("수집된 리뷰 데이터를 AI 모델(GPT-4o)로 분석합니다.")
    if st.button("AI 분석 시작하기", type="primary"):
        script_path = "analyze.py"
        if not os.path.exists(script_path):
            st.error(f"`{script_path}` 파일을 찾을 수 없습니다.")
        else:
            st.info("AI 분석을 시작합니다...")
            log_area = st.empty()
            log_text = ""
            try:
                process = subprocess.Popen([sys.executable, "-u", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in iter(process.stdout.readline, ''):
                    log_text += line
                    log_area.code(log_text, language='log')
                process.stdout.close()
                process.wait()
                if process.returncode == 0:
                    st.success("✅ AI 분석이 성공적으로 완료되었습니다!")
                    st.balloons()
                else:
                    st.error("❌ 크롤링 중 오류가 발생했습니다.")
            except Exception as e:
                st.error(f"스크립트 실행 중 예외가 발생했습니다: {e}")
    st.divider()
    st.subheader("참고: 데이터베이스 초기화")
    if st.button("모든 데이터 삭제하기", type="secondary"):
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE reviews, review_analysis, review_sentence_analysis RESTART IDENTITY;")
                conn.commit()
            st.success("✅ 모든 데이터가 성공적으로 삭제되었습니다.")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"데이터 삭제 중 오류 발생: {e}")