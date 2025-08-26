# analyze.py
import os
import sys
import io
import json
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from openai import APIError

# Windows 환경의 subprocess에서 한글 출력이 깨지는 현상 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- 환경설정 및 상수 정의 ---

# .env 파일에서 환경 변수를 불러옵니다
load_dotenv()
# 분석에 사용할 LLM 모델을 지정합니다
MODEL_NAME = "gpt-4o"
# 한 번에 데이터베이스에서 가져올 리뷰의 수를 정의합니다
BATCH_SIZE = 100
# 데이터베이스 연결 정보를 설정합니다
PG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5432")),
    dbname=os.getenv("PGDATABASE", "kurlydb"),
    user=os.getenv("PGUSER", "kurlyuser"),
    password=os.getenv("PGPASSWORD", "kurlypassword"),
)
# AI가 분석할 리뷰의 주요 카테고리 목록을 정의합니다
CATEGORIES = ["맛", "가격/가성비", "품질/신선도", "양", "포장/배송", "조리 편의성", "재구매 의사"]

# --- 데이터베이스 유틸리티 함수 ---

# 데이터베이스 연결 객체를 생성하는 함수입니다
def get_conn():
    return psycopg2.connect(**PG, client_encoding='utf8')



# 아직 분석되지 않은 리뷰를 데이터베이스에서 가져오는 함수입니다
def fetch_unanalyzed_reviews(limit=BATCH_SIZE):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT r.id, r.content
            FROM reviews r
            LEFT JOIN review_analysis a ON r.id = a.review_id AND a.model_name = %s
            WHERE a.id IS NULL
            ORDER BY r.id
            LIMIT %s
        """, (MODEL_NAME, limit))
        return cur.fetchall()

# AI 분석 결과를 데이터베이스의 각 테이블에 저장하는 함수입니다
def save_analysis_results(analysis_data, sentence_data):
    with get_conn() as conn, conn.cursor() as cur:
        # 리뷰 전체 분석 결과를 저장합니다
        execute_values(
            cur,
            """
            INSERT INTO review_analysis (
                review_id, model_name, is_actual_review, keywords, categories, sentiment_label,
                sentiment_score, star_rating, is_recipe_like, raw_json,
                summary, user_persona, improvement_suggestion, sentiment_reason, has_question
            ) VALUES %s
            ON CONFLICT (review_id, model_name) DO UPDATE SET
                is_actual_review = EXCLUDED.is_actual_review,
                keywords = EXCLUDED.keywords,
                categories = EXCLUDED.categories,
                sentiment_label = EXCLUDED.sentiment_label,
                sentiment_score = EXCLUDED.sentiment_score,
                star_rating = EXCLUDED.star_rating,
                is_recipe_like = EXCLUDED.is_recipe_like,
                raw_json = EXCLUDED.raw_json,
                summary = EXCLUDED.summary,
                user_persona = EXCLUDED.user_persona,
                improvement_suggestion = EXCLUDED.improvement_suggestion,
                sentiment_reason = EXCLUDED.sentiment_reason,
                has_question = EXCLUDED.has_question,
                analyzed_at = now()
            """,
            analysis_data,
            page_size=100
        )
        # 문장 단위 분석 결과를 저장합니다
        if sentence_data:
            execute_values(
                cur,
                """
                INSERT INTO review_sentence_analysis (
                    review_id, sent_index, sentence, sentiment_label, sentiment_score, model_name, categories, keywords
                ) VALUES %s
                ON CONFLICT (review_id, sent_index, model_name) DO UPDATE SET
                    sentence = EXCLUDED.sentence,
                    sentiment_label = EXCLUDED.sentiment_label,
                    sentiment_score = EXCLUDED.sentiment_score,
                    categories = EXCLUDED.categories,
                    keywords = EXCLUDED.keywords,
                    analyzed_at = now()
                """,
                sentence_data,
                page_size=500
            )
        conn.commit()

# --- LangChain 모델 및 프롬프트 설정 ---

# AI가 문장 단위로 출력할 데이터의 구조를 정의합니다
class SentenceAnalysis(BaseModel):
    sentence: str = Field(description="분석된 문장")
    sentiment_label: str = Field(description="문장의 감성 (positive, neutral, negative)")
    sentiment_score: float = Field(description="문장의 감성 점수 (-1.0 ~ 1.0)")
    categories: Optional[List[str]] = Field(description=f"이 문장이 해당하는 카테고리 목록. 선택 가능: {CATEGORIES}")
    keywords: Optional[List[str]] = Field(description="이 문장의 핵심 키워드 (3개 이내)")

# AI가 리뷰 전체에 대해 최종적으로 출력할 데이터의 구조를 정의합니다
class ReviewAnalysis(BaseModel):
    is_actual_review: bool = Field(description="텍스트가 실제 상품 경험에 대한 리뷰가 맞는지 여부.")
    star_rating: Optional[float] = Field(description="리뷰 전체 내용 기반으로 추정된 별점 (1.0 ~ 5.0, 0.5 단위)")
    is_recipe_like: Optional[bool] = Field(description="상품 자체보다 요리법/레시피에 초점을 맞춘 후기인지 여부")
    summary: Optional[str] = Field(description="리뷰의 핵심 내용을 한 문장으로 요약")
    user_persona: Optional[str] = Field(description="리뷰 작성자의 유형 추정 (예: 1인 가구, 요리 초보 등)")
    sentiment_reason: Optional[str] = Field(description="리뷰의 전반적인 감성이 결정된 핵심 이유를 짧은 구문으로 요약.")
    improvement_suggestion: Optional[str] = Field(description="부정적 리뷰 내용 기반의 구체적인 제품/서비스 개선 제안. 긍정 리뷰의 경우 null.")
    has_question: Optional[bool] = Field(description="리뷰에 판매자나 다른 고객에게 묻는 질문이 포함되어 있는지 여부 (true/false)")
    sentences: Optional[List[SentenceAnalysis]] = Field(description="리뷰를 문장 단위로 나누어, 각 문장별로 감성, 카테고리, 키워드를 상세 분석한 결과")

# OpenAI 모델을 설정합니다. 오류 시 3번까지 재시도합니다
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0, max_retries=3, request_timeout=120)
# LLM이 지정된 JSON 구조로 답변하도록 설정합니다
structured_llm = llm.with_structured_output(ReviewAnalysis)

# AI에게 역할을 부여하고, 상세한 분석 방법을 지시하는 프롬프트입니다
system_prompt_text = (
    '당신은 주어진 사용자 리뷰를 문장 단위로 매우 상세하게 분석하는 전문 AI입니다. 다음 지침을 엄격히 준수하여 결과를 JSON 형식으로 출력하세요.\n\n'
    '1. **리뷰 유형 판단**: `is_actual_review` 필드에 실제 상품 리뷰가 맞는지 판단하세요. (광고, 문의 등은 false)\n'
    '   - 실제 리뷰가 아니라면, `is_actual_review`를 `false`로 설정하고 다른 모든 필드는 `null`로 비워두세요.\n'
    '2. **리뷰 전체 종합 분석 (실제 리뷰일 경우에만 수행)**:\n'
    '   - `star_rating`: 리뷰 내용 전체를 바탕으로 1.0에서 5.0 사이의 별점을 0.5점 단위로 예측하세요.\n'
    '   - `summary`: 리뷰 전체의 핵심 내용을 한 문장으로 요약하세요.\n'
    '   - `user_persona`: 리뷰 전체 내용으로 미루어 보아, 어떤 사용자가 작성했을지 (예: 1인 가구, 대가족, 요리 초보 등) 추정하여 기입하세요.\n'
    '   - `sentiment_reason`: 리뷰의 전반적인 감성(긍정/부정)이 결정된 핵심적인 이유를 짧은 구문으로 요약하세요. (예: "신선도가 뛰어나고 맛이 좋음", "포장 불량 및 배송 지연")\n'
    '   - `has_question`: 리뷰 내용에 명확한 질문(예: "~인가요?", "~어떻게 하나요?")이 포함되어 있는지 `true`/`false`로 판단하세요.\n'
    '   - `is_recipe_like`: 상품 자체보다 요리법/레시피에 초점을 맞춘 후기인지 여부를 판단하세요.\n\n'
    '3. **문장 단위 상세 분석 (가장 중요)**:\n'
    '   - `sentences` 필드에 리뷰를 문장 단위로 나누어 각각 분석 결과를 채워주세요.\n'
    '   - 각 문장마다, `sentiment_label`, `sentiment_score`, `categories`, `keywords`를 반드시 모두 분석해야 합니다.\n'
    '   - `categories`: 해당 문장이 어떤 카테고리({categories})에 대한 내용인지 명확히 판단하여 목록으로 만드세요. 관련 없는 문장은 빈 목록 `[]`으로 두세요.\n'
    '   - `keywords`: 해당 문장의 핵심 키워드를 3개 이내로 추출하세요.'
)

# 위에서 정의한 시스템 프롬프트와 사용자 입력을 결합하여 최종 프롬프트를 구성합니다
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text.format(categories=", ".join(CATEGORIES))),
    ("human", "다음 리뷰를 분석해 주세요:\n\n---\n{review_text}\n---")
])

# 프롬프트와 구조화된 출력을 지원하는 LLM을 연결하여 분석 체인을 완성합니다
chain = prompt | structured_llm

# --- 메인 실행 로직 ---

# 이 스크립트가 직접 실행될 때 동작하는 메인 함수입니다
def main():
    print("리뷰 분석을 시작합니다...")
    
    # 분석할 리뷰가 없을 때까지 무한 루프를 돕니다
    while True:
        # DB에서 아직 분석되지 않은 리뷰를 배치 사이즈만큼 가져옵니다
        reviews_to_analyze = fetch_unanalyzed_reviews(BATCH_SIZE)
        # 더 이상 분석할 리뷰가 없으면 루프를 종료합니다
        if not reviews_to_analyze:
            print("분석할 리뷰가 더 이상 없습니다. 종료합니다.")
            break

        print(f"DB에서 {len(reviews_to_analyze)}개의 리뷰를 가져왔습니다. 분석을 시작합니다...")

        # 분석 결과를 임시로 저장할 리스트를 초기화합니다
        analysis_results_for_db = []
        sentence_results_for_db = []

        # 가져온 리뷰들을 하나씩 순회하며 분석합니다
        for review in tqdm(reviews_to_analyze, desc="리뷰 분석 중"):
            try:
                # LangChain 체인을 실행하여 AI에게 리뷰 분석을 요청합니다
                result: ReviewAnalysis = chain.invoke({"review_text": review["content"]})

                # AI가 실제 리뷰가 아니라고 판단한 경우, 최소 정보만 기록하고 넘어갑니다
                if not result.is_actual_review:
                    analysis_results_for_db.append(
                        (
                            review["id"], MODEL_NAME, False, None, None, 'not_a_review',
                            None, None, None, json.dumps(result.dict()),
                            None, None, None, None, None
                        )
                    )
                    continue
                
                # 문장별 분석 결과로부터 리뷰 전체의 키워드, 카테고리, 감성 점수 등을 종합합니다
                all_keywords, all_categories, sentiment_scores = [], [], []
                if result.sentences:
                    for sent in result.sentences:
                        if sent.keywords: all_keywords.extend(sent.keywords)
                        if sent.categories: all_categories.extend(sent.categories)
                        if sent.sentiment_score: sentiment_scores.append(sent.sentiment_score)
                
                # Pandas 경고를 피하고, 순서를 유지하는 방식으로 중복을 제거합니다
                final_keywords = list(dict.fromkeys(all_keywords)) if all_keywords else None
                final_categories = list(dict.fromkeys(all_categories)) if all_categories else None
                
                # 문장별 감성 점수의 평균을 내어 리뷰 전체의 감성을 결정합니다
                avg_sentiment_score = float(np.mean(sentiment_scores)) if sentiment_scores else None
                if avg_sentiment_score is not None:
                    if avg_sentiment_score > 0.1: sentiment_label = 'positive'
                    elif avg_sentiment_score < -0.1: sentiment_label = 'negative'
                    else: sentiment_label = 'neutral'
                else:
                    sentiment_label = None

                # 리뷰 전체 분석 결과를 DB에 저장하기 위해 튜플 형태로 준비합니다
                analysis_results_for_db.append((
                    review["id"], MODEL_NAME, True, final_keywords, final_categories,
                    sentiment_label, avg_sentiment_score, result.star_rating,
                    result.is_recipe_like, json.dumps(result.dict()),
                    result.summary, result.user_persona, result.improvement_suggestion,
                    result.sentiment_reason, result.has_question
                ))

                # 문장별 분석 결과를 DB에 저장하기 위해 튜플 리스트에 추가합니다
                if result.sentences:
                    for i, sent in enumerate(result.sentences):
                        sentence_results_for_db.append(( 
                            review["id"], i, sent.sentence, sent.sentiment_label, 
                            sent.sentiment_score, MODEL_NAME, sent.categories, sent.keywords
                        ))

            except (psycopg2.Error, APIError) as e:
                review_content_safe = review['content'][:200].encode('utf-8', 'replace').decode('utf-8')
                print(f"\n--- ERROR: 리뷰 ID {review['id']} 처리 중 DB 또는 API 오류 발생 ---")
                print(f"리뷰 내용 (앞 200자): {review_content_safe}...")
                print(f"오류 유형: {type(e).__name__}")
                print(f"오류 메시지: {e}")
                traceback.print_exc()
                print("-------------------------------------------------")
                continue
            except Exception as e:
                review_content_safe = review['content'][:200].encode('utf-8', 'replace').decode('utf-8')
                print(f"\n--- ERROR: 리뷰 ID {review['id']} 분석 중 예상치 못한 오류 발생 ---")
                print(f"리뷰 내용 (앞 200자): {review_content_safe}...")
                print(f"오류 유형: {type(e).__name__}")
                print(f"오류 메시지: {e}")
                traceback.print_exc()
                print("-------------------------------------------------")
                continue

        # 한 배치의 분석이 모두 끝나면, 결과를 DB에 한꺼번에 저장합니다
        if analysis_results_for_db:
            print(f"{len(analysis_results_for_db)}개 리뷰의 분석 결과를 DB에 저장합니다...")
            save_analysis_results(analysis_results_for_db, sentence_results_for_db)
            print("저장 완료.")

# 이 스크립트가 직접 실행될 때 main 함수를 호출합니다
if __name__ == "__main__":
    main()
