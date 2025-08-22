-- 1. 수집된 리뷰 원본 테이블
CREATE TABLE reviews (
  id BIGSERIAL PRIMARY KEY,
  product_name VARCHAR(255),
  content TEXT NOT NULL,
  review_date DATE,
  content_hash CHAR(64) NOT NULL UNIQUE,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. 리뷰 전체에 대한 AI 분석 결과 테이블 
CREATE TABLE review_analysis (
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

-- 3. 문장 단위 세부 분석 결과 테이블 
CREATE TABLE review_sentence_analysis (
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
