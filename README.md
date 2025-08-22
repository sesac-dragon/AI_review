# AI 리뷰 분석 대시보드

이 프로젝트는 컬리 (Kurly) 웹사이트에서 상품 리뷰 데이터를 수집하고, OpenAI의 GPT 모델을 통해 긍정/부정 감성, 주요 키워드, 카테고리 등을 분석하여 대시보드 형태로 시각화하는 스트림릿 앱입니다. 사용자는 이 대시보드를 통해 특정 상품에 대한 고객들의 반응을 직관적으로 파악하고, AI가 생성한 종합 평가 및 개선 제안을 확인할 수 있습니다.

## 🌟 주요 기능

- **웹 크롤링**: Selenium을 사용하여 컬리 웹사이트의 상품 후기 페이지에서 동적으로 리뷰 데이터를 수집합니다.

- **AI 기반 리뷰 분석**: LangChain과 OpenAI GPT 모델을 활용하여 각 리뷰를 다각도로 심층 분석합니다.
  - 긍/부정/중립 감성 분류 및 점수화
  - 리뷰 내용 기반의 AI 예측 별점
  - 주요 카테고리 및 키워드 추출
  - 사용자 페르소나 추정 및 개선점 제안
- **데이터베이스 연동**: 수집된 원본 리뷰와 AI 분석 결과를 PostgreSQL 데이터베이스에 저장하고 관리합니다.

- **인터랙티브 대시보드**: Streamlit으로 구축된 웹 대시보드를 통해 분석 결과를 시각화하고, 사용자가 직접 필터링하며 데이터를 탐색할 수 있습니다.

## 🛠️ 기술 스택

- **Language**: Python
- **Crawling**: Selenium, WebDriver Manager
- **AI & NLP**: LangChain, OpenAI API
- **Database**: PostgreSQL
- **Web Framework**: Streamlit
- **Containerization**: Docker, Docker Compose

---

## 🚀 프로젝트 설정 및 실행 방법

새로운 데스크톱 환경에서 이 프로젝트를 실행하기 위해 아래의 단계를 따르세요.

### 1. 사전 준비 사항

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)을 설치하고 실행합니다.

### 2. 환경 변수 설정

프로젝트 루트 디렉터리에 `.env` 파일을 생성하고, 함께 제공된 `.env.example` 파일의 내용을 복사하여 실제 값으로 채워넣습니다.

```
# .env 파일 예시

# PostgreSQL Docker Container Connection Info
PGHOST=db
PGPORT=5432
PGDATABASE=kurlydb
PGUSER=kurlyuser
PGPASSWORD=kurlypassword

# OpenAI API Key
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

- `OPENAI_API_KEY`에는 자신의 OpenAI API 키 입력

### 3. Docker 컨테이너 실행

터미널에서 아래 명령어를 실행하여 Docker 컨테이너를 빌드하고 실행합니다. 이 과정에서 필요한 모든 환경(PostgreSQL, Chrome, Python 라이브러리 등)이 자동으로 구성됩니다.

```bash
docker-compose up -d --build
```

### 4. 애플리케이션 사용

컨테이너 실행이 완료되면, 웹 브라우저에서 아래 주소로 접속하여 애플리케이션을 사용합니다.

- **URL**: `http://localhost:8501`

최초 실행 시 데이터베이스에 테이블이 자동으로 생성됩니다. 이후 아래 순서에 따라 데이터를 수집하고 분석하세요.

1.  **데이터 수집**: **[📥 데이터 수집]** 탭으로 이동하여 **'크롤링 시작하기'** 버튼을 클릭합니다.
2.  **AI 분석**: 수집이 완료되면 **[🧠 AI 분석]** 탭으로 이동하여 **'AI 분석 시작하기'** 버튼을 클릭합니다.
3.  **대시보드 확인**: 분석이 완료되면 **[📊 통합 대시보드]**에서 시각화된 분석 결과를 확인할 수 있습니다.