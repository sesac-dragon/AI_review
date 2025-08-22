# Python 3.11 버전의 공식 이미지를 기반으로 합니다.
FROM python:3.11-slim

# Install system dependencies for Google Chrome
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    fonts-liberation \
    fonts-nanum \

    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    wget \
    xdg-utils

# Download and install Google Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y ./google-chrome-stable_current_amd64.deb \
    && rm google-chrome-stable_current_amd64.deb

# 컨테이너 내 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app


# requirements.txt 파일을 먼저 복사하여 의존성 변경 시에만 재설치하도록 합니다.
COPY requirements.txt .

# pip를 최신 버전으로 업그레이드하고, requirements.txt에 명시된 라이브러리를 설치합니다.
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 모든 파일을 컨테이너의 /app 디렉토리로 복사합니다.
COPY . .

# Streamlit이 사용하는 8501 포트를 외부에 노출하도록 설정합니다.
EXPOSE 8501

# 컨테이너가 시작될 때 Streamlit 앱을 실행하는 명령입니다.
CMD ["streamlit", "run", "app.py"]
