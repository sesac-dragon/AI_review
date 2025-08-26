import sys
import io

# Windows 환경의 subprocess에서 한글 출력이 깨지는 현상 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 필요한 라이브러리들을 불러옵니다
from dotenv import load_dotenv
import os, re, time, hashlib, json
from dataclasses import dataclass
from typing import List, Optional, Set
from datetime import datetime
import argparse

# 웹 브라우저 제어를 위한 Selenium 관련 라이브러리들을 불러옵니다
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# .env 파일에서 환경 변수를 불러옵니다
load_dotenv()

# 크롤링할 최대 페이지 수를 정의합니다
MAX_PAGES   = 30
# 웹 페이지 로딩을 기다릴 최대 시간(초)을 정의합니다
WAIT_SEC    = 15
# 페이지 이동 등 작업 후 잠시 기다릴 시간(초)을 정의합니다
SLEEP_SEC   = 2.0

# 데이터 클래스를 사용하여 리뷰 데이터의 구조를 정의합니다
@dataclass
class Review:
    product_name: Optional[str]
    content: str
    date: Optional[str]

# Selenium 웹드라이버를 설정하고 생성하는 함수입니다
def setup_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1400,2200")
    opts.add_argument("--lang=ko-KR")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64" \
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    return driver

# 여러 XPath 중 하나라도 나타날 때까지 기다리는 함수입니다
def wait_any(driver, xpaths: list, timeout=WAIT_SEC):
    end = time.time() + timeout
    last_err = None
    while time.time() < end:
        for xp in xpaths:
            try:
                el = driver.find_element(By.XPATH, xp)
                if el: return el
            except Exception as e:
                last_err = e
        time.sleep(0.2)
    if last_err: raise TimeoutException(str(last_err))
    raise TimeoutException("wait_any timeout")

# 웹페이지의 팝업 창을 닫는 함수입니다
def close_popups(driver):
    for xp in [
        "//button[contains(., '닫기') or contains(., 'Close')]",
        "//button[@aria-label='닫기' or @aria-label='Close']",
        "//div[@role='dialog']//button[contains(., '닫기') or contains(., 'Close')]"
    ]:
        try:
            for el in driver.find_elements(By.XPATH, xp):
                driver.execute_script("arguments[0].click();", el)
                time.sleep(0.1)
        except Exception:
            pass

# 상품 페이지의 '후기' 탭으로 이동하는 함수입니다
def goto_review_tab(driver, url):
    driver.get(url + "#review")
    WebDriverWait(driver, WAIT_SEC).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(SLEEP_SEC)
    close_popups(driver)
    for xp in [
        "//button[contains(., '후기') or contains(., 'Review')]",
        "//a[contains(., '후기') or contains(., 'Review')]",
        "//nav//li[.//span[contains(., '후기') or contains(., 'Review')] or contains(., '후기') or contains(., 'Review')]"
    ]:
        try:
            btn = driver.find_element(By.XPATH, xp)
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.2)
            btn.click()
            break
        except NoSuchElementException:
            continue
        except Exception:
            pass
    for _ in range(8):
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(0.25)
    wait_any(driver, [
        "//h2[contains(., '상품 후기') or contains(., 'Product Review')]",
        "//*[contains(text(),'도움돼요') or contains(text(),'Helpful')]",
        "//section[.//h2[contains(., '후기') or contains(., 'Review')]]//li"
    ], timeout=WAIT_SEC)

# 페이지에서 상품 이름을 가져오는 함수입니다
def get_product_name(driver) -> Optional[str]:
    for xp in ["//h1", "//header//h1", "//h2"]:
        try:
            t = driver.find_element(By.XPATH, xp).text.strip()
            if 2 < len(t) < 120:
                return t
        except Exception:
            continue
    return None

# 브라우저에서 JavaScript를 실행하여 리뷰 데이터를 파싱하는 코드입니다
JS_PARSE = r"""(() => {
  try {
    const header = Array.from(document.querySelectorAll('h2, h3'))
      .find(h => /상품\s*후기|후기|Review/.test(h.textContent || ''));
    if (!header) return JSON.stringify({ ok: true, cards: 0, items: [], note: 'no-header' });

    const root = header.closest('section') || header.parentElement || document;
    const nodes = root.querySelectorAll('li, article, div[role="listitem"], div[data-testid*="review"]');
    const dateRe = /(20\d{2}[-./]\d{2}[-./]\d{2})/;
    const items = [];
    for (const el of nodes) {
      const text = (el.innerText || '').trim();
      if (!text) continue;

      const lines = text.split(/\n+/).map(s => s.trim()).filter(Boolean);
      let content = '', maxLen = 0;
      for (const s of lines) {
        if (s.length <= maxLen) continue;
        if (/도움돼요|베스트|옵션|선택|작성자|관리자|Helpful|Best|Option|Select|Author|Admin/.test(s)) continue;
        if (dateRe.test(s)) continue;
        content = s; maxLen = s.length;
      }

      const mDate = text.match(dateRe);
      const date = mDate ? mDate[1].replace(/[-/]/g, '.') : null;

      if (content && content.length >= 6) {
        items.push({ date, content });
      }
    }

    return JSON.stringify({ ok: true, cards: nodes.length, items });
  } catch (e) {
    return JSON.stringify({ ok: false, error: String(e) });
  }
})();"""

# 현재 페이지의 리뷰들을 파싱하는 메인 함수입니다
def parse_page_reviews(driver, product_name: Optional[str]) -> List[Review]:
    data = None
    try:
        raw = driver.execute_script(JS_PARSE)
        if not raw: raise RuntimeError("execute_script returned empty")
        data = json.loads(raw) if not isinstance(raw, dict) else raw
    except Exception as e:
        print(f"[WARN] JS parse failed: {e}")
        data = None

    if not isinstance(data, dict) or not data.get("items"):
        return parse_page_reviews_fallback(driver, product_name)

    out: List[Review] = []
    for it in (data.get("items") or []):
        out.append(Review(
            product_name=product_name,
            content=preprocess_content(it.get("content") or "", product_name),
            date=it.get("date")
        ))
    return out

# JS 파싱 실패 시 사용될 Selenium 기반의 폴백 파서 함수입니다
def parse_page_reviews_fallback(driver, product_name: Optional[str]) -> List[Review]:
    out: List[Review] = []
    date_re = re.compile(r"(20\d{2}\.\d{2}\.\d{2})")
    sections = driver.find_elements(By.XPATH, "//section[.//h2[contains(., '후기') or contains(., '상품 후기')]]")
    root = sections[0] if sections else driver
    items = root.find_elements(By.XPATH, ".//*[self::li or self::article or @role='listitem']")
    if not items:
        items = driver.find_elements(By.XPATH, "//*[self::li or self::article or @role='listitem']")
    for el in items:
        try:
            text = el.text.strip()
            if not text: continue
            m_date = date_re.search(text)
            date = m_date.group(1) if m_date else None
            lines = [s.strip() for s in re.split(r"\n+", text) if s.strip()]
            body_lines = []
            for s in lines:
                if any(k in s for k in ("도움돼요", "베스트", "옵션", "선택", "작성자", "관리자")): continue
                if date_re.search(s): continue
                body_lines.append(s)
            body = " ".join(body_lines)
            body = preprocess_content(body, product_name)
            if len(body) < 6: continue
            out.append(Review(product_name=product_name, content=body, date=date))
        except Exception:
            continue
    print(f"[DEBUG] fallback parsed: {len(out)}")
    return out

# 정규식을 사용하여 불필요한 텍스트 패턴을 정의합니다
URL_RE       = re.compile(r"https?://\S+")
EMOJI_RE     = re.compile(r"[𐀀-􏿿]", flags=re.UNICODE)
OPTION_LINE  = re.compile(r"(?m)^(옵션|선택|구성|상품명)\s*[:：].*$")
BRACKET_LINE = re.compile(r"(?m)^\s*\[[^\]]+\]\s*$")
MULTI_WS     = re.compile(r"\s+")

# 리뷰 내용에서 불필요한 부분을 제거하는 전처리 함수입니다
def preprocess_content(text: str, product_name: Optional[str]) -> str:
    s = text or ""
    s = URL_RE.sub("", s)
    s = EMOJI_RE.sub("", s)
    s = OPTION_LINE.sub("", s)
    s = BRACKET_LINE.sub("", s)
    if product_name:
        s = s.replace(product_name, "")
        pn_fuzzy = re.sub(r"\s+", r"\\s*", re.escape(product_name))
        s = re.sub(pn_fuzzy, "", s, flags=re.IGNORECASE)
    s = MULTI_WS.sub(" ", s).strip()
    return s

# 문자열의 해시값을 계산하여 중복 여부를 판단하는 데 사용합니다
def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# '다음' 페이지 버튼을 찾아 클릭하는 함수입니다
def click_next(driver) -> bool:
    candidates = [
        "//button[normalize-space(.)='다음']",
        "//a[normalize-space(.)='다음']",
        "//*[@aria-label='다음' or @aria-label='다음 페이지']",
        "//button[.//span[contains(., '다음')]]",
        "//button[.//*[text()='›'] or .//*[text()='»'] or .//*[@aria-label='Next']"]
    
    for xp in candidates:
        try:
            btns = driver.find_elements(By.XPATH, xp)
            if not btns: continue
            btn = btns[0]
            if btn.get_attribute("disabled"): return False
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.2)
            btn.click()
            return True
        except Exception:
            continue
    return False

# 지정된 URL에서 전체 리뷰를 크롤링하는 메인 함수입니다
def crawl(url: str, max_pages=MAX_PAGES) -> List[Review]:
    drv = setup_driver(headless=True)
    all_reviews: List[Review] = []
    seen: Set[str] = set()
    try:
        goto_review_tab(drv, url)
        product_name = get_product_name(drv)
        print(f"[INFO] product_name: '{product_name!r}'")
        page = 1
        while True:
            time.sleep(0.6)
            page_reviews = parse_page_reviews(drv, product_name)
            kept = 0
            for r in page_reviews:
                if not r.content or len(r.content) < 6: continue
                h = digest(r.content)
                if h in seen: continue
                seen.add(h); kept += 1; all_reviews.append(r)
            print(f"[DEBUG] page {page} parsed:{len(page_reviews)} kept:{kept} total:{len(all_reviews)}")
            if max_pages and page >= max_pages: break
            if not click_next(drv): break
            page += 1
            try:
                wait_any(drv, [
                    "//h2[contains(., '상품 후기') or contains(., '후기')]",
                    "//section[.//h2[contains(., '후기')]]//*[self::li or self::article or @role='listitem']"
                ], timeout=WAIT_SEC)
            except Exception:
                pass
    finally:
        drv.quit()
    return all_reviews

# 날짜 형식의 문자열을 datetime 객체로 변환하는 함수입니다
def parse_date_ymd(date_text: Optional[str]):
    if not date_text: return None
    for fmt in ("%Y.%m.%d","%Y-%m-%d","%Y/%m/%d"):
        try: return datetime.strptime(date_text, fmt).date()
        except: pass
    return None

# PostgreSQL 데이터베이스 연결 객체를 생성하는 함수입니다
def get_pg_conn():
    import psycopg2
    # .env 파일에 정의된 환경 변수를 사용하여 로컬 Docker DB에 접속합니다
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=int(os.getenv("PGPORT")),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        client_encoding='utf8'
    )

# 크롤링한 리뷰 데이터를 PostgreSQL에 저장하는 함수입니다
def save_postgres(reviews: List[Review]):
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except Exception as e:
        print(f"[WARN] psycopg2 미설치로 Postgres 저장 생략: {e}")
        return
    rows = []
    for r in reviews:
        content = (r.content or "").strip()
        if not content: continue
        rows.append((r.product_name, content, parse_date_ymd(r.date), hashlib.sha256(content.encode("utf-8")).hexdigest()))
    if not rows:
        print("[INFO] Postgres에 저장할 행이 없습니다.")
        return
    conn = get_pg_conn()
    try:
        with conn, conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO reviews (product_name, content, review_date, content_hash)
                VALUES %s
                ON CONFLICT (content_hash) DO NOTHING
            """, rows, page_size=1000)
        print(f"[INFO] Postgres 저장 완료: {len(rows)}건 시도(중복은 무시)")
    finally:
        conn.close()

# 이 스크립트가 직접 실행될 때 호출되는 메인 블록입니다
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='마켓컬리 상품 리뷰 크롤러')
    parser.add_argument('url', type=str, help='크롤링할 상품 페이지 URL')
    parser.add_argument('--max_pages', type=int, default=MAX_PAGES, help=f'최대 크롤링 페이지 수 (기본값: {MAX_PAGES})')
    args = parser.parse_args()

    reviews = crawl(args.url, max_pages=args.max_pages)
    print("Collected (after dedupe):", len(reviews))
    if reviews:
        save_postgres(reviews)
