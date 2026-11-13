# 생성형 AI 활용을 위한 머신러닝 / 딥러닝

## 📑 목차
- [1. 데이터 수집](#data_collect)

## <a id="data_collect"></a> 1. 데이터 수집

### 인공지능 데이터 처리 프로세스

#### 인공지능 데이터 처리 파이프라인(Data Pipeline)

1. 데이터 수집(Data Collection):
    - API 호출, 웹 스크래핑, DB 조회, 센서 데이터 등
2. 데이터 저장(Data Storage):
    - CSV, JSON, SQL(RDBMS), NoSQL, Data Lake 등
3. 데이터 전처리(Data Preprocessing):
    - 결측치 채우기, 이상치 제거, 스케일링, 인코딩
4. 피처 엔지니어링(Feature Engineering)
    - 파생변수 생성, 차원 축소, 피처 선택
5. 모델 학습 및 평가(Modeling & Evaluation):
    - 알고리즘 학습 및 하이퍼파라미터 튜닝
6. 서빙 및 배포(Serving & Deployment):
    - API 서버 배포, 배치 추론 파이프라인 구축

### 데이터 수집 방법 개요

- 데이터 수집은 데이터를 효율적으로 모으고 활용하기 위해 중요한 단계이다.
- 데이터는 다양한 형식과 출처에서 수집될 수 있으며, 목적에 따라 방법도 달라진다.<br>
    → 목적과 환경에 따라 적합한 방법을 선택해야 한다.
- 데이터 수집 방법
    - 웹 스크래핑: HTML 데이터 파싱
    - API 활용: JSON 데이터 가져오기
    - 데이터베이스: 저장된 데이터 검색
    - 공개 데이터셋: Kaggle 등에서 다운로드
    - IoT 센서: 실시간 데이터 수집
    - 소셜 미디어: 트위터 API 활용

1. 웹 스크래핑(Web Scraping)
    - 웹 스크래핑은 웹 사이트에서 HTML데이터를 가져와 필요한 정보를 추출하는 방법이다.
    - 자동화해서 데이터를 수집할 수 있다.
    - 예시
        - 목적: 온라인 쇼핑몰의 상품 가격 정보 수집
        - 방법
            1. BeautifulSoup와 같은 Python 라이브러리를 사용
            2. 웹 페이지의 HTML을 파싱한다.
            3. 상품명과 가격을 가져온다.
2. API 활용
    - API(Application Programming Interface)는 데이터를 쉽게 요청하고 응답 받을 수 있도록 제공되는 인터페이스 → 주로 REST API를 통해 JSON 형태의 데이터를 제공
    - 예시
        - 목적: 날씨 데이터를 수집
        - 방법
            - requests 모듈을 이용하여 OpenWeatherMap API를 호출
            - 공공데이터 포털(https://www.data.go.kr/)
                - 휴게음식점 인허가
                - URL로 들어가기
                - 식품_휴게음식점
3. 데이터베이스에서 데이터 가져오기
    - 이미 수집된 데이터를 데이터베이스에서 쿼리를 통해 가져오는 방법<br>
        → RDBMS(SQL)이나 NoSQL 데이터베이스를 사용
    - 예시
        - 목적: 고객 정보 조회
        - 방법
            - RDBMS를 이용하여 select를 이용해서 데이터를 조회<br> 
            → 데이터베이스 관련 모듈 이용
4. 공개된 데이터셋 활용
    - 정부나 기업, 연구기관에서 제공하는 공개 데이터셋을 다운로드하거나 API를 통해 사용<br>
        → Kaggle, Google Datase Search 등이 자주 사용
    - 예시
        - 목적: 인구 통계 데이터 분석
        - 방법
            - Kaggle에서 데이터를 다운로드한 뒤 Pandas로 읽어 분석
    - 국내 데이터 사이트

        | 사이트 | 특징 |
        | :--- | :--- |
        | **[AI Hub](https://www.aihub.or.kr/)** | 국내 최대 AI 학습용 데이터 플랫폼. 이미지, 음성, 자연어, 의료, 제조, 금융 등 다양한 데이터 제공. 일부 데이터는 회원가입 및 활용 신청 필요 |
        | **[공공데이터포털 (Data.go.kr)](https://www.data.go.kr/)** | 대한민국 정부가 제공하는 공공 데이터. CSV, JSON, Open API 형태 제공 |
        | **[KDX 한국데이터거래소](https://kdx.kr/)** | 금융, 유통, 통신 등 산업 데이터 제공. 무료와 유료 데이터 모두 포함 |
        | **[국가통계포털 KOSIS](https://kosis.kr/)** | 통계청에서 제공하는 국가 통계 데이터 |
        | **[서울 열린데이터광장](https://data.seoul.go.kr/)** | 서울시 교통, 환경, 행정, 문화 등 다양한 데이터 제공 |
        | **[농림축산식품 공공데이터포털](https://data.mafra.go.kr/)** | 농업 및 식품 관련 데이터 제공 |
        ---

    - 해외 데이터 사이트

        | 사이트 | 특징 |
        | :--- | :--- |
        | **[Kaggle Datasets](https://www.kaggle.com/datasets)** | 머신러닝 학습자들이 가장 많이 사용하는 데이터셋 저장소. 튜토리얼과 대회도 함께 제공. |
        | **[UCI Machine Learning Repository](https://archive.ics.uci.edu/)** | 머신러닝 교육에서 가장 많이 사용하는 고전 데이터셋 저장소(Iris, Wine, Adult 등). |
        | **[Google Dataset Search](https://datasetsearch.research.google.com/)** | 전 세계 공개 데이터셋을 검색하는 검색엔진 |
        | **[Hugging Face Datasets](https://huggingface.co/datasets)** | LLM, NLP, 음성, 이미지 등 최신 AI 데이터셋 제공 |
        | **[OpenML](https://www.openml.org/)** | 머신러닝 실험과 데이터셋 공유 플랫폼 |
        | **[Papers with Code Datasets](https://paperswithcode.com/datasets)** | 최신 AI 논문에서 사용하는 데이터셋과 성능 비교 |
        | **[AWS Open Data Registry](https://registry.opendata.aws/)** | AWS에서 제공하는 대용량 공개 데이터 |
        | **[Microsoft Research Open Data](https://msropendata.com/)** | Microsoft 연구용 공개 데이터 |
        | **[Data.gov (미국 정부)](https://data.gov/)** | 미국 정부의 공공 데이터 포털 |
        | **[data.world](https://data.world/)** | 다양한 분야의 공개 데이터와 협업 기능 제공 |

    - 학습 분야 사이트

        | 학습 분야 | 추천 사이트 |
        | :--- | :--- |
        | **머신러닝 입문** | Kaggle, UCI ML Repository |
        | **데이터 분석** | Kaggle, Data.gov, KOSIS |
        | **딥러닝(이미지)** | AI Hub, Kaggle, Hugging Face |
        | **자연어처리(NLP)** | AI Hub, Hugging Face |
        | **음성 인식** | AI Hub |
        | **시계열 데이터** | Kaggle, Data.gov |
        | **공공 데이터 분석** | 공공데이터포털, 서울 열린데이터광장, Data.gov |
        | **최신 AI 연구** | Papers with Code, Hugging Face |
5. IoT 또는 센서 데이터 수집
    - 센서를 이용해 실시간으로 데이터를 수집한다.<br>
        → 온도 센서, 카메라, GPS 장치 등이 사용
    - 예시
        - 목적: 스마트 홈의 온도 데이터를 수집
        - 방법
            - MQTT 프로토콜을 이용하여 센서 데이터를 수집
6. 소셜 미디어 데이터 수집
    - 소셜 미디어 플랫폼에서 사용자 활동 데이터를 수집하는 방법
        → 해당 플랫폼의 API를 사용
    - 예시
        - 목적: 트위터의 특정 키워드로 트윗 수집
        - 방법
            - Tweepy 라이브러리를 사용하여 트위터 API 호출

### requests 모듈을 활용한 데이터 수집

- Python의 requests 모듈은 HTTP요청을 보내고 응답을 처리하는 데 사용되는 라이브러리
- 웹 서버와 통신하여 데이터를 가져오거나 전송하는 작업을 쉽게 수행
    - 다양한 HTTP 메서드(GET, POST, PUT, DELETE 등)를 지원
    - 데이터를 요청하고 응답을 처리하는 과정을 간단하게 처리
- 웹 페이지나 Open API는 HTTP 프로토콜을 이용해 클라이언트와 데이터를 주고 받는다.
- Python의 requests 라이브러리는 쉽고 직관적인 HTTP 통신을 지원하는 표준적인 데이터 수집 도구이다.

#### HTTP 요청의 주요 메서드
- GET 요청: 서버에서 데이터를 가져오는 데 사용 → Read
- POST: CREATE
- UPDATE: UPDATE
= DELETE: DELETE
- 목적: 웹 페이지의 내용을 가져오기
    ```python
    import requests

    # 요청 URL
    url = 'https://jsonplaceholder.typicode.com/posts'
    response = requests.get(url)

    if response.status_code == 200:
        # 상태코드 200은 성공을 의미
        print("데이터 가져오기 성공!")
        # 응답 본문의 일부 출력
        print(response.text[:200])
    else:                                                   # 4XX, 5XX
        print(f"오류 발생: {response.status_code}")
    
    # 데이터 가져오기 성공!
    # [
    # {
    #     "userId": 1,
    #     "id": 1,
    #     "title": "sunt aut facere repellat ...",
    #     "body": "quia et suscipit\nsuscipit ..."
    # },...
    # ]
    ```

#### 응답 데이터 처리

- response.text
    - 응답 본문을 문자열로 반환
- response.json()
    - JSON 형식의 응답을 딕셔너리로 변환
- response.status_code
    - HTTP 상태 코드 반환
    ```python
    if response.status_code == 200:
        # JSON 응답을 딕셔너리로 변환
        data = response.json()

        for datum in data:
            # print("datum: {datum}")
            print(f"제목: {datum['title']}")
            print(f"내용: {datum['body']}")

    else:
        print(f"오류 발생: {response.status_code}")

    # 제목: sunt aut facere repellat provident occaecati excepturi optio reprehenderit
    # 내용: quia et suscipit
    # suscipit recusandae consequuntur expedita et cum ...
    ```
- 실제 응용 사례
    - 웹 페이지 크롤링: 뉴스 헤드라인 가져오기
    - API 활용: 날씨, 주식 정보, 소셜 미디어 데이터 수집
    - 자동화: 특정 시간에 정기적으로 데이터 요청
- 주의사항
    - API 호출 시 요청 제한(Request Limit)을 초과하지 않도록 주의
    - 민감한 데이터(API 키, 인증 정보)는 환경 변수로 관리하거나 안전하게 저장
    - 웹 스크래핑 사이트의 robots.txt 규칙을 준수해야 한다.
- 쿼리 파라미터 사용하기
    - 서버에 데이터를 전달할 때 URL에 파라미터를 포함해서 전달<br>
        → 특정 데이터 가져오기
    ```python
    # userId가 1인 게시글 가져오기
    params = {'userId': 1}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        print("사용자 게시글 가져오기 성공!")

        for data in responst.json()[:5]:
            print(f"userId = {data['userId']}")
            print(f"title = {data['title'][:10]}")
            print(f"body = {data['body'][:10]}")
    else:
        print(f"오류 발생: {response.status_code}")

    # 사용자 게시글 가져오기 성공!
    # userId = 1
    # title = sunt aut f
    # body = quia et su
    # ...
    ```

### HTML 구조 이해(태그, 속성, DOM)

#### HTML이란?

- HTML(HyperText Markup Language)은 웹 페이지의 구조를 정의하는 마크업 언어
- HTML 문서는 브라우저가 읽어서 화면에 표시하는 기본 구조를 제공
- HTML
    - **태그(tag)** 를 사용하여 요소를 정의
    - 각 요소는 **속성(attribute)** 과 콘텐츠를 포함
    - 웹 페이지의 텍스트, 이미지, 링크, 버튼 등 다양한 요소를 구성

#### 태그(Tags)

- 태그는 HTML 요소를 정의하며, 다음과 같은 형식
    ```html
    <태그이름 속성="값">내용(Contents)</태그이름>
    ```
    - 시작 태그: <태그이름>
    - 종료 태그: </태그이름>(일부 태그는 종료 태그가 없음. 예: <img>)
    - 내용: 시작 태그와 종료 태그 사이에 들어가는 컨텐츠 데이터
        ```html
        <p>Hello, World!</p>
        ```
    - \<p>: 문단(paragraph)을 나타내는 태그
    - Hello, World!: 태그 사이의 내용

#### 속성(Attributes)

- 속성은 태그에 추가 정보를 제공
- 속성="값" 형식으로 작성, 속성은 시작 태그에만 추가
    - id: 요소의 고유 식별자
    - class: CSS 스타일링이나 JavaScript 제어를 위한 클래스 이름
    - href: 링크를 나타냄(예: \<a> 태그에서 사용)
    - src: 이미지나 외부 리소스의 경로(예: \<img> 태그에서 사용)
        ```html
        <a href="https://example.com" target="_blank">Click Here</a>
        ```

#### DOM(Document Object Model)

- DOM은 브라우저가 HTML문서를 파싱하여 생성한 객체 모델
- HTML 문서를 트리 구조로 표현하며, 각 요소는 노드(Node)로 구성
- DOM의 구성 요소
    - HTML 요소(Element Nodes): 태그들
    - 속성(Attribute Nodes): 태그의 속성들
    - 텍스트(Text Nodes): 태그 안의 내용
        ```html
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
        </head>
        <body>
            <h1 id="main-title">
                Welcome to Web Scraping    
            </h1>
            <p class="description">
                This is an example paragraph.
            </p>
            <a href="https://example.com" target="_blank">
                Visit Example
            </a>
        </body>
        </html>
        ```

### BeautifulSoup을 활용한 웹 스크래핑

- Python에서 HTML과 XML 문서를 파싱하고 원하는 데이터를 추출할 때 사용하는 라이브러리<br>
→ HTML의 DOM 구조를 탐색하고 태그, 속성, 텍스트를 쉽게 추출
- 설치
    - pip install beautifulsoup4

    ```python
    from bs4 import BeautifulSoup

    html_doc = """
    <html>
        <head>
            <title>Example Page</title>
        </head>
        <body>
            <h1>Welcome to Web Scraping</h1>
            <p class="description">This is an example paragraph.</p>
            <a href="https://example.com">Visit Example</a>
        </body>
    </html>
    """

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_doc, 'html.parser')

    # 객체의 속성값을 이용해서 데이터 추출
    print(soup.title.string)    # Example Page
    print(soup.h1.string)       # Welcome to Web Scraping
    print(soup.a['href'])       # https://example.com
    print(type(soup.find('a'))) # <class 'bs4.element.Tag'>
    ```

#### 주요 기능

1. 특정 태그 찾기
    - 태그 이름으로 요소를 쉽게 찾을 수 있다.
        ```python
        # 첫 번째 <p> 태그 찾기
        print(soup.p)           # <p class="description">This is an example paragraph.</p>
        print(soup.p['class'])  # ['description']
        ```
2. 여러 태그 찾기
    - find_all()을 사용하면 특정 태그를 모두 찾을 수 있다.
        ```python
        # 모든 <a> 태그 찾기
        links = soup.find_all('a')
        for link in links:
            print(link['href'])     # https://example.com
        ```
3. CSS 선택자 활용
    - select()를 사용해 CSS 선택자를 기반으로 요소를 찾음
    - select는 CSS 선택자를 사용하므로 복잡한 선택 조건을 간단하게 표현
    - find나 find_all 보다 가독성이 좋고 직관적이다.
        - tag: 특정 태그명 선택(예: h1)
        - .class: 클래스명으로 선택(예: .price)
        - #id: 고유 ID명으로 선택(예: #main-title)
        - tag.class 또는 tag > tag: 부모-자식 경로 구조
            ```python
            # 클래스가 "description"인 <p>태그 찾기
            description = soup.select_one('.description')
            print(description.text)     # This is an example paragraph.
            ```

#### BeautifulSoup을 활용한 웹 스크래핑

**순서 주의**

```python
# URL 요청
url = 'https://news.ycombinator.com/'
response = requests.get(url)

# BeautifulSoup 객체 생성
soup = BeautifulSoup(response.text, 'html.parser')

# 제목과 링크 추출(href 있는 것만 필터링)
for item in soup.find_all('a'):
    href = item.get('href')
    if href:                            # href가 존재할 때만 출력
            print(f"Title: {item.get_text(strip=True)}")
            print(f"Link: {href}")

# Title: Hacker News
# Link: news
# Title: new
# Link: newest
```

#### 웹 크롤링 주의사항 및 윤리

1. robots.txt란 무엇인가?
    - 정의: 웹사이트 루트 경로(도메인/robots.txt)에 위치하여 크롤러의 접근 허용, 차단 범위를 명시한 텍스트 파일
    - 목적: 서버 과부하 방지, 민감 경로 및 불필요한 페이지 수집 차단, 트래픽 리소스 보호
2. 핵심 지시어(Directives) 4가지
    - User-agent: 규칙을 적용할 크롤러 지정('*'는 모든 크롤러, Googlebot등은 특정 봇 지칭)
    - Disallow: 수집 및 접근을 금지하는 URL 경로(예: /admin/, /private/)
    - Allow: 차단된 상위 경로 내에서 예외적으로 접근을 허용하는 하위 경로(예: /public/)
    - Crawl-delay: 연속 요청 사이의 최소 대기 시간(초) 지정으로 서버 부하 방지
3. 왜 반드시 준수해야 하는가?
    - 기술적 측면(안정성): 과도한 요청으로 인한 타겟 서버 장애(DoS 유사 상태) 예방 및 자사 IP 영구 차단 방지
    - 법적·기술적 측면: 업무 방해죄·저작권 침해·부정경쟁방지법 분쟁 시 선의의 수집 여부를 판단하는 핵심 기준
4. 실무 스크래핑 적용 3단계 수칙
    - 사전 파싱 자동화: 크럴러 실행 시 urllib.robotparser 등을 활용해 대상 사이트의 robots.txt를 먼저 조회
    - 요청 속도 제어: 명시된 Crawl-delay를 준수하고, 설정이 없더라도 요청 간 1~3초 이상의 간격(Throttling) 유지
    - 식별 정보 명시: 요청 헤더의 User-Agent에 봇 이름과 담당자 연락처(또는 안내 페이지)를 투명하게 기재

- Robots.txt
    - User-Agent 설정
        - 브라우저를 통한 정상적인 접근처럼 보이도록 HTTP 헤더에 브라우저 정보(User-Agent)를 주입해야 응답 거부를 방지할 수 있다.
            ```python
            import time

            # User-Agent 헤더 주입 예시
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            url = "https://httpbin.org/user-agent"
            response = requests.get(url, headers=headers)
            print(response.json())      # {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            ```
        
### REST API란 무엇인가?

#### REST(Representational State Transfer)

- 웹 서비스와 클라이언트 간의 통신을 설계하는 아키텍처 스타일
- REST API는 REST 원칙을 따르는 API(Application Programming Interface)
- 클라이언트(사용자)와 서버(서비스 제공자) 간 데이터(JSON 또는 XML)를 주고 받는 데 사용

#### REST API의 구조

- URL구조
    - REST API의 자원 접근은 URL로 구성 → 기본 형식: https://baseurl/resource/{id}
    - 모든 사용자의 데이터: https://api.example.com/users
    - 특정 사용자 데이터: https://api.example.com/users/123
- HTTP 메서드
    - 서버에 요청을 보낼 때 HTTP 메서드와 URL을 조합해 사용

        | 메서드 | 동작 | 설명 |
        | :---: | :---: | :--- |
        | **GET** | 데이터 조회 | 서버에서 데이터를 가져옵니다. |
        | **POST** | 데이터 생성 | 서버에 새 데이터를 추가합니다. |
        | **PUT** | 데이터 수정 | 기존 데이터를 업데이트합니다. |
        | **DELETE** | 데이터 삭제 | 서버에서 데이터를 삭제합니다. |
- Headers와 Body
    - Headers: 요청 또는 응답의 부가 정보를 담는다. (예: Content-Type, Authorization)
    - Body: 요청 시 보내는 데이터. 주로 JSON 형식을 사용한다.
        ```
        POST /users HTTP/1.1
        Host: api.example.com
        Content-Type: application/json
        Authorization: Bearer token123
        {
        "name": "king",
        "email": "john.doe@example.com"
        }
        ```
### 공공데이터 포털 또는 오픈 API 활용

#### 공공 데이터 포털

- 정부나 공공기관이 제공하는 데이터를 모아둔 플랫폼
- 대한민국의 대표적인 공공 데이터 포털: 공공데이터포털(data.go.kr)
- 다양한 분야의 데이터를 무료로 제공하며, REST API 형식으로 데이터에 접근
- 주요 데이터
    - 교통 정보(버스, 지하철 실시간 위치)
    - 기상 정보(날씨 예보)
    - 인구 통계(연령별, 지역별 인구)

#### 오픈 API

- 기업이나 기관이 데이터를 제공하기 위해 공개한 API
- 주요 데이터
    - 네이버 검색, 카카오 검색
    - 구글 지도 API(위치 데이터 제공)
    - 트위터 API(트윗 데이터 제공)
    - NASA API(우주 관련 데이터 제공)

### CSV, Excel, JSON 파일에 데이터 저장

#### CSV(Comma-Separated Values) 파일

- 데이터를 쉼표로 구분하여 저장하는 텍스트 기반의 파일 형식이다.
- 간단한 데이터 저장과 교환에 적합하며, 대부분의 데이터 분석 도구와 호환된다.
- Python에서 CSV파일을 저장할 수 있다.
    ```python
    import csv

    # 데이터 준비
    data = [
        ['Name', 'Age', 'City'],
        ['Alice', 30, 'New York'],
        ['Bob', 25, 'Log Angeles'],
        ['Charlie', 35, 'Chicago']
    ]

    # CSV 파일에 저장
    with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print('CSV 파일이 저장되었습니다.')
    ```
- CSV 파일 읽기
    - pandas 라이브러리를 주로 사용
        - pd.read_csv('파일 경로'): CSV파일을 읽어 데이터프레임 형태로 반환
            ```python
            import pandas as pd

            # CSV 파일 읽기
            data = pd.read_csv('data.csv')
            print(data)
            ```
        - data.head(): 상위 몇 개의 데이터를 미리보기.
- CSV 파일은 단순 텍스트이므로, open()을 사용한다.
    ```python
    # open()을 사용한 CSV 파일 읽기
    with open('data.csv', 'r') as file:
        for line in file:
            print(line.strip())
    ```

#### Excel 파일

- 데이터 시각화와 분석에 강력
- 표 형태의 데이터를 저장할 수 있음
- openpyxl 또는 pandas 라이브러리를 주로 사용
- Excel 파일 쓰기
    ```python
    import pandas as pd

    # 데이터 준비
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [30, 25, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']
    }

    # DataFrame 생성
    df = pd.DataFrame(data)

    # Excel 파일로 저장
    df.to_excel('data.xlsx', index=False)

    print("Excel 파일이 저장 되었습니다!")
    ```
- Excel 파일 읽기
    - pd.read_excel('파일경로'): 엑셀 파일 읽기.
        ```python
        data = pd.read_excel('data.xlsx')
        print(data)

        #       Name  Age         City
        # 0    Alice   30     New York
        # 1      Bob   25  Los Angeles
        # 2  Charlie   35      Chicago
        ```

#### JSON(JavaScript Object Notation) 파일

- 키-값 쌍으로 데이터를 저장
- API와 데이터 교환에서 주로 사용
- 계층적 데이터 구조를 표현할 수 있어 XML이 대안으로 자주 사용
- Python에서 JSON 파일 저장
    - json.dump(): 데이터를 JSON 형식으로 저장
    - indent=4를 사용하여 사람이 읽기 쉽게 데이터를 정렬
- Python에서 JSON파일 읽기
    - json.load(): JSON 파일을 파이썬의 딕셔너리로 변환.
        ```python
        # JSON 파일 읽기
        with open('example.json', 'r') as file:
            data = json.load(file)
        
        print(data)
        ```

#### 각 형식의 비교

| 형식 | 장점 | 단점 |
| :---: | :--- | :--- |
| **CSV** | 간단하고 가볍다. 대부분의 도구와 호환 가능. **대규모 데이터**에 적합 | 계층적 데이터 저장 불가. |
| **Excel** | 시각화화 다중 시트 지원. 구조화된 데이터 관리. | 무겁고 대규모 데이터 교환에 적합하지 않음. |
| **JSON** | 계층적 데이터 저장 가능. API 데이터 교환에 적합. | 사람이 직접 읽기 어렵고 편집 도구가 필요함. |

#### 활용 팁

- **CSV**: 대량 데이터 저장 및 분석 초기 단계에 적합
- **Excel**: 데이터 요약, 보고서 작성, 분석 작업에 유용
- **JSON**: 웹 및 모바일 애플리케이션과의 데이터 교환에 적합

### SQLite를 활용한 간단한 데이터베이스 관리

#### SQLite

- 수집한 대용량 데이터를 매번 로컬 CSV나 JSON파일로만 다루면 여러 프로세스가 동시에 조회/수정할 때 꼬이거나 속도가 크게 느려진다.
- AI 실무 환경에서는 이를 위해 정형 데이터를 효과적으로 조회하고 관리할 ㅜ 있는 RDBMS(관계형 데이터베이스)를 사용한다.

#### SQLite의 특징

- 가볍고 파일 기반의 관계형 데이터베이스 관리 시스템(RDBMS)이다.
- 서버 설치가 필요 없으며, Python 내장 모듈로 쉽게 사용할 수 있다.
- 데이터 저장, 조회, 관리 등을 SQLite를 활용한다.

#### 관계형 데이터베이스와 SQL 기초

- RDBMS(Relational Database Management System)
    - 테이블(행과 열) 단위로 데이터를 관리하며 고유 키(PK)로 테이블 간 관계를 설정
- SQL(Strutured Query Language): DB에 질의하기 위한 공통 언어
    - CREATE: 테이블 구조 정의
    - INSERT: 데이터 신규 삽입
    - SELECT: 데이터 검색 및 필터링
    - UPDATE: 기존 데이터 수정
    - DELETE: 데이터 삭제

#### SQLite 사용하기

1. SQLite와 Python 연동
- Python의 sqlite3 모듈을 사용하면 SQLite 데이터베이스를 쉽게 사용할 수 있다.
- 데이터베이스 연결
    - example.db<br>
        SQLite 데이터베이스 파일 이름<br>파일이 없으면 새로 생성

        ```python
        import sqlite3

        # SQLite 데이터베이스 연결(파일 생성)
        connection = sqlite3.connect("example.db")
        print("데이터베이스에 연결되었습니다!")

        # 연결 닫기
        connection.close()
            ```
2. 테이블 생성
    - 데이터를 저장하기 위해 먼저 테이블을 생성
    - 테이블 저장 코드
        - CREATE TABLE IF NOT EXISTS: 테이블이 없을 경우에만 생성한다.
        - PRIMARY KEY AUTOINCREMENT: 기본 키로 자동 증가 값을 사용한다.
        - 순서 주의: `연결 → cursor → execute → close`

            ```python
            import sqlite3

            # 데이터베이스 연결
            connection = sqlite3.connect("example.db")

            # 데이터베이스 실행하는 도구 → cursor
            cursor = connection.cursor()

            # SQL 명령어로 테이블 생성
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id  INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER,
                city TEXT
            )
            """)
            print("테이블이 생성되었습니다.")

            # 연결 종료
            connection.close()
            ```
3. 데이터 삽입
    - Python 코드로 데이터 삽입
        - ?는 값의 자리표시자: SQL인젝션 공격을 방지할 수 있다.
        - executemany(): 여러 데이터를 한 번에 삽입할 수 있다.
            ```python
            # 데이터베이스 연결
            connection = sqlite3.connect("example.db")
            cursor = connection.cursor()

            # 데이터 삽입
            cursor.execute("INSERT INTO users (name, age, city) VALUES (?, ?, ?)", ("Alice", 30, "New York"))

            # 여러 데이터 삽입
            data = [
                ("Bob", 25, "Los Angeles"),
                ("Charlie", 35, "Chicago"),
                ("Diana", 28, "Houston")
            ]
            cursor.executemany("INSERT INTO users (name, age, city) VALUES (?, ?, ?)", data)

            # 변경사항 저장
            connection.commit()
            print("데이터가 삽입되었습니다.")

            # 연결 종료
            connection.close()
            ```
4. 데이터 조회
    - Python  코드로 데이터 조회
        - fetchall(): 모든 조회 결과를 가져온다.
        - fetchone(): 한 행씩 가져온다.
            ```python
            # 데이터베이스 연결
            connection = sqlite3.connect("example.db")
            cursor = connection.cursor()

            # 데이터 조회
            cursor.execute("SELECT * FROM users")
            rows = cursor.fetchall()

            # 조회된 데이터 출력
            for id, name, age, city in rows:
                print(f'name: {name}, age: {age}, city: {city}')

            # name: Alice, age: 30, city: New York
            # name: Bob, age: 25, city: Los Angeles
            # name: Charlie, age: 35, city: Chicago
            # name: Diana, age: 28, city: Houston

            # 연결 종료
            connection.close()
            ```

#### Pandas와 Database의 결합

- 실제 데이터 분석 환경에서는 sqlite3 커서로 로우를 다루기보다는, Pandas 데이터프레임과 통째로 통신하는 방식을 선호한다.
    - df.to_sql(name, conn)
        - 데이터프레임 내용을 DB 테이블로 즉시 적재
    - pd. read_sql(query, conn)
        - SQL 쿼리 실행 결과를 데이터프레임 객체로 즉시 반환

