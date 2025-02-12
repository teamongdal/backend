backend 사용 방법 
Python 3.13.2 사용 - 다른 버전도 문제없는지 모름

0. FE & BE 연동 테스트 
pip install -r requirement.txt
(1) backend/hyub_google_cloud_key.json, (2) backend/llm/.env 있는지 확인 (없으면 요청)
python main.py -- http://localhost:8000/docs 접속 확인
python ./tools/test_modified_backend.py -- status code 다 200(성공)으로 나오는지 확인

1. 주요 파일
requirement.txt -- python3.13.2 / require libraries / pip install -r requirement.txt 실행
main.py         -- backend using FastAPI
database.py     -- SQLalchemy DB initialization 
app_data.db     -- sqlite3 db
tools/make_toy_db.py    -- 실험용 DB 생성

2. DB setup/usage 설명 
(TODO: crawling작업 끝나고 DB 새로 만들 예정!)
python database.py                   -- app_data.db 생성함 (app_data.db 이미 있으면 커맨드 실행 X)
sqlite3 app_data.db                  -- DB 접속 방법 1
DB Browser for SQLite                -- DB 접속 방법 2
.tables                              -- table 목록 조회
PRAGMA table_info(<table>);          -- checking info of specific table
DROP TABLE IF EXISTS <table>;        -- removing tables
SELECT * FROM <table>                -- find all entries from table
python modify_db.py                  -- DB 추가/삭제 macro (직접 수정해서 실행)

3. backend server 실행
python main.py                          -- localhost:8000로 backend server 접속

4. app_data.db
(추후 작업 예정)

5. tools/
(추후 작업 예정)

6. crawling/
chromedriver.exe 설치하고 crawling/ 디렉토리로 이동
cloth.py 실행하면 crawling 실행

check_valid_image_2_csv.py -- 사진 2개 이상있는 entry
check_valid_image_2_csv.py -- 사진 4개 있는 entry