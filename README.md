backend 사용 방법 (Python 3.13.2 사용 - 다른 버전도 문제없는지 모름)

0. 패치 노트 (Patch Notes):
!! 업데이트 할 예정 !!
(a) product db 업데이트:
    (삭제) detail / detail_url
    (추가) 상품 추가 사진

(b) backend 함수 return에서 message삭제, JSON만 돌려주기 업데이트

(c) FE / BE API 맞추고 return type sync 작업

(d) AI 모델 sync

(e) @app.get("/api/similar_product_list") -> 모든 상품에서 유사 상품 return



1. 주요 파일 설명:
requirement.txt -- python3.13.1 / 
[중요] 명령어 사용: pip install -r requirement.txt
main.py         -- backend using FastAPI
database.py     -- SQLalchemy DB initialization 
app_data.db     -- sqlite3 db: 사용 방법은 (2) 참고
modify_db.py    -- Dummy DB에 추가 / 삭제 (e.g. 실험용 찜 목록)
post_format.py  -- backend에 POST request하기
get_format.py   -- backend에 GET request하기

2. DB setup/usage 설명:
python database.py                   -- (최초 1번 실행)
sqlite3 app_data.db                  -- DB 접속 방법 1
DB Browser for SQLite                -- DB 접속 방법 2
.tables                              -- table 목록 조회
PRAGMA table_info(<table>);          -- checking info of specific table
DROP TABLE IF EXISTS <table>;        -- removing tables
SELECT * FROM <table>                -- find all entries from table
python modify_db.py                  -- DB 추가/삭제 macro (직접 수정해서 실행)

3. backend server 실행:
python main.py -- localhost:5000로 backend server 접속
python test_backend.py -- backend에서 잘 호출하는지 확인 가능

4. local_data
4.1 local_data/highlight_video_pic
images는 "highlight_<video_id>_<highlight_idx>_<product_id>" 이름을 사용함
