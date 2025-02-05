backend 사용 방법

1. 주요 파일 설명:
requirement.txt -- python3.13.1 / 명령어 사용: pip install -r requirement.txt
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
