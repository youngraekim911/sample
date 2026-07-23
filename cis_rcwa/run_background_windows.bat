@echo off
chcp 65001 >nul
rem ============================================================
rem  CIS RCWA — "걸어두고 퇴근" 백그라운드 서버 (윈도우)
rem
rem  오래 걸리는 DOE/surrogate 스윕을 걸어두고 자리를 뜰 때 사용.
rem  * 계산은 브라우저가 아니라 '이 서버 창'에서 돕니다.
rem  * 회사 정책으로 브라우저(Chrome/Edge/Explorer)가 꺼져도 계산은 계속됩니다.
rem  * 결과는 끝나는 즉시 out\surrogate_cache 에 자동 저장(DB)됩니다.
rem  * 나중에 브라우저를 다시 열면 '설정' 페이지에서 자동 재접속되고,
rem    끝난 결과는 [💾 저장된 surrogate] 목록에서 바로 불러올 수 있습니다.
rem
rem  !! 이 창(서버)만은 닫지 마세요. 닫으면 진행 중 계산이 중단됩니다 !!
rem  (완료되어 캐시에 저장된 결과는 창을 닫아도 그대로 남습니다.)
rem ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [오류] python 을 찾을 수 없습니다. https://python.org 에서 설치 후
  echo        설치 시 "Add Python to PATH" 를 체크하세요.
  pause & exit /b 1
)
if not exist .venv (
  echo [설치] 가상환경 생성 + 패키지 설치중... (수 분 소요, 최초 1회)
  python -m venv .venv || (echo venv 생성 실패 & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo 설치 실패 & pause & exit /b 1)
)

rem --- 계산 도중 모니터/절전으로 프로세스가 죽지 않도록 (정책상 막히면 무시됨) ---
powercfg -change -monitor-timeout-ac 0 >nul 2>nul
powercfg -change -standby-timeout-ac 0 >nul 2>nul

echo.
echo ============================================================
echo  서버 시작: http://127.0.0.1:8787
echo  - 브라우저가 곧 열립니다. DOE 를 Run 하고 그냥 두고 가세요.
echo  - 브라우저는 닫혀도 됩니다. 다시 열면 자동으로 이어서 보여줍니다.
echo  - 계산이 끝나면 결과가 자동 저장되어 언제든 불러올 수 있습니다.
echo  * 이 창은 닫지 마세요 (최소화는 OK).
echo ============================================================
echo.
start "" http://127.0.0.1:8787
.venv\Scripts\python app.py --no-browser
echo.
echo [서버 종료됨] 진행 중이던 계산이 있었다면 중단되었습니다.
pause
