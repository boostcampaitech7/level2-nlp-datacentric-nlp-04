@echo off
:loop
cls

REM 이 프로그램은 리눅스에서의 watch -n 1 nvidia-smi 명령어를 모방합니다.
REM 윈도우에서 실시간 VRAM 상황을 모니터링합니다.

echo ===== GPU Information =====
REM GPU 이름과 인덱스만 표시합니다.(vscode, pycharm의 terminal 사이즈 고려)
nvidia-smi --query-gpu=name,index --format=csv,noheader

echo.
echo ===== Real-time GPU Usage =====
REM GPU의 기본 프로세스에서의 실시간 사용량 표시
nvidia-smi | findstr "Default"

echo.
REM 실시간 메모리 사용량 (GB 단위) 표시
REM FreePhysicalMemory (KB)와 TotalVisibleMemorySize (KB) 값을 가져와 메모리 정보를 계산

REM 사용 가능한 메모리(KB) 가져오기
for /f "tokens=2 delims==" %%A in ('wmic OS get FreePhysicalMemory /Value') do set FreeMem=%%A
REM 총 메모리(KB) 가져오기
for /f "tokens=2 delims==" %%A in ('wmic OS get TotalVisibleMemorySize /Value') do set TotalMem=%%A

REM 메모리 값을 GB 단위로 변환
set /a FreeMemGB=%FreeMem% / 1024 / 1024
set /a TotalMemGB=%TotalMem% / 1024 / 1024
set /a UsedMemGB=%TotalMemGB% - %FreeMemGB%

REM 메모리 정보 출력
echo Total Memory: %TotalMemGB% GB
echo Used Memory : %UsedMemGB% GB
echo Free Memory : %FreeMemGB% GB

REM 1초 동안 대기 후 루프 반복
timeout /t 1 >nul
goto loop
