@echo off
REM run_training.bat — Launch RL training and save output to a timestamped log file
REM
REM Usage:
REM   run_training.bat              -> Phase 2 (3M steps)
REM   run_training.bat phase3       -> Phase 3 (8M steps, domain randomization)
REM   run_training.bat dummy        -> smoke-test with random data

setlocal enabledelayedexpansion

REM Build args
set ARGS=
if /i "%1"=="phase3" set ARGS=--phase3
if /i "%1"=="dummy"  set ARGS=--dummy

REM Create logs directory
if not exist training_logs mkdir training_logs

REM Timestamped filename using wmic
for /f "tokens=1-6 delims=/:. " %%a in ("%date% %time%") do (
    set YYYY=%%d
    set MM=%%b
    set DD=%%c
    set HH=%%e
    set MIN=%%f
)
set PHASE=phase2
if /i "%1"=="phase3" set PHASE=phase3
set LOGFILE=training_logs\train_%PHASE%_%YYYY%-%MM%-%DD%_%HH%-%MIN%.txt

echo Logging to: %LOGFILE%
echo Command: python train_rl.py %ARGS%
echo.

python train_rl.py %ARGS% 2>&1 | tee "%LOGFILE%"

echo.
echo Training complete. Log saved to: %LOGFILE%
