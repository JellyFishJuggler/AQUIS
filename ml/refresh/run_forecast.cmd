@echo off
REM AQUIS ML - forecast refresh cycle (fetch -> features -> inference -> publish).
REM Intended for manual, interactive runs where you want to watch the output.
REM
REM The AQUIS-ML-Refresh scheduled task does NOT run this file. It calls
REM refresh\run_forecast.pyw through pythonw.exe instead: a console process is
REM killed with 0xC000013A ("STATUS_CONTROL_C_EXIT") when the user session
REM closes or the machine sleeps, which is what broke the 01:00 run on
REM 2026-11-05. pythonw allocates no console, so there is nothing to signal.
REM
REM The refresh lock makes overlapping invocations a safe no-op.
cd /d "%~dp0.."
echo [%date% %time%] refresh start >> "data\refresh\run_forecast.log"
".venv\Scripts\python.exe" -u -m refresh.cli forecast >> "data\refresh\run_forecast.log" 2>&1
echo [%date% %time%] refresh exit %ERRORLEVEL% >> "data\refresh\run_forecast.log"
