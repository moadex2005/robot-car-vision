@echo off
setlocal
rem Starts the Robot Car Vision demo page and opens it in the browser.
rem Extra words are passed on to the program, for example: start_demo.cmd --source phone
rem Closing this window stops the program.

title Robot Car Vision - close this window to stop
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The environment is not installed: the Install section of README.md says how to install it.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m car_vision.app --open %*
set "code=%errorlevel%"

if not "%code%"=="0" (
    echo.
    echo The program stopped with an error. Read the message above.
    pause
)
exit /b %code%
