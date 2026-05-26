@echo off
setlocal EnableDelayedExpansion

rem ================================================================
rem  Python interpreter fallback list — first existing path wins.
rem  Add entries as PYTHON_PATH_1, PYTHON_PATH_2, ... (up to 9).
rem ================================================================
set PYTHON_PATH_1=C:\WinPython\WPy64-31241\python-3.12.4.amd64\python.exe
set PYTHON_PATH_2=%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\python3.13.exe

set PYTHON_EXE=
for /L %%i in (1,1,9) do (
    if "!PYTHON_EXE!"=="" if not "!PYTHON_PATH_%%i!"=="" if exist "!PYTHON_PATH_%%i!" (
        set PYTHON_EXE=!PYTHON_PATH_%%i!
    )
)
if not "!PYTHON_EXE!"=="" echo [setup] Using Python: !PYTHON_EXE!

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%.venv
set EXIT_CODE=0

if not exist "%VENV_DIR%\" goto :create_venv

set /p RECREATE=[setup] .venv already exists. Recreate? [y/N]:
if /i "!RECREATE!"=="y" (
    echo [setup] Removing existing .venv ...
    rmdir /s /q "%VENV_DIR%"
) else (
    echo [setup] Keeping existing .venv.
    goto :run_install
)

:create_venv
if "!PYTHON_EXE!"=="" (
    echo [setup] ERROR: No Python interpreter found. Add your path to the fallback list in this script.
    set EXIT_CODE=1
    goto :end
)
echo [setup] Creating .venv using !PYTHON_EXE! ...
"%PYTHON_EXE%" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [setup] ERROR: venv creation failed.
    set EXIT_CODE=1
    goto :end
) 
echo [setup] .venv created.

:run_install
echo [setup] Writing root_imports.pth ...
set SITE_PKG=%VENV_DIR%\Lib\site-packages
echo %SCRIPT_DIR:~0,-1%>"%SITE_PKG%\root_imports.pth"

echo [setup] Running install-missing.py ...
"%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%install-missing-packages.py"
set EXIT_CODE=%errorlevel%

:end
pause
exit /b %EXIT_CODE%
