@echo off
cd /d "%~dp0"
echo ==================================================
echo    先小测：每场景 30 条（确认 API 能通、环境没问题）
echo    本版本无需 pip / scipy，任何 Python 3 直接跑
echo ==================================================
echo.
echo [1/3] 正在寻找 Python ...
set "PYEXE="
where py >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE where python3 >nul 2>&1 && set "PYEXE=python3"
if not defined PYEXE for /d %%D in ("%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.1*") do if exist "%%D\python.exe" set "PYEXE=%%D\python.exe"
if not defined PYEXE (
  echo [失败] 没找到 Python。建议把整个文件夹交给 Hermes 运行。
  pause
  exit /b
)
echo     使用：%PYEXE%
echo.
echo [2/3] 请粘贴你的 DeepSeek API Key（右键粘贴），然后按回车：
set /p KEY=API Key: 
set "DEEPSEEK_API_KEY=%KEY%"
echo.
echo [3/3] 开始小测（每场景 30 条）...
echo.
%PYEXE% run_thesis_experiment.py --per-scenario 30
echo.
echo 小测完成。若 thesis_stats.json 里 api_available=true，就可以去双击 “① 双击跑实验.bat” 跑全量 833 条。
echo.
pause
