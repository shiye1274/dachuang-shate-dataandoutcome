@echo off
cd /d "%~dp0"
echo ==================================================
echo    生成四段式输出样例图（不需要 API Key）
echo ==================================================
echo.
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
%PYEXE% make_sample_figure.py "第一次往沙特发女装，需要做哪些合规准备？"
echo.
echo 样例已生成到 results 文件夹：sample_output.png / sample_output.html
echo.
pause
