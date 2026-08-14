@echo off
cd /d "%~dp0"
echo ==================================================
echo    自检：检查程序是否正确（不联网 / 不调用 API / 不花钱）
echo ==================================================
echo.
set "PYEXE="
where py >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE where python3 >nul 2>&1 && set "PYEXE=python3"
if not defined PYEXE for /d %%D in ("%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.1*") do if exist "%%D\python.exe" set "PYEXE=%%D\python.exe"
if not defined PYEXE (
  echo [失败] 没找到 Python。请把整个文件夹交给 Hermes 运行，或截图发给小白。
  pause
  exit /b
)
%PYEXE% self_check.py
echo.
echo 上面每一项都显示 [OK] 就说明这套程序的统计与数据是可信的。
echo （若出现 [FAIL]，请把本窗口整屏截图发给小白。）
echo.
pause
