@echo off
cd /d "%~dp0"
echo ==================================================
echo    ArabTrade 论文实验 - 一键运行（全量 833 条）
echo    本版本无需 pip / scipy / 联网装库，任何 Python 3 直接跑
echo    评分：默认用 LLM 评委打分（更可信）；省钱可加 --no-judge
echo ==================================================
echo.
echo [1/3] 正在寻找 Python ...
set "PYEXE="
where py >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE where python >nul 2>&1 && set "PYEXE=python"
if not defined PYEXE where python3 >nul 2>&1 && set "PYEXE=python3"
if not defined PYEXE for /d %%D in ("%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.1*") do if exist "%%D\python.exe" set "PYEXE=%%D\python.exe"
if not defined PYEXE (
  echo.
  echo [失败] 没找到 Python。请把整个 arabtrade_thesis_kit 文件夹交给 Hermes 运行，
  echo    或把这个黑窗口整屏截图发给小白。
  echo.
  pause
  exit /b
)
echo     使用：%PYEXE%
echo.
echo [2/3] 请粘贴你的 DeepSeek API Key（窗口里右键粘贴），然后按回车：
set /p KEY=API Key: 
set "DEEPSEEK_API_KEY=%KEY%"
echo.
echo [3/3] 开始跑全量 833 条（先自动测 API，通了才开跑，中途别关窗口）...
echo.
%PYEXE% run_thesis_experiment.py
if errorlevel 1 (
  echo.
  echo ==================================================
  echo    [出错] 实验中途中断，结果【未生成】。
  echo    常见原因：API Key 错 / 没网 / 余额不足。
  echo    请把这个黑窗口整屏截图发给小白。
  echo ==================================================
  echo.
  pause
  exit /b
)
echo.
echo ==================================================
echo    跑完了！结果在 results 文件夹：
echo      thesis_chapter_6_results.md   （粘进论文第六章）
echo      thesis_stats.json             （统计数据）
echo      audit_per_question.csv        （逐题留痕）
echo    验收：打开 thesis_stats.json，确认 api_available 是 true。
echo ==================================================
echo.
pause
