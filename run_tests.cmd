@echo off
setlocal
cd /d "%~dp0"
if not defined E3_PYTHON set "E3_PYTHON=python"
"%E3_PYTHON%" -m pytest tests -q -o addopts=
exit /b %errorlevel%

