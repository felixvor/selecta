@echo off
cd /d "%~dp0"
wsl.exe -e bash setup.sh %*
