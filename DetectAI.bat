@echo off
rem Abre o DetectAI numa janela propria, com dois cliques (Windows).
rem Este arquivo precisa ficar na mesma pasta do app.py e da pasta .venv.
cd /d "%~dp0"
call .venv\Scripts\activate
python abrir_detectai.py
if errorlevel 1 pause
