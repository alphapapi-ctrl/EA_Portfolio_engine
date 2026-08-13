@echo off
rem EA Portfolio Engine launcher - double-click to start (or focus) the app.
cd /d "%~dp0"
powershell -NoProfile -Command ^
  "try { $c = New-Object Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8504); $c.Close(); Start-Process 'http://localhost:8504'; exit 0 } catch { }; Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','streamlit','run','app.py' -WorkingDirectory '%~dp0' -WindowStyle Hidden; Start-Sleep 4; Start-Process 'http://localhost:8504'"
if errorlevel 1 pause
