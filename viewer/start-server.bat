@echo off
echo Starting AFE Viewer server...
echo.
echo Open your browser to: http://127.0.0.1:8080
echo.
cd /d "C:\Users\david\Documents\GitHub\arrow-field-embeddings\viewer\dist"
python -m http.server 8080 --bind 127.0.0.1
pause
