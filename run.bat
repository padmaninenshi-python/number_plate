@echo off
echo.
echo  =====================================================
echo   PlateVision - Indian Number Plate Reader
echo   Hello Lakhay Jai!
echo  =====================================================
echo.
echo  STEP 1: Installing Python packages...
py -3 -m pip install -r requirements.txt
echo.
echo  STEP 2: Starting server...
echo  (First run downloads AI model weights - needs internet,
echo   takes a few seconds, then it's cached for future runs)
echo  Open browser at: http://localhost:5000
echo.
py -3 app.py
pause
