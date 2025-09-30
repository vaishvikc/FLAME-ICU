@echo off
REM Simple environment setup for FLAME-ICU
python -m venv flameICU
call flameICU\Scripts\activate.bat
pip install -r requirements.txt
echo Setup complete. Run 'flameICU\Scripts\activate.bat' to use.