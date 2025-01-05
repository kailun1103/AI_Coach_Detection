After venv is built

1. pip install -r requirements.txt

2. cd feedback

3. $env:OPENAI_API_KEY = "{API-KEY}"

4. upload file in the feedback/__data__/raw_data
    *naming rules : {standard or rookie}_raw_{number}
    *e.g. "standard_raw_01.json" or "rookie_raw_01.json"

5. apply main.py


