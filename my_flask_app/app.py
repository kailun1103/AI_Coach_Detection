from flask import Flask, render_template, request, redirect, url_for, flash
import requests

app = Flask(__name__)
app.secret_key = 'some_secret_key_for_flask'  # 供 flask flash 使用

# FastAPI 伺服器位址 (根路徑)，請依實際情況修改
FASTAPI_BASE_URL = "http://192.168.0.149:8000"

@app.route("/")
def index():
    """
    首頁，顯示呼叫各個 API 的表單
    """
    return render_template("index.html")

@app.route("/call_input_data", methods=["POST"])
def call_input_data():
    """
    呼叫 /input_data?name=xxx&height=xxx&dominant_hand=xxx
    """
    name = request.form.get("name")
    height = request.form.get("height")
    dominant_hand = request.form.get("dominant_hand")  # 0 or 1

    try:
        url = f"{FASTAPI_BASE_URL}/input_data"
        params = {
            "name": name,
            "height": height,
            "dominant_hand": dominant_hand
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        flash(f"呼叫成功: {data}", "success")
    except Exception as e:
        flash(f"呼叫失敗: {str(e)}", "danger")
    
    return redirect(url_for("index"))

@app.route("/call_take_photo", methods=["POST"])
def call_take_photo():
    """
    呼叫 /take_photo
    """
    try:
        url = f"{FASTAPI_BASE_URL}/take_photo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        flash(f"呼叫成功: {data}", "success")
    except Exception as e:
        flash(f"呼叫失敗: {str(e)}", "danger")

    return redirect(url_for("index"))

@app.route("/call_start_recording", methods=["POST"])
def call_start_recording():
    """
    呼叫 /start_recording
    """
    try:
        url = f"{FASTAPI_BASE_URL}/start_recording"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        flash(f"呼叫成功: {data}", "success")
    except Exception as e:
        flash(f"呼叫失敗: {str(e)}", "danger")

    return redirect(url_for("index"))

@app.route("/call_stop_recording", methods=["POST"])
def call_stop_recording():
    """
    呼叫 /stop_recording
    """
    try:
        url = f"{FASTAPI_BASE_URL}/stop_recording"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        flash(f"呼叫成功: {data}", "success")
    except Exception as e:
        flash(f"呼叫失敗: {str(e)}", "danger")

    return redirect(url_for("index"))

@app.route("/call_stop_recording_and_download", methods=["POST"])
def call_stop_recording_and_download():
    """
    呼叫 /stop_recording_and_download
    """
    try:
        url = f"{FASTAPI_BASE_URL}/stop_recording_and_download"
        response = requests.get(url, timeout=30)  # 下載可能花較久時間
        response.raise_for_status()
        data = response.json()
        flash(f"呼叫成功: {data}", "success")
    except Exception as e:
        flash(f"呼叫失敗: {str(e)}", "danger")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
