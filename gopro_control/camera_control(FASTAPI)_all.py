from fastapi import FastAPI
import aiohttp
import asyncio
import uvicorn

app = FastAPI(title="GoPro Controller API")

async def post_gopro(session, url):
    try:
        async with session.post(url) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e)}
    
async def get_gopro(session, url):
    try:
        async with session.get(url) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/connect")
async def connect_gopro():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            get_gopro(session, "http://localhost:3253/connect"),
            get_gopro(session, "http://localhost:9436/connect")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/take_photo")
async def take_photo():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/take_photo"),
            post_gopro(session, "http://localhost:9436/take_photo")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/start_recording")
async def start_recording():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/start_recording"),
            post_gopro(session, "http://localhost:9436/start_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/stop_recording")
async def stop_recording():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/stop_recording"),
            post_gopro(session, "http://localhost:9436/stop_recording")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/stop_recording_and_download")
async def stop_recording_and_download():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            post_gopro(session, "http://localhost:3253/stop_recording_and_download"),
            post_gopro(session, "http://localhost:9436/stop_recording_and_download")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

@app.get("/download_last")
async def download_last_media():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            get_gopro(session, "http://localhost:3253/download_last"),
            get_gopro(session, "http://localhost:9436/download_last")
        )
    return {
        "gopro1": results[0],
        "gopro2": results[1]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)