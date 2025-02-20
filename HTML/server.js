const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname)));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// API: 讀取 assets/video 資料夾中的 MP4 檔案清單
app.get('/getVideos', (req, res) => {
    const videoDir = path.join(__dirname, 'assets', 'video');
    fs.readdir(videoDir, (err, files) => {
        if (err) {
            console.error("讀取影片資料夾失敗：", err);
            return res.status(500).json({ error: '無法讀取影片資料夾' });
        }
        // 過濾出副檔名為 .mp4 的檔案
        const mp4Files = files.filter(file => file.toLowerCase().endsWith('.mp4'));
        res.json(mp4Files);
    });
});

app.get("/getjson", (req, res) => {
    const dir = req.query.dir;
    if (!dir) {
        return res.status(400).json({ error: "缺少 dir 參數" });
    }

    const directoryPath = path.join(__dirname, dir);

    fs.readdir(directoryPath, (err, files) => {
        if (err) {
            return res.status(500).json({ error: "無法讀取目錄", details: err.message });
        }

        // 僅回傳 .json 檔案
        const jsonFiles = files.filter(file => file.endsWith(".json"));
        res.json(jsonFiles);
    });
});


app.listen(port, () => {
    console.log(`伺服器運行於 http://localhost:${port}`);
});
