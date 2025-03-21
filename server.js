const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(express.static(path.join(__dirname)));

app.listen(port, () => {
    console.log(`伺服器運行於 http://localhost:${port}`);
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'drawing_2D_chart_js.html'));
});

app.get('/getFolders', (req, res) => {
    const assetsDir = path.join(__dirname, 'trajectory');
    fs.readdir(assetsDir, (err, files) => {
        if (err) {
            // console.error("讀取資料夾失敗：", err);
            return res.status(500).json({ error: '無法讀取資料夾' });
        }
        // 只保留目錄（資料夾）
        const folders = files.filter(file => {
            const filePath = path.join(assetsDir, file);
            return fs.statSync(filePath).isDirectory();
        });
        res.json(folders);
    });
});

app.get('/getVideos', (req, res) => {
    const folder = req.query.folder;
    if (!folder) {
        return res.status(400).json({ error: '請提供 folder 參數' });
    }
    const videoDir = path.join(__dirname, 'trajectory', folder);
    fs.readdir(videoDir, (err, files) => {
        if (err) {
            // console.error(`讀取 ${folder} 資料夾失敗：`, err);
            return res.status(500).json({ error: `無法讀取 ${folder} 資料夾` });
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