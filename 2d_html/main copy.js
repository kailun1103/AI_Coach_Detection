const videoPlayer = document.getElementById('videoPlayer');
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');



// -----------------------------------------------------------------------------
// -----Video Process-----------------------------------------------------------
// -----------------------------------------------------------------------------
const folderSelect = document.getElementById('folderSelect');
const videoSelect = document.getElementById('videoSelect');
const basePath = "./assets/";

async function fetchFolderList() {
    try {
        const response = await fetch('/getFolders');
        const folders = await response.json();
        folderSelect.innerHTML = '<option value="">Choose Player Name</option>' + folders.map(folder => `<option value="${folder}">${folder}</option>`).join('');
    } catch (error) {
        console.error("無法獲取資料夾清單：", error);
    }
}

async function fetchVideoList(folder) {
    try {
        const response = await fetch(`/getVideos?folder=${folder}`);
        const videos = await response.json();
        videoSelect.innerHTML = '<option value="">Choose Video</option>' + videos.filter(video => video.endsWith('.mp4')).map(video => `<option value="${basePath}${folder}/${video}">${video}</option>`).join('');
    } catch (error) {
        console.error("無法獲取影片清單：", error);
    }
}

folderSelect.addEventListener('change', e => {
    const selectedFolder = e.target.value;
    selectedFolder ? fetchVideoList(selectedFolder) : videoSelect.innerHTML = '<option value="">Choose Video</option>';
});

document.addEventListener('DOMContentLoaded', fetchFolderList);

videoSelect.addEventListener('change', e => {
    const selectedVideo = e.target.value;
    if (selectedVideo) {
        // 播放影片
        videoPlayer.src = selectedVideo;
        videoPlayer.play();

        // 假設 selectedVideo 格式為 "./assets/{folder selected}/{prefix}_full_video.mp4"
        const pathParts = selectedVideo.split('/');
        // 例如: [".", "assets", "folderName", "prefix_full_video.mp4"]
        const folderName = pathParts[2];
        const fileName = pathParts[3];

        // 取得 prefix (移除 "_full_video.mp4")
        const prefix = fileName.replace('_full_video.mp4', '');

        // 產生 JSON 路徑變數
        const Json_45_Path = `${basePath}${folderName}/${prefix}_45(2D_trajectory_smoothed).json`;
        const Json_side_Path = `${basePath}${folderName}/${prefix}_side(2D_trajectory_smoothed).json`;

        handleFileSelection(Json_45_Path, Json_side_Path);
    }
});


async function handleFileSelection(filePath45, filePathSide) {
    // 處理 45 度的 JSON 檔案
    try {
        const response45 = await fetch(filePath45);
        if (!response45.ok) {
            console.error("找不到 45 度檔案，請檢查路徑是否正確：", filePath45);
            throw new Error(`HTTP error! Status: ${response45.status}`);
        }
        const data45 = await response45.json();
        const filename45 = filePath45.split('/').pop(); // 取出檔名
        console.log('45 度 JSON 檔案載入成功，檔名：', filename45);
        // 將 45 度檔案導入至 filename2 與 trajectoryChart2
        document.getElementById('filename2').textContent = filename45;
        createChart('trajectoryChart2', data45);
    } catch (error) {
        console.error("載入 45 度 JSON 檔案失敗：", error);
    }

    // 處理 Side 的 JSON 檔案
    try {
        const responseSide = await fetch(filePathSide);
        if (!responseSide.ok) {
            console.error("找不到 Side 檔案，請檢查路徑是否正確：", filePathSide);
            throw new Error(`HTTP error! Status: ${responseSide.status}`);
        }
        const dataSide = await responseSide.json();
        const filenameSide = filePathSide.split('/').pop(); // 取出檔名
        console.log('Side JSON 檔案載入成功，檔名：', filenameSide);
        // 將 Side 檔案導入至 filename1 與 trajectoryChart1
        document.getElementById('filename1').textContent = filenameSide;
        createChart('trajectoryChart1', dataSide);
    } catch (error) {
        console.error("載入 Side JSON 檔案失敗：", error);
    }
}


function createChart(canvasId, data) {
    console.log("createChart 被執行", canvasId, data);
    const points = data
        .map(frame => ({
            x: frame.right_wrist?.x,
            y: frame.right_wrist?.y,
            frame: frame.frame
        }))
        .filter(point => point.x != null && point.y != null);
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Wrist Trajectory',
                data: points,
                borderColor: '#4DC4C0',
                backgroundColor: 'rgba(77, 196, 192, 0.5)',
                showLine: true,
                pointRadius: 3,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: { display: true, text: 'X Position' },
                    grid: { color: '#E5E5E5' }
                },
                y: {
                    type: 'linear',
                    reverse: true,
                    title: { display: true, text: 'Y Position' },
                    grid: { color: '#E5E5E5' }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `Frame: ${context.raw.frame}, X: ${context.raw.x.toFixed(2)}, Y: ${context.raw.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}


let charts = {
    chart1: null,
    chart2: null
};