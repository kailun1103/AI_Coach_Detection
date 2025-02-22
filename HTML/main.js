const videoPlayer = document.getElementById('videoPlayer');
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');
const Json_45_Path = "./assets/player01/";
const Json_side_Path = "./assets/player01/";


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
        videoPlayer.src = selectedVideo;
        videoPlayer.play();
    }
});


// -----------------------------------------------------------------------------
// -----Json Process-----------------------------------------------------------
// -----------------------------------------------------------------------------

let charts = {
    chart1: null,
    chart2: null
};

// 創建軌跡圖表
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


async function fetchJsonList(directory) {
    try {
        const response = await fetch(`/getjson?dir=${encodeURIComponent(directory)}`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const files = await response.json();
        return files;
    } catch (error) {
        console.error(`無法獲取目錄 ${directory} 中的 JSON 檔案：`, error);
        return [];
    }
}


async function populateSideSelect() {
    const sideSelect = document.getElementById('jsonFileSelectSide');
    if (!sideSelect) return;
    const files = await fetchJsonList(Json_side_Path);
    sideSelect.innerHTML = '<option value="">Choose Side file</option>';
    files.forEach(file => {
        const option = document.createElement('option');
        option.value = `${Json_side_Path}/${file}`;
        option.textContent = file;
        sideSelect.appendChild(option);
    });
}

// 填充 45 degree 檔案下拉選單
async function populate45Select() {
    const degreeSelect = document.getElementById('jsonFileSelect45');
    if (!degreeSelect) return;
    const files = await fetchJsonList(Json_45_Path);
    degreeSelect.innerHTML = '<option value="">Choose 45 degree file</option>';
    files.forEach(file => {
        const option = document.createElement('option');
        option.value = `${Json_45_Path}/${file}`;
        option.textContent = file;
        degreeSelect.appendChild(option);
    });
}

// 全域的檔案選取處理函式，根據所選檔案載入 JSON 並生成圖表
async function handleFileSelection(filePath) {
    if (!filePath) return; // 如果未選擇檔案，則不執行任何動作

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const filename = filePath.split('/').pop(); // 從檔案路徑取出檔名

        // 根據 select 元素的 ID 判斷更新哪個區塊
        if (filePath.includes("/45/")) {
            document.getElementById('filename2').textContent = filename;
            createChart('trajectoryChart2', data);
        } else {
            document.getElementById('filename1').textContent = filename;
            createChart('trajectoryChart1', data);
        }
    } catch (error) {
        console.error("載入 JSON 檔案失敗：", error);
    }
}


// 當 DOM 載入完成後，初始化下拉選單
document.addEventListener('DOMContentLoaded', () => {
    populateSideSelect();
    populate45Select();
});