const videoPlayer = document.getElementById('videoPlayer');
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');

const folderSelect = document.getElementById('folderSelect');
const videoSelect = document.getElementById('videoSelect');
const basePath = "./assets/";

async function fetchFolderList() {
    try {
        const response = await fetch('/getFolders');
        const folders = await response.json();
        folderSelect.innerHTML = '<option value="">Choose Player Name</option>' + folders.map(folder => `<option value="${folder}">${folder}</option>`).join('');
    } catch (error) {
        console.error("Unable to fetch folder list:", error);
    }
}

async function fetchVideoList(folder) {
    try {
        const response = await fetch(`/getVideos?folder=${folder}`);
        const videos = await response.json();
        videoSelect.innerHTML = '<option value="">Choose Video</option>' + videos.filter(video => video.endsWith('.mp4')).map(video => `<option value="${basePath}${folder}/${video}">${video}</option>`).join('');
    } catch (error) {
        console.error("Unable to fetch video list:", error);
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
        const pathParts = selectedVideo.split('/');
        const folderName = pathParts[2];
        const fileName = pathParts[3];
        const prefix = fileName.replace('_full_video.mp4', '');
        const Json_45_Path = `${basePath}${folderName}/${prefix}_45(2D_trajectory_smoothed).json`;
        const Json_side_Path = `${basePath}${folderName}/${prefix}_side(2D_trajectory_smoothed).json`;
        handleFileSelection(Json_45_Path, Json_side_Path);
    }
});

async function handleFileSelection(filePath45, filePathSide) {
    try {
        const response45 = await fetch(filePath45);
        if (!response45.ok) {
            console.error("45 degree file not found, please check if path is correct:", filePath45);
            throw new Error(`HTTP error! Status: ${response45.status}`);
        }
        const data45 = await response45.json();
        const filename45 = filePath45.split('/').pop();
        console.log("45 degree JSON file loaded successfully, filename:", filename45);
        document.getElementById('filename2').textContent = filename45;
        createChart('trajectoryChart2', data45);
    } catch (error) {
        console.error("Failed to load 45 degree JSON file:", error);
    }
    try {
        const responseSide = await fetch(filePathSide);
        if (!responseSide.ok) {
            console.error("Side file not found, please check if path is correct:", filePathSide);
            throw new Error(`HTTP error! Status: ${responseSide.status}`);
        }
        const dataSide = await responseSide.json();
        const filenameSide = filePathSide.split('/').pop();
        console.log("Side JSON file loaded successfully, filename:", filenameSide);
        document.getElementById('filename1').textContent = filenameSide;
        createChart('trajectoryChart1', dataSide);
    } catch (error) {
        console.error("Failed to load Side JSON file:", error);
    }
}

function createChart(canvasId, data) {
    console.log("createChart executed", canvasId, data);
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
