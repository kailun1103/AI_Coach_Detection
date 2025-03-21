const videoPlayer = document.getElementById('videoPlayer');
// --- Speed ---------------------------------------------
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');

function updateSpeed() {
    const speed = Number(speedControl.value);
    videoPlayer.playbackRate = speed;
    speedValue.textContent = `${speed.toFixed(2)}x`;

    const min = Number(speedControl.min);
    const max = Number(speedControl.max);
    const percent = ((speed - min) / (max - min)) * 100;
    speedControl.style.setProperty('--val', percent + '%');
}

speedControl.addEventListener('input', updateSpeed);
updateSpeed();


// ---Video to Json---------------------------------------------
const folderSelect = document.getElementById('folderSelect');
const videoSelect = document.getElementById('videoSelect');
const basePath = "./trajectory/";

async function fetchFolderList() {
    try {
        const response = await fetch('/getFolders');
        const folders_full = await response.json();
        const folders = folders_full.map(folder_full => folder_full.split('__')[0]);
        folderSelect.innerHTML = '<option value="">Player Name</option>' + folders.map(folder => `<option value="${folder}">${folder}</option>`).join('');
    } catch (error) {
        console.error("Unable to fetch folder list:", error);
    }
}

async function fetchVideoList(folder) {
    // 新增預設選項
    videoSelect.innerHTML = `<option value="">select trajectory</option>`;
    for (let i = 1; i <= 100; i++) {
        try {

            const response = await fetch(`/getVideos?folder=${folder}__trajectory/trajectory__${i}`);
            if (!response.ok) {
                console.warn(`trajectory_${i} not found, skip.`);
                continue;
            }
            const videos_all = await response.json();
            const videos = videos_all.filter(v => v.includes('full_video'));
            console.log(videos);
            videoSelect.innerHTML += videos
                .filter(video => video.endsWith('.mp4'))
                .map(video => `<option value="${basePath}${folder}/trajectory_${i}/${video}">${video}</option>`)
                .join('');
        } catch (error) {
            // console.error(`Error fetching trajectory_${i}:`, error);
            continue;
        }
    }
}

folderSelect.addEventListener('change', e => {
    const selectedFolder = e.target.value;
    selectedFolder ? fetchVideoList(selectedFolder) : videoSelect.innerHTML = '<option value="">Choose Video</option>';
});

document.addEventListener('DOMContentLoaded', fetchFolderList);

videoSelect.addEventListener('change', e => {
    const selected_path = e.target.value;
    let pathParts = selected_path.split('/');
    pathParts[2] = pathParts[2] + '__trajectory';
    pathParts[3] = pathParts[3].replace('trajectory_', 'trajectory__');
    const selectedVideo = pathParts.join('/');
    console.log(selectedVideo);

    if (selectedVideo) {
        videoPlayer.src = selectedVideo;
        console.log("TARGET DEBUG", selectedVideo)

        const playPromise = videoPlayer.play();
        if (playPromise !== undefined) {
            playPromise.catch(error => { error });
        }

        const pathParts = selectedVideo.split('/');
        if (pathParts.length >= 4) {
            const folderName = pathParts[2];
            const fileName = pathParts[3];
            const trajectory = pathParts[4];
            const prefix = trajectory.replace('_full_video.mp4', '');
            console.log("資料夾名稱：", folderName, "檔案名稱：", fileName, "tra：", trajectory, "prefix：", prefix, "basePath：", basePath);

            const Json_45_Path = `${basePath}${folderName}/${fileName}/${prefix}_45(2D_trajectory_smoothed).json`;
            const Json_side_Path = `${basePath}${folderName}/${fileName}/${prefix}_side(2D_trajectory_smoothed).json`;

            handleFileSelection(Json_45_Path, Json_side_Path);
        }
    }
});




document.addEventListener('DOMContentLoaded', fetchFolderList);


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