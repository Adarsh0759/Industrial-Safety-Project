// ============================================
// APP.JS - Safety Monitor Pro
// ============================================

const appState = {
    recording: false,
    startTime: Date.now(),
    alerts: [],
    detectionStats: {
        hardhats: 0,
        vests: 0,
        gestures: 0
    }
};

document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    startTimers();
    startFPSCounter();
    updateStatistics();
    updateDetectionInfo();
    updateUptime();
    showToast('Safety monitoring system initialized', 'success');
});

function initializeEventListeners() {
    const snapshot = document.getElementById('snapshot');
    const record = document.getElementById('record');
    const clearBtn = document.getElementById('clear-alerts');
    const videoFeed = document.getElementById('video-feed');

    if (snapshot) snapshot.addEventListener('click', captureSnapshot);
    if (record) record.addEventListener('click', toggleRecording);
    if (clearBtn) clearBtn.addEventListener('click', clearAlerts);
    if (videoFeed) videoFeed.addEventListener('load', updateFPSFrame);

    document.addEventListener('keydown', handleKeyboard);
}

function handleKeyboard(e) {
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    switch (e.key.toLowerCase()) {
        case 's':
            captureSnapshot();
            break;
        case 'r':
            toggleRecording();
            break;
        case 'c':
            clearAlerts();
            break;
    }
}

function captureSnapshot() {
    const videoFeed = document.getElementById('video-feed');
    if (!videoFeed) return;

    if (!videoFeed.naturalWidth || !videoFeed.naturalHeight) {
        showToast('No frame available to capture', 'warning');
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoFeed.naturalWidth;
    canvas.height = videoFeed.naturalHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoFeed, 0, 0);

    const timestamp = new Date().toLocaleString().replace(/[/:,\s]/g, '-');
    const link = document.createElement('a');
    link.download = `safety-snapshot-${timestamp}.jpg`;
    link.href = canvas.toDataURL('image/jpeg', 0.95);
    link.click();

    showToast('Snapshot captured', 'success');
}

function toggleRecording() {
    const recordBtn = document.getElementById('record');
    if (!recordBtn) return;

    appState.recording = !appState.recording;

    if (appState.recording) {
        recordBtn.classList.add('recording');
        recordBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Recording';
        showToast('Recording started', 'success');
    } else {
        recordBtn.classList.remove('recording');
        recordBtn.innerHTML = '<i class="fas fa-video"></i> Start Recording';
        showToast('Recording stopped', 'success');
    }
}

function addAlert(message, type = 'warning') {
    const now = Date.now();
    const isDuplicate = appState.alerts.some(alert =>
        alert.message === message && (now - alert.id) < 5000
    );

    if (isDuplicate) return;

    appState.alerts.unshift({
        id: now,
        message,
        type,
        timestamp: new Date()
    });

    if (appState.alerts.length > 10) {
        appState.alerts.pop();
    }

    updateAlertsList();
    showToast(message, type === 'danger' ? 'danger' : 'warning');
}

function updateAlertsList() {
    const alertsList = document.getElementById('alerts-list');
    if (!alertsList) return;

    if (appState.alerts.length === 0) {
        alertsList.innerHTML = '<p class="no-alerts">No alerts</p>';
        return;
    }

    alertsList.innerHTML = appState.alerts.map(alert => `
        <div class="alert-item ${alert.type}">
            <i class="fas ${alert.type === 'danger' ? 'fa-exclamation-circle' : 'fa-exclamation-triangle'}"></i>
            <div>
                <strong>${escapeHTML(alert.message)}</strong>
                <small style="display:block; opacity:0.8;">${alert.timestamp.toLocaleTimeString()}</small>
            </div>
        </div>
    `).join('');
}

function clearAlerts() {
    appState.alerts = [];
    updateAlertsList();
    showToast('Alerts cleared', 'success');
}

function escapeHTML(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

async function updateStatistics() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) {
            throw new Error(`Stats request failed with ${response.status}`);
        }

        const data = await response.json();

        appState.detectionStats = {
            hardhats: data.hardhats || 0,
            vests: data.vests || 0,
            gestures: data.hand_gestures || 0
        };

        setElementText('hardhat-count', data.hardhats || 0);
        setElementText('vest-count', data.vests || 0);
        setElementText('gesture-count', data.hand_gestures || 0);

        setElementText('ppe-hardhat', (data.hardhats || 0) > 0 ? 'Detected' : 'Not Detected');
        setElementText('ppe-vest', (data.vests || 0) > 0 ? 'Detected' : 'Not Detected');
        setElementText('gesture-status', (data.hand_gestures || 0) > 0 ? 'Active' : 'Idle');

        if (data.gesture_details && data.gesture_details.length > 0) {
            const gesture = data.gesture_details[0];
            setElementText('gesture-type', gesture.gesture || 'Unknown');
            setElementText(
                'gesture-conf',
                gesture.confidence ? `${(gesture.confidence * 100).toFixed(1)}%` : 'N/A'
            );
        } else {
            setElementText('gesture-type', 'None');
            setElementText('gesture-conf', '0%');
        }

        syncAlerts(data);
        updateDetectionInfo();
    } catch (error) {
        console.error('Failed to fetch stats:', error);
        showToast('Unable to fetch live statistics', 'warning');
    }
}

function syncAlerts(data) {
    if ((data.people || 0) > 0 && (data.hardhats || 0) === 0) {
        addAlert('People detected without hard hats', 'danger');
    }

    if ((data.people || 0) > 0 && (data.vests || 0) === 0) {
        addAlert('People detected without safety vests', 'warning');
    }
}

function updateDetectionInfo() {
    const detectionInfo = document.getElementById('detection-info');
    if (!detectionInfo) return;

    const { hardhats, vests, gestures } = appState.detectionStats;
    detectionInfo.innerHTML = `
        <span>Hardhats: <strong>${hardhats}</strong></span>
        <span>Vests: <strong>${vests}</strong></span>
        <span>Gestures: <strong>${gestures}</strong></span>
    `;
}

function startTimers() {
    setInterval(updateStatistics, 2000);
    setInterval(updateUptime, 1000);
    setInterval(updateDetectionInfo, 1000);
    setInterval(updateConnectionStatus, 5000);
}

function updateConnectionStatus() {
    const el = document.getElementById('connection-status');
    if (!el) return;

    el.textContent = 'Connected';
    el.style.color = '#27ae60';
}

function updateUptime() {
    const elapsed = Date.now() - appState.startTime;
    const hours = Math.floor(elapsed / 3600000);
    const minutes = Math.floor((elapsed % 3600000) / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    const pad = value => String(value).padStart(2, '0');

    setElementText('uptime', `${pad(hours)}:${pad(minutes)}:${pad(seconds)}`);
    setElementText('timestamp', new Date().toLocaleTimeString());
}

let frameTimestamps = [];

function startFPSCounter() {
    setInterval(() => {
        const now = Date.now();
        frameTimestamps.push(now);
        frameTimestamps = frameTimestamps.filter(timestamp => now - timestamp < 1000);
        setElementText('fps', frameTimestamps.length);
    }, 100);
}

function updateFPSFrame() {
    frameTimestamps.push(Date.now());
}

let toastTimer = null;

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    if (!toast || !toastMessage) return;

    if (toastTimer) clearTimeout(toastTimer);

    toastMessage.textContent = message;
    toast.className = `toast show ${type === 'danger' ? 'danger' : type === 'warning' ? 'warning' : ''}`;

    toastTimer = setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function setElementText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}
