// Real-time stats update (simulated)
setInterval(() => {
    const alertList = document.getElementById('alert-list');
    alertList.innerHTML = `
        <li>Hardhat detected: ${Math.floor(Math.random() * 10)}</li>
        <li>Safety vest missing: ${Math.floor(Math.random() * 5)}</li>
        <li>Warning gestures: ${Math.floor(Math.random() * 3)}</li>
    `;
}, 2000);

// Snapshot functionality
document.getElementById('snapshot').addEventListener('click', () => {
    const videoFeed = document.querySelector('img');
    const canvas = document.createElement('canvas');
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoFeed, 0, 0);
    
    const link = document.createElement('a');
    link.download = 'safety-snapshot.jpg';
    link.href = canvas.toDataURL('image/jpeg');
    link.click();
});

// Camera toggle
document.getElementById('toggle-cam').addEventListener('click', () => {
    alert('Camera toggle functionality would be implemented here');
});
