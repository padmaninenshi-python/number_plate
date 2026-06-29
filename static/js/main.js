/* PlateVision v3 — Perspective-accurate plate detection + Yellow plate support */

let stream = null, uploadData = null;
let currentMode = 'logo';
let cameraActive = false;   // ← NEW: track camera state

function setMode(m) {
    currentMode = m;
    document.querySelectorAll('.mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === m);
    });
}

function switchTab(t) {
    ['camera','upload'].forEach(id => {
        document.getElementById('tab-'+id).classList.toggle('active', id === t);
        document.getElementById('panel-'+id).classList.toggle('hidden', id !== t);
    });
    hide('result');
}

// ── NEW: update capture button lock state ─────────────────────────────────────
function _updateCaptureLock() {
    const btn      = document.getElementById('btnCapture');
    const lockMsg  = document.getElementById('captureLockMsg');
    if (cameraActive) {
        btn.disabled = false;
        btn.classList.remove('locked');
        if (lockMsg) lockMsg.classList.add('hidden');
    } else {
        btn.disabled = true;
        btn.classList.add('locked');
        if (lockMsg) lockMsg.classList.remove('hidden');
    }
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        });
        const v = document.getElementById('videoEl');
        v.srcObject = stream; v.play();
        document.getElementById('placeholderIcon').textContent = '📷';
        document.getElementById('placeholderText').textContent = 'Click Start Camera';
        document.getElementById('placeholder').classList.add('hidden');
        document.getElementById('scanLine').classList.add('active');
        dis('btnStart', true); dis('btnStop', false);
        cameraActive = true;        // ← unlock
        _updateCaptureLock();
    } catch(e) { alert('Camera error: ' + e.message); }
}

function stopCamera() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    const v = document.getElementById('videoEl');
    v.pause(); v.srcObject = null;
    cameraActive = false;
    // Auto-refresh page after stop
    window.location.reload();
}

function resumeCamera() {
    const v = document.getElementById('videoEl');
    if (stream && v.paused) {
        v.play();
        document.getElementById('scanLine').classList.add('active');
    }
}

function captureFrame() {
    if (!cameraActive) return;      // extra guard
    const v = document.getElementById('videoEl'), c = document.getElementById('canvas');
    c.width = v.videoWidth || 1280; c.height = v.videoHeight || 720;
    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);

    // Freeze video on captured frame
    v.pause();
    document.getElementById('scanLine').classList.remove('active');

    send(c.toDataURL('image/jpeg', 0.98));
}

function handleFile(e) {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = ev => {
        uploadData = ev.target.result;
        document.getElementById('uploadImg').src = uploadData;
        document.getElementById('uploadPreview').classList.remove('hidden');
        document.getElementById('dropzone').style.display = 'none';
    };
    r.readAsDataURL(f);
}

function detectFromUpload() {
    if (!uploadData) { alert('Select an image first.'); return; }
    send(uploadData);
}

document.addEventListener('DOMContentLoaded', () => {
    // Enforce lock on page load
    cameraActive = false;
    _updateCaptureLock();

    const dz = document.getElementById('dropzone');
    dz.addEventListener('dragover',  e => { e.preventDefault(); dz.style.borderColor = 'var(--o)'; });
    dz.addEventListener('dragleave', () => dz.style.borderColor = '');
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.style.borderColor = '';
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) handleFile({ target: { files: [f] } });
    });
});

async function send(imageData) {
    show('loading'); hide('result');
    try {
        const res = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, mode: currentMode })
        });
        const d = await res.json();
        hide('loading');
        if (d.error) { alert('Error: ' + d.error + '\n\n' + (d.trace||'')); return; }
        showResult(d);
    } catch(e) { hide('loading'); alert('Network error: ' + e.message); }
}

function showResult(d) {
    show('result');
    document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });

    const plate = d.plate_text || '—';
    document.getElementById('plateNumber').textContent = plate;

    // Plate type badge
    const colorLabel = d.plate_color === 'yellow'
        ? '<span class="plate-type-badge badge-yellow">🟡 Yellow / Rear</span>'
        : '<span class="plate-type-badge badge-white">⬜ White / Front</span>';

    const modeBadge = d.mode === 'border' ? '🔲 Border' : '🔒 Logo';
    const confBadge = d.conf != null ? `  •  conf ${d.conf}` : '';
    const enhBadge  = d.enhanced ? '  •  ⚡ enhanced' : '';

    document.getElementById('plateHello').innerHTML =
        d.status === 'success'
            ? `✅ ${plate}  •  ${modeBadge}${confBadge}${enhBadge} ${colorLabel}`
            : `${d.message}  •  ${modeBadge}${confBadge}${enhBadge} ${colorLabel}`;

    // Corner detection info
    const ci = document.getElementById('cornerInfo');
    if (d.perspective_quad) {
        ci.textContent = '✅ Perspective quad fitted — logo perfectly matches plate angle';
        ci.className = 'corner-info corner-quad';
    } else {
        ci.textContent = '📐 Bounding box used — plate edge not fully detected, but logo is placed correctly';
        ci.className = 'corner-info corner-bbox';
    }

    document.getElementById('resultImg').src = d.result_image;

    const dl = document.getElementById('dlBtn');
    dl.href     = d.result_image;
    dl.download = plate !== '—' ? `plate_${plate}.png` : 'plate_result.png';
}

function resetAll() {
    hide('result');
    uploadData = null;
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('dropzone').style.display = '';
    document.getElementById('fi').value = '';
    resumeCamera();   // unfreeze video after Try Another
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }
function dis(id, v) { document.getElementById(id).disabled = v; }
