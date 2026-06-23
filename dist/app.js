/**
 * app.js — カメラ + DeviceOrientation ベースの AR グリッド描画
 *
 * WebXR を廃止し、全 iOS ブラウザ (Safari / Chrome 両方) で動作する
 * getUserMedia + DeviceOrientationEvent に切り替え。
 *
 * フレームごとの処理:
 *   DeviceOrientationEvent (beta/gamma)
 *     → Kalman1D フィルタ (JS 実装)
 *     → Three.js カメラ回転を更新
 *     → fallbackBuildVertices() でグリッド頂点生成
 *     → BufferGeometry 更新 → renderer.render()
 */

/* ── 定数 ─────────────────────────────────────────────────────────────── */

const MAX_FLOATS      = 12_000;
const DEFAULT_SPACING = 0.60;   // 初期グリッド間隔 [m]
const DEFAULT_EXTENT  = 2.50;   // グリッド半幅 [m]
const EYE_HEIGHT      = 1.60;   // 仮定カメラ高さ [m]
const CAM_FOV         = 60;     // iPhone リアカメラの垂直 FOV 近似値 [deg]

/* ── JS Kalman フィルタ (internal/grid_core.py の _Kalman1D に対応) ─── */

class Kalman1D {
    /**
     * @param {number} q プロセスノイズ — 角度の変化しやすさ
     * @param {number} r 観測ノイズ — センサー読み値のばらつき [deg]
     */
    constructor(q = 0.1, r = 1.0) {
        this._q = q;  this._r = r;
        this._x = 0;  this._p = 1;  this._ready = false;
    }
    update(z) {
        if (!this._ready) { this._x = z; this._ready = true; return z; }
        const pPred = this._p + this._q;
        const k     = pPred / (pPred + this._r);
        this._x    += k * (z - this._x);
        this._p     = (1 - k) * pPred;
        return this._x;
    }
}

// beta (前後傾き 0-180°) と gamma (左右傾き -90〜90°) を個別にフィルタ
const kBeta  = new Kalman1D(0.1, 1.0);
const kGamma = new Kalman1D(0.1, 1.0);
let filteredBeta  = 60;   // 初期値: カメラが約 30° 下向き
let filteredGamma = 0;

/* ── Three.js セットアップ ────────────────────────────────────────────── */

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);   // 背景を透明にしてビデオを透過
document.getElementById('canvas-container').appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(CAM_FOV,
    window.innerWidth / window.innerHeight, 0.01, 20);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ── グリッドジオメトリ ───────────────────────────────────────────────── */

const gridPositions = new Float32Array(MAX_FLOATS);
const posAttr       = new THREE.BufferAttribute(gridPositions, 3);
posAttr.setUsage(THREE.DynamicDrawUsage);

const gridGeo  = new THREE.BufferGeometry();
gridGeo.setAttribute('position', posAttr);
gridGeo.setDrawRange(0, 0);

const gridMat  = new THREE.LineBasicMaterial({ color: 0x00ff88 });
const gridMesh = new THREE.LineSegments(gridGeo, gridMat);
scene.add(gridMesh);

/* ── カメラ姿勢更新 ──────────────────────────────────────────────────── */

function updateCamera() {
    // カメラを目線の高さに固定
    camera.position.set(0, EYE_HEIGHT, 0);
    camera.rotation.order = 'YXZ';

    // beta:  0° = スマホ水平(上向き) / 90° = 垂直(水平を見る) / 45° = 床を見る
    // pitch: 正 = 上向き / 負 = 下向き
    camera.rotation.x = -(90 - filteredBeta) * Math.PI / 180;

    // gamma: 正 = 右傾き / 負 = 左傾き (符号を反転して自然な見え方に)
    camera.rotation.z = -filteredGamma * Math.PI / 180;
}

/* ── グリッド頂点生成 (JS 純実装) ───────────────────────────────────── */

const fallback = { plane: null };

function fallbackBuildVertices(spacing, extentX, extentZ) {
    if (!fallback.plane) return 0;
    const { nx, ny, nz, d } = fallback.plane;
    const p0 = [nx * d, ny * d, nz * d];

    // 基底ベクトル (Y-up)
    let ux, uy, uz;
    if (Math.abs(ny) < 0.9) { ux = -nz; uy = 0;  uz = nx; }
    else                     { ux =   0; uy = nz;  uz = -ny; }
    const ul = Math.hypot(ux, uy, uz);
    ux /= ul; uy /= ul; uz /= ul;
    const vx = ny*uz - nz*uy, vy = nz*ux - nx*uz, vz = nx*uy - ny*ux;

    const stepsU = Math.max(1, Math.round(extentX / spacing));
    const stepsV = Math.max(1, Math.round(extentZ / spacing));
    let idx = 0;

    for (let i = -stepsV; i <= stepsV; i++) {
        const t = i * spacing;
        gridPositions[idx++] = p0[0] + t*vx - extentX*ux;
        gridPositions[idx++] = p0[1] + t*vy - extentX*uy;
        gridPositions[idx++] = p0[2] + t*vz - extentX*uz;
        gridPositions[idx++] = p0[0] + t*vx + extentX*ux;
        gridPositions[idx++] = p0[1] + t*vy + extentX*uy;
        gridPositions[idx++] = p0[2] + t*vz + extentX*uz;
    }
    for (let i = -stepsU; i <= stepsU; i++) {
        const t = i * spacing;
        gridPositions[idx++] = p0[0] + t*ux - extentZ*vx;
        gridPositions[idx++] = p0[1] + t*uy - extentZ*vy;
        gridPositions[idx++] = p0[2] + t*uz - extentZ*vz;
        gridPositions[idx++] = p0[0] + t*ux + extentZ*vx;
        gridPositions[idx++] = p0[1] + t*uy + extentZ*vy;
        gridPositions[idx++] = p0[2] + t*uz + extentZ*vz;
    }
    return idx;
}

function updateGridGeometry() {
    const n = fallbackBuildVertices(uiState.spacing, DEFAULT_EXTENT, DEFAULT_EXTENT);
    if (n > 0) {
        posAttr.needsUpdate = true;
        gridGeo.setDrawRange(0, n / 3);
    }
}

/* ── メインループ ────────────────────────────────────────────────────── */

function renderLoop() {
    requestAnimationFrame(renderLoop);
    updateCamera();
    if (gridMesh.visible) updateGridGeometry();
    renderer.render(scene, camera);
}

/* ── 起動フロー ──────────────────────────────────────────────────────── */

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function start() {
    /* 1. リアカメラ映像取得 */
    setStatus('カメラを起動中…');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width:  { ideal: 1920 },
                height: { ideal: 1080 },
            },
            audio: false,
        });
        const video = document.getElementById('cam-feed');
        video.srcObject = stream;
        video.style.display = 'block';
    } catch (e) {
        alert('カメラへのアクセスが拒否されました。\nブラウザの設定でカメラを許可してください。');
        setStatus('カメラ許可が必要です');
        return;
    }

    /* 2. 傾きセンサー (iOS 13+ は明示的許可が必要) */
    setStatus('傾きセンサーを初期化中…');
    try {
        if (typeof DeviceOrientationEvent !== 'undefined' &&
            typeof DeviceOrientationEvent.requestPermission === 'function') {
            // iOS 13+ — ユーザーアクション内でのみ呼び出し可能
            const perm = await DeviceOrientationEvent.requestPermission();
            if (perm !== 'granted') throw new Error('denied');
        }
        window.addEventListener('deviceorientation', (e) => {
            if (e.beta  !== null) filteredBeta  = kBeta.update(e.beta);
            if (e.gamma !== null) filteredGamma = kGamma.update(e.gamma);
        });
    } catch (_) {
        // デスクトップや古い端末: 固定の俯瞰アングルで表示
        setStatus('傾きセンサー未対応 — 固定視点で表示');
    }

    /* 3. 床面を固定 (y=0 の水平平面) */
    fallback.plane = { nx: 0, ny: 1, nz: 0, d: 0 };

    /* 4. UI 切り替え */
    document.getElementById('btn-start').hidden = true;
    document.getElementById('ui').classList.add('visible');
    setStatus('');

    /* 5. レンダーループ開始 */
    renderLoop();
}

document.getElementById('btn-start').addEventListener('click', start);

/* ── UI イベント ─────────────────────────────────────────────────────── */

const uiState = { spacing: DEFAULT_SPACING };

document.getElementById('slider-spacing').addEventListener('input', (e) => {
    uiState.spacing = parseFloat(e.target.value);
    document.getElementById('label-spacing').textContent =
        `${uiState.spacing.toFixed(2)} m`;
});

document.getElementById('slider-smoothing').addEventListener('input', (e) => {
    const r = parseFloat(e.target.value);
    document.getElementById('label-smoothing').textContent = r.toFixed(2);
    // R が大きいほど平滑化が強い (計測値を信用しない)
    kBeta._r  = r * 5;
    kGamma._r = r * 5;
});

document.getElementById('color-picker').addEventListener('input', (e) => {
    gridMat.color.set(e.target.value);
});

document.getElementById('toggle-grid').addEventListener('change', (e) => {
    gridMesh.visible = e.target.checked;
    document.getElementById('label-toggle').textContent =
        e.target.checked ? '表示' : '非表示';
    if (!e.target.checked) gridGeo.setDrawRange(0, 0);
});
