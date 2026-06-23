/**
 * app.js
 *
 * 座標系: local-floor (y=0 = 床面) を世界座標として扱う。
 * グリッドは y=0 の水平平面に固定された THREE.Mesh であり、
 * カメラがどこへ移動してもグリッドは地面に張り付いたまま。
 *
 * グリッドの描画はフラグメントシェーダーで vWorldPos.xz を使って
 * 世界座標上に格子を引くため、頂点バッファの再生成は一切不要。
 * 大きな PlaneGeometry をカメラ直下に追従させるだけでよい。
 *
 * Primary:  WebXR immersive-ar (local-floor) → ARKit 6DOF カメラポーズ
 * Fallback: getUserMedia + DeviceOrientation → 3DOF (回転のみ、並進なし)
 */

/* ═══════════════════════════════════════════════════════════
   §1  グリッドシェーダー
   ═══════════════════════════════════════════════════════════ */

const GRID_VERT = /* glsl */`
varying vec3 vWorldPos;
void main() {
    // ワールド座標を varying に渡す (modelMatrix = mesh の World Transform)
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

const GRID_FRAG = /* glsl */`
#extension GL_OES_standard_derivatives : enable
precision highp float;

uniform float uSpacing;   // グリッド間隔 [m]
uniform vec3  uColor;
uniform float uOpacity;

varying vec3 vWorldPos;

void main() {
    // ワールド XZ 座標を spacing で割り、格子パターンを計算
    // fwidth() による自動アンチエイリアシング（距離に応じた線幅）
    vec2 coord = vWorldPos.xz / uSpacing;
    vec2 grid  = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line  = min(grid.x, grid.y);
    float alpha = (1.0 - clamp(line, 0.0, 1.0)) * uOpacity;
    if (alpha < 0.005) discard;
    gl_FragColor = vec4(uColor, alpha);
}
`;

/* ═══════════════════════════════════════════════════════════
   §2  定数
   ═══════════════════════════════════════════════════════════ */

const DEFAULT_SPACING = 0.60;   // [m] 模擬歩幅
const PLANE_SIZE      = 200;    // [m] 床面メッシュサイズ (常にカメラ直下をカバー)
const FALLBACK_EYE_H  = 1.40;  // [m] DeviceOrientation 時のカメラ高さ

/* ═══════════════════════════════════════════════════════════
   §3  Three.js セットアップ
   ═══════════════════════════════════════════════════════════ */

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);
renderer.xr.enabled = true;   // WebXR 対応 (非 XR セッション時も通常描画可)
document.getElementById('canvas-container').appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    60, window.innerWidth / window.innerHeight, 0.01, 200
);
camera.position.set(0, FALLBACK_EYE_H, 0);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ═══════════════════════════════════════════════════════════
   §4  ワールド座標グリッドメッシュ
   ═══════════════════════════════════════════════════════════ */

const gridUniforms = {
    uSpacing: { value: DEFAULT_SPACING },
    uColor:   { value: new THREE.Color(0x00ff88) },
    uOpacity: { value: 0.9 },
};

const gridMat = new THREE.ShaderMaterial({
    uniforms:       gridUniforms,
    vertexShader:   GRID_VERT,
    fragmentShader: GRID_FRAG,
    transparent:    true,
    side:           THREE.DoubleSide,
    depthWrite:     false,
    extensions:     { derivatives: true },  // WebGL1 で fwidth を有効化
});

// PlaneGeometry はデフォルトで XY 平面 → -90° 回転して XZ 水平床
const gridMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE, 1, 1),
    gridMat
);
gridMesh.rotation.x = -Math.PI / 2;
gridMesh.position.y = 0;   // ワールド Y=0 に固定
scene.add(gridMesh);

/* ═══════════════════════════════════════════════════════════
   §5  DeviceOrientation フォールバック
   ═══════════════════════════════════════════════════════════ */

// 1D Kalman フィルタ (python 側 _Kalman1D の JS 移植)
class Kalman1D {
    constructor(q, r) {
        this.q = q; this.r = r;
        this.x = 0; this.p = 1; this.ready = false;
    }
    update(z) {
        if (!this.ready) { this.x = z; this.ready = true; return z; }
        const pp = this.p + this.q;
        const k  = pp / (pp + this.r);
        this.x  += k * (z - this.x);
        this.p   = (1 - k) * pp;
        return this.x;
    }
}

const kAlpha = new Kalman1D(0.5, 2.0);
const kBeta  = new Kalman1D(0.1, 1.0);
const kGamma = new Kalman1D(0.1, 1.0);
const devOri = { alpha: 0, beta: 60, gamma: 0 };

// DeviceOrientation (alpha/beta/gamma) → Four.js カメラ回転
// Three.js DeviceOrientationControls と同じ変換式
const _q0    = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));
const _euler = new THREE.Euler();
const _qOri  = new THREE.Quaternion();

function applyOrientationToCamera() {
    _euler.set(
        THREE.MathUtils.degToRad(devOri.beta),
        THREE.MathUtils.degToRad(devOri.alpha),
        THREE.MathUtils.degToRad(-devOri.gamma),
        'YXZ'
    );
    _qOri.setFromEuler(_euler);
    camera.quaternion.copy(_qOri).multiply(_q0);
    // 並進は推定できないので初期位置に固定
    camera.position.set(0, FALLBACK_EYE_H, 0);
}

/* ═══════════════════════════════════════════════════════════
   §6  フレームループ
   ═══════════════════════════════════════════════════════════ */

let isXrMode   = false;
let xrRefSpace = null;

function onFrame(_timestamp, frame) {
    let camX = 0, camZ = 0;

    if (isXrMode && frame && xrRefSpace) {
        // WebXR: ARKit から 6DOF カメラポーズを取得
        const pose = frame.getViewerPose(xrRefSpace);
        if (pose) {
            camX = pose.transform.position.x;
            camZ = pose.transform.position.z;
        }
    } else {
        // DeviceOrientation: 並進なし (カメラは常に原点)
        applyOrientationToCamera();
        camX = 0;
        camZ = 0;
    }

    // グリッド平面をカメラ直下に追従させる
    // ——重要——
    // mesh は XZ 方向に動くが、シェーダーは vWorldPos.xz (絶対座標) で
    // 格子を描くため、グリッド線は常に世界座標に固定されて見える。
    // カメラが前進するとグリッド線が足元を後ろへ流れる動きになる。
    gridMesh.position.set(camX, 0, camZ);

    renderer.render(scene, camera);
}

/* ═══════════════════════════════════════════════════════════
   §7  起動フロー
   ═══════════════════════════════════════════════════════════ */

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function tryWebXR() {
    if (!navigator.xr) return false;
    const supported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
    if (!supported) return false;

    const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hit-test'],
    });

    renderer.xr.setReferenceSpaceType('local-floor');
    await renderer.xr.setSession(session);
    xrRefSpace = await session.requestReferenceSpace('local-floor');

    session.addEventListener('end', () => {
        xrRefSpace = null;
        isXrMode   = false;
        document.getElementById('btn-start').hidden = false;
        document.getElementById('ui').classList.remove('visible');
    });

    isXrMode = true;
    renderer.setAnimationLoop(onFrame);
    setStatus('ARKit トラッキング中');
    return true;
}

async function startDeviceOrientationFallback() {
    // カメラ映像をビデオ要素に流す (AR 感を出すための背景)
    const video = document.getElementById('cam-feed');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 } },
            audio: false,
        });
        video.srcObject = stream;
        video.style.display = 'block';
    } catch (_) {
        setStatus('カメラなし — グリッドのみ表示');
    }

    // iOS 13+ は明示的な許可が必要 (ユーザーアクション内でのみ有効)
    try {
        if (typeof DeviceOrientationEvent.requestPermission === 'function') {
            if (await DeviceOrientationEvent.requestPermission() !== 'granted') throw 0;
        }
        window.addEventListener('deviceorientation', (e) => {
            if (e.alpha !== null) devOri.alpha = kAlpha.update(e.alpha);
            if (e.beta  !== null) devOri.beta  = kBeta.update(e.beta);
            if (e.gamma !== null) devOri.gamma = kGamma.update(e.gamma);
        });
        setStatus('ジャイロ モード (並進なし)');
    } catch (_) {
        setStatus('固定視点 — 傾きセンサー未許可');
    }

    renderer.setAnimationLoop(onFrame);
}

document.getElementById('btn-start').addEventListener('click', async () => {
    document.getElementById('btn-start').hidden = true;
    document.getElementById('ui').classList.add('visible');

    const gotXr = await tryWebXR();
    if (!gotXr) await startDeviceOrientationFallback();
});

/* ═══════════════════════════════════════════════════════════
   §8  UI イベント
   ═══════════════════════════════════════════════════════════ */

const uiState = { spacing: DEFAULT_SPACING };

document.getElementById('slider-spacing').addEventListener('input', (e) => {
    uiState.spacing = parseFloat(e.target.value);
    // uniform を直接更新するだけ — 頂点バッファ再生成は不要
    gridUniforms.uSpacing.value = uiState.spacing;
    document.getElementById('label-spacing').textContent =
        `${uiState.spacing.toFixed(2)} m`;
});

document.getElementById('slider-smoothing').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    // R が大きいほど計測値を信用しない → 平滑化強
    kBeta.r = kGamma.r = v * 5 + 0.1;
    document.getElementById('label-smoothing').textContent = v.toFixed(2);
});

document.getElementById('color-picker').addEventListener('input', (e) => {
    gridUniforms.uColor.value.set(e.target.value);
});

document.getElementById('toggle-grid').addEventListener('change', (e) => {
    gridMesh.visible = e.target.checked;
    document.getElementById('label-toggle').textContent =
        e.target.checked ? '表示' : '非表示';
});
