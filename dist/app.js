/**
 * app.js — World-space floor grid
 *
 * グリッドは THREE.Mesh として World Y=0 に固定される。
 * カメラが前進すると格子線が足元を後方へ流れる。
 *
 * Primary:  WebXR immersive-ar + local-floor (ARKit 6DOF)
 * Fallback: getUserMedia + DeviceOrientation (3DOF 回転のみ)
 *
 * シェーダーは fwidth / GL_OES_standard_derivatives を使わず
 * mod() のみで格子を描画 → iOS Safari/Chrome 両対応
 */

/* ════════════════════════════════════════════════════════
   §1  シェーダー — ワールド座標で格子を描画
   ════════════════════════════════════════════════════════ */

const GRID_VERT = `
varying vec3 vWorldPos;
void main() {
    // modelMatrix でローカル→ワールド変換し、XZ をフラグメントへ渡す
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

// fwidth を使わない互換シェーダー
// mod() で各セル内の正規化座標を求め、端 (格子線) かどうか step() で判定する
const GRID_FRAG = `
precision mediump float;

uniform float uSpacing;   // 格子間隔 [m]
uniform vec3  uColor;
uniform float uOpacity;

varying vec3 vWorldPos;

void main() {
    // vWorldPos.xz をセル幅で割った余り → [0, uSpacing) の繰り返し
    vec2 cell = mod(vWorldPos.xz, uSpacing);

    // セル幅の 5% を線幅とする
    float lw = uSpacing * 0.05;

    // セルの左端・右端にいれば 1.0、それ以外は 0.0
    float onX = step(cell.x, lw) + step(uSpacing - lw, cell.x);
    float onZ = step(cell.y, lw) + step(uSpacing - lw, cell.y);

    float alpha = clamp(onX + onZ, 0.0, 1.0) * uOpacity;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(uColor, alpha);
}
`;

/* ════════════════════════════════════════════════════════
   §2  定数
   ════════════════════════════════════════════════════════ */

const DEFAULT_SPACING = 0.60;   // [m]
const PLANE_SIZE      = 200;    // [m] カメラ直下をカバーする広さ
const FALLBACK_EYE_H  = 1.40;  // [m] DeviceOrientation 時の仮カメラ高さ

/* ════════════════════════════════════════════════════════
   §3  Three.js レンダラー
   注: xr.enabled はグローバルに true にしない。
       WebXR セッションを開始する直前にのみ true にする。
   ════════════════════════════════════════════════════════ */

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);   // 透明 (カメラ映像を透過)
// renderer.xr.enabled は WebXR 使用時のみ true に切り替える
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

/* ════════════════════════════════════════════════════════
   §4  グリッドメッシュ (World Y=0 固定)
   ════════════════════════════════════════════════════════ */

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
});

// PlaneGeometry はデフォルト XY 平面 → X 軸 -90° 回転で水平床 (XZ) にする
const gridMesh = new THREE.Mesh(
    new THREE.PlaneGeometry(PLANE_SIZE, PLANE_SIZE, 1, 1),
    gridMat
);
gridMesh.rotation.x = -Math.PI / 2;
gridMesh.position.y = 0;
scene.add(gridMesh);

/* ════════════════════════════════════════════════════════
   §5  DeviceOrientation フォールバック
   ════════════════════════════════════════════════════════ */

class Kalman1D {
    constructor(q, r) {
        this.q = q; this.r = r; this.x = 0; this.p = 1; this.ready = false;
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

// Three.js DeviceOrientationControls と同じ変換式 (portrait モード)
const _q0    = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5)); // -90° around X
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
    // _qOri * _q0 : デバイスの自然方向 (平置き=天井向き) を補正
    camera.quaternion.copy(_qOri).multiply(_q0);
    camera.position.set(0, FALLBACK_EYE_H, 0);
}

/* ════════════════════════════════════════════════════════
   §6  フレームループ
   ════════════════════════════════════════════════════════ */

let isXrMode   = false;
let xrRefSpace = null;

function onFrame(_ts, frame) {
    let camX = 0, camZ = 0;

    if (isXrMode && frame && xrRefSpace) {
        // WebXR: ARKit 6DOF ポーズからカメラ XZ を取得
        const pose = frame.getViewerPose(xrRefSpace);
        if (pose) {
            camX = pose.transform.position.x;
            camZ = pose.transform.position.z;
        }
        // camera 姿勢は Three.js WebXR が自動更新するので手動設定不要
    } else {
        // DeviceOrientation: カメラ回転を手動更新 (並進なし)
        applyOrientationToCamera();
    }

    // ── グリッドをカメラ直下に追従 ─────────────────────────────────
    // 平面メッシュが XZ 方向に動いても、シェーダーは vWorldPos.xz
    // (絶対ワールド座標) で格子を描くためグリッド線は世界に固定される。
    // → カメラが前進するとグリッドが足元を後方へ流れる
    gridMesh.position.set(camX, 0, camZ);

    renderer.render(scene, camera);
}

/* ════════════════════════════════════════════════════════
   §7  起動フロー
   ════════════════════════════════════════════════════════ */

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function tryWebXR() {
    if (!navigator.xr) return false;
    const ok = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
    if (!ok) return false;

    // WebXR を使う直前にのみ xr.enabled を true にする
    renderer.xr.enabled = true;

    const session = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hit-test'],
    });

    renderer.xr.setReferenceSpaceType('local-floor');
    await renderer.xr.setSession(session);
    xrRefSpace = await session.requestReferenceSpace('local-floor');

    session.addEventListener('end', () => {
        renderer.xr.enabled = false;
        xrRefSpace = null;
        isXrMode   = false;
        document.getElementById('btn-start').hidden = false;
        document.getElementById('ui').classList.remove('visible');
        setStatus('');
    });

    isXrMode = true;
    renderer.setAnimationLoop(onFrame);
    setStatus('ARKit トラッキング中');
    return true;
}

async function startFallback() {
    // カメラ映像を背景に流す
    const video = document.getElementById('cam-feed');
    try {
        video.srcObject = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }, audio: false,
        });
        video.style.display = 'block';
    } catch (_) {
        setStatus('カメラなし — グリッドのみ表示');
    }

    // iOS 13+: DeviceOrientationEvent の明示的な許可
    try {
        if (typeof DeviceOrientationEvent.requestPermission === 'function') {
            if (await DeviceOrientationEvent.requestPermission() !== 'granted') throw 0;
        }
        window.addEventListener('deviceorientation', (e) => {
            if (e.alpha !== null) devOri.alpha = kAlpha.update(e.alpha);
            if (e.beta  !== null) devOri.beta  = kBeta.update(e.beta);
            if (e.gamma !== null) devOri.gamma = kGamma.update(e.gamma);
        });
        setStatus('ジャイロ モード');
    } catch (_) {
        setStatus('固定視点');
    }

    // DeviceOrientation モード: xr.enabled は false のまま
    // requestAnimationFrame ベースのループを使う
    renderer.setAnimationLoop(onFrame);
}

document.getElementById('btn-start').addEventListener('click', async () => {
    document.getElementById('btn-start').hidden = true;
    document.getElementById('ui').classList.add('visible');
    setStatus('初期化中…');

    const gotXr = await tryWebXR();
    if (!gotXr) await startFallback();
});

/* ════════════════════════════════════════════════════════
   §8  UI イベント
   ════════════════════════════════════════════════════════ */

const uiState = { spacing: DEFAULT_SPACING };

document.getElementById('slider-spacing').addEventListener('input', (e) => {
    uiState.spacing = parseFloat(e.target.value);
    gridUniforms.uSpacing.value = uiState.spacing;
    document.getElementById('label-spacing').textContent =
        `${uiState.spacing.toFixed(2)} m`;
});

document.getElementById('slider-smoothing').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
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
