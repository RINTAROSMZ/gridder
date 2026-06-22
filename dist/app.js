/**
 * app.js — Three.js + WebXR + Wasm グリッド描画
 *
 * 毎フレームの呼び出し順:
 *   XRFrame.detectedPlanes
 *     → _grid_update_plane(nx, ny, nz, d)   // Wasm: Kalman フィルタ更新
 *     → _grid_build_vertices(...)            // Wasm: 頂点バッファ生成
 *     → Float32Array view (ゼロコピー)
 *     → BufferGeometry.attributes.position  // Three.js へコピー(1回)
 *     → renderer.render(scene, camera)
 */

/* ── 定数 ─────────────────────────────────────────────────────────────── */

const MAX_FLOATS      = 12_000;   // 最大 4000 頂点 = 2000 線分
const INIT_Q          = 1e-4;     // Kalman プロセスノイズ (床面の動きやすさ)
const INIT_R          = 5e-2;     // Kalman 計測ノイズ (WebXR 平面検出精度)
const DEFAULT_SPACING = 0.60;     // 初期グリッド間隔 [m] — 平均歩幅相当
const DEFAULT_EXTENT  = 2.50;     // グリッド半幅 [m]

/* ── Wasm 状態 ────────────────────────────────────────────────────────── */

let wasm   = null;   // Emscripten Module（grid_* 関数群を持つ）
let outPtr = 0;      // Wasm 線形メモリ上の出力バッファ (byte offset)

async function loadWasm() {
    if (window.__wasmLoadFailed || typeof createGridModule === 'undefined') {
        setStatus('Wasm 未ビルド — フィルタなしで動作');
        return;
    }
    try {
        wasm   = await createGridModule();
        outPtr = wasm._malloc(MAX_FLOATS * 4);   // float = 4 bytes
        const rc = wasm._grid_init(INIT_Q, INIT_R);
        if (rc !== 0) throw new Error(`grid_init → ${rc}`);
        setStatus('Wasm OK');
    } catch (e) {
        console.warn('[Wasm]', e);
        wasm = null;
        setStatus('Wasm エラー — フィルタなしで動作');
    }
}

/* ── Three.js セットアップ ────────────────────────────────────────────── */

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.getElementById('canvas-container').appendChild(renderer.domElement);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
    70, window.innerWidth / window.innerHeight, 0.01, 20);

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ── グリッドジオメトリ（フレーム間で使い回す）──────────────────────── */

const gridPositions = new Float32Array(MAX_FLOATS);
const posAttr       = new THREE.BufferAttribute(gridPositions, 3);
posAttr.setUsage(THREE.DynamicDrawUsage);   // GPU に「頻繁に更新する」と伝える

const gridGeo = new THREE.BufferGeometry();
gridGeo.setAttribute('position', posAttr);
gridGeo.setDrawRange(0, 0);   // 平面未検出時は描画しない

const gridMat  = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 1 });
const gridMesh = new THREE.LineSegments(gridGeo, gridMat);
scene.add(gridMesh);

/* ── WebXR セッション管理 ─────────────────────────────────────────────── */

let xrSession    = null;
let xrRefSpace   = null;
let hitTestSource = null;   // plane-detection 非対応時のフォールバック

async function startAR() {
    if (!navigator.xr) {
        alert('WebXR 非対応: iOS 16+ の Safari をお使いください。');
        return;
    }
    const ok = await navigator.xr.isSessionSupported('immersive-ar');
    if (!ok) {
        alert('immersive-ar 非対応端末です。');
        return;
    }

    xrSession = await navigator.xr.requestSession('immersive-ar', {
        requiredFeatures: ['local-floor'],
        // plane-detection: iOS 16+ ARKit 連携で有効
        // hit-test       : フォールバック用 (将来拡張)
        optionalFeatures: ['plane-detection', 'hit-test'],
    });

    renderer.xr.setReferenceSpaceType('local-floor');
    await renderer.xr.setSession(xrSession);
    xrRefSpace = await xrSession.requestReferenceSpace('local-floor');

    xrSession.addEventListener('end', onSessionEnd);

    // hit-test ソースを作成 (plane-detection が空の場合のフォールバック)
    // viewer 空間から前方にレイを飛ばして床面との交点を検出する
    if ('requestHitTestSource' in xrSession) {
        try {
            const viewerSpace = await xrSession.requestReferenceSpace('viewer');
            hitTestSource = await xrSession.requestHitTestSource({ space: viewerSpace });
        } catch (e) {
            console.warn('[WebXR] hit-test source 作成失敗:', e);
        }
    }

    document.getElementById('btn-start').hidden = true;
    document.getElementById('ui').classList.add('visible');

    renderer.setAnimationLoop(onFrame);
    setStatus('平面を探しています…');
}

function onSessionEnd() {
    if (hitTestSource) { hitTestSource.cancel(); hitTestSource = null; }
    xrSession  = null;
    xrRefSpace = null;
    planeFound = false;
    gridGeo.setDrawRange(0, 0);
    renderer.setAnimationLoop(null);
    document.getElementById('btn-start').hidden = false;
    document.getElementById('ui').classList.remove('visible');
    setStatus('セッション終了');
}

/* ── ユーティリティ ───────────────────────────────────────────────────── */

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

/**
 * XRPlane の planeSpace を基準としたポーズの向きクォータニオン q から
 * 世界座標系での平面法線 (Y 軸を回転したもの) を計算する。
 *
 * 公式: rotate((0,1,0), q) を展開すると
 *   nx = 2(qx·qy − qw·qz)
 *   ny = 1 − 2(qx² + qz²)
 *   nz = 2(qy·qz + qw·qx)
 */
function planeNormal(q) {
    return {
        x:  2 * (q.x * q.y - q.w * q.z),
        y:  1 - 2 * (q.x * q.x + q.z * q.z),
        z:  2 * (q.y * q.z + q.w * q.x),
    };
}

/* ── フレームループ ───────────────────────────────────────────────────── */

let planeFound = false;

function onFrame(_timestamp, frame) {
    if (!frame || !xrRefSpace) {
        renderer.render(scene, camera);
        return;
    }

    /* ①  平面検出 → Wasm Kalman フィルタ更新 */
    const planes = frame.detectedPlanes;   // XRPlaneSet | undefined

    if (planes && planes.size > 0) {
        for (const plane of planes) {
            const pose = frame.getPose(plane.planeSpace, xrRefSpace);
            if (!pose) continue;

            const n = planeNormal(pose.transform.orientation);

            // |ny| ≥ 0.7 → ほぼ水平 → 床面として採用
            if (Math.abs(n.y) < 0.7) continue;

            const p = pose.transform.position;
            const d = n.x * p.x + n.y * p.y + n.z * p.z;

            if (wasm) {
                wasm._grid_update_plane(n.x, n.y, n.z, d);
            } else {
                // Wasm 未ロード時はフォールバック: 仮の水平床を設定
                _fallbackUpdatePlane(n.x, n.y, n.z, d);
            }

            if (!planeFound) {
                planeFound = true;
                setStatus('グリッド表示中');
            }
            break;   // 最初に見つかった床面のみ使用
        }
    }

    // plane-detection が空または未対応の場合: hit-test で床面を推定
    if ((!planes || planes.size === 0) && hitTestSource) {
        const hits = frame.getHitTestResults(hitTestSource);
        if (hits.length > 0) {
            const pose = hits[0].getPose(xrRefSpace);
            if (pose) {
                const n = planeNormal(pose.transform.orientation);
                if (Math.abs(n.y) >= 0.7) {   // ほぼ水平な面のみ採用
                    const p = pose.transform.position;
                    const d = n.x * p.x + n.y * p.y + n.z * p.z;
                    wasm ? wasm._grid_update_plane(n.x, n.y, n.z, d)
                         : _fallbackUpdatePlane(n.x, n.y, n.z, d);
                    if (!planeFound) { planeFound = true; setStatus('グリッド表示中'); }
                }
            }
        }
    }

    /* ②  頂点バッファ構築 → Three.js へ転送 */
    if (gridMesh.visible) {
        updateGridGeometry();
    }

    renderer.render(scene, camera);
}

/* ── グリッドジオメトリ更新 ───────────────────────────────────────────── */

function updateGridGeometry() {
    let nWritten = 0;

    if (wasm && outPtr) {
        /* Wasm 経由: Python Kalman → C++ → Wasm 線形メモリ */
        nWritten = wasm._grid_build_vertices(
            uiState.spacing, DEFAULT_EXTENT, DEFAULT_EXTENT,
            outPtr, MAX_FLOATS,
        );

        if (nWritten > 0) {
            // Wasm 線形メモリをゼロコピーで view し Three.js バッファへコピー
            // new Float32Array(buffer, byteOffset, length) は view なのでアロケートなし
            const view = new Float32Array(wasm.HEAPF32.buffer, outPtr, nWritten);
            gridPositions.set(view);           // Wasm → JS: 1回の memcpy
        }
    } else if (fallback.plane) {
        /* Wasm 未ビルド時: JS 側でグリッド頂点を直接生成 */
        nWritten = fallbackBuildVertices(uiState.spacing, DEFAULT_EXTENT, DEFAULT_EXTENT);
    }

    if (nWritten > 0) {
        posAttr.needsUpdate = true;
        gridGeo.setDrawRange(0, nWritten / 3);   // 3 floats = 1 頂点
    }
}

/* ── Wasm 未ビルド時のフォールバック (JS 純実装) ─────────────────────── */
// Three.js 連携を Wasm なしで確認するためのスタブ。
// Kalman フィルタは簡易 Lerp で代替する。

const fallback = {
    plane: null,           // { nx, ny, nz, d } 平滑化済み
    alpha: 0.15,           // Lerp 重み
};

function _fallbackUpdatePlane(nx, ny, nz, d) {
    if (!fallback.plane) {
        fallback.plane = { nx, ny, nz, d };
        return;
    }
    const a = fallback.alpha;
    fallback.plane = {
        nx: fallback.plane.nx + a * (nx - fallback.plane.nx),
        ny: fallback.plane.ny + a * (ny - fallback.plane.ny),
        nz: fallback.plane.nz + a * (nz - fallback.plane.nz),
        d:  fallback.plane.d  + a * (d  - fallback.plane.d),
    };
    // 法線を再正規化
    const len = Math.hypot(fallback.plane.nx, fallback.plane.ny, fallback.plane.nz);
    if (len > 1e-7) {
        fallback.plane.nx /= len;
        fallback.plane.ny /= len;
        fallback.plane.nz /= len;
    }
}

function fallbackBuildVertices(spacing, extentX, extentZ) {
    const { nx, ny, nz, d } = fallback.plane;
    const p0 = [nx * d, ny * d, nz * d];

    // 基底ベクトル (Y-up)
    let ux, uy, uz;
    if (Math.abs(ny) < 0.9) { ux = -nz; uy = 0; uz = nx; }
    else                     { ux =   0; uy = nz; uz = -ny; }
    const ul = Math.hypot(ux, uy, uz);
    ux /= ul; uy /= ul; uz /= ul;
    const vx = ny*uz - nz*uy, vy = nz*ux - nx*uz, vz = nx*uy - ny*ux;

    const stepsU = Math.max(1, Math.round(extentX / spacing));
    const stepsV = Math.max(1, Math.round(extentZ / spacing));
    let idx = 0;

    // u に平行な線
    for (let i = -stepsV; i <= stepsV; i++) {
        const t = i * spacing;
        gridPositions[idx++] = p0[0] + t*vx - extentX*ux;
        gridPositions[idx++] = p0[1] + t*vy - extentX*uy;
        gridPositions[idx++] = p0[2] + t*vz - extentX*uz;
        gridPositions[idx++] = p0[0] + t*vx + extentX*ux;
        gridPositions[idx++] = p0[1] + t*vy + extentX*uy;
        gridPositions[idx++] = p0[2] + t*vz + extentX*uz;
    }
    // v に平行な線
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

/* ── UI 状態 & イベント ───────────────────────────────────────────────── */

const uiState = { spacing: DEFAULT_SPACING };

document.getElementById('btn-start').addEventListener('click', async () => {
    await loadWasm();
    await startAR();
});

document.getElementById('slider-spacing').addEventListener('input', (e) => {
    uiState.spacing = parseFloat(e.target.value);
    document.getElementById('label-spacing').textContent =
        `${uiState.spacing.toFixed(2)} m`;
});

document.getElementById('slider-smoothing').addEventListener('input', (e) => {
    const alpha = parseFloat(e.target.value);
    document.getElementById('label-smoothing').textContent = alpha.toFixed(2);
    fallback.alpha = alpha;
    if (wasm) wasm._grid_set_smoothing(alpha);
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

/* ── 起動時チェック ────────────────────────────────────────────────────── */

// HTTP 経由のアクセスは WebXR AR が動かないため警告を表示
// localhost は例外 (Mac での動作確認用)
if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
    document.getElementById('https-warn').style.display = 'block';
}
