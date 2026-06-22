"""
grid_core.py  — 空間グリッド生成のメインロジック

責務:
  - _Kalman1D     : 単次元スカラーカルマンフィルタ（フレーム間状態保持）
  - _PlaneKalman  : Plane の4成分 (nx,ny,nz,d) へ _Kalman1D を適用
  - GridCore      : C++ / JS が直接インスタンス化するエントリポイント

時間的連続性:
  GridCore._kalman はインスタンス変数として生き続けるため、
  C++ 側が pyCore_ を保持している限りフィルタ状態（推定値・誤差共分散）
  はフレームを跨いで維持される。
"""

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# §1  カルマンフィルタ
# ─────────────────────────────────────────────────────────────────────────────

class _Kalman1D:
    """
    定常 1D カルマンフィルタ。

    状態方程式:  x[k] = x[k-1] + w,  w ~ N(0, Q)
    観測方程式:  z[k] = x[k] + v,    v ~ N(0, R)

    Q (process_noise)     : 床面がフレーム間でどれだけ動くか
                            小さいほど追従が遅く安定、大きいほど速く追従
    R (measurement_noise) : センサー（WebXR 平面検出）のノイズ大きさ
                            大きいほど計測値を信用しない
    """

    __slots__ = ("_x", "_p", "_q", "_r", "_ready")

    def __init__(self, q: float, r: float) -> None:
        self._q     = float(q)
        self._r     = float(r)
        self._x     = 0.0   # 推定値
        self._p     = 1.0   # 誤差共分散（初期値は大きめ = 不確かさが高い）
        self._ready = False

    def update(self, z: float) -> float:
        # 初回は計測値をそのまま採用してフィルタを「着地」させる
        if not self._ready:
            self._x     = z
            self._ready = True
            return self._x

        # ── 予測ステップ ──────────────────────────────────────────────
        p_pred = self._p + self._q

        # ── 更新ステップ ──────────────────────────────────────────────
        k      = p_pred / (p_pred + self._r)   # カルマンゲイン
        self._x = self._x + k * (z - self._x)
        self._p = (1.0 - k) * p_pred

        return self._x


class _PlaneKalman:
    """Plane の nx, ny, nz, d それぞれに独立した _Kalman1D を適用する。"""

    def __init__(self, q: float, r: float) -> None:
        # 4成分それぞれに独立したフィルタを割り当て
        self._fn, self._fy, self._fz, self._fd = (
            _Kalman1D(q, r), _Kalman1D(q, r),
            _Kalman1D(q, r), _Kalman1D(q, r),
        )

    def update(self, nx: float, ny: float,
               nz: float, d: float) -> tuple:
        snx = self._fn.update(nx)
        sny = self._fy.update(ny)
        snz = self._fz.update(nz)
        sd  = self._fd.update(d)

        # フィルタ後の法線を正規化（各成分を独立に平滑化すると長さが崩れる）
        length = math.sqrt(snx*snx + sny*sny + snz*snz)
        if length < 1e-7:
            return (nx, ny, nz, d)   # 縮退: 入力値をそのまま返す
        inv = 1.0 / length
        return (snx * inv, sny * inv, snz * inv, sd)


# ─────────────────────────────────────────────────────────────────────────────
# §2  グリッド頂点生成
# ─────────────────────────────────────────────────────────────────────────────

def _orthonormal_basis(
    nx: float, ny: float, nz: float
) -> tuple[tuple, tuple]:
    """
    法線 n⃗ に直交する正規直交基底 (u⃗, v⃗) を返す。
    WebXR 座標系 (Y-up, 右手系) を前提とする。

    - |n⃗ × Y⃗| が十分大きい → u⃗ = n⃗ × Y⃗ を正規化
    - n⃗ ≈ ±Y⃗ (垂直壁の真上を向く稀ケース) → u⃗ = n⃗ × X⃗ を正規化
    """
    if abs(ny) < 0.9:
        # cross(n, Y) = (ny*0 - nz*1, nz*0 - nx*0, nx*1 - ny*0)
        #             = (-nz, 0, nx)
        ux, uy, uz = -nz, 0.0, nx
    else:
        # cross(n, X) = (ny*0 - nz*0, nz*1 - nx*0, nx*0 - ny*1)
        #             = (0, nz, -ny)
        ux, uy, uz = 0.0, nz, -ny

    u_len = math.sqrt(ux*ux + uy*uy + uz*uz)
    ux, uy, uz = ux / u_len, uy / u_len, uz / u_len

    # v⃗ = n⃗ × u⃗  （右手系; 単位ベクトル同士の外積なので自動的に正規化済）
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux

    return (ux, uy, uz), (vx, vy, vz)


def _build_grid_vertices(
    nx: float, ny: float, nz: float, d: float,
    spacing: float,
    extent_x: float, extent_z: float,
) -> np.ndarray:
    """
    平滑化済み平面上にグリッド（line-list）の頂点バッファを生成する。

    平面上の基準点: p0 = d * n⃗
    格子方向:
      u⃗ 方向に -extent_x ～ +extent_x, spacing 間隔で線を引く
      v⃗ 方向に -extent_z ～ +extent_z, spacing 間隔で線を引く

    戻り値: float32 の 1D 配列 [x0,y0,z0, x1,y1,z1, ...] (C-contiguous)
    Three.js の new THREE.BufferAttribute(arr, 3) に直接渡せる。
    """
    p0 = np.array([nx * d, ny * d, nz * d], dtype=np.float32)
    (ux, uy, uz), (vx, vy, vz) = _orthonormal_basis(nx, ny, nz)
    u = np.array([ux, uy, uz], dtype=np.float32)
    v = np.array([vx, vy, vz], dtype=np.float32)

    # u 方向のステップ数 (u に平行な線は v 方向に並ぶ → steps_v 本)
    steps_v = max(1, int(round(extent_z / spacing)))
    # v 方向のステップ数 (v に平行な線は u 方向に並ぶ → steps_u 本)
    steps_u = max(1, int(round(extent_x / spacing)))

    # ── u に平行な線群 (v 方向オフセット) ────────────────────────────
    # ts_v: [-steps_v*spacing, ..., 0, ..., +steps_v*spacing]
    ts_v     = np.arange(-steps_v, steps_v + 1,
                         dtype=np.float32) * spacing   # shape (2N_v+1,)
    centers  = p0 + ts_v[:, None] * v[None, :]         # shape (2N_v+1, 3)
    u_par    = np.empty((len(ts_v) * 2, 3), dtype=np.float32)
    u_par[0::2] = centers - extent_x * u   # 始点
    u_par[1::2] = centers + extent_x * u   # 終点

    # ── v に平行な線群 (u 方向オフセット) ────────────────────────────
    ts_u     = np.arange(-steps_u, steps_u + 1,
                         dtype=np.float32) * spacing   # shape (2N_u+1,)
    centers  = p0 + ts_u[:, None] * u[None, :]         # shape (2N_u+1, 3)
    v_par    = np.empty((len(ts_u) * 2, 3), dtype=np.float32)
    v_par[0::2] = centers - extent_z * v   # 始点
    v_par[1::2] = centers + extent_z * v   # 終点

    # 結合してフラット化: shape → (total_verts * 3,)
    return np.concatenate([u_par, v_par], axis=0).ravel()


# ─────────────────────────────────────────────────────────────────────────────
# §3  GridCore  — C++ wrapper.cpp が直接呼び出すエントリポイント
# ─────────────────────────────────────────────────────────────────────────────

class GridCore:
    """
    フレーム間の時間的連続性を担保するメインクラス。

    self._kalman が保持する _Kalman1D の推定値・誤差共分散が
    「前フレームのキャッシュ」として機能する。
    C++ 側が pyCore_ (Owned ref) を解放しない限りこの状態は消えない。
    """

    def __init__(self, process_noise: float,
                 measurement_noise: float) -> None:
        self._kalman          = _PlaneKalman(process_noise, measurement_noise)
        self._smoothing_alpha = 0.15     # 将来の Lerp モード用; 現在は未使用
        self._plane: tuple | None = None # 最新の平滑化済み平面

    # ── C++ から毎フレーム呼び出される ───────────────────────────────

    def update_plane(self, nx: float, ny: float,
                     nz: float, d: float) -> None:
        """WebXR の平面検出結果をフィルタに通して内部状態を更新する。"""
        self._plane = self._kalman.update(nx, ny, nz, d)

    def build_vertices(self, spacing: float,
                       extent_x: float,
                       extent_z: float) -> np.ndarray:
        """
        平滑化済み平面上のグリッド頂点を float32 の 1D 配列で返す。
        未初期化時（update_plane 未呼出）は長さ 0 の配列を返す。
        """
        if self._plane is None:
            return np.empty(0, dtype=np.float32)

        nx, ny, nz, d = self._plane
        return _build_grid_vertices(nx, ny, nz, d,
                                    spacing, extent_x, extent_z)

    def set_smoothing(self, alpha: float) -> None:
        """Lerp α を [0, 1] にクランプして保存する（将来の Lerp モード用）。"""
        self._smoothing_alpha = max(0.0, min(1.0, float(alpha)))
