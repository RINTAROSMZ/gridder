"""grid_core.py のユニットテスト。"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "internal"))
from grid_core import (
    _Kalman1D,
    _PlaneKalman,
    _build_grid_vertices,
    _orthonormal_basis,
    GridCore,
)


# ── _Kalman1D ────────────────────────────────────────────────────────────────

class TestKalman1D:
    def test_first_update_equals_measurement(self):
        # 初回は計測値をそのまま返す (フィルタの「着地」)
        f = _Kalman1D(1e-4, 5e-2)
        assert f.update(3.14) == pytest.approx(3.14)

    def test_converges_to_constant_signal(self):
        f = _Kalman1D(1e-4, 5e-2)
        for _ in range(200):
            f.update(1.0)
        assert f.update(1.0) == pytest.approx(1.0, abs=1e-4)

    def test_smoothing_reduces_variance(self):
        # ガウスノイズ信号に対してフィルタ出力の分散が低減されること
        rng = np.random.default_rng(42)
        f = _Kalman1D(1e-4, 5e-2)
        meas = rng.normal(0.0, 0.1, 200).tolist()
        ests = [f.update(m) for m in meas]
        assert float(np.var(ests)) < float(np.var(meas))

    def test_higher_q_tracks_faster(self):
        # Q が大きいほど急変に素早く追従する
        slow = _Kalman1D(q=1e-6, r=5e-2)
        fast = _Kalman1D(q=1e-1, r=5e-2)
        for _ in range(50):
            slow.update(0.0)
            fast.update(0.0)
        # ステップ変化
        slow_est = slow.update(1.0)
        fast_est = fast.update(1.0)
        assert fast_est > slow_est


# ── _PlaneKalman ─────────────────────────────────────────────────────────────

class TestPlaneKalman:
    def test_normal_is_unit_after_filtering(self):
        pk = _PlaneKalman(1e-4, 5e-2)
        for _ in range(30):
            nx, ny, nz, _ = pk.update(0.0, 1.0, 0.0, -1.5)
        length = math.sqrt(nx**2 + ny**2 + nz**2)
        assert length == pytest.approx(1.0, abs=1e-5)

    def test_first_frame_matches_input(self):
        pk = _PlaneKalman(1e-4, 5e-2)
        nx, ny, nz, d = pk.update(0.0, 1.0, 0.0, -1.5)
        assert ny == pytest.approx(1.0, abs=1e-6)
        assert d  == pytest.approx(-1.5, abs=1e-6)

    def test_degenerate_normal_falls_back(self):
        # ほぼゼロ法線が来ても正規化でゼロ除算しないこと
        pk = _PlaneKalman(1e-4, 5e-2)
        pk.update(0.0, 1.0, 0.0, -1.5)           # 初期化
        result = pk.update(1e-10, 1e-10, 1e-10, 0.0)  # 縮退入力
        assert all(math.isfinite(v) for v in result)


# ── _orthonormal_basis ────────────────────────────────────────────────────────

class TestOrthonormalBasis:
    def _check_basis(self, nx, ny, nz):
        (ux, uy, uz), (vx, vy, vz) = _orthonormal_basis(nx, ny, nz)
        n = np.array([nx, ny, nz])
        u = np.array([ux, uy, uz])
        v = np.array([vx, vy, vz])
        assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-5)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
        assert np.dot(u, n) == pytest.approx(0.0, abs=1e-5)   # u ⊥ n
        assert np.dot(v, n) == pytest.approx(0.0, abs=1e-5)   # v ⊥ n
        assert np.dot(u, v) == pytest.approx(0.0, abs=1e-5)   # u ⊥ v

    def test_horizontal_floor(self):
        self._check_basis(0.0, 1.0, 0.0)          # ほぼ Y-up

    def test_tilted_floor(self):
        n = np.array([0.2, 0.9, 0.3])
        n /= np.linalg.norm(n)
        self._check_basis(*n)

    def test_near_vertical_wall(self):
        n = np.array([1.0, 0.01, 0.0])
        n /= np.linalg.norm(n)
        self._check_basis(*n)                      # abs(ny) < 0.9 のパス


# ── _build_grid_vertices ──────────────────────────────────────────────────────

class TestBuildGridVertices:
    def _floor(self, spacing=0.5, ext=1.0):
        return _build_grid_vertices(0.0, 1.0, 0.0, -1.5,
                                    spacing=spacing,
                                    extent_x=ext, extent_z=ext)

    def test_dtype_is_float32(self):
        assert self._floor().dtype == np.float32

    def test_output_is_1d(self):
        assert self._floor().ndim == 1

    def test_length_multiple_of_6(self):
        # line-list: 1線 = 2頂点 × 3成分 = 6 floats
        assert len(self._floor()) % 6 == 0

    def test_c_contiguous(self):
        # C++ 側 PyBUF_C_CONTIGUOUS が要求するレイアウト
        assert self._floor().flags["C_CONTIGUOUS"]

    def test_vertices_lie_on_plane(self):
        nx, ny, nz, d = 0.0, 1.0, 0.0, -1.5
        verts = _build_grid_vertices(nx, ny, nz, d, 0.3, 1.5, 1.5)
        pts = verts.reshape(-1, 3)
        dot = pts[:, 0] * nx + pts[:, 1] * ny + pts[:, 2] * nz
        np.testing.assert_allclose(dot, d, atol=1e-5)

    def test_line_count(self):
        # spacing=0.5, extent=1.0
        #   steps_v = round(1.0/0.5) = 2 → u平行線: 2*2+1 = 5本
        #   steps_u = round(1.0/0.5) = 2 → v平行線: 2*2+1 = 5本
        #   計 10線 × 2頂点 × 3成分 = 60 floats
        assert len(self._floor(spacing=0.5, ext=1.0)) == 60

    def test_smaller_spacing_more_lines(self):
        coarse = self._floor(spacing=1.0, ext=2.0)
        fine   = self._floor(spacing=0.5, ext=2.0)
        assert len(fine) > len(coarse)

    def test_empty_extent_returns_minimum(self):
        # extent < spacing でも最低1本は生成する (steps = max(1, ...))
        verts = _build_grid_vertices(0.0, 1.0, 0.0, 0.0, 1.0, 0.1, 0.1)
        assert len(verts) > 0


# ── GridCore (統合) ────────────────────────────────────────────────────────────

class TestGridCore:
    def test_build_before_update_returns_empty(self):
        core = GridCore(1e-4, 5e-2)
        assert len(core.build_vertices(0.6, 2.0, 2.0)) == 0

    def test_build_after_update_returns_data(self):
        core = GridCore(1e-4, 5e-2)
        core.update_plane(0.0, 1.0, 0.0, -1.5)
        assert len(core.build_vertices(0.6, 2.0, 2.0)) > 0

    def test_output_dtype_is_float32(self):
        core = GridCore(1e-4, 5e-2)
        core.update_plane(0.0, 1.0, 0.0, -1.5)
        assert core.build_vertices(0.6, 2.0, 2.0).dtype == np.float32

    def test_set_smoothing_clamps(self):
        core = GridCore(1e-4, 5e-2)
        core.set_smoothing(5.0)
        assert core._smoothing_alpha == pytest.approx(1.0)
        core.set_smoothing(-1.0)
        assert core._smoothing_alpha == pytest.approx(0.0)

    def test_temporal_continuity_via_kalman(self):
        # 50フレーム水平に収束させた後、急変を入れても出力は急変しない
        core = GridCore(process_noise=1e-4, measurement_noise=5e-2)
        for _ in range(50):
            core.update_plane(0.0, 1.0, 0.0, -1.5)

        core.update_plane(1.0, 0.0, 0.0, -1.5)   # 90° 急変
        nx_smoothed = core._plane[0]
        # Kalman により nx はまだ 1.0 に達していないはず
        assert nx_smoothed < 0.5

    def test_multiple_updates_accumulate_state(self):
        # update を重ねるごとにフィルタが収束していく (誤差共分散が減少)
        core = GridCore(1e-4, 5e-2)
        core.update_plane(0.0, 1.0, 0.0, -1.5)
        p_after_1 = core._kalman._fy._p

        for _ in range(99):
            core.update_plane(0.0, 1.0, 0.0, -1.5)
        p_after_100 = core._kalman._fy._p

        assert p_after_100 < p_after_1
