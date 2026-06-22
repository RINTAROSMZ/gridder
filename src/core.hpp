#pragma once

/**
 * core.hpp  — C++ / Emscripten ラッパー層の型定義と公開 API 宣言
 *
 * アーキテクチャ上の役割:
 *   JavaScript (Three.js)
 *     └─ Wasm (core.wasm)  ← emcc -s WITH_PYTHON=1 でビルド
 *          └─ C++ wrapper  (このファイル / wrapper.cpp)
 *               └─ CPython  (internal/grid_core.py が実態)
 *
 * 時間的連続性の担保戦略:
 *   フィルタ状態（前フレームの推定値・誤差共分散）は
 *   Python 側の GridCore インスタンスが保持する。
 *   C++ は PyObject* pyCore_ をシングルトンで持ち続けることで
 *   フレームを跨いで状態が破棄されないことを保証する。
 */

#include <cstddef>
#include <cstdint>

/* ─────────────────────────────────────────────────────────────────────
   §1  共有 POD 型  — Python の numpy 配列と 1:1 対応するレイアウト
   ───────────────────────────────────────────────────────────────────── */

/**
 * Hessian 法線形式の平面: n⃗·x⃗ = d
 *   nx,ny,nz : 単位法線ベクトル  (WebXR の XRPlane.normal 相当)
 *   d        : 原点からの符号付き距離 [m]
 */
struct Plane {
    float nx, ny, nz;
    float d;
};

/**
 * グリッド描画パラメータ (JS の UI 状態から毎フレーム渡す)
 *   spacing   : グリッド間隔 [m]  ← スライダー値
 *   extentX/Z : グリッドの半幅 [m]
 *   colorRGBA : 0xRRGGBBAA  ← カラーピッカー値
 */
struct GridConfig {
    float    spacing;
    float    extentX;
    float    extentZ;
    uint32_t colorRGBA;
};

/* ─────────────────────────────────────────────────────────────────────
   §2  C リンケージ API  — Emscripten が JS へエクスポートする関数群
   ─────────────────────────────────────────────────────────────────────
   JavaScript 側呼び出し例:
     const mod = await createGridModule();
     mod._grid_init(0.001, 0.05);
     mod._grid_update_plane(nx, ny, nz, d);
     const n = mod._grid_build_vertices(
         0.6, 2.0, 2.0, outPtr, maxFloats);
   ───────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * 起動時に一度だけ呼ぶ。Python インタープリタを初期化し
 * GridCore インスタンスを生成する。
 *
 *   processNoise      : カルマンフィルタ Q — 床面の動きやすさ (推奨: 1e-4)
 *   measurementNoise  : カルマンフィルタ R — センサーノイズ  (推奨: 5e-2)
 *
 * 戻り値: 0=成功, 負=エラー
 */
int grid_init(float processNoise, float measurementNoise);

/**
 * 毎フレーム、WebXR から得た平面検出結果を渡す。
 * Python 側フィルタが状態を更新し、平滑化済み Plane を内部保持する。
 *
 * 戻り値: 0=成功, -1=縮退入力 (法線がゼロベクトル等), -2=未初期化
 */
int grid_update_plane(float nx, float ny, float nz, float d);

/**
 * 平滑化済み平面上にグリッドの頂点バッファ (line-list) を生成する。
 *
 *   spacing    : グリッド間隔 [m]
 *   extentX/Z  : グリッドの半幅 [m]
 *   out        : 書き込み先 float 配列 (呼び出し元が確保)
 *   max_floats : out のキャパシティ (floats 単位)
 *
 * 戻り値: 書き込んだ float 数 (XYZ×2頂点/線), 負=エラー
 *
 * 頂点フォーマット: [x0,y0,z0, x1,y1,z1,  x2,y2,z2, x3,y3,z3, ...]
 * Three.js の BufferAttribute(Float32Array, 3) に直接渡せる。
 */
int grid_build_vertices(float spacing, float extentX, float extentZ,
                        float* out, int max_floats);

/**
 * Lerp 重み α を動的変更する (スライダーで実行時に調整可能)。
 *   alpha=0.0 : 完全固定 (前フレームのみ)
 *   alpha=1.0 : 平滑化なし (検出値をそのまま使用)
 */
void grid_set_smoothing(float alpha);

/**
 * シャットダウン時に一度だけ呼ぶ。Python インタープリタを終了し
 * 内部バッファを解放する。
 */
void grid_destroy(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ─────────────────────────────────────────────────────────────────────
   §3  C++ 内部クラス  — JS には直接露出しない
   ───────────────────────────────────────────────────────────────────── */
#ifdef __cplusplus

#include <Python.h>

/**
 * GridWrapper  — Python GridCore オブジェクトへの唯一の C++ 窓口。
 *
 * シングルトンにすることで:
 *   - pyCore_ が Wasm ライフサイクル全体で生き続ける
 *     (= Python 側フィルタ状態が毎フレーム維持される)
 *   - 複数スレッドからの多重初期化を防ぐ
 */
class GridWrapper {
public:
    static GridWrapper& instance() noexcept;

    /* noncopyable */
    GridWrapper(const GridWrapper&)            = delete;
    GridWrapper& operator=(const GridWrapper&) = delete;

    /** Python インタープリタと GridCore を起動する */
    bool init(float processNoise, float measurementNoise);

    /**
     * 平面計測値をフィルタに通す。
     * Python の GridCore.update_plane(nx, ny, nz, d) を呼び出す。
     * 戻り値: false = 縮退入力 or 未初期化
     */
    bool updatePlane(const Plane& measured);

    /**
     * 頂点バッファを構築する。
     * Python の GridCore.build_vertices(cfg) を呼び出し、
     * 返却された numpy 配列を out に memcpy する。
     * 戻り値: 書き込み float 数, 負=エラー
     */
    int buildVertices(const GridConfig& cfg, float* out, int maxFloats);

    /** Lerp α の動的変更 */
    void setSmoothing(float alpha);

    /** Python インタープリタを終了し pyCore_ / pyModule_ を解放する */
    void destroy();

    bool isReady() const noexcept { return ready_; }

private:
    GridWrapper() = default;

    /* ── フレーム間で保持する状態 ────────────────────────────────────
     * pyCore_  : Python の GridCore インスタンス (Borrowed ref 禁止)
     *            Py_INCREF 済みの Owned ref として管理する。
     *            このポインタが生きている限り Python 側の
     *            フィルタ状態 (estimate, error_cov) は消えない。
     * pyModule_: internal.grid_core モジュール (同上)
     * ─────────────────────────────────────────────────────────────── */
    PyObject* pyModule_{ nullptr };
    PyObject* pyCore_  { nullptr };
    bool      ready_   { false   };
};

#endif /* __cplusplus */
