/**
 * wrapper.cpp  — GridWrapper 実装 + Emscripten エクスポート関数
 *
 * ビルド例:
 *   emcc wrapper.cpp -O3 -s WASM=1 -s WITH_PYTHON=1 \
 *        -s EXPORTED_FUNCTIONS='["_grid_init","_grid_update_plane", \
 *                                "_grid_build_vertices","_grid_set_smoothing", \
 *                                "_grid_destroy"]' \
 *        --preload-file internal \
 *        -o ../dist/core.js
 */

#include "core.hpp"

#include <cstring>   // memcpy
#include <cstdio>    // fprintf / stderr
#include <cmath>     // sqrtf

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#  define GRID_EXPORT EMSCRIPTEN_KEEPALIVE
// Emscripten 仮想 FS 上の Python ソース配置パス (--preload-file internal)
#  define GRID_PYTHON_PATH "/internal"
#else
#  define GRID_EXPORT
#  define GRID_PYTHON_PATH "internal"
#endif

/* ─────────────────────────────────────────────────────────────────────
   内部ユーティリティ
   ───────────────────────────────────────────────────────────────────── */

/** Python 例外をログ出力して false を返す。エラーハンドリングの定型を集約。 */
static bool pyFail(const char* ctx) {
    fprintf(stderr, "[GridWrapper] error in %s:\n", ctx);
    PyErr_Print();   // Python トレースバックを stderr へ
    PyErr_Clear();
    return false;
}

/** PyObject* の Null チェック。Null なら pyFail() を呼ぶ。 */
static bool pyCheck(PyObject* obj, const char* ctx) {
    return obj ? true : pyFail(ctx);
}

/* ─────────────────────────────────────────────────────────────────────
   GridWrapper — シングルトン実装
   ───────────────────────────────────────────────────────────────────── */

GridWrapper& GridWrapper::instance() noexcept {
    static GridWrapper inst;
    return inst;
}

/* ---------- init ---------------------------------------------------- */

bool GridWrapper::init(float processNoise, float measurementNoise) {
    if (ready_) return true;

    Py_Initialize();

    // sys.path に内部モジュールのパスを追加
    // Emscripten 仮想 FS では GRID_PYTHON_PATH = "/internal"
    PyObject* sys  = PyImport_ImportModule("sys");
    PyObject* path = sys ? PyObject_GetAttrString(sys, "path") : nullptr;
    if (!pyCheck(sys,  "import sys"))       { Py_XDECREF(sys); return false; }
    if (!pyCheck(path, "sys.path getter"))  { Py_DECREF(sys);  return false; }

    PyObject* pyPath = PyUnicode_FromString(GRID_PYTHON_PATH);
    PyList_Insert(path, 0, pyPath);   // 先頭挿入で他の path より優先
    Py_DECREF(pyPath);
    Py_DECREF(path);
    Py_DECREF(sys);

    // import grid_core
    pyModule_ = PyImport_ImportModule("grid_core");
    if (!pyCheck(pyModule_, "import grid_core")) return false;

    // GridCore(process_noise, measurement_noise) — Python 側コンストラクタ
    pyCore_ = PyObject_CallMethod(
        pyModule_, "GridCore", "ff",
        (double)processNoise, (double)measurementNoise);
    if (!pyCheck(pyCore_, "GridCore.__init__")) {
        Py_DECREF(pyModule_);
        pyModule_ = nullptr;
        return false;
    }
    // pyCore_ は Owned ref として管理 (Py_DECREF は destroy() で行う)

    ready_ = true;
    return true;
}

/* ---------- updatePlane -------------------------------------------- */

bool GridWrapper::updatePlane(const Plane& p) {
    if (!ready_) return false;

    // 縮退入力ガード: ほぼゼロ法線は無効な平面検出を意味する
    const float len2 = p.nx*p.nx + p.ny*p.ny + p.nz*p.nz;
    if (len2 < 1e-6f) return false;

    // Python: self.update_plane(nx, ny, nz, d)
    // 戻り値は None または bool; 参照カウントのために受け取って即解放
    PyObject* ret = PyObject_CallMethod(
        pyCore_, "update_plane", "ffff",
        (double)p.nx, (double)p.ny, (double)p.nz, (double)p.d);
    if (!pyCheck(ret, "update_plane")) return false;
    Py_DECREF(ret);
    return true;
}

/* ---------- buildVertices ------------------------------------------ */

int GridWrapper::buildVertices(const GridConfig& cfg,
                               float* out, int maxFloats) {
    if (!ready_) return -2;

    // Python: self.build_vertices(spacing, extent_x, extent_z)
    // 戻り値は numpy.ndarray (dtype=float32, shape=[N,3], C-contiguous)
    PyObject* arr = PyObject_CallMethod(
        pyCore_, "build_vertices", "fff",
        (double)cfg.spacing, (double)cfg.extentX, (double)cfg.extentZ);
    if (!pyCheck(arr, "build_vertices")) return -1;

    // バッファプロトコルで numpy 配列の生メモリポインタを取得
    // PyBUF_C_CONTIGUOUS: C 連続メモリを要求 (memcpy 前提)
    // PyBUF_FORMAT      : フォーマット文字列を取得 ('f' = float32 の確認用)
    Py_buffer view;
    if (PyObject_GetBuffer(arr, &view,
                           PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
        pyFail("PyObject_GetBuffer");
        Py_DECREF(arr);
        return -1;
    }

    // フォーマット検証: 'd'(float64)が来た場合は受け入れない
    if (view.format && view.format[0] != 'f') {
        fprintf(stderr, "[GridWrapper] build_vertices must return float32 "
                        "(got '%s')\n", view.format);
        PyBuffer_Release(&view);
        Py_DECREF(arr);
        return -1;
    }

    const int nFloats = static_cast<int>(view.len / sizeof(float));
    const int toCopy  = (nFloats < maxFloats) ? nFloats : maxFloats;
    std::memcpy(out, view.buf, static_cast<size_t>(toCopy) * sizeof(float));

    PyBuffer_Release(&view);
    Py_DECREF(arr);
    return toCopy;
}

/* ---------- setSmoothing ------------------------------------------- */

void GridWrapper::setSmoothing(float alpha) {
    if (!ready_) return;
    // Python: self.set_smoothing(alpha)
    PyObject* ret = PyObject_CallMethod(
        pyCore_, "set_smoothing", "f", (double)alpha);
    Py_XDECREF(ret);   // None でも解放; エラーは無視（非致命的）
}

/* ---------- destroy ------------------------------------------------- */

void GridWrapper::destroy() {
    if (!ready_) return;
    Py_XDECREF(pyCore_);    // GridCore インスタンス (フィルタ状態) を解放
    Py_XDECREF(pyModule_);
    pyCore_   = nullptr;
    pyModule_ = nullptr;
    ready_    = false;
    Py_Finalize();
}

/* ─────────────────────────────────────────────────────────────────────
   extern "C" API  — Emscripten が JS へエクスポートする薄いラッパー群
   ─────────────────────────────────────────────────────────────────────
   GRID_EXPORT (= EMSCRIPTEN_KEEPALIVE) を付けることで、
   最適化によるデッドコード削除を防ぐ。
   ───────────────────────────────────────────────────────────────────── */

extern "C" {

GRID_EXPORT
int grid_init(float processNoise, float measurementNoise) {
    return GridWrapper::instance().init(processNoise, measurementNoise) ? 0 : -1;
}

GRID_EXPORT
int grid_update_plane(float nx, float ny, float nz, float d) {
    if (!GridWrapper::instance().isReady()) return -2;
    return GridWrapper::instance().updatePlane({nx, ny, nz, d}) ? 0 : -1;
}

GRID_EXPORT
int grid_build_vertices(float spacing, float extentX, float extentZ,
                        float* out, int max_floats) {
    const GridConfig cfg{ spacing, extentX, extentZ, 0xFFFFFFFFu };
    return GridWrapper::instance().buildVertices(cfg, out, max_floats);
}

GRID_EXPORT
void grid_set_smoothing(float alpha) {
    GridWrapper::instance().setSmoothing(alpha);
}

GRID_EXPORT
void grid_destroy(void) {
    GridWrapper::instance().destroy();
}

} /* extern "C" */
