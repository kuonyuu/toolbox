"""SAI2 Toolbox 核心（不依賴 Qt，可單獨測試；插件也可以 from tbcore import ...）

插件規格
--------
放進 plugins/ 資料夾（或使用者插件資料夾，預設 data/plugins/，可在介面改路徑）。檔名／資料夾名以 _ 或 . 開頭的會被忽略。
兩種形式：
  1. 單一 .py 檔
  2. 資料夾插件：資料夾內有 plugin.py（或 __init__.py），可再放其他 .py、資料檔，
     並用相對匯入（from . import helper）；整個資料夾視為一個插件
一般資料夾（內沒有 plugin.py / __init__.py）只是分類：裡面的插件會依資料夾名稱歸類，可多層巢狀。
INFO["category"] 若有寫會優先；用 "光效/柔化" 這種寫法也能建立巢狀分類。
每個插件需要提供：

INFO    : dict。必填 id、name；選填 category、version、author、description、generator
          generator: True＝永遠生成圖片；"auto"＝有圖時當濾鏡、沒有圖時單獨生成（process 收到 None 時用 width/height 參數）
PRESETS : 選填。內建預設 [{"name": ..., "params": {...}}]；params 可只寫要改的項目
PARAMS  : list[dict]。每項需有 key、type、default；
          int / float 另需 min、max（可選 step）；choice 另需 choices；
          label 為介面顯示名稱
process : process(img, params, ctx) -> ndarray
            img    uint8 (H, W, 4) RGBA；generator 插件收到 None
            params dict，key 對應 PARAMS
            ctx    {"scale": 預覽縮放比例（1.0 = 原尺寸）, "preview": 是否為預覽}
            回傳   uint8 (H, W, 4)；一般濾鏡必須與輸入同尺寸
PARAMS 每一項可加 "show_if"：依另一個勾選框／下拉選單／數值參數決定這個參數要不要顯示，例如
          {"key": "color", ..., "show_if": {"param": "fill", "is": "solid"}}、{"param": "outline", "gt": 0}（寫法見 _norm_show_if）。
          隱藏的參數只是不顯示：值仍照常保存、傳給插件，不會因此重新計算。
INFO["skip_if_empty"] : 選填。圖片或文字參數的名稱。該參數是空的（沒選圖片／沒有文字）時，這一層等於不存在：
          不會被計算、也不會讓後面的層重算，畫布上也不能拖曳移動它。
INFO["drag"] : 選填。{"x": 參數名, "y": 參數名, "unit": "percent"（畫面寬高的 %，預設）或 "px"}。
          宣告後，選取這個插件時可用畫布上的「✥ 移動」在預覽上拖曳來改這兩個數值參數（int 或 float）。
output_size : 選填。output_size(params, w, h) -> (新寬, 新高)；會改變圖片尺寸的插件（例如畫布大小）才需要。
          w、h 是目前「完整尺寸」（不是預覽縮小後的）；回傳的也是完整尺寸。預覽時 process 的 ctx["scale"]
          用來把這個尺寸換算成預覽的大小（round(新寬 * scale)）。

支援的參數 type：int, float, choice, bool, gradient, curve, color（值為 [r, g, b]）
共用工具：from tbcore import blend, BLEND_MODES, gaussian_fast, apply_curve ...；色彩空間見 colorspace.py

generator 插件若要讓預覽縮小計算，請提供 width / height 參數；管線靠它們推算完整尺寸。
"""
import copy
import importlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np

import colorspace
import par

# 打包成 exe（PyInstaller）時，程式碼在暫存／內部資料夾裡，data/ 與 plugins/ 要放在 exe 旁邊
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent   # core/ 的上一層＝程式根目錄
USER_DIR = APP_DIR / "data"                       # 設定、預設、漸層庫、使用者插件都放在程式資料夾旁的 data/
LEGACY_DIR = Path.home() / ".sai2_toolbox"        # 舊版存放位置（會自動搬移）
SETTINGS_PATH = USER_DIR / "settings.json"

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
PARAM_TYPES = {"int", "float", "choice", "bool", "gradient", "curve", "color", "image", "text", "font"}
CURVE_KEYS = ("rgb", "r", "g", "b")


# ------------------------------------------------------------------ 漸層 / 曲線
def stops_to_lut(stops, reverse=False):
    """stops: [[pos, r, g, b, a], ...] -> (256, 4) float32，值域 0..255。"""
    st = sorted(stops, key=lambda s: s[0])
    pos = [s[0] for s in st]
    xs = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(xs, pos, [s[i] for s in st]) for i in (1, 2, 3, 4)], axis=1)
    lut = lut.astype(np.float32)
    return lut[::-1] if reverse else lut


# 曲線資料格式：{"space": "rgb", "curves": {"rgb": {"rgb": pts, "r": pts}, "hsl": {"l": pts}}}
# 只儲存「不是對角線」的通道；space 是目前作用的色彩空間（只套用該空間的曲線）。
# 舊格式 {"rgb": pts, "r": pts, "g": pts, "b": pts} 會自動轉換。
CURVE_SPACES = list(colorspace.SPACES)
IDENTITY_POINTS = [[0.0, 0.0], [1.0, 1.0]]


def default_curve(space="rgb"):
    return {"space": space, "curves": {}}


def is_identity_points(pts):
    return all(abs(float(x) - float(y)) < 1e-6 for x, y in pts)


def normalize_curve(v):
    """任何格式（舊／新）-> 新格式，並清理數值。"""
    if not isinstance(v, dict):
        raise ValueError("curve 必須是 dict")
    if "curves" in v or "space" in v:
        space, curves = v.get("space", "rgb"), v.get("curves", {})
    else:
        space, curves = "rgb", {"rgb": {k: v[k] for k in CURVE_KEYS if k in v}}
    if space not in colorspace.SPACES:
        raise ValueError(f"不支援的色彩空間：{space}")
    out = {}
    for sp, chans in (curves.items() if isinstance(curves, dict) else []):
        if sp not in colorspace.SPACES or not isinstance(chans, dict):
            continue
        keys = colorspace.channel_keys(sp)
        d = {}
        for k, pts in chans.items():
            if k not in keys:
                continue
            pts = [[min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)] for x, y in pts]
            if len(pts) < 2:
                raise ValueError("曲線至少需要兩個點")
            if not is_identity_points(pts):
                d[k] = sorted(pts)
        if d:
            out[sp] = d
    return {"space": space, "curves": out}


def curve_points(curve, space, ch):
    """取得某空間某通道的控制點（沒設定過就是對角線）。"""
    pts = curve.get("curves", {}).get(space, {}).get(ch)
    return [list(p) for p in pts] if pts else [list(p) for p in IDENTITY_POINTS]


def set_curve_points(curve, space, ch, pts):
    """就地修改 curve；設回對角線時會移除該通道。"""
    curves = curve.setdefault("curves", {})
    if is_identity_points(pts):
        curves.get(space, {}).pop(ch, None)
        if space in curves and not curves[space]:
            del curves[space]
    else:
        curves.setdefault(space, {})[ch] = [list(p) for p in pts]


def curve_lut(points):
    """控制點 [[x, y], ...]（0..1）-> (256,) float，值域 0..255。
    使用單調三次插值（Fritsch–Carlson），不會產生過衝。"""
    pts = {}
    for x, y in points:
        pts[round(min(1.0, max(0.0, float(x))), 4)] = min(1.0, max(0.0, float(y)))
    keys = sorted(pts)
    xs = np.array(keys, dtype=np.float64)
    ys = np.array([pts[k] for k in keys], dtype=np.float64)
    x = np.linspace(0, 1, 256)
    n = len(xs)
    if n == 1:
        return np.full(256, ys[0] * 255.0)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.empty(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    xc = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, n - 2)
    t = (xc - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    y = ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
         + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1])
    return np.clip(y, 0.0, 1.0) * 255.0


def curve_luts(curve):
    """曲線 -> 三張 uint8 LUT (R, G, B)，只看 RGB 空間的曲線。先套各通道，再套 RGB 總曲線。"""
    chans = normalize_curve(curve)["curves"].get("rgb", {})
    ident = IDENTITY_POINTS
    master = curve_lut(chans.get("rgb", ident))
    luts = []
    for k in ("r", "g", "b"):
        ch = curve_lut(chans.get(k, ident))
        out = np.interp(ch, np.arange(256), master)
        luts.append(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    return luts


def apply_curve(rgb, curve):
    """對 uint8 (H, W, 3) 套用曲線（作用中的色彩空間），回傳 uint8 (H, W, 3)。"""
    c = normalize_curve(curve)
    space, chans = c["space"], c["curves"].get(c["space"], {})
    if not chans:
        return rgb.copy()
    if space == "rgb":
        luts = curve_luts(c)
        return np.stack([luts[i][rgb[..., i]] for i in range(3)], axis=-1)
    arr = colorspace.to_space(space, rgb.astype(np.float32) / 255.0)
    grid = np.linspace(0, 1, 256)
    for i, k in enumerate(colorspace.channel_keys(space)):
        if k in chans:
            arr[..., i] = np.interp(arr[..., i], grid, curve_lut(chans[k]) / 255.0)
    out = colorspace.from_space(space, arr)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 卷積 / 模糊工具
def _fast_len(n):
    """不小於 n 的最小 5-smooth 數，讓 FFT 更快。"""
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def convolve_fft(arr, kernel):
    """arr (H, W, C) float32、kernel (kh, kw) 奇數尺寸 -> (H, W, C)。
    out[i] = sum_d arr[i + d] * kernel[d + center]；邊緣以最近像素延伸。"""
    if arr.ndim == 3 and arr.shape[2] >= 2 and par.threads() > 1:       # 通道彼此獨立：分給多條執行緒
        return par.map_channels(lambda a: _convolve_fft(a, kernel), arr)
    return _convolve_fft(arr, kernel)


def _convolve_fft(arr, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = arr.shape[:2]
    padded = np.pad(arr, ((ph, ph), (pw, pw), (0, 0)), mode="edge")
    fh, fw = _fast_len(padded.shape[0]), _fast_len(padded.shape[1])
    F = np.fft.rfft2(padded, s=(fh, fw), axes=(0, 1))
    K = np.fft.rfft2(kernel[::-1, ::-1].astype(np.float32), s=(fh, fw))
    out = np.fft.irfft2(F * K[..., None], s=(fh, fw), axes=(0, 1))
    return out[kh - 1:kh - 1 + h, kw - 1:kw - 1 + w].astype(np.float32)


def blur_with_kernel(img, kernel, blur_alpha=True, weight=None):
    """以任意核心模糊 RGBA（依透明度加權，邊緣不會變暗）。
    weight：選填 (H, W, 1) 權重圖，用於讓高光在模糊中佔更多比重（散景）。回傳 uint8。"""
    kernel = kernel.astype(np.float32) / max(float(kernel.sum()), 1e-8)
    a = img[..., 3:4].astype(np.float32) / 255.0
    rgb = img[..., :3].astype(np.float32)
    wa = a if weight is None else a * weight
    res = convolve_fft(np.concatenate([rgb * wa, wa, a], axis=2), kernel)
    den = res[..., 3:4]
    out_rgb = np.where(den > 1e-5, res[..., :3] / np.maximum(den, 1e-5), rgb)
    out_a = (res[..., 4:5] if blur_alpha else a) * 255.0
    return np.clip(np.concatenate([out_rgb, out_a], axis=2) + 0.5, 0, 255).astype(np.uint8)


def gaussian_kernel(sigma):
    sigma = max(float(sigma), 0.1)
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    return np.outer(k, k)


def motion_kernel(length, angle_deg, symmetric=True):
    """直線動態模糊核心。angle_deg：模糊方向（0°＝向右，逆時針）。"""
    L = max(float(length), 0.0)
    r = int(np.ceil(L / 2 if symmetric else L)) + 1
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a), -np.sin(a)
    n = max(2, int(L * 2) + 1)
    ts = np.linspace(-L / 2, L / 2, n) if symmetric else np.linspace(0, L, n)
    xs, ys = r + ts * dx, r + ts * dy
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for (ox, oy, wgt) in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + oy, x0 + ox), wgt)
    return k


def aperture_kernel(radius, sides=0, rotation_deg=0.0):
    """鏡頭光圈核心：sides < 3 為圓形，否則為正 n 邊形（radius＝外接圓半徑）。"""
    R = max(float(radius), 0.5)
    r = int(np.ceil(R)) + 1
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    if sides < 3:
        return np.clip(R - np.hypot(xs, ys) + 0.5, 0, 1)
    apothem = R * np.cos(np.pi / sides)
    d = np.full(xs.shape, -1e9, np.float32)
    for i in range(sides):
        t = np.deg2rad(rotation_deg) + 2 * np.pi * i / sides
        d = np.maximum(d, xs * np.cos(t) + ys * np.sin(t))
    return np.clip(apothem - d + 0.5, 0, 1)


def smoothstep(a, b, x):
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def resize_bilinear(arr, nh, nw):
    """(H, W, C) float -> (nh, nw, C)，雙線性（像素中心對齊）。"""
    h, w = arr.shape[:2]
    if (h, w) == (nh, nw):
        return arr

    def axis(n_old, n_new):
        pos = np.clip((np.arange(n_new) + 0.5) * n_old / n_new - 0.5, 0, n_old - 1)
        i0 = np.floor(pos).astype(np.int64)
        return i0, np.minimum(i0 + 1, n_old - 1), (pos - i0).astype(np.float32)

    y0, y1, fy = axis(h, nh)
    x0, x1, fx = axis(w, nw)
    a = arr[y0] * (1 - fy)[:, None, None] + arr[y1] * fy[:, None, None]
    return (a[:, x0] * (1 - fx)[None, :, None] + a[:, x1] * fx[None, :, None]).astype(np.float32)


def gaussian_fast(arr, sigma):
    """高斯模糊 (H, W, C) float32。sigma 大時先縮小再模糊再放大，速度不隨半徑增加。"""
    sigma = float(sigma)
    arr = np.asarray(arr, np.float32)
    if sigma < 0.3:
        return arr

    def kern(sg):
        k = gaussian_kernel(sg)
        return k / k.sum()

    f = int(sigma // 4)
    if f < 2:
        return convolve_fft(arr, kern(sigma))
    h, w, c = arr.shape
    nh, nw = -(-h // f), -(-w // f)
    p = np.pad(arr, ((0, nh * f - h), (0, nw * f - w), (0, 0)), mode="edge")
    small = p.reshape(nh, f, nw, f, c).mean(axis=(1, 3))
    small = convolve_fft(small, kern(sigma / f))
    return resize_bilinear(small, nh * f, nw * f)[:h, :w]


# ------------------------------------------------------------------ 混合模式
BLEND_MODES = [
    ("normal", "正常"), ("multiply", "色彩增值"), ("screen", "濾色"), ("overlay", "覆蓋"),
    ("soft_light", "柔光"), ("hard_light", "實光"), ("color_dodge", "加亮顏色"),
    ("color_burn", "加深顏色"), ("add", "線性加亮（相加）"), ("linear_burn", "線性加深"),
    ("darken", "變暗"), ("lighten", "變亮"), ("difference", "差異化"), ("exclusion", "排除"),
    ("linear_light", "線性光源"), ("hue", "色相"), ("saturation", "飽和度"),
    ("color", "顏色"), ("luminosity", "明度"),
]


def _hard_light(cb, cs):
    return np.where(cs <= 0.5, cb * 2 * cs, 1 - (1 - cb) * (1 - (2 * cs - 1)))


_HEAVY_BLEND = {"soft_light", "color_dodge", "color_burn", "overlay", "hard_light", "linear_light",
                "hue", "saturation", "color", "luminosity"}


def blend(base, top, mode):
    """混合模式（W3C 公式）。大張圖的重運算模式會分給多條執行緒。"""
    cb, cs = np.asarray(base, np.float32), np.asarray(top, np.float32)
    if mode in _HEAVY_BLEND and cb.ndim == 3 and cb.shape == cs.shape and par.threads() > 1:
        return par.map_rows(lambda x, y: _blend(x, y, mode), cb, cs)
    return _blend(cb, cs, mode)


def _blend(base, top, mode):
    """混合模式（W3C 公式）。base、top 為 float (..., 3)，值域 0..1；回傳混合後的顏色（不含透明度）。
    hue / saturation / color / luminosity 以 HSL 近似。"""
    cb, cs = np.asarray(base, np.float32), np.asarray(top, np.float32)
    if mode == "normal":
        out = cs
    elif mode == "multiply":
        out = cb * cs
    elif mode == "screen":
        out = 1 - (1 - cb) * (1 - cs)
    elif mode == "overlay":
        out = _hard_light(cs, cb)
    elif mode == "hard_light":
        out = _hard_light(cb, cs)
    elif mode == "soft_light":
        d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(cb))
        out = np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))
    elif mode == "color_dodge":
        out = np.where(cb <= 0, 0.0, np.where(cs >= 1, 1.0, np.minimum(1.0, cb / np.maximum(1 - cs, 1e-6))))
    elif mode == "color_burn":
        out = np.where(cb >= 1, 1.0, np.where(cs <= 0, 0.0, 1 - np.minimum(1.0, (1 - cb) / np.maximum(cs, 1e-6))))
    elif mode == "add":
        out = cb + cs
    elif mode == "linear_burn":
        out = cb + cs - 1
    elif mode == "darken":
        out = np.minimum(cb, cs)
    elif mode == "lighten":
        out = np.maximum(cb, cs)
    elif mode == "difference":
        out = np.abs(cb - cs)
    elif mode == "exclusion":
        out = cb + cs - 2 * cb * cs
    elif mode == "linear_light":
        out = cb + 2 * cs - 1
    elif mode in ("hue", "saturation", "color", "luminosity"):
        hb, ht = colorspace.rgb_to_hsl(cb), colorspace.rgb_to_hsl(cs)
        h, sat, l = hb[..., 0], hb[..., 1], hb[..., 2]
        if mode == "hue":
            h = ht[..., 0]
        elif mode == "saturation":
            sat = ht[..., 1]
        elif mode == "color":
            h, sat = ht[..., 0], ht[..., 1]
        else:
            l = ht[..., 2]
        out = colorspace.hsl_to_rgb(np.stack([h, sat, l], -1))
    else:
        raise ValueError(f"未知的混合模式：{mode}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ 漸層組（列表 / 匯入匯出）
GRADIENT_FORMAT = "sai2toolbox-gradients"


def gradient_lib_path():
    return USER_DIR / "gradients.json"


def parse_gradient_set(data):
    """JSON 資料 -> [{"name", "stops"}, ...]；不合法的項目會被略過，全部不合法則丟出 ValueError。"""
    items = data.get("gradients") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("不是有效的漸層組檔案")
    out = []
    for g in items:
        try:
            name = str(g["name"]).strip()
            if not name:
                continue
            out.append({"name": name, "stops": _clean_value({"type": "gradient"}, g["stops"])})
        except Exception:
            continue
    if not out:
        raise ValueError("檔案中沒有可用的漸層")
    return out


def read_gradient_set(path):
    return parse_gradient_set(json.loads(Path(path).read_text(encoding="utf-8")))


def write_gradient_set(path, items):
    data = {"format": GRADIENT_FORMAT, "version": 1, "gradients": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def merge_gradients(existing, incoming):
    """依名稱合併，同名者以 incoming 覆蓋。回傳 (合併後清單, 新增數, 覆蓋數)。"""
    by = {g["name"]: g for g in existing}
    added = replaced = 0
    for g in incoming:
        if g["name"] in by:
            replaced += 1
        else:
            added += 1
        by[g["name"]] = g
    return list(by.values()), added, replaced


def rename_gradient(lib, old, new):
    """回傳新的列表；new 已存在或 old 不存在時丟出 ValueError。"""
    new = new.strip()
    if not new:
        raise ValueError("名稱不能是空的")
    if new != old and any(g["name"] == new for g in lib):
        raise ValueError(f"「{new}」已存在")
    if not any(g["name"] == old for g in lib):
        raise ValueError(f"找不到「{old}」")
    return [dict(g, name=new) if g["name"] == old else g for g in lib]


def delete_gradient(lib, name):
    return [g for g in lib if g["name"] != name]


# 依 "name" 欄位操作清單的通用版本（漸層、堆疊預設共用）
rename_named = rename_gradient
delete_named = delete_gradient
merge_named = merge_gradients


STACK_FORMAT = "sai2toolbox-stacks"


def stacks_path():
    return USER_DIR / "stacks.json"


def clean_layer_name(name):
    """堆疊層的自訂名稱：去掉多餘空白、最長 60 字；不是字串或空白就視為沒有名稱（回傳空字串）。"""
    return " ".join(name.split())[:60] if isinstance(name, str) else ""


def _clean_layers(layers):
    out = []
    for e in layers if isinstance(layers, list) else []:
        if not (isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"]):
            continue
        params = e.get("params")
        preset = e.get("preset")
        item = {"id": e["id"], "enabled": bool(e.get("enabled", True)),
                "params": copy.deepcopy(params) if isinstance(params, dict) else {},
                "preset": preset if isinstance(preset, str) and preset else None}
        name = clean_layer_name(e.get("name"))
        if name:
            item["name"] = name
        out.append(item)
    return out


def parse_stack_set(data):
    """JSON 資料 -> [{"name", "layers"}, ...]；不合法項目略過，全部不合法則丟出 ValueError。"""
    items = data.get("stacks") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("不是有效的堆疊預設檔案")
    out = []
    for s in items:
        try:
            name = str(s["name"]).strip()
            layers = _clean_layers(s["layers"])
        except Exception:
            continue
        if name and layers:
            out.append({"name": name, "layers": layers})
    if not out:
        raise ValueError("檔案裡沒有可用的堆疊預設")
    return out


def write_stack_set(path, items):
    data = {"format": STACK_FORMAT, "version": 1, "stacks": items}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def read_stack_set(path):
    return parse_stack_set(json.loads(Path(path).read_text(encoding="utf-8")))


def load_stack_library():
    try:
        return read_stack_set(stacks_path())
    except Exception:
        return []


def save_stack_library(items):
    write_stack_set(stacks_path(), items)


def load_gradient_library():
    try:
        return read_gradient_set(gradient_lib_path())
    except Exception:
        return []


def save_gradient_library(items):
    write_gradient_set(gradient_lib_path(), items)


# ------------------------------------------------------------------ 預設資料庫（每個插件各自的參數組）
PRESET_FORMAT = "sai2toolbox-presets"


def presets_path():
    return USER_DIR / "presets.json"


def parse_preset_file(data):
    """JSON 資料 -> {plugin_id: [{"name", "params"}, ...]}；不合法項目略過，全部不合法則丟出 ValueError。"""
    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, dict):
        raise ValueError("不是有效的預設檔案")
    out = {}
    for pid, items in plugins.items():
        if not isinstance(items, list):
            continue
        by = {}
        for it in items:
            if isinstance(it, dict) and str(it.get("name", "")).strip() and isinstance(it.get("params"), dict):
                by[str(it["name"]).strip()] = {"name": str(it["name"]).strip(), "params": copy.deepcopy(it["params"])}
        if by:
            out[str(pid)] = list(by.values())
    if not out:
        raise ValueError("檔案中沒有可用的預設")
    return out


def read_preset_file(path):
    return parse_preset_file(json.loads(Path(path).read_text(encoding="utf-8")))


def write_preset_file(path, db):
    data = {"format": PRESET_FORMAT, "version": 1, "plugins": {k: v for k, v in db.items() if v}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_presets():
    try:
        return read_preset_file(presets_path())
    except Exception:
        return {}


def save_presets(db):
    write_preset_file(presets_path(), db)


def upsert_preset(db, pid, name, params):
    """新增或覆蓋；回傳 True 表示覆蓋了同名預設。"""
    items = db.setdefault(pid, [])
    entry = {"name": name, "params": copy.deepcopy(params)}
    for i, it in enumerate(items):
        if it["name"] == name:
            items[i] = entry
            return True
    items.append(entry)
    return False


def delete_preset(db, pid, name):
    db[pid] = [it for it in db.get(pid, []) if it["name"] != name]
    if not db[pid]:
        del db[pid]


def rename_preset(db, pid, old, new):
    items = db.get(pid, [])
    if any(it["name"] == new for it in items):
        raise ValueError(f"「{new}」已存在")
    for it in items:
        if it["name"] == old:
            it["name"] = new
            return
    raise ValueError(f"找不到「{old}」")


def merge_presets(existing, incoming):
    """依 (插件, 名稱) 合併，同名以 incoming 覆蓋。回傳 (合併後資料庫, 新增數, 覆蓋數)。"""
    merged = copy.deepcopy(existing)
    added = replaced = 0
    for pid, items in incoming.items():
        for it in items:
            if upsert_preset(merged, pid, it["name"], it["params"]):
                replaced += 1
            else:
                added += 1
    return merged, added, replaced


# ------------------------------------------------------------------ 參數驗證
def choice_info_for(p, value, label):
    """某個下拉選項的說明（用選項值或顯示文字對應）；沒有就回傳 None。"""
    ci = p.get("choice_info")
    if not isinstance(ci, dict):
        return None
    return ci.get(str(value)) or ci.get(str(label)) or None


def param_info_text(p):
    """參數的說明文字（ⓘ 的提示）：info，加上下拉選單每個有說明的選項（「選項：說明」一行一項）。沒有任何說明回傳 None。"""
    lines = []
    info = p.get("info")
    if isinstance(info, str) and info.strip():
        lines.append(info.strip())
    if p.get("type") == "choice":
        for ch in p.get("choices") or []:
            try:
                value, label = ch[0], ch[1]
            except (TypeError, IndexError):
                continue
            text = choice_info_for(p, value, label)
            if text:
                lines.append(f"{label}：{text}")
    return "\n".join(lines) or None


def normalize_choices(choices):
    out = []
    for c in choices:
        if isinstance(c, (list, tuple)):
            out.append((c[0], str(c[1])))
        else:
            out.append((c, str(c)))
    return out


def _validate_params(params):
    if not isinstance(params, (list, tuple)):
        raise ValueError("PARAMS 必須是 list")
    out, keys = [], set()
    for p in params:
        if not isinstance(p, dict):
            raise ValueError("PARAMS 的每一項必須是 dict")
        k, t = p.get("key"), p.get("type")
        if not isinstance(k, str) or not k:
            raise ValueError("參數缺少 key")
        if k in keys:
            raise ValueError(f"參數 key 重複：{k}")
        keys.add(k)
        if t not in PARAM_TYPES:
            raise ValueError(f"參數 {k} 的 type 不支援：{t}")
        if "default" not in p:
            raise ValueError(f"參數 {k} 缺少 default")
        q = dict(p)
        q.setdefault("label", k)
        if t in ("int", "float") and not ("min" in q and "max" in q):
            raise ValueError(f"參數 {k} 需要 min / max")
        if t == "choice":
            q["choices"] = normalize_choices(q.get("choices") or [])
            if not q["choices"]:
                raise ValueError(f"參數 {k} 需要 choices")
        if "info" in q and not isinstance(q["info"], str):          # 說明文字：label 旁的 ⓘ 滑過時顯示
            q.pop("info")
        ci = q.get("choice_info")
        if ci is not None:                                          # 每個下拉選項的說明 {選項值或顯示文字: 說明}
            if isinstance(ci, dict):
                q["choice_info"] = {str(a): str(b) for a, b in ci.items() if isinstance(b, str)}
            else:
                q.pop("choice_info")
        if t in ("image", "text", "font") and not isinstance(q["default"], str):
            raise ValueError(f"參數 {k}（{t}）的 default 必須是字串")
        if q.get("group") is not None and not isinstance(q["group"], str):
            raise ValueError(f"參數 {k} 的 group 必須是字串")
        out.append(q)
    by_key = {q["key"]: q for q in out}
    for q in out:                                                   # show_if：要等所有參數都解析完才能檢查「依據哪個參數」
        if "show_if" in q:
            conds = _norm_show_if(q["show_if"], q["key"], by_key)
            if conds:
                q["show_if"] = conds
            else:
                q.pop("show_if")                                    # 寫錯的條件直接忽略，不讓整個插件失敗
    return out


def _norm_show_if(raw, key, by_key):
    """PARAM["show_if"]：一個條件 dict 或一串條件（全部成立才顯示）。條件的寫法：
      {"param": 依據的參數, "in": [值…]}  /  "not_in": [值…]  /  "is": 值  /  "is_not": 值
      {"param": 數值參數, "gt"|"ge"|"lt"|"le": 數字}       （大於／大於等於／小於／小於等於）
      {"param": 勾選框參數}                               （勾選時顯示）
    依據的參數必須是 bool、choice、int 或 float，而且不能是自己。回傳整理後的 [{"param", "op", "value"}]。"""
    out = []
    for c in raw if isinstance(raw, (list, tuple)) else [raw]:
        if not isinstance(c, dict):
            continue
        ctl = c.get("param")
        q = by_key.get(ctl)
        if q is None or ctl == key or q["type"] not in ("bool", "choice", "int", "float"):
            continue
        if "in" in c or "not_in" in c:
            op = "in" if "in" in c else "not_in"
            v = c[op]
            out.append({"param": ctl, "op": op, "value": list(v) if isinstance(v, (list, tuple, set)) else [v]})
        elif "is" in c:
            out.append({"param": ctl, "op": "in", "value": [c["is"]]})
        elif "is_not" in c:
            out.append({"param": ctl, "op": "not_in", "value": [c["is_not"]]})
        elif any(k in c for k in ("gt", "ge", "lt", "le")):
            k = next(k for k in ("gt", "ge", "lt", "le") if k in c)
            if q["type"] in ("int", "float") and isinstance(c[k], (int, float)) and not isinstance(c[k], bool):
                out.append({"param": ctl, "op": k, "value": float(c[k])})
        elif q["type"] == "bool":
            out.append({"param": ctl, "op": "in", "value": [True]})
    return out


def eval_show_if(params, values):
    """依目前的參數值算出每個參數該不該顯示（跟 params 同順序的 list[bool]）。
    依據的參數自己被隱藏時，依賴它的參數也一併隱藏（可以連鎖）；循環依賴視為顯示。"""
    by_key = {p["key"]: p for p in params}
    memo = {}

    def vis(k):
        if k in memo:
            return memo[k]
        memo[k] = True                                              # 先當成顯示：碰到循環依賴時不會無窮遞迴
        ok = True
        for c in by_key[k].get("show_if") or []:
            ctl = c["param"]
            if ctl in by_key and not vis(ctl):
                ok = False
                break
            v, op, ref = values.get(ctl), c["op"], c["value"]
            if op == "in":
                good = v in ref
            elif op == "not_in":
                good = v not in ref
            else:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    good = True
                else:
                    good = {"gt": f > ref, "ge": f >= ref, "lt": f < ref, "le": f <= ref}[op]
            if not good:
                ok = False
                break
        memo[k] = ok
        return ok

    return [vis(p["key"]) for p in params]


def _clean_value(p, v):
    t = p["type"]
    if t == "int":
        return int(min(max(int(round(float(v))), p["min"]), p["max"]))
    if t == "float":
        return float(min(max(float(v), p["min"]), p["max"]))
    if t == "bool":
        return bool(v)
    if t == "choice":
        if v not in [c[0] for c in p["choices"]]:
            raise ValueError("choice 不在選項內")
        return v
    if t == "color":
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            raise ValueError("color 必須是 [r, g, b]")
        return [int(min(max(round(float(c)), 0), 255)) for c in v[:3]]
    if t == "gradient":
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            raise ValueError("gradient 至少需要兩個色標")
        out = []
        for s in v:
            if len(s) != 5:
                raise ValueError("色標格式應為 [pos, r, g, b, a]")
            out.append([min(max(float(s[0]), 0.0), 1.0)]
                       + [int(min(max(round(float(c)), 0), 255)) for c in s[1:]])
        return out
    if t == "curve":
        return normalize_curve(v)
    if t == "image":
        if not isinstance(v, str):
            raise ValueError("image 必須是圖片路徑字串")
        return v
    if t in ("text", "font"):
        if not isinstance(v, str):
            raise ValueError(f"{t} 必須是字串")
        return v[:5000]
    raise ValueError(f"未知 type：{t}")


def assign_groups(params, groups):
    """替 PARAMS 標上分類（參數面板會把同一分類收成可折疊的區塊）。
    groups：[("分類名稱", ["key1", "key2", ...]), ...]。回傳 params 本身，方便接在定義後面。"""
    by_key = {p["key"]: p for p in params}
    for name, keys in groups:
        for k in keys:
            if k in by_key:
                by_key[k]["group"] = name
    return params


# ------------------------------------------------------------------ 圖片參數（type "image"）
IMAGE_LOADER = None            # 由主程式設定：path -> uint8 (H, W, 4) 陣列（可用 Qt 解碼）
_IMG_CACHE = OrderedDict()     # path -> (mtime, size, array)
_IMG_LOCK = threading.Lock()
_IMG_CACHE_MAX = 6


def images_dir():
    return USER_DIR / "images"      # 從剪貼簿貼進圖片參數的圖會存在這裡


def load_image(path):
    """讀圖片參數的檔案 -> uint8 (H, W, 4)（唯讀）；路徑空白回傳 None；讀不到丟出 ValueError。"""
    if not path:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        raise ValueError(f"找不到圖片：{path}") from None
    key = str(p)
    with _IMG_LOCK:
        hit = _IMG_CACHE.get(key)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            _IMG_CACHE.move_to_end(key)
            return hit[2]
    if IMAGE_LOADER is None:
        raise ValueError("程式沒有提供圖片讀取功能")
    arr = IMAGE_LOADER(key)
    if not (isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8):
        raise ValueError(f"無法讀取圖片：{path}")
    arr.flags.writeable = False        # 快取共用，插件不能改動
    with _IMG_LOCK:
        _IMG_CACHE[key] = (st.st_mtime, st.st_size, arr)
        while len(_IMG_CACHE) > _IMG_CACHE_MAX:
            _IMG_CACHE.popitem(last=False)
    return arr


# ------------------------------------------------------------------ 文字繪製（給「加入文字」這類插件用）
TEXT_RENDERER = None           # 由主程式設定（用 Qt 畫字）；插件請呼叫 render_text()
_TXT_CACHE = OrderedDict()


# 直書時的標點處理
V_CORNER = set("。，、．：；､｡,.")      # 貼到字格的右上角
V_SMALL = set("ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶ")   # 小假名：往右上偏一點
V_ROTATE = set("ー—―－-–〜～…⋯（）()〔〕［］[]｛｝{}【】《》〈〉「」『』‹›«»＜＞<>＝=")   # 順時針轉 90°（橫排的括號／長音／刪節號 → 直書的樣子）
NO_LINE_START = set("。，、．：；！？）］｝〕】》〉」』”’）,.;:!?)]}%‰ゝゞーァィゥェォッャュョヮヵヶぁぃぅぇぉっゃゅょゎ々〻")


def vertical_kind(ch):
    """直書時這個字的處理方式：'corner'（貼右上角）／'rotate'（轉 90°）／'small'（小假名）／'upright'（直立置中）。"""
    if ch in V_CORNER:
        return "corner"
    if ch in V_ROTATE:
        return "rotate"
    if ch in V_SMALL:
        return "small"
    return "upright"


def _tokens(par):
    """把一段文字切成可換行的單位：連續的英數字（一個單字）、空白、其他每個字各自一個。"""
    out, word = [], ""
    for ch in par:
        if ch.isascii() and (ch.isalnum() or ch in "_'-"):
            word += ch
        else:
            if word:
                out.append(word)
                word = ""
            out.append(ch)
    if word:
        out.append(word)
    return out


def wrap_text(text, measure, limit):
    """依寬度（橫排）或高度（直書）自動分行。measure(s) 回傳字串 s 沿排列方向的長度；limit <= 0 表示不換行。
    原有的換行會保留；英文單字不會從中間切開（除非單字本身就比 limit 長）；
    句號、逗號、右括號等不會出現在行首（改掛在上一行行尾）。回傳行的 list。"""
    paras = text.split("\n")
    if limit is None or limit <= 0:
        return paras
    lines = []
    for par in paras:
        cur = ""
        for tok in _tokens(par):
            if not cur:
                if measure(tok) <= limit or len(tok) == 1:
                    cur = tok
                else:                                  # 單字比整行還長：只好在字中間切
                    for ch in tok:
                        if cur and measure(cur + ch) > limit:
                            lines.append(cur)
                            cur = ""
                        cur += ch
                continue
            if measure(cur + tok) <= limit:
                cur += tok
            elif tok[0] in NO_LINE_START and len(tok) == 1:
                cur += tok                              # 禁則：標點掛在行尾，不放行首
            elif tok.isspace():
                continue                                # 換行處的空白吃掉
            else:
                lines.append(cur.rstrip())
                cur = ""
                if measure(tok) <= limit or len(tok) == 1:
                    cur = tok
                else:
                    for ch in tok:
                        if cur and measure(cur + ch) > limit:
                            lines.append(cur)
                            cur = ""
                        cur += ch
        lines.append(cur.rstrip())
    return lines


def vertical_layout(lines, cell, pitch, align="left", total_h=None):
    """直書排版（純計算，不依賴 Qt）：每個字佔一格（高 cell），一行是一直欄，第一行在最右邊、往左排。
    align：left＝靠上、center＝置中、right＝靠下（對齊的是各欄的字數差；total_h 指定時以它為準，例如固定高度的框）。
    回傳 ([(行, 字索引, 該欄中心 x, 該格上緣 y), ...], 總寬, 總高)。"""
    n = len(lines)
    cell = max(float(cell), 1.0)
    longest = max((len(ln) for ln in lines), default=0)
    total_h = longest * cell if total_h is None else float(total_h)
    out = []
    for i, ln in enumerate(lines):
        extra = total_h - len(ln) * cell
        off = {"center": extra / 2.0, "right": extra}.get(align, 0.0)
        cx = (n - 1 - i) * pitch + pitch / 2.0
        for j in range(len(ln)):
            out.append((i, j, cx, off + j * cell))
    return out, n * pitch, total_h


def render_text(text, family="", px=48.0, bold=False, italic=False, spacing=0.0, line_spacing=120.0,
                align="left", outline=0.0, vertical=False, box_w=0.0, box_h=0.0, valign="top", clip=True):
    """把文字畫成遮罩，回傳 (fill, outline_mask)：兩個 uint8 (h, w) 陣列（0..255，同尺寸，四周留了空白）。
    outline_mask 是「文字加外框」的範圍（outline <= 0 時與 fill 相同）；文字是空的時回傳 None。
    px：字高（像素）；spacing：字距（px）；line_spacing：行距（% 字高）；align：left / center / right；
    vertical：True＝直書（字由上往下，換行往左；每個字直立排成一格；標點會貼右上角或轉向）。
    box_w / box_h：文字框的寬／高（px，0＝自動）。橫排時 box_w 決定在哪裡自動換行，直書時 box_h 決定；
    設定的那一邊遮罩就是框的大小（align 對齊的是框，超出框的部分會被裁掉），沒設定的那邊依文字大小。
    valign：框高（橫排）或框寬（直書）有指定時，文字在框裡的位置 top / middle / bottom（直書時 top＝靠右、bottom＝靠左）。
    clip：False 時超出框的文字不裁掉；遮罩會以框的中心為中心、向兩側對稱加大到放得下（框本身的位置不變）。"""
    if not text or not text.strip():
        return None
    if TEXT_RENDERER is None:
        raise ValueError("程式沒有提供文字繪製功能")
    key = (text, family, round(float(px), 2), bool(bold), bool(italic), round(float(spacing), 2),
           round(float(line_spacing), 1), align, round(float(outline), 2), bool(vertical),
           round(float(box_w), 1), round(float(box_h), 1), valign, bool(clip))
    with _IMG_LOCK:
        hit = _TXT_CACHE.get(key)
        if hit is not None:
            _TXT_CACHE.move_to_end(key)
            return hit
    res = TEXT_RENDERER(*key)
    if res is not None:
        for a in res:
            a.flags.writeable = False
        with _IMG_LOCK:
            _TXT_CACHE[key] = res
            while len(_TXT_CACHE) > 16:
                _TXT_CACHE.popitem(last=False)
    return res


def resolve_image_params(specs, params):
    """把參數裡 type 為 image 的路徑換成圖片陣列（或 None）；其他值不動。回傳新的 dict。"""
    out = dict(params)
    for s in specs or []:
        if s.get("type") == "image":
            out[s["key"]] = load_image(out.get(s["key"], s.get("default", "")))
    return out


def default_params(params):
    return {p["key"]: copy.deepcopy(p["default"]) for p in params}


def sanitize(params, values):
    """依規格整理參數：缺少或不合法的值回到預設，數值夾在範圍內。"""
    values = values if isinstance(values, dict) else {}
    out = {}
    for p in params:
        try:
            out[p["key"]] = _clean_value(p, values.get(p["key"], p["default"]))
        except Exception:
            out[p["key"]] = _clean_value(p, copy.deepcopy(p["default"]))
    return out


# ------------------------------------------------------------------ 插件載入
class PluginInfo:
    def __init__(self, path):
        self.path = path              # 主檔案（單檔插件＝該 .py；資料夾插件＝plugin.py / __init__.py）
        self.source_name = path.name  # 錯誤時顯示用（資料夾插件會改為資料夾名稱）
        self.is_package = False
        self.id = "err:" + path.name
        self.name = path.stem
        self.category = "其他"
        self.category_path = ("其他",)
        self.version = ""
        self.author = ""
        self.description = ""
        self.generator = False        # True：永遠憑空產生圖片
        self.can_generate = False     # generator = "auto"：有圖時當濾鏡，沒有圖時自己生成
        self.params = []
        self.process = None
        self.skip_if_empty = None  # 選填：某個圖片／文字參數是空的時候這一層等於不存在（例如還沒選圖的疊加圖片），調它的其他參數不會觸發重新計算
        self.drag = None          # 選填：{'x': 參數名, 'y': 參數名, 'unit': 'percent'|'px'}，讓畫布上的「移動」工具能改這兩個參數
        self.output_size = None   # 選填：插件會改變圖片尺寸時，用它回報輸出的完整尺寸
        self.presets = []   # 插件內建預設：[{'name', 'params'}]
        self.error = None


def _validate_presets(raw):
    out = []
    for x in raw if isinstance(raw, (list, tuple)) else []:
        if isinstance(x, dict) and str(x.get("name", "")).strip() and isinstance(x.get("params"), dict):
            out.append({"name": str(x["name"]).strip(), "params": dict(x["params"])})
    return out


def _validate_skip(raw, params):
    """INFO['skip_if_empty'] = 參數名（必須是 image 或 text 參數）；寫錯就忽略。"""
    ok = {q["key"] for q in params if q["type"] in ("image", "text")}
    return raw if isinstance(raw, str) and raw in ok else None


def _validate_drag(raw, params):
    """INFO['drag'] = {"x": "offset_x", "y": "offset_y", "unit": "percent"}；寫錯就忽略（不讓整個插件失敗）。"""
    if not isinstance(raw, dict):
        return None
    nums = {q["key"] for q in params if q["type"] in ("int", "float")}
    if raw.get("x") in nums and raw.get("y") in nums and raw.get("x") != raw.get("y"):
        return {"x": raw["x"], "y": raw["y"], "unit": "px" if raw.get("unit") == "px" else "percent"}
    return None


def _purge_modules(prefix):
    for k in [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]:
        del sys.modules[k]


def _category_parts(raw):
    parts = tuple(x.strip() for x in str(raw).replace("\\", "/").split("/") if x.strip())
    return parts or ("其他",)


def _load_one(path, seen, folders=(), package_dir=None):
    """path：主檔案；folders：它所在的分類資料夾（相對於插件根目錄）；
    package_dir：資料夾插件的資料夾（會以套件方式載入，可用相對匯入）。"""
    p = PluginInfo(path)
    key = "/".join(folders + ((package_dir.name,) if package_dir else (path.stem,)))
    modname = "tb_plugin_" + re.sub(r"\W", "_", key)
    if package_dir is not None:
        p.is_package = True
        p.source_name = package_dir.name
    _purge_modules(modname)
    old_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True   # 不產生 .pyc：重新載入時一定讀到最新的原始碼
    try:
        importlib.invalidate_caches()
        kw = {"submodule_search_locations": [str(package_dir)]} if package_dir is not None else {}
        spec = importlib.util.spec_from_file_location(modname, path, **kw)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        info = getattr(mod, "INFO", None)
        if not isinstance(info, dict) or not info.get("id"):
            raise ValueError("缺少 INFO['id']")
        if not callable(getattr(mod, "process", None)):
            raise ValueError("缺少 process() 函式")
        params = _validate_params(getattr(mod, "PARAMS", []))
        pid = str(info["id"])
        if pid in seen:
            raise ValueError(f"插件 id 重複：{pid}")
        seen.add(pid)
        p.id = pid
        p.name = str(info.get("name", pid))
        p.category_path = _category_parts(info["category"]) if info.get("category") else (tuple(folders) or ("其他",))
        p.category = "/".join(p.category_path)
        p.version = str(info.get("version", ""))
        p.author = str(info.get("author", ""))
        p.description = str(info.get("description", ""))
        gen = info.get("generator", False)
        p.can_generate = gen == "auto"
        p.generator = bool(gen) and not p.can_generate
        p.params = params
        p.presets = _validate_presets(getattr(mod, "PRESETS", []))
        p.drag = _validate_drag(info.get("drag"), params)
        p.skip_if_empty = _validate_skip(info.get("skip_if_empty"), params)
        p.process = mod.process
        fn = getattr(mod, "output_size", None)
        p.output_size = fn if callable(fn) else None
    except Exception as e:  # 單一插件壞掉不影響其他插件
        p.error = f"{type(e).__name__}: {e}"
        p.category_path = tuple(folders) or ("其他",)
        p.category = "/".join(p.category_path)
        _purge_modules(modname)
        traceback.print_exc()
    finally:
        sys.dont_write_bytecode = old_flag
    return p


def _skip_name(name):
    return name.startswith(("_", ".")) or name == "__pycache__"


def _scan(d, folders, plugins, seen, depth=0):
    if depth > 8:
        return
    try:
        entries = sorted(d.iterdir(), key=lambda e: e.name.lower())
    except OSError:
        return
    for e in entries:
        if _skip_name(e.name):
            continue
        if e.is_file() and e.suffix == ".py":
            plugins.append(_load_one(e, seen, folders))
        elif e.is_dir():
            main = next((e / n for n in ("plugin.py", "__init__.py") if (e / n).is_file()), None)
            if main is not None:
                plugins.append(_load_one(main, seen, folders, package_dir=e))
            else:
                _scan(e, folders + (e.name,), plugins, seen, depth + 1)


def load_plugins(dirs):
    plugins, seen = [], set()
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            _scan(d, (), plugins, seen)
    return plugins


# ------------------------------------------------------------------ 設定檔
def load_settings():
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def default_user_plugin_dir():
    return USER_DIR / "plugins"


def migrate_legacy_data():
    """第一次使用新位置時，把舊版 ~/.sai2_toolbox 裡的資料複製進 data/（只複製 data/ 裡還沒有的；不刪除舊檔）。
    回傳複製的項目名稱清單。"""
    import shutil
    done = []
    try:
        if not LEGACY_DIR.is_dir() or LEGACY_DIR.resolve() == USER_DIR.resolve():
            return done
        USER_DIR.mkdir(parents=True, exist_ok=True)
        for item in LEGACY_DIR.iterdir():
            dest = USER_DIR / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
            done.append(item.name)
    except Exception:
        traceback.print_exc()
    return done


def save_settings(data):
    """寫入設定檔。先寫暫存檔再取代，寫到一半當掉也不會弄壞原本的設定（自動存檔會頻繁寫入，這點很重要）。"""
    try:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)
        return True
    except Exception:
        return False


def settings_corrupt():
    """設定檔存在但讀不出來（壞掉）時回傳 True。"""
    try:
        if not SETTINGS_PATH.is_file():
            return False
        return not isinstance(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")), dict)
    except Exception:
        return True


DEFAULT_BACKUP_COUNT = 5


def backup_dir():
    return SETTINGS_PATH.parent / "backups"


def list_backups():
    """設定備份檔，新的在前。"""
    try:
        return sorted(backup_dir().glob("settings_*.json"), reverse=True)
    except Exception:
        return []


def backup_settings(keep=DEFAULT_BACKUP_COUNT):
    """啟動時把目前的設定檔複製成帶時間的備份，只保留最近 keep 份（0＝不備份）。
    壞掉的設定檔不備份（免得把好的備份擠掉）；內容跟最新一份備份相同時也不重複備份。回傳新備份的路徑或 None。"""
    try:
        keep = int(keep)
    except (TypeError, ValueError):
        keep = DEFAULT_BACKUP_COUNT
    if keep <= 0 or not SETTINGS_PATH.is_file():
        return None
    try:
        raw = SETTINGS_PATH.read_bytes()
        if not isinstance(json.loads(raw.decode("utf-8")), dict):
            return None
        backup_dir().mkdir(parents=True, exist_ok=True)
        old = list_backups()
        if old and old[0].read_bytes() == raw:
            return None
        path = backup_dir() / time.strftime("settings_%Y%m%d_%H%M%S.json")
        n = 1
        while path.exists():
            path = backup_dir() / time.strftime(f"settings_%Y%m%d_%H%M%S_{n}.json")
            n += 1
        path.write_bytes(raw)
        for extra in list_backups()[keep:]:
            try:
                extra.unlink()
            except OSError:
                pass
        return path
    except Exception:
        return None


def restore_backup(path, keep=DEFAULT_BACKUP_COUNT):
    """用某份備份取代目前的設定檔（取代前先把目前的設定也備份起來）。成功回傳 True。"""
    try:
        raw = Path(path).read_bytes()
        if not isinstance(json.loads(raw.decode("utf-8")), dict):
            return False
    except Exception:
        return False
    backup_settings(max(int(keep), 1) + 1)        # 目前的設定先留一份（多留一份，免得被修剪時擠掉要還原的那份）
    try:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_name(SETTINGS_PATH.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, SETTINGS_PATH)
        return True
    except Exception:
        return False
