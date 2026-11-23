import json
import os
from pathlib import Path
import time
from datetime import datetime
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.ops import unary_union

API_HOST = "https://mv4nmv3f8m.re.qweatherapi.com"
API_KEY = (
    os.getenv("QWEATHER_API_KEY")
    or os.getenv("WEATHER_API_KEY")
    or ""
).strip()

CITY_GEOJSON_URL = "https://geo.datav.aliyun.com/areas_v3/bound/440000_full.json"
CITY_BOUNDARY_PATH = "guangdong_cities.geojson"
BORDER_CACHE_PATH = "gd_inland_borders_cache.json"

EAST_SHIFT_DEG = 0.01
MIN_MAINLAND_AREA_DEG2 = 0.03

TEMP_TIERS = [
    (36.0, 99.0, "#FF8A80", "極端酷熱"),
    (32.5, 36.0, "#FFB4A2", "酷熱"),
    (28.0, 32.5, "#FFCC80", "炎熱"),
    (23.0, 28.0, "#FFE082", "溫暖"),
    (18.0, 23.0, "#FFF59D", "和暖"),
    (12.5, 18.0, "#C5E1A5", "清涼"),
    (8.0, 12.5, "#A7FFEB", "寒冷"),
    (5.0, 8.0, "#90CAF9", "嚴寒"),
    (-50.0, 5.0, "#CE93D8", "極端嚴寒"),
]


def get_temp_c(t):
    if t is None:
        return "#E0E0E0", "black"
    if t >= 36:
        return "#FF8A80", "black"
    elif t >= 32.5:
        return "#FFB4A2", "black"
    elif t >= 28:
        return "#FFCC80", "black"
    elif t >= 23:
        return "#FFE082", "black"
    elif t >= 18:
        return "#FFF59D", "black"
    elif t >= 12.5:
        return "#C5E1A5", "black"
    elif t >= 8:
        return "#A7FFEB", "black"
    elif t > 5:
        return "#90CAF9", "black"
    else:
        return "#CE93D8", "black"


STATIONS = [
    ("廣州", 23.1291, 113.2644), ("深圳", 22.5431, 114.0579), ("珠海", 22.2710, 113.5767),
    ("汕頭", 23.3540, 116.6815), ("佛山", 23.0215, 113.1214), ("韶關", 24.8104, 113.5972),
    ("湛江", 21.2707, 110.3594), ("肇慶", 23.0472, 112.4651), ("江門", 22.5787, 113.0815),
    ("茂名", 21.6630, 110.9254), ("惠州", 23.1108, 114.4162), ("梅州", 24.2766, 116.1176),
    ("汕尾", 22.7857, 115.3752), ("河源", 23.7435, 114.7004), ("陽江", 21.8579, 111.9822),
    ("清遠", 23.6988, 113.0560), ("東莞", 23.0207, 113.7518), ("中山", 22.5176, 113.3926),
    ("潮州", 23.6570, 116.6229), ("揭陽", 23.5438, 116.3728), ("雲浮", 22.9151, 112.0445),
]


def download_city_boundary(save_path: str, force: bool = False) -> Path | None:
    path = Path(save_path)
    if path.exists() and not force:
        return path
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://datav.aliyun.com/",
    }
    try:
        resp = requests.get(CITY_GEOJSON_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "features" not in data:
            return None
        path.write_text(resp.text, encoding="utf-8")
        return path
    except Exception as e:
        print(f"下載市界失敗：{e}", flush=True)
        return None


def filter_mainland_geometry(geom, min_area: float):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area >= min_area else None
    if isinstance(geom, MultiPolygon):
        kept = [poly for poly in geom.geoms if poly.area >= min_area]
        if not kept:
            return None
        return MultiPolygon(kept) if len(kept) > 1 else kept[0]
    return geom


def get_or_build_inland_borders(mainland_gdf: gpd.GeoDataFrame, cache_path: str):
    p = Path(cache_path)
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            return shape(raw["city_border"]), shape(raw["provincial_border"])
        except Exception:
            pass
    ocean_feat = cfeature.NaturalEarthFeature("physical", "ocean", "10m")
    ocean_geoms = [g for g in ocean_feat.geometries() if g is not None and not g.is_empty]
    ocean_union = unary_union(ocean_geoms).buffer(0.015) if ocean_geoms else None
    city_lines = unary_union([geom.boundary for geom in mainland_gdf.geometry if geom is not None])
    if ocean_union is not None:
        city_lines = city_lines.difference(ocean_union)
    gd_union = unary_union(mainland_gdf.geometry)
    prov_lines = gd_union.boundary.difference(ocean_union) if ocean_union is not None else gd_union.boundary
    try:
        payload = {
            "city_border": mapping(city_lines),
            "provincial_border": mapping(prov_lines),
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass
    return city_lines, prov_lines


def fetch_now_by_coord(session: requests.Session, name: str, lat: float, lon: float):
    url = f"{API_HOST}/v7/weather/now"
    params = {
        "location": f"{lon:.4f},{lat:.4f}",
        "key": API_KEY,
        "lang": "zh-hant",
        "unit": "m",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    try:
        resp = session.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"❌ {name} 請求失敗，狀態碼: {resp.status_code}，回應: {resp.text}", flush=True)
            return None
        data = resp.json()
        if data.get("code") == "200" and "now" in data:
            now = data["now"]
            return name, lat, lon, float(now["temp"]), now.get("obsTime", "")
        else:
            print(f"❌ {name} 失敗，和風回傳: {resp.text}", flush=True)
    except Exception as e:
        print(f"❌ {name} 請求異常：{e}", flush=True)
    return None


def declutter_positions(records, min_dx=0.28, min_dy=0.18, step_x=0.16, step_y=0.12):
    placed, adjusted = [], []
    for name, lat, lon, t_val in sorted(records, key=lambda r: (-r[1], r[2])):
        dx_cand, dy_cand = lon, lat
        if any(abs(dx_cand - px) < min_dx and abs(dy_cand - py) < min_dy for px, py in placed):
            offsets = [
                (step_x, 0.0), (-step_x, 0.0), (0.0, step_y), (0.0, -step_y),
                (step_x, step_y), (-step_x, step_y), (step_x, -step_y), (-step_x, -step_y),
                (1.6 * step_x, 0.0), (-1.6 * step_x, 0.0), (0.0, 1.6 * step_y), (0.0, -1.6 * step_y),
            ]
            for ox, oy in offsets:
                tx, ty = lon + ox, lat + oy
                if not any(abs(tx - px) < min_dx and abs(ty - py) < min_dy for px, py in placed):
                    dx_cand, dy_cand = tx, ty
                    break
        placed.append((dx_cand, dy_cand))
        adjusted.append((name, lat, lon, dx_cand, dy_cand, t_val))
    return adjusted


def main():
    t_total = time.perf_counter()
    if not API_KEY:
        print("❌ 未設定 QWEATHER_API_KEY 或 WEATHER_API_KEY 環境變數，無法查詢天氣。", flush=True)
        return

    print("[進度] 1. 準備市界資料...", flush=True)
    city_path = download_city_boundary(CITY_BOUNDARY_PATH, force=False)
    records, obs_time = [], ""

    print(f"[進度] 2. 查詢 {len(STATIONS)} 個地級市站點氣溫...", flush=True)
    with requests.Session() as session:
        for name, lat, lon in STATIONS:
            res = fetch_now_by_coord(session, name, lat, lon)
            if res:
                _, _, _, t_val, o_time = res
                records.append((name, lat, lon, t_val))
                if o_time and not obs_time:
                    obs_time = o_time
            time.sleep(0.3)

    if not records:
        print("沒有抓到有效站點資料，請檢查上方日誌中的詳細原因。", flush=True)
        return

    lon_min, lon_max = 109.5, 117.5
    lat_min, lat_max = 20.0, 25.6
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "Microsoft JhengHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    print("[進度] 3. 繪製底圖、內陸市界與省界...", flush=True)
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(13, 9.5), dpi=150)
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#C5E1F5")
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#F7EED6")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.9, edgecolor="#555555", zorder=2.3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle="--", linewidth=0.8, edgecolor="#666666")

    if city_path and city_path.exists():
        try:
            city_gdf = gpd.read_file(city_path)
            if city_gdf.crs and city_gdf.crs.to_string() != "EPSG:4326":
                city_gdf = city_gdf.to_crs("EPSG:4326")
            mainland_geoms = []
            for geom in city_gdf.geometry:
                shifted = translate(geom, xoff=EAST_SHIFT_DEG, yoff=0.0) if geom is not None else None
                filtered = filter_mainland_geometry(shifted, MIN_MAINLAND_AREA_DEG2)
                if filtered is not None and not filtered.is_empty:
                    mainland_geoms.append(filtered)
            if mainland_geoms:
                mainland_gdf = gpd.GeoDataFrame(geometry=mainland_geoms, crs="EPSG:4326")
                mainland_gdf = mainland_gdf.cx[lon_min:lon_max, lat_min:lat_max]
                city_lines, prov_lines = get_or_build_inland_borders(mainland_gdf, BORDER_CACHE_PATH)
                ax.add_geometries(
                    [city_lines],
                    crs=proj,
                    facecolor="none",
                    edgecolor="#222222",
                    linewidth=0.75,
                    linestyle="-",
                    zorder=2.4,
                )
                ax.add_geometries(
                    [prov_lines],
                    crs=proj,
                    facecolor="none",
                    edgecolor="#D32F2F",
                    linewidth=1.25,
                    linestyle="-",
                    zorder=2.6,
                )
        except Exception as e:
            print(f"處理市界與省界失敗：{e}", flush=True)

    draw_items = declutter_positions(records)
    for name, lat, lon, draw_lon, draw_lat, t_val in draw_items:
        bg_color, fg_color = get_temp_c(t_val)
        label_str = f"{t_val:g}"
        ax.text(
            draw_lon,
            draw_lat,
            label_str,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=fg_color,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=bg_color,
                edgecolor="#666666",
                linewidth=0.5,
                alpha=0.95,
            ),
            transform=proj,
            zorder=4,
        )

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=label)
        for _, _, color, label in TEMP_TIERS
    ]
    leg = ax.legend(
        handles=legend_handles,
        title="圖例",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#888888",
        framealpha=0.92,
        fontsize=9,
        title_fontsize=10.5,
    )
    leg.get_title().set_ha("center")

    update_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if obs_time:
        try:
            dt = datetime.fromisoformat(obs_time)
            update_str = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass

    ax.text(
        0.025,
        0.03,
        f"更新時間：{update_str}\n制圖：港澳天氣站",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#AAAAAA",
            alpha=0.92,
        ),
        zorder=5,
    )
    ax.set_title("廣東現時氣溫分佈圖", fontsize=15, fontweight="bold", pad=14)
    plt.tight_layout()

    output_path = "guangdong_temp_latest.png"
    plt.savefig(output_path, dpi=150)
    print(f"[完成] 已輸出 {output_path}（共 {len(records)} 個有效城市，耗時 {time.perf_counter() - t_total:.2f}s）", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
