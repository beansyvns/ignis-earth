# IGNIS Earth: replay a real evening of a Greek fire season the model never saw.
# Reads the small demo_* files made by 03_demo.ipynb. Nothing is trained or predicted here: the page replays
# the predictions frozen in notebook 2, so nothing on it can be tuned after seeing the test seasons.
# Code drafted with help from Claude (Anthropic), reviewed by the author.
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import plotly.io as pio
import plotly.utils
from matplotlib.figure import Figure as MplFigure   # only used to trace the smooth risk bands on the map
from matplotlib.path import Path as MplPath

# One dark chart style for the whole site: see-through backgrounds, quiet grid, readable hover boxes
pio.templates["ignis"] = go.layout.Template(pio.templates["plotly_dark"])
pio.templates["ignis"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, Inter, system-ui, sans-serif", color="#b6bfcc", size=14),
    xaxis=dict(gridcolor="#222935", zerolinecolor="#2f3846"), yaxis=dict(gridcolor="#222935", zerolinecolor="#2f3846"),
    title=dict(font=dict(color="#eceff4", size=16)),
    hoverlabel=dict(bgcolor="#181d25", bordercolor="#2f3846", namelength=-1,
                    font=dict(family="Geist, Inter, system-ui, sans-serif", color="#eceff4", size=13)))
pio.templates.default = "ignis"
UP, DOWN = "#d9381e", "#8a94a6"        # raises risk (fire red) / lowers risk (grey)
BONE = "#f1ebe1"                       # what really burned
ACCENT = "#f5a524"                     # our model, and the main action

P = pd.read_parquet("demo_preds.parquet")
V = pd.read_parquet("demo_villages.parquet")
H = pd.read_parquet("demo_history.parquet")         # burning cells per day (for the animated replay)
RES = pd.read_csv("demo_results.csv")                # scores of every method on this season
COEF = pd.read_csv("demo_coef.csv")                  # M1 coefficients (per 1 standard deviation)
RC = pd.read_csv("demo_riskcov.csv")                 # risk-coverage curve of the trust gate
PER_DAY = pd.read_csv("demo_per_day.csv")            # newly burned cells per day, caught by model / wind rule
S = json.load(open("demo_settings.json"))
THR = S["threshold"]                                 # warning threshold (tuned on the validation season)
SEASON_LABEL = S["season_label"]
P["day_str"] = P.day.dt.strftime("%Y-%m-%d")
DAYS = sorted(P.day_str.unique())
PRESETS = S.get("presets", [])
# open on a clear, trustworthy example: the day our model beat the wind rule
PRESETS = sorted(PRESETS, key=lambda p: p["label"] != "Model beats the wind rule")
DEFAULT_DAY = PRESETS[0]["day"] if PRESETS else S.get("default_day")
GATE_COLOR = {"Supported": "#b6bfcc", "Limited": "#f5a524", "Unsupported": "#d9381e"}
GATE_ACTION = {"Supported": "Today's weather looks like the summers the model learned from (2014-2022), so the forecast can be used.",
               "Limited": "Some of today's weather ({r}) is more extreme than the summers the model learned from. Use the map with care.",
               "Unsupported": "Today's {r} {v} outside anything the model has seen. Don't rely on this map: "
                              "use the simple wind rule and reports from the ground instead."}
PLAIN = {"tmax": "temperature", "rh_min": "humidity", "wind_max": "wind", "gust_max": "wind gusts", "vpd_max": "dry air",
         "days_since_rain": "days without rain", "rain_7d": "recent rain", "et0_7d": "drought", "upslope": "slope",
         "dist_to_fire": "distance to the fire", "wind_alignment": "wind direction", "cluster_size": "fire size"}
FEATURE_WORDS = {"dist_to_fire": "distance from the fire", "wind_alignment": "downwind of the fire",
                 "upslope": "uphill of the fire", "wind_max": "wind speed", "rh_min": "humidity",
                 "vpd_max": "dry air", "tmax": "temperature", "days_since_rain": "days without rain",
                 "cluster_size": "size of the fire", "fire_today": "square already burning"}
METHOD_NAMES = {"M1_logistic": "IGNIS (our model)", "M2_boosting": "Complex model (tested, not used)",
                "B5_downwind": "Wind rule", "B1_persistence": "Same as today",
                "B3_neighbours": "Next to the fire", "B2_climatology": "Where fires burned before"}
PRESET_NAMES = {"Model beats the wind rule": "Our model beats the wind rule", "Failure day": "A day it failed",
                "Big false alarm": "Big false alarm", "Big fire day": "Big fire day"}
FONT = dict(family="Geist, Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=14, color="#b6bfcc")
C_COLS = [c for c in P.columns if c.startswith("c_")]
# Map tiles without place labels (OpenStreetMap labels in Greece are in Greek); we draw our own village names in Latin script
MAP_STYLE = os.environ.get("IGNIS_MAP_STYLE", "carto-darkmatter-nolabels")
LAT0 = float((P.lat - (P.row + 0.5) * 0.05).median()); LON0 = float((P.lon - (P.col + 0.5) * 0.05).median())
V = V.assign(lat=LAT0 + (V.row + 0.5) * 0.05, lon=LON0 + (V.col + 0.5) * 0.05)
MAP_DIR = Path("maps").resolve(); MAP_DIR.mkdir(exist_ok=True)     # the animated maps, saved as small web pages
gr.set_static_paths(paths=[str(MAP_DIR)])


def topo_background():
    """Faint terrain-style contour lines for the Overview background (drawn once at startup, nothing is real elevation)."""
    out = MAP_DIR / "topo.png"
    if out.exists():
        return out
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:1:360j, 0:1.75:630j]
    z = np.zeros_like(x)
    for _ in range(18):   # a sum of soft random waves looks like rolling hills
        k = rng.uniform(1.5, 9.0); a = rng.uniform(0, 2 * np.pi)
        z += np.cos(2 * np.pi * k * (x * np.cos(a) + y * np.sin(a)) + rng.uniform(0, 2 * np.pi)) / k
    fig = MplFigure(figsize=(14, 8), dpi=100); fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.contour(x, y, z, levels=28, colors="#f1ebe1", linewidths=0.7, alpha=0.16)
    fig.savefig(out, transparent=True)
    return out

TOPO_URL = f"/gradio_api/file={topo_background()}"

# Map traces: plotly >= 5.24 has the token-free "Scattermap"; older versions use Scattermapbox
if hasattr(go, "Scattermap"):
    MapTrace, MAP_KEY = go.Scattermap, "map"
else:
    MapTrace, MAP_KEY = go.Scattermapbox, "mapbox"


def compass(deg):
    return ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"][int((deg + 22.5) % 360 // 45)]


def nearest_village(rows, cols):
    # the most populated place within ~3 cells (15 km) of the fire, else the closest one
    r, c = float(np.mean(rows)), float(np.mean(cols))
    d = np.hypot(V.row - r, V.col - c)
    near = V[d <= 3]
    return (near.sort_values("population", ascending=False).iloc[0] if len(near) else V.iloc[int(d.argmin())])["name"]


def fire_label(fid):
    b = P[(P.fire_id == fid) & (P.fire_today == 1)]
    d = pd.Timestamp(b.day.iloc[0]).strftime("%-d %b %Y")
    n = len(b)
    size = "big fire" if n >= 8 else "medium fire" if n >= 4 else "small fire"
    return f"{d}, near {nearest_village(b.row, b.col)}: {size} ({n} squares burning)"


def all_fires():
    # every fire with at least 3 burning squares, plus the example fires, in date order
    n = P[P.fire_today == 1].groupby("fire_id").size()
    keep = set(n[n >= 3].index) | {p.get("fire_id") for p in PRESETS if p.get("fire_id") in n.index}
    first_day = P.groupby("fire_id").day.min()
    return [(fire_label(fid), fid) for fid in sorted(keep, key=lambda x: (first_day[x], x))]


def fires_on(day_str):
    # one dropdown entry per fire that day, biggest first: (what people see, the fire id the app uses)
    d = P[(P.day_str == day_str) & (P.fire_today == 1)]
    out = []
    for fid, b in sorted(d.groupby("fire_id"), key=lambda kv: -len(kv[1])):
        n = len(b)
        out.append((f"Fire near {nearest_village(b.row, b.col)} ({n} burning square{'s' if n != 1 else ''})", fid))
    return out


def arrow(lat, lon, deg, length, color, name):
    # a line with a marker at the tip, pointing toward compass direction `deg`
    r = math.radians(deg)
    lat2 = lat + length * math.cos(r)
    lon2 = lon + length * math.sin(r) / math.cos(math.radians(lat))
    return MapTrace(lat=[lat, lat2], lon=[lon, lon2], mode="lines+markers", name=name, hoverinfo="name",
                    line=dict(width=4, color=color), marker=dict(size=[0, 12], color=color))


def empty(name):
    return MapTrace(lat=[], lon=[], mode="markers", name=name, showlegend=False)


def square_geojson(lats, lons):
    # one 5 km square (0.05 degrees) per cell centre, so the map shows the real grid
    feats = []
    for la, lo in set(zip(np.round(lats, 3), np.round(lons, 3))):
        h = 0.025
        feats.append({"type": "Feature", "id": f"{la:.3f}_{lo:.3f}",
                      "geometry": {"type": "Polygon", "coordinates": [[[lo - h, la - h], [lo + h, la - h], [lo + h, la + h],
                                                                        [lo - h, la + h], [lo - h, la - h]]]}})
    return {"type": "FeatureCollection", "features": feats}


def keys(d):
    return [f"{la:.3f}_{lo:.3f}" for la, lo in zip(np.round(d.lat, 3), np.round(d.lon, 3))]


CHART_HEAD = ('<head><meta charset="utf-8"><link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap" '
              'rel="stylesheet"><style>html,body{margin:0;background:transparent;overflow:hidden}</style>')

# Charts inside a hidden tab get drawn at zero size, and Plotly then remembers those zero text widths, so labels
# pile on top of each other. So each chart page waits to draw until it is actually on screen and its font has loaded,
# then tells the map it may start its replay.
MAP_CSS = (".maplibregl-ctrl-bottom-right,.mapboxgl-ctrl-bottom-right{top:0 !important;bottom:auto !important}"
           ".maplibregl-ctrl-bottom-left,.mapboxgl-ctrl-bottom-left{display:none !important}")
WAIT_JS = ("<script>window.__ignisPlot=function(){var a=arguments;return new Promise(function(res){function go(){"
           "var fe=window.frameElement,shown=fe?fe.getClientRects().length>0&&fe.offsetWidth>50:innerWidth>50;"
           "if(!window.Plotly||!shown){setTimeout(go,150);return;}"
           "(document.fonts?document.fonts.ready:Promise.resolve()).then(function(){var p=Plotly.newPlot.apply(Plotly,a);"
           "window.dispatchEvent(new Event('ignis-ready'));p.then(res,function(){res();});});}go();});};</script>")


def wait_until_shown(page):
    return page.replace("Plotly.newPlot(", "window.__ignisPlot(").replace("</head>", WAIT_JS + "</head>", 1)


BANDS = [(0.2, "#fbbf24", "lower risk"), (0.42, "#f5a524", "medium risk"), (0.62, "#d9381e", "highest risk")]


def risk_bands(cand, score):
    """Smooth, see-through hazard bands (like a hazard map) instead of hard squares.
    Each square's risk score (0-1) is spread out with a soft blur, then traced as yellow, orange and red outlines."""
    pad = 0.12
    gy = np.arange(cand.lat.min() - pad, cand.lat.max() + pad, 0.005)
    gx = np.arange(cand.lon.min() - pad, cand.lon.max() + pad, 0.005)
    X, Y = np.meshgrid(gx, gy)
    k = math.cos(math.radians(float(cand.lat.mean())))
    sig = 0.032
    num, den = np.zeros_like(X), np.zeros_like(X)
    for la, lo, r in zip(cand.lat.values, cand.lon.values, score):
        w = np.exp(-(((Y - la) ** 2) + ((X - lo) * k) ** 2) / (2 * sig ** 2))
        num += w * r; den += w
    field = num / np.maximum(den, 1e-9) * np.clip(den / 1.2, 0, 1)   # fades out where there are no scored squares
    feats = []
    ax = MplFigure().subplots()
    for b, (lvl, _, _) in enumerate(BANDS):
        polys = []
        cs = ax.contourf(X, Y, field, levels=[lvl, 10])
        for path in cs.get_paths():
            rings = [r for r in path.to_polygons(closed_only=True) if len(r) >= 4]
            if not rings:
                continue
            area = [0.5 * np.sum(r[:-1, 0] * r[1:, 1] - r[1:, 0] * r[:-1, 1]) for r in rings]
            outer_sign = np.sign(area[int(np.argmax(np.abs(area)))])
            outers = [[r] for r, a in zip(rings, area) if np.sign(a) == outer_sign]
            for r, a in zip(rings, area):
                if np.sign(a) != outer_sign:    # a hole: attach it to the outline that contains it
                    for o in outers:
                        if MplPath(o[0]).contains_point(r[0]):
                            o.append(r); break
            polys += [[ring.round(4).tolist() for ring in o] for o in outers]
        feats.append({"type": "Feature", "id": f"band{b}", "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    return {"type": "FeatureCollection", "features": feats}


def map_html(f, reveal, mini=False):
    """Animated replay: the 3 days before, tonight at 6 PM, the forecast, then (Reveal) what really burned."""
    burn, cand = f[f.fire_today == 1], f[f.fire_today == 0]
    c_lat, c_lon = burn.lat.mean(), burn.lon.mean()
    day = pd.Timestamp(f.day.iloc[0])
    box = (H.lat.between(f.lat.min() - 0.05, f.lat.max() + 0.05)) & (H.lon.between(f.lon.min() - 0.05, f.lon.max() + 0.05))
    hist = H[box & H.fire_day.between(day - pd.Timedelta(days=3), day - pd.Timedelta(days=1))]
    wind_to = (float(f.wind_dir.iloc[0]) + 180) % 360
    new = cand[cand.fire_tomorrow == 1]
    geo = square_geojson(new.lat.values, new.lon.values) if len(new) else {"type": "FeatureCollection", "features": []}

    def dots(d, name, color, size, opacity=0.95):   # fire as glowing dots, so the map underneath stays visible
        return MapTrace(lat=d.lat, lon=d.lon, mode="markers", marker=dict(size=size, color=color, opacity=opacity),
                        hovertemplate=name + "<extra></extra>", showlegend=False)

    def none_pts():
        return MapTrace(lat=[], lon=[], mode="markers", showlegend=False)

    def earlier(upto):  # squares that burned on earlier days of the replay
        return dots(hist[hist.fire_day < upto], "Burned in the days before", "#8a94a6", 9, 0.6)

    # risk score per square: its rank among tonight's squares around this fire (0 = lowest, 1 = highest)
    score = cand.prob.rank(pct=True).values if len(cand) else np.array([])
    tier = np.minimum((score * 3).astype(int), 2) if len(cand) else score
    bgeo = risk_bands(cand, score) if len(cand) >= 2 else None

    def band(b, show=True):
        if bgeo is None or not show:
            return go.Choroplethmap(geojson={"type": "FeatureCollection", "features": []}, locations=[], z=[], showscale=False)
        return go.Choroplethmap(geojson=bgeo, locations=[f"band{b}"], z=[1], featureidkey="id", showscale=False, showlegend=False,
                                colorscale=[[0, BANDS[b][1]], [1, BANDS[b][1]]], marker=dict(opacity=0.34, line=dict(width=0)),
                                hoverinfo="skip")

    # invisible points on each square so hovering anywhere near one shows its numbers in plain words
    TIER_NAMES = ["Lower risk", "Medium risk", "Highest risk"]
    hover = MapTrace(lat=cand.lat, lon=cand.lon, mode="markers", marker=dict(size=26, color="#ffffff", opacity=0.01),
                     text=[f"<b>{TIER_NAMES[t]}</b>: {p:.0%} chance of fire tomorrow<br>{(w or 'no strong reason').capitalize()}"
                           f"<br>{d:.0f} km from the fire" for t, p, w, d in zip(tier, cand.prob, cand.why, cand.dist_to_fire)],
                     hovertemplate="%{text}<extra></extra>", showlegend=False)

    vb = V[V.lat.between(f.lat.min() - 0.03, f.lat.max() + 0.03) & V.lon.between(f.lon.min() - 0.03, f.lon.max() + 0.03)]
    vb = vb.sort_values("population", ascending=False).head(10)
    towns = MapTrace(lat=vb.lat, lon=vb.lon, mode="markers+text", text=vb["name"], textposition="top right",
                     textfont=dict(size=12, color="#eceff4"), marker=dict(size=5, color="#eceff4"), showlegend=False,
                     hovertext=[f"{n}, {p:,} people" for n, p in zip(vb["name"], vb.population)], hoverinfo="text")
    span_lat, span_lon = f.lat.max() - f.lat.min() + 0.1, f.lon.max() - f.lon.min() + 0.1
    zoom = float(np.clip(min(np.log2(600 * 360 / (256 * span_lon)), np.log2(470 * 360 * math.cos(math.radians(c_lat)) / (256 * span_lat))) - 0.7, 8, 11.5))
    # a field of small, faint arrows showing tonight's wind (like a weather map); longer arrows = stronger wind
    speed = float(f.wind_max.iloc[0]) if "wind_max" in f else 20.0
    L = 0.018 * float(np.clip(speed / 20, 0.6, 1.8))
    r = math.radians(wind_to); k = math.cos(math.radians(c_lat))
    alat, alon = [], []
    for la in np.linspace(f.lat.min(), f.lat.max(), 4):
        for lo in np.linspace(f.lon.min(), f.lon.max(), 4):
            tip = (la + L * math.cos(r), lo + L * math.sin(r) / k)
            alat += [la - L * math.cos(r), tip[0], None]; alon += [lo - L * math.sin(r) / k, tip[1], None]
            for side in (-1, 1):   # the two short lines of the arrow head
                b2 = r + math.pi + side * 0.5
                alat += [tip[0], tip[0] + 0.4 * L * math.cos(b2), None]; alon += [tip[1], tip[1] + 0.4 * L * math.sin(b2) / k, None]
    wind = MapTrace(lat=alat, lon=alon, mode="lines", line=dict(width=1.5, color="#b6bfcc"), opacity=0.45,
                    hoverinfo="skip", showlegend=False)
    no_line = MapTrace(lat=[], lon=[], mode="lines", showlegend=False)

    def outline(d):
        if not len(d):
            return go.Choroplethmap(geojson=geo, locations=[], z=[], showscale=False, showlegend=False)
        return go.Choroplethmap(geojson=geo, locations=keys(d), z=[1] * len(d), showscale=False, showlegend=False,
                                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                marker=dict(line=dict(color=BONE, width=3)), hovertemplate="Really burned the next day<extra></extra>")

    tonight = dots(burn, "Burning tonight at 6 PM", "#ff6a2b", 14)
    halo = dots(burn, "Burning tonight at 6 PM", "#ff6a2b", 30, 0.18)

    # traces in every frame: earlier, halo, burning, 3 risk bands, hover points, wind, real outline, towns
    def layers(early, glow, burning, b=(False, False, False), hov=False, w=False, real=None):
        return [early, glow, burning, band(0, b[0]), band(1, b[1]), band(2, b[2]), hover if hov else none_pts(), wind if w else no_line,
                outline(real if real is not None else new.head(0)), towns]

    # Each frame: (name, traces, counter text, how long it stays in ms, shown on the slider?)
    frames = []
    for kd in (3, 2, 1):
        dk = day - pd.Timedelta(days=kd)
        that = dots(hist[hist.fire_day == dk], "Burning that day", "#ff8a4c", 12)
        frames.append((f"{kd} day{'s' if kd > 1 else ''} before", layers(earlier(dk), none_pts(), that), "", 750, True))
    frames.append(("Tonight 6 PM", layers(earlier(day), halo, tonight, w=True), "", 1000, True))
    # the forecast "heats up": the yellow band spreads in first, then orange, then red
    frames.append(("_heat0", layers(earlier(day), halo, tonight, (True, False, False), True, True), "", 320, False))
    frames.append(("_heat1", layers(earlier(day), halo, tonight, (True, True, False), True, True), "", 320, False))
    frames.append(("Forecast", layers(earlier(day), halo, tonight, (True, True, True), True, True), "", 1100, True))
    full = (True, True, True)
    if reveal and len(new):
        # reveal sweep: the squares that really burned get their blue outline a few at a time, riskiest first,
        # with a running count of how many our top 10 had caught
        top10 = set(cand.sort_values("prob", ascending=False).head(10).index)
        order = new.sort_values("prob", ascending=False)
        cuts = np.unique(np.linspace(1, len(order), min(len(order), 8)).round().astype(int))
        for i, c in enumerate(cuts):
            shown = order.head(c); hit = len(top10 & set(shown.index))
            last = i == len(cuts) - 1
            txt = (f"<b>{hit} of {len(order)}</b> new squares were in our top 10" if last
                   else f"Checking: <b>{hit} of {c}</b> in our top 10")
            frames.append(("What burned" if last else f"_sweep{i}", layers(earlier(day), halo, tonight, full, True, True, shown),
                           txt, 1000 if last else 260, last))
    elif reveal:
        frames.append(("What burned", layers(earlier(day), halo, tonight, full, True, True),
                       "The fire did not spread to new squares", 1000, True))

    def counter(txt):
        return dict(annotations=[dict(text=txt, x=0.99, y=0.99, xref="paper", yref="paper", xanchor="right", yanchor="top",
                                      showarrow=False, bgcolor="#12161c", bordercolor=BONE, borderwidth=1, borderpad=8,
                                      font=dict(size=15, color="#eceff4"))] if txt else [])

    names = [n for n, *_ in frames]
    durations = [dict(duration=d, redraw=True) for *_, d, _ in frames]
    # with Reveal on, skip the history and play only the reveal sweep
    first = names.index("Forecast") + 1 if reveal else 0
    play_names, play_durations = names[first:], durations[first:]
    start = frames[-1]   # the page first shows the finished map, so it's complete even if the animation can't run
    fig = go.Figure(data=start[1], layout=counter(start[2]))
    frames_json = json.dumps([go.Frame(name=n, data=d, layout=counter(t)).to_plotly_json() for n, d, t, *_ in frames],
                             cls=plotly.utils.PlotlyJSONEncoder)
    anim = dict(frame=durations, transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        **{MAP_KEY: dict(style=MAP_STYLE, center=dict(lat=c_lat, lon=c_lon), zoom=zoom, domain=dict(x=[0, 1], y=[0, 1]))},
        height=580,
        # a soft dark band at the bottom so the slider reads on top of the map
        shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0, y1=0.2, line=dict(width=0),
                     fillcolor="rgba(10,12,16,0.55)", layer="above")], margin=dict(l=0, r=0, t=0, b=0), font=FONT, showlegend=False, paper_bgcolor="#0a0c10",
        updatemenus=[dict(type="buttons", active=-1, pad=dict(r=6, t=6), showactive=False, x=0.01, y=0.99, xanchor="left", yanchor="top", direction="left",
                          bgcolor="#181d25", bordercolor="#2f3846", font=dict(color="#eceff4", size=13),
                          buttons=[dict(label="Replay", method="animate", args=[names, anim]),
                                   dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=sum(1 for *_, on in frames if on) - 1, x=0.08, y=0.01, len=0.82, yanchor="bottom", pad=dict(t=6, b=8),
                      currentvalue=dict(prefix="Showing: ", font=dict(size=15, color="#ffffff"), offset=12), font=dict(size=11, color="#eceff4"),
                      bgcolor="#ffffff", activebgcolor="#ffffff", bordercolor="#ffffff", borderwidth=1,
                      tickcolor="rgba(255,255,255,0.6)", ticklen=4, minorticklen=0,
                      steps=[dict(label=n, method="animate", args=[[n], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for n, *_, on in frames if on])])
    # Plays once when the map opens (skipped for people who turn off animations); Replay and the slider replay or scrub it.
    # The frames are added by our own script (not chained to the map's loading), so a slow map tile or font
    # can't stop the replay. It plays once when the map opens, unless the viewer turned animations off.
    # The replay starts the first time at least half the map is actually on screen (not while it sits below the fold),
    # and the guided demo can call window.ignisReplay() to play it again.
    play_js = ("<script>window.addEventListener('ignis-ready',function(){var gd=document.querySelector('.js-plotly-plot');"
               f"Plotly.addFrames(gd,{frames_json}).then(function(){{"
               "var last=0;window.ignisReplay=function(){if(Date.now()-last<4000)return;last=Date.now();"
               f"Plotly.animate(gd,{json.dumps(play_names)},{{frame:{json.dumps(play_durations)},transition:{{duration:0}},mode:'immediate'}}).catch(function(){{}});}};"
               "if(window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;"
               "var io=new IntersectionObserver(function(es){if(es[0].intersectionRatio>=0.5){io.disconnect();window.ignisReplay();}},"
               "{threshold:[0.5]});io.observe(gd);"
               "});},{once:true});</script>")
    if mini:   # the still preview on the Overview page: tonight's forecast only, no controls, no scroll zoom
        fig.layout.updatemenus, fig.layout.sliders, fig.layout.shapes = (), (), ()
        fig.update_layout(height=460)
        play_js = ""
    # Gradio's plot component drops animation frames, so each map is saved as its own small web page
    # (once per fire and reveal setting) and shown in an iframe.
    page = fig.to_html(include_plotlyjs="cdn", full_html=True, config={"displaylogo": False, "displayModeBar": False, "scrollZoom": not mini, "responsive": True},
                       auto_play=False).replace("<head>", CHART_HEAD.replace("transparent", "#0a0c10", 1).replace("</style>", MAP_CSS + "</style>", 1), 1).replace("</body>", play_js + "</body>", 1)
    page = wait_until_shown(page)
    path = MAP_DIR / f"{f.fire_id.iloc[0]}_{int(reveal)}{'_mini' if mini else ''}.html"
    path.write_text(page, encoding="utf-8")
    return f'<iframe src="/gradio_api/file={path}" style="width:100%;height:{470 if mini else 590}px;border:0;border-radius:12px;display:block"></iframe>'



def chart(fig, name):
    """A chart as its own small page: no toolbar, drag to move, scroll to zoom, double-click to reset."""
    if fig is None:
        return ""
    fig.update_layout(dragmode="pan", autosize=True)
    h = int(fig.layout.height or 380)
    page = fig.to_html(include_plotlyjs="cdn", full_html=True, default_width="100%", default_height=f"{h}px",
                       config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset", "responsive": True}
                       ).replace("<head>", CHART_HEAD, 1)
    page = wait_until_shown(page)
    path = MAP_DIR / f"chart_{name}.html"
    path.write_text(page, encoding="utf-8")
    return (f'<iframe src="/gradio_api/file={path}?v={abs(hash(page)) % 10**8}" '
            f'style="width:100%;height:{h + 4}px;border:0;display:block;background:transparent"></iframe>')


def push_words(v, more=""):
    # a push on the risk in plain words, with the rounded number for anyone who wants it
    size = "a little" if abs(v) < 0.1 else "somewhat" if abs(v) < 0.3 else "a lot"
    return f"{more}{'raises' if v > 0 else 'lowers'} the risk {size} ({v:+.2f})"


def why_figure(row):
    # one horizontal bar per feature: how hard it pushed this cell toward (red) or away from (blue) fire tomorrow
    vals = pd.Series({c[2:]: row[c] for c in C_COLS}).sort_values()
    fig = go.Figure(go.Bar(x=vals.values, y=[FEATURE_WORDS.get(k, k) for k in vals.index], orientation="h",
                           marker_color=[UP if v > 0 else DOWN for v in vals.values],
                           customdata=[push_words(v) for v in vals.values],
                           hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
    fig.update_layout(height=320, margin=dict(l=10, r=40, t=56, b=48), font=FONT,
                      title=dict(text=f"Why {row.prob:.0%}? Red raises the risk, grey lowers it", x=0.02, font=dict(size=15)),
                      xaxis_title="← lowers the risk · raises the risk →")
    return fig


def show(fire_choice, reveal):
    try:
        return _show(fire_choice, reveal)
    except Exception as e:   # never show a bare "Error" to a judge
        print("show() failed for", fire_choice, repr(e))
        return ("", "", "<div class='brief'>Sorry, this fire could not be drawn. Please pick another one.</div>",
                pd.DataFrame(), "", "", "")


def _show(fire_choice, reveal):
    if not fire_choice:
        return "", "", "<div class='brief'>Pick a fire above.</div>", pd.DataFrame(), "", "", ""
    fid = fire_choice
    f = P[P.fire_id == fid].copy()
    day_str = pd.Timestamp(f.day.iloc[0]).strftime("%-d %B %Y")
    burn, cand = f[f.fire_today == 1], f[f.fire_today == 0]
    place = nearest_village(burn.row, burn.col)

    # trust check: the worst label among this fire's cells, with what to do about it
    order = ["Supported", "Limited", "Unsupported"]
    g = max(f.gate, key=order.index)
    rs = sorted({PLAIN.get(r, r.replace("_", " ")) for x in f.gate_reasons for r in str(x).split(", ") if r and r != "nan"}) or ["weather"]
    reasons = rs[0] if len(rs) == 1 else ", ".join(rs[:-1]) + " and " + rs[-1]
    verb = "is" if len(rs) == 1 else "are"
    gate_html = (f"<div class='trust' style='border-left-color:{GATE_COLOR[g]}'><span class='state' style='color:{GATE_COLOR[g]}'>"
                 f"Trust check: {g}</span> "
                 f"{GATE_ACTION[g].format(r=reasons, v=verb)}</div>")

    # villages in cells above the warning threshold
    warned = cand[cand.prob >= THR]
    vv = V.merge(warned[["row", "col", "prob", "dist_to_fire", "wind_alignment"]], on=["row", "col"]) \
          .sort_values(["prob", "population"], ascending=False)

    # the briefing: a fixed template filled from the numbers above (no language model, so every word is checkable)
    wind_to = compass((float(f.wind_dir.iloc[0]) + 180) % 360)
    top = cand.sort_values("prob", ascending=False).head(5)
    common = pd.Series(", ".join(top.why).split(", ")).value_counts()
    common = [w for w in common.index if w][:2]
    vill_txt = (f"villages inside them: " + ", ".join(vv.name.head(3)) + (f" and {len(vv) - 3} more" if len(vv) > 3 else "")) \
               if len(vv) else "no villages inside them"
    nb, nw = len(burn), len(warned)
    briefing = (f"<div class='brief'><div class='brief-h'>Briefing for {day_str}, 6 PM</div>"
                f"A fire is burning near <b>{place}</b> ({nb} square{'s' if nb != 1 else ''} of 5 km on fire). "
                f"Tonight's wind blows toward the {wind_to}"
                f"{f' at about {float(f.wind_max.iloc[0]):.0f} km/h' if 'wind_max' in f else ''}. The squares most at risk are mostly "
                f"<b>{' and '.join(common) or 'close to the fire'}</b>.<br>"
                f"<b>{nw}</b> square{'' if nw == 1 else 's'} flagged as at risk for tomorrow, {vill_txt}.</div>")

    table = pd.DataFrame({"#": range(1, len(top) + 1),
                          "chance": (100 * top.prob).round(0).astype(int).astype(str) + "%",
                          "why": top.why.replace("", "-"),
                          "km away": top.dist_to_fire.round(0).astype(int)})

    if len(vv):
        lines = [f"- **{r['name']}** ({r.population:,} people): {r.prob:.0%} chance, "
                 f"{r.dist_to_fire:.0f} km {'downwind of' if r.wind_alignment > 0.5 else 'from'} the fire"
                 for _, r in vv.head(6).iterrows()]
        villages = "### Villages in the path\n" + "\n".join(lines)
    else:
        villages = "### Villages in the path\nNo village is inside a square flagged as at risk."

    score = ""
    if reveal:
        new = cand[cand.fire_tomorrow == 1]
        caught_model = int((new.prob >= THR).sum())
        caught_rule = int((new.upwind_nbr_fire == 1).sum())
        top10 = set(cand.sort_values("prob", ascending=False).head(10).index)
        score = (f"<div class='score'><b>What really happened:</b> the fire reached {len(new)} new square{'s' if len(new) != 1 else ''} the next day.<br>"
                 f"<b>Our model</b> flagged {caught_model} of them ({len(warned)} squares flagged in total; "
                 f"{len(top10 & set(new.index))} were in its top 10).<br>"
                 f"<b>The wind rule</b> (flag every square just downwind of the fire) caught {caught_rule} of them "
                 f"({int((cand.upwind_nbr_fire == 1).sum())} squares flagged).</div>")
    why_fig = chart(why_figure(top.iloc[0]), f"why_{fid}_1") if len(top) else ""
    if not len(cand):
        briefing += "<div class='explain'>No squares around this fire were scored for tomorrow.</div>"
    return map_html(f, reveal), gate_html, briefing, table, villages, score, why_fig


def explain(fire_choice, rank):
    if not fire_choice:
        return ""
    f = P[(P.fire_id == fire_choice) & (P.fire_today == 0)]
    top = f.sort_values("prob", ascending=False).head(5)
    i = min(int(rank), len(top))
    return chart(why_figure(top.iloc[i - 1]), f"why_{fire_choice}_{i}") if len(top) else ""


# ---------- "How it works" charts ----------
def coef_figure():
    c = COEF.assign(words=COEF.feature.map(FEATURE_WORDS).fillna(COEF.feature)).sort_values("coef")
    fig = go.Figure(go.Bar(x=c.coef, y=c.words, orientation="h",
                           marker_color=[UP if v > 0 else DOWN for v in c.coef],
                           customdata=[push_words(v, "More of it ") for v in c.coef],
                           hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=56, b=48), font=FONT, title="Which conditions push fire risk up or down?",
                      xaxis_title="← lowers risk · raises risk →")
    return fig


def riskcov(cov_pct):
    cov = cov_pct / 100
    i = int(np.abs(RC.coverage - cov).argmin())
    tip = "<b>%{fullData.name}</b><br>Keep the most normal %{x:.0%} of forecasts<br>Average error %{y:.3f}<extra></extra>"
    fig = go.Figure([go.Scatter(x=RC.coverage, y=RC.risk_v2, name="Trust check", line=dict(width=3, color=ACCENT), hovertemplate=tip),
                     go.Scatter(x=RC.coverage, y=RC.risk_v1, name="Simpler version", line=dict(width=2, dash="dot", color="#8a94a6"),
                                hovertemplate=tip),
                     go.Scatter(x=[RC.coverage[i]], y=[RC.risk_v2[i]], mode="markers", marker=dict(size=14, color=ACCENT),
                                showlegend=False, hoverinfo="skip")])
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=56, b=110), font=FONT, title="Is our model more accurate on normal days?",
                      xaxis=dict(title="share of forecasts we keep (most normal days first)", tickformat=".0%"),
                      yaxis_title="average error (lower is better)", legend=dict(orientation="h", x=0, y=-0.28, yanchor="top"))
    full = RC.risk_v2.iloc[-1]
    txt = (f"If we only trust the most familiar **{cov:.0%}** of forecasts, the average error is **{RC.risk_v2[i]:.3f}**, "
           f"compared with **{full:.3f}** for all of them.")
    return chart(fig, f"riskcov_{cov_pct}"), txt


# ---------- Scorecard charts ----------
METRICS = {"Caught in the top 10": "top10_share_caught", "False alarms per day": "false_alarms_per_day"}
METRIC_HELP = {"Caught in the top 10": "If crews can cover only the 10 riskiest squares, how much of the new fire do they catch? Higher is better.",
               "Ranking score": "How well the method puts the squares that really burned at the top of its list (0 to 1). Higher is better.",
               "New fire flagged": "Share of the squares that really burned the next day that were flagged. Higher is better.",
               "False alarms per day": "Squares flagged that did not burn, per fire-day. Lower is better."}
SUBSETS = [("Squares the fire hasn't reached yet", "spread"), ("Squares already burning", "continuing")]


def num(v, col):
    # a number that counts up when it appears (the final value is already in the text, so nothing breaks without animation)
    pct = col in ("top10_share_caught", "recall")
    val, dec, suf = (100 * v, 0, "%") if pct else (v, 1, "")
    return f"<span class='num' data-count='{val:.{dec}f}' data-dec='{dec}' data-suffix='{suf}'>{val:.{dec}f}{suf}</span>"


def score_figure(subset, metric):
    # one bar per method, best at the top; the bars grow from zero each time they're shown
    col = METRICS[metric]
    lower_better = "False" in metric
    r = RES[(RES.subset == subset) & RES[col].notna() & (RES.method != "B1_persistence")].copy()
    r["name"] = r.method.map(METHOD_NAMES).fillna(r.method)
    r = r.sort_values(col, ascending=lower_better)
    top = float(r[col].max()) or 1.0
    title = ("Which method catches the most new fire in its top 10 squares?" if not lower_better
             else "Which method raises the fewest false alarms?")
    rows = []
    for i, (_, x) in enumerate(r.iterrows()):
        ours = x.method == "M1_logistic"
        shown = f"{x[col]:.0%}" if col == "top10_share_caught" else f"{x[col]:.1f}"
        rows.append(f"<div class='bar-row{' ours' if ours else ''}' title='{x['name']}: {shown}'>"
                    f"<div class='bar-name'>{x['name']}</div>"
                    f"<div class='bar-track'><div class='bar-fill' style='--w:{x[col] / top:.4f};--i:{i}'></div></div>"
                    f"<div class='bar-val'>{num(x[col], col)}</div></div>")
    return (f"<div class='bars'><div class='bars-t'>{title}</div>{''.join(rows)}"
            f"<div class='bars-n'>{'Lower' if lower_better else 'Higher'} is better. Amber = our model.</div></div>")


def score_help(metric):
    return METRIC_HELP[metric]


def timeline_figure():
    d = PER_DAY.copy()
    d = d[d.new > 0].sort_values("day")
    x = pd.to_datetime(d.day).dt.strftime("%-d %b %Y")
    tip = "%{y} squares<extra></extra>"
    fig = go.Figure([go.Bar(x=x, y=d.new, name="New fire that day", marker_color="#3b4352", hovertemplate=tip),
                     go.Bar(x=x, y=d.model, name="Warned by our model", marker_color=ACCENT, hovertemplate=tip),
                     go.Bar(x=x, y=d.rule, name="Warned by the wind rule", marker_color="#b6bfcc", hovertemplate=tip)])
    fig.update_layout(barmode="group", hovermode="x unified", height=420, margin=dict(l=10, r=10, t=72, b=90), font=FONT,
                      title="On each day, how much of the new fire<br>did each method see coming?",
                      legend=dict(orientation="h", x=0, y=-0.18, yanchor="top"), xaxis=dict(type="category", showticklabels=False, title="days, in order"))
    return fig


CSS = """
/* Dark, quiet instrument look: near-black ground, one amber accent for actions and our model, blue second.
   Serif for the big question headlines, mono for numbers. */
:root{--bg:#0a0c10;--surface:#12161c;--raised:#181d25;--hover:#1d232d;--line:#222935;--line2:#2f3846;
      --text:#eceff4;--text2:#b6bfcc;--muted:#7f8a9a;--accent:#f5a524;--accent2:#ffcf7a;--red:#d9381e;--bone:#f1ebe1}
body, .gradio-container{background:var(--bg) !important}
.gradio-container{background:radial-gradient(1100px 420px at 70% -120px, rgba(245,165,36,.10), transparent 70%), var(--bg) !important;
                  max-width:1440px !important; font-size:15px; color:var(--text)}
.gradio-container, .gradio-container *{font-family:'Geist', 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important}
.gradio-container .mono, .gradio-container .num, .gradio-container .bar-val, .gradio-container table td{
  font-family:'Geist Mono', ui-monospace, Menlo, Consolas, monospace !important; font-variant-numeric:tabular-nums}
.gradio-container .vh h1{font-family:'Instrument Serif', Georgia, serif !important}
.gradio-container .prose{font-size:15px; line-height:1.6; color:var(--text2)}
.gradio-container .prose h3{color:var(--text); font-size:17px; font-weight:620}
/* top bar */
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:6px 2px 14px;border-bottom:1px solid var(--line)}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:.14em;font-size:16px;color:var(--text)}
.mark{display:inline-flex;gap:2px}.mark i{width:7px;height:16px;border-radius:2px;background:#fbbf24}
.mark i:nth-child(2){background:#f5a524}.mark i:nth-child(3){background:#d9381e}
/* tabs as a nav: the active tab gets the amber outline */
.nav > .tab-wrapper button, .nav [role=tab]{font-size:14px !important;color:var(--text2) !important;border-radius:10px !important;
  padding:7px 14px !important;border:1px solid transparent !important;background:none !important}
.nav [role=tab][aria-selected=true]{color:var(--text) !important;border-color:var(--accent) !important;background:var(--raised) !important}
.nav [role=tab][aria-selected=true]::after{display:none !important}
/* page heads: a big serif question, muted subtitle, a stats line */
.vh{padding:26px 0 14px}
.vh h1{font-size:clamp(34px,3.4vw,52px) !important;font-weight:400 !important;letter-spacing:-.01em;line-height:1.05;margin:0 0 12px !important;color:var(--text)}
.vh p{font-size:16.5px;line-height:1.55;color:var(--text2);max-width:72ch;margin:0 !important}
.statline{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:16px;padding-top:12px;border-top:1px solid var(--line);font-size:14px;color:var(--text2)}
.statline b{color:var(--text);font-weight:600;font-size:15px;margin-right:4px}
.statline i{font-style:normal;color:var(--muted)}
/* left sidebar with the inputs */
.side{background:rgba(255,255,255,.02) !important;border:1px solid var(--line) !important;border-radius:12px !important;padding:16px !important;
      align-self:flex-start;position:sticky;top:8px}
.side-sub{font-size:13px;color:var(--muted);margin:10px 0 2px}
.side-t{font-size:14px;font-weight:600;color:var(--text);margin:18px 0 2px;padding-top:14px;border-top:1px solid var(--line)}
.side-note{font-size:12.5px;line-height:1.45;color:var(--muted);margin:2px 0 6px}
.gradio-container button.chip{justify-content:flex-start !important;text-align:left;border-radius:999px !important;
  background:var(--bg) !important;border:1px solid var(--line2) !important;color:var(--text) !important;font-weight:500 !important}
.gradio-container button.chip:hover{border-color:var(--accent) !important}
.chip-d{font-size:12.5px;line-height:1.4;color:var(--muted);margin:-4px 2px 4px}
.legend{font-size:13.5px;line-height:2;color:var(--text2)}
.legend .sw, .legend .dot{display:inline-block;width:14px;height:14px;vertical-align:-2px;margin-right:9px;border-radius:3px;box-sizing:border-box}
.legend .dot{border-radius:50%;width:11px;height:11px;margin:0 10px 0 2px}
.legend .ln{display:inline-block;width:16px;height:2px;background:#b6bfcc;opacity:.7;vertical-align:4px;margin-right:9px}
/* Overview: a left-aligned editorial opening with a live map on the right, then the one number that matters */
.ov{align-items:center !important;gap:48px !important;padding:56px 0 24px}
.ov-pill{display:inline-flex;align-items:center;gap:10px;font-family:'Geist Mono', monospace !important;font-size:12.5px;color:var(--text2);
         border:1px solid var(--line2);border-radius:999px;padding:5px 14px}
.ov-pill i{width:7px;height:7px;border-radius:50%;background:var(--accent)}
.gradio-container .ov-h{font-family:'Fraunces', 'Instrument Serif', Georgia, serif !important;font-weight:600 !important;
       font-size:clamp(44px,5.4vw,80px) !important;line-height:.98 !important;letter-spacing:-.035em;margin:22px 0 18px !important;
       color:var(--text);max-width:11ch;animation:rise .7s cubic-bezier(0.22,1,0.36,1) .05s both}
.ov-sub{font-size:21px;line-height:1.45;color:var(--text);max-width:30em;margin:0 0 12px !important;animation:rise .7s cubic-bezier(0.22,1,0.36,1) .15s both}
.ov-sub b{color:var(--accent);font-weight:600}
.ov-p{font-size:16px;line-height:1.6;color:var(--text2);max-width:34em;margin:0 !important;animation:rise .7s cubic-bezier(0.22,1,0.36,1) .25s both}
.ov-cta{justify-content:flex-start !important;gap:12px !important;margin-top:22px}
/* Gradio dims an HTML panel while it updates; keep the map and charts solid so toggling doesn't flash */
.gradio-container .html-container.pending{opacity:1 !important}
#ig-tour-start{white-space:nowrap}
/* fire picker: keep the long fire name clear of the dropdown arrow */
#ig-fire input{padding-right:30px !important;text-overflow:ellipsis !important;overflow:hidden !important;white-space:nowrap !important}
.ov-note{font-size:13px;color:var(--muted);margin:10px 0 0 !important}
.ov-map{position:relative;border:1px solid var(--line);border-radius:14px;overflow:hidden;
        box-shadow:0 30px 80px -30px rgba(245,165,36,.25);animation:rise .8s cubic-bezier(0.22,1,0.36,1) .3s both}
.ov-cap{display:flex;align-items:center;gap:6px;padding:10px 14px;font-size:13px;color:var(--text2);border-top:1px solid var(--line);background:var(--surface)}
.ov-res{display:flex;align-items:baseline;gap:28px;flex-wrap:wrap;padding:28px 0 40px;margin-top:8px;border-top:1px solid var(--line)}
.ov-big{font-family:'Geist Mono', monospace !important;font-size:clamp(64px,8vw,112px);font-weight:600;line-height:1;color:var(--accent);letter-spacing:-.04em}
.ov-big .num{font-family:'Geist Mono', monospace !important;color:var(--accent) !important}
.ov-res-t{flex:1;min-width:280px;max-width:40em;font-size:19px;line-height:1.5;color:var(--text2)}
.ov-res-t b{color:var(--text);font-weight:600}
/* the top-5 table fits its column: no sideways scrolling, long reasons wrap */
#ig-top5 table{table-layout:fixed;width:100% !important}
#ig-top5 th:nth-child(1), #ig-top5 td:nth-child(1){width:34px}
#ig-top5 th:nth-child(2), #ig-top5 td:nth-child(2){width:78px}
#ig-top5 th:nth-child(4), #ig-top5 td:nth-child(4){width:84px}
#ig-top5 td{white-space:normal !important}
/* table headers stay on one line */
.gradio-container table th, .gradio-container table th *{white-space:nowrap !important}
/* fewer cards: most blocks are plain text set off by thin rules; only the briefing and sidebar keep a faint box */
.brief{background:rgba(255,255,255,.025) !important;border:1px solid var(--line) !important;border-radius:12px !important;
       padding:16px 20px;font-size:16px;line-height:1.6;color:var(--text2)}
.brief b{color:var(--text)}
.brief-h{font-weight:600;font-size:14px;color:var(--accent);margin-bottom:4px}
.trust{padding:2px 0 2px 16px;border-left:2px solid;font-size:15px;line-height:1.55;color:var(--text2)}
.state{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;font-weight:500;padding:3px 10px 3px 8px;border-radius:999px;
       border:1px solid currentColor;white-space:nowrap;line-height:1.4;margin-right:8px}
.state::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;flex:none}
.s-muted{color:var(--muted)}
.score{padding:4px 0 4px 16px;border-left:2px solid var(--bone);font-size:15px;line-height:1.6;color:var(--text2);margin-top:6px}
.score b{color:var(--text)}
.panel{background:none !important;border:0 !important;border-top:1px solid var(--line) !important;border-radius:0 !important;padding:14px 0 0 !important}
.explain{font-size:14px;color:var(--muted);margin:2px 0 -4px 0}
/* "How it works": five numbered columns under one rule, like a magazine sidebar */
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0;margin:10px 0 18px;border-top:1px solid var(--line2)}
.steps div{padding:14px 18px 6px 0;font-size:15px;line-height:1.45;color:var(--text2)}
.steps div + div{padding-left:18px;border-left:1px solid var(--line)}
.steps b{display:block;font-family:'Geist Mono', monospace !important;font-size:13px;color:var(--accent);margin-bottom:8px;letter-spacing:.04em}
.gradio-container button{transition:transform .16s cubic-bezier(0.22,1,0.36,1), background-color .15s cubic-bezier(0.22,1,0.36,1),
                         border-color .15s cubic-bezier(0.22,1,0.36,1)}
.gradio-container button:active{transform:scale(.97)}
.gradio-container :focus-visible{outline:2px solid var(--accent) !important;outline-offset:2px}

/* ---------- Page motion ----------
   Each animation has a job: show what just changed, or tell the story in reading order.
   Only opacity, transform and blur move. Nothing loops. */
@keyframes rise{from{opacity:0;transform:translateY(8px);filter:blur(4px)}}
@keyframes settle{from{opacity:0;transform:translateY(4px)}}
/* 1. Page load (once): the question headline, then its subtitle and numbers, come into focus */
.vh h1{animation:rise .6s cubic-bezier(0.22,1,0.36,1) .1s both}
.vh p{animation:rise .6s cubic-bezier(0.22,1,0.36,1) .2s both}
.statline{animation:rise .6s cubic-bezier(0.22,1,0.36,1) .3s both}
/* 2. A new fire is picked: the trust check, then the briefing, come into focus in reading order */
.trust{animation:rise .38s cubic-bezier(0.22,1,0.36,1) both}
.brief{animation:rise .38s cubic-bezier(0.22,1,0.36,1) .06s both}
/* 3. "Show what really happened": the result card arrives (the map does its own reveal sweep) */
.score{animation:rise .45s cubic-bezier(0.22,1,0.36,1) both}
/* 4. Switching tabs: the new tab settles in; the "How it works" steps follow the pipeline in order */
.tabitem{animation:settle .24s cubic-bezier(0.22,1,0.36,1) both}
.steps div{animation:settle .32s cubic-bezier(0.22,1,0.36,1) calc(var(--i, 0) * 50ms) both}
.steps div:nth-child(2){--i:1}.steps div:nth-child(3){--i:2}.steps div:nth-child(4){--i:3}.steps div:nth-child(5){--i:4}
/* 5. Hover lift only where there is a real mouse */
@media (hover:hover) and (pointer:fine){
  .gradio-container button:hover{transform:translateY(-1px)}
  .gradio-container button:active{transform:scale(.97)}
}
.num{font-variant-numeric:tabular-nums}
/* racing bars on the Results tab */
.bars{padding:14px 0 0;border-top:1px solid var(--line)}
.bars-t{font-weight:620;font-size:17px;margin-bottom:14px;color:var(--text)}
.bar-row{display:grid;grid-template-columns:160px 1fr 60px;gap:12px;align-items:center;padding:6px 0}
.bar-name{font-size:14px;line-height:1.25;color:var(--text2)}
.bar-row.ours .bar-name{font-weight:600;color:var(--text)}
.bar-track{height:8px;background:var(--raised);border-radius:999px;overflow:hidden}
.bar-fill{height:100%;width:calc(var(--w) * 100%);background:#4b5567;border-radius:999px;transform-origin:left center;
  transform:scaleX(0);transition:transform .7s cubic-bezier(.22,1,.36,1) calc(var(--i) * 70ms)}
.bar-row.ours .bar-fill{background:var(--accent)}
.bar-fill.in{transform:scaleX(1)}
.bar-val{font-size:14px;font-weight:600;text-align:right;color:var(--text)}
.bars-n{font-size:13px;color:var(--muted);margin-top:12px}
@media (prefers-reduced-motion: reduce){.bar-fill{transform:none !important}}
@media (prefers-reduced-motion: reduce){*{animation:none !important;transition:none !important}
  .gradio-container button:hover, .gradio-container button:active{transform:none !important}}
"""
_D = dict(body_background_fill="#0a0c10", block_background_fill="transparent", block_border_color="transparent",
          border_color_primary="#222935", background_fill_primary="#12161c", background_fill_secondary="#0d1015",
          input_background_fill="#181d25", input_border_color="#2f3846", body_text_color="#eceff4",
          body_text_color_subdued="#7f8a9a", block_label_text_color="#b6bfcc", block_title_text_color="#b6bfcc",
          block_label_background_fill="transparent", button_primary_background_fill="#f5a524",
          button_primary_background_fill_hover="#ffcf7a", button_primary_text_color="#1b1204",
          button_secondary_background_fill="#181d25", button_secondary_background_fill_hover="#1d232d",
          button_secondary_text_color="#eceff4", button_secondary_border_color="#2f3846",
          table_even_background_fill="#12161c", table_odd_background_fill="#0d1015", table_border_color="#222935",
          checkbox_background_color_selected="#f5a524", checkbox_border_color_selected="#f5a524",
          checkbox_label_background_fill="#181d25", checkbox_label_background_fill_selected="#1d232d",
          checkbox_label_text_color_selected="#eceff4", slider_color="#f5a524", color_accent_soft="#1d232d")
THEME = gr.themes.Base(primary_hue="amber", neutral_hue="slate", radius_size=gr.themes.sizes.radius_lg,
                       font=[gr.themes.GoogleFont("Geist"), "Inter", "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"],
                       text_size=gr.themes.sizes.text_md).set(color_accent="#f5a524", **_D, **{k + "_dark": v for k, v in _D.items()})


def explain_line(text):
    # one plain sentence above a chart
    return gr.Markdown(f"<div class='explain'>{text}</div>")

MOTION_JS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Fraunces:opsz,wght@9..144,600&family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script>
(() => {
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ease = t => 1 - Math.pow(1 - t, 3);
  function countUp(el) {
    const to = parseFloat(el.dataset.count), dec = +el.dataset.dec || 0, suf = el.dataset.suffix || '';
    if (still || isNaN(to)) return;
    const t0 = performance.now(), dur = 900;
    const step = now => { const k = Math.min((now - t0) / dur, 1);
      el.textContent = (to * ease(k)).toFixed(dec) + suf; if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return; io.unobserve(e.target);
    if (e.target.classList.contains('bar-fill')) requestAnimationFrame(() => e.target.classList.add('in'));
    else countUp(e.target);
  }));
  const scan = () => document.querySelectorAll('.num:not([data-seen]), .bar-fill:not([data-seen])')
    .forEach(el => { el.dataset.seen = 1; io.observe(el); });
  new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
  addEventListener('DOMContentLoaded', scan);
})();
</script>
"""
FORCE_DARK = "() => { document.body.classList.add('dark'); document.documentElement.classList.add('dark'); }"

EXAMPLES = {"Model beats the wind rule": ("Our model got it right", "Red squares where the fire really went, and the wind rule missed it."),
            "Big fire day": ("The biggest spread day", "The fire jumped into the most new squares in one night."),
            "Failure day": ("Our model got it wrong", "The fire went where we said it was less likely. We show this on purpose."),
            "Big false alarm": ("A false alarm", "We warned of fire that never came. Firefighters may have stopped it.")}
FIRES = all_fires()


def hero_stats():
    # the two headline results, read from results.csv (2024-2025 seasons the model never saw); the numbers count up
    r = RES[RES.subset == "spread"].set_index("method")
    if not {"M1_logistic", "B5_downwind"} <= set(r.index):
        return ""
    m, w = r.loc["M1_logistic"], r.loc["B5_downwind"]
    return ("<div class='statline'>"
            f"<span><b class='mono'>{num(m.top10_share_caught, 'top10_share_caught')}</b> of new fire was in our 10 riskiest squares "
            f"<i>(wind rule {w.top10_share_caught:.0%})</i></span>"
            f"<span><b class='mono'>{num(m.false_alarms_per_day, 'false_alarms_per_day')}</b> false alarms a day "
            f"<i>(wind rule {w.false_alarms_per_day:.1f})</i></span></div>")


HERO_STATS = hero_stats()


def overview_result():
    # the one number that matters, read from results.csv: how much real new spread landed in our 10 riskiest squares
    r = RES[RES.subset == "spread"].set_index("method")
    if not {"M1_logistic", "B5_downwind"} <= set(r.index):
        return ""
    m, w = r.loc["M1_logistic"], r.loc["B5_downwind"]
    return ("<div class='ov-res'><div class='ov-big'>" + num(m.top10_share_caught, "top10_share_caught") + "</div>"
            f"<div class='ov-res-t'><b>of the real new spread</b> in the 2024-2025 test seasons landed in our 10 riskiest squares, "
            f"against <b class='mono'>{w.top10_share_caught:.0%}</b> for the simple wind rule, with "
            f"<b class='mono'>{m.false_alarms_per_day:.1f}</b> false alarms a day instead of "
            f"<b class='mono'>{w.false_alarms_per_day:.1f}</b>.</div></div>")


OVERVIEW_RESULT = overview_result()
FIRE_IDS = [v for _, v in FIRES]
START = next((p["fire_id"] for p in PRESETS if p.get("fire_id") in FIRE_IDS), FIRE_IDS[0] if FIRE_IDS else None)


def overview_map():
    # a live preview: tonight's forecast for the opening example fire, with its soft hazard zones
    try:
        f = P[P.fire_id == START].copy()
        return (f"<div class='ov-map'>{map_html(f, False, mini=True)}"
                f"<div class='ov-cap'><span class='state' style='color:#f5a524'>Forecast</span>{fire_label(START)}</div></div>")
    except Exception as e:
        print("overview map failed", repr(e))
        return ""


OVERVIEW_MAP = overview_map()


def tour_steps():
    """The guided demo: 11 steps in 5 parts. Every number is read from the data files when the app starts, nothing is typed in."""
    r = RES[RES.subset == "spread"].set_index("method")
    have = {"M1_logistic", "B5_downwind"} <= set(r.index)
    m, w = (r.loc["M1_logistic"], r.loc["B5_downwind"]) if have else (None, None)
    top10 = f"{m.top10_share_caught:.0%}" if have else "n/a"
    wtop10 = f"{w.top10_share_caught:.0%}" if have else "n/a"
    fa, wfa = (f"{m.false_alarms_per_day:.1f}", f"{w.false_alarms_per_day:.1f}") if have else ("n/a", "n/a")
    good = next((p for p in PRESETS if p["label"] == "Model beats the wind rule" and p.get("fire_id") in FIRE_IDS), None)
    bad = next((p for p in PRESETS if p["label"] == "Failure day" and p.get("fire_id") in FIRE_IDS), None)
    gid = good["fire_id"] if good else START
    f = P[P.fire_id == gid]
    cand = f[f.fire_today == 0].sort_values("prob", ascending=False)
    burn = f[f.fire_today == 1]
    new = cand[cand.fire_tomorrow == 1]
    hits = len(set(cand.head(10).index) & set(new.index))
    first = cand.iloc[0] if len(cand) else None
    day = pd.Timestamp(f.day.iloc[0]).strftime("%-d %B %Y")
    place = nearest_village(burn.row, burn.col) if len(burn) else "the fire"
    order = ["Supported", "Limited", "Unsupported"]
    gate = max(f.gate, key=order.index) if len(f) else "Supported"
    i = int(np.abs(RC.coverage - 0.5).argmin())
    bf = P[P.fire_id == bad["fire_id"]] if bad else None
    bday = pd.Timestamp(bf.day.iloc[0]).strftime("%-d %B %Y") if bad else ""
    bc = bf[bf.fire_today == 0] if bad else None
    bnew = bc[bc.fire_tomorrow == 1] if bad else None
    bmiss = int((bnew.prob < THR).sum()) if bad else 0
    good_chip = EXAMPLES["Model beats the wind rule"][0] if good else None
    bad_chip = EXAMPLES["Failure day"][0] if bad else None
    return [
        dict(part="1 / Problem", tab="Overview", target=".ov-h", title="Fires don't wait for morning",
             text="A Greek wildfire can jump into new valleys overnight, and crews have to decide where to stand by evening. "
                  "IGNIS Earth gives them a forecast for every 5 km square around the fire, for tomorrow."),
        dict(part="1 / Problem", tab="Overview", target=".ov-res", title="The proof, up front",
             text=f"In the 2024-2025 seasons the model never saw, {top10} of the real new spread landed in our 10 riskiest squares. "
                  f"The simple wind rule caught {wtop10}."),
        dict(part="2 / The forecast", tab="Forecast", target=".side", title="Pick any fire",
             text="On the left you can open any real fire from the 2024-2025 seasons, or jump to four we picked: "
                  "our best day, the biggest spread, a clear miss and a false alarm. The key at the bottom explains every colour on the map."),
        dict(part="2 / The forecast", tab="Forecast", chip=good_chip, fire=fire_label(gid), replay=True, target="#ig-map", title="A real fire, replayed",
             text=f"This is the fire near {place} on {day}. The map replays the three days before it, then tonight at 6 PM."),
        dict(part="2 / The forecast", tab="Forecast", target="#ig-map", title="Tomorrow's risk zones",
             text=f"Red, orange and yellow zones rank tomorrow's risk; the faint arrows are tonight's wind. "
                  f"The riskiest square has a {first.prob:.0%} chance of fire tomorrow." if first is not None else
                  "Red, orange and yellow zones rank tomorrow's risk; the faint arrows are tonight's wind."),
        dict(part="3 / Check us", tab="Forecast", reveal=True, target="#ig-map", title="What really happened",
             text=f"Now the real next day is drawn in. The fire reached {len(new)} new squares, and {hits} of them were in our top 10."),
        dict(part="3 / Check us", tab="Forecast", target="#ig-why", title="Every score has reasons",
             text=f"For the riskiest square the main reasons are {first.why or 'its distance to the fire'}. "
                  "Red bars push the risk up, grey bars push it down." if first is not None else
                  "Red bars push the risk up, grey bars push it down."),
        dict(part="4 / Honesty", tab="Forecast", target="#ig-gate", title="A trust check on every day",
             text=f"Each fire gets a trust label from how unusual its weather is. This day is rated {gate}, "
                  "and the page says plainly what to do when it isn't safe to rely on."),
        dict(part="4 / Honesty", tab="How it works", target="#ig-trust", title="Normal days, smaller errors",
             text=f"If we keep only the most normal half of forecasts, the average error is {RC.risk_v2[i]:.3f}, "
                  f"against {RC.risk_v2.iloc[-1]:.3f} for all of them. The trust check is measured on real seasons, not assumed."),
        dict(part="4 / Honesty", tab="Forecast", chip=bad_chip, fire=fire_label(bad["fire_id"]) if bad else "", reveal=True, target="#ig-map", title="A miss, shown on purpose",
             text=f"On {bday} the fire went where we said it was less likely: {bmiss} of {len(bnew)} new squares were below our warning level. "
                  "We keep this example in the demo on purpose." if bad else "We keep our failures in the demo on purpose."),
        dict(part="5 / Results", tab="Results", target="#ig-bars", title="It says when not to trust it",
             text=f"Tested once on 2024-2025: {top10} of new fire caught in the top 10 against {wtop10} for the wind rule, "
                  f"with {fa} false alarms a day against {wfa}. And it says when not to trust it."),
    ]


TOUR_JS = """
<script>
(() => {
  const STEPS = __STEPS__;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const wait = ms => new Promise(r => setTimeout(r, ms));
  let i = -1, card = null, ring = null, target = null;
  const byText = (sel, t) => [...document.querySelectorAll(sel)].find(e => e.textContent.trim() === t);
  function build() {
    card = document.createElement('div'); card.className = 'tour'; card.setAttribute('role', 'dialog'); card.setAttribute('aria-label', 'Guided demo');
    card.innerHTML = "<div class='tour-bar'><i></i></div><div class='tour-top'><span class='tour-n'></span><button class='tour-x' type='button'>Exit</button></div>"
      + "<h3 class='tour-h'></h3><p class='tour-p'></p><div class='tour-nav'><button class='tour-b' type='button'>Back</button>"
      + "<button class='tour-f' type='button'>Next</button></div>";
    ring = document.createElement('div'); ring.className = 'tour-ring';
    document.body.append(ring, card);
    card.querySelector('.tour-x').onclick = stop;
    card.querySelector('.tour-b').onclick = () => go(i - 1);
    card.querySelector('.tour-f').onclick = () => { if (!busy) (i >= STEPS.length - 1 ? stop() : go(i + 1)); };
  }
  function place() {
    if (!card || i < 0) return;
    // Gradio can swap an element out under us; find it again if the stored one is gone or empty
    if (!target || !target.isConnected || target.getBoundingClientRect().height < 2) target = document.querySelector(STEPS[i].target) || target;
    if (!target) return;
    const r = target.getBoundingClientRect();
    Object.assign(ring.style, {left: r.left - 6 + 'px', top: r.top - 6 + 'px', width: r.width + 12 + 'px', height: r.height + 12 + 'px'});
    const W = innerWidth, H = innerHeight, cw = card.offsetWidth, ch = card.offsetHeight, m = 16;
    if (W < 700) { Object.assign(card.style, {left: m + 'px', right: m + 'px', top: 'auto', bottom: m + 'px'}); return; }
    // put the card in whichever corner covers the highlighted part least
    const corners = [[m, m], [W - cw - m, m], [m, H - ch - m], [W - cw - m, H - ch - m]];
    const cover = ([x, y]) => Math.max(0, Math.min(x + cw, r.right) - Math.max(x, r.left)) * Math.max(0, Math.min(y + ch, r.bottom) - Math.max(y, r.top));
    const best = corners.reduce((a, b) => (cover(b) < cover(a) ? b : a));
    Object.assign(card.style, {left: best[0] + 'px', top: best[1] + 'px', right: 'auto', bottom: 'auto'});
  }
  let busy = false, lastChip = null;
  const waitFor = async (ok, max) => { const t0 = Date.now(); while (!ok() && Date.now() - t0 < max) await wait(80); };
  async function settleScroll() {   // wait until a smooth scroll has stopped, so the highlight lands once
    let y = -1, same = 0; const t0 = Date.now();
    while (same < 3 && Date.now() - t0 < 1500) { await wait(60); const ny = scrollY; same = ny === y ? same + 1 : 0; y = ny; }
  }
  async function mapReload(trigger) {   // click, then wait until the new map page has loaded
    const old = (document.querySelector('#ig-map iframe') || {}).src;
    trigger();
    await waitFor(() => { const f = document.querySelector('#ig-map iframe'); return f && f.src !== old; }, 6000);
    const f = document.querySelector('#ig-map iframe');
    if (f) await new Promise(r => { if (f.contentDocument && f.contentDocument.readyState === 'complete' && f.contentWindow.Plotly) r(); else { f.addEventListener('load', r, {once: true}); setTimeout(r, 4000); } });
    await wait(300);
  }
  async function go(n) {
    if (busy || n < 0 || n >= STEPS.length) return;
    busy = true; i = n; const s = STEPS[n];
    // fade the card and the highlight out while the page changes underneath
    card.classList.add('out'); ring.style.opacity = 0;
    await wait(still ? 0 : 180);
    card.querySelector('.tour-n').textContent = `Step ${n + 1} of ${STEPS.length}  ·  ${s.part}`;
    card.querySelector('.tour-h').textContent = s.title;
    card.querySelector('.tour-p').textContent = s.text;
    card.querySelector('.tour-bar i').style.transform = `scaleX(${(n + 1) / STEPS.length})`;
    card.querySelector('.tour-b').disabled = n === 0;
    card.querySelector('.tour-f').textContent = n === STEPS.length - 1 ? 'Finish' : 'Next';
    try {
      // 1. open the right page
      const tab = byText('[role=tab]', s.tab);
      if (tab && tab.getAttribute('aria-selected') !== 'true') { tab.click(); await wait(450); }
      // 2. set up the scene: load the example fire (only if it isn't loaded already), then show the real outcome
      const cur = (document.querySelector('#ig-fire input') || {}).value;
      if (s.chip && s.chip !== lastChip && cur !== s.fire) { const c = byText('button', s.chip); if (c) { await mapReload(() => c.click()); lastChip = s.chip; } }
      const rev = byText('button', 'Show what really happened');
      if (s.reveal && rev) await mapReload(() => rev.click());
      // 3. scroll to the feature and wait for the scroll to finish
      const big = () => { const e = document.querySelector(s.target); return e && e.getBoundingClientRect().height > 20; };
      await waitFor(big, 3000);
      target = document.querySelector(s.target);
      if (target) { target.scrollIntoView({block: target.getBoundingClientRect().height > innerHeight * 0.8 ? 'start' : 'center', inline: 'nearest', behavior: still ? 'auto' : 'smooth'}); await settleScroll(); }
    } catch (e) {}
    // 4. place the highlight and the card while invisible, then fade both in together
    ring.classList.add('snap'); place(); void ring.offsetWidth; ring.classList.remove('snap');
    ring.style.opacity = target ? 1 : 0; card.classList.remove('out');
    setTimeout(() => { if (i === n) place(); }, 350);
    if (s.replay) { const fr = document.querySelector('#ig-map iframe'); try { fr && fr.contentWindow.ignisReplay && fr.contentWindow.ignisReplay(); } catch (e) {} }
    busy = false;
  }
  function start() { if (!card) build(); card.hidden = false; ring.hidden = false; lastChip = null; busy = false; document.body.classList.add('touring'); go(0); }
  function stop() { if (!card) return; card.hidden = true; ring.hidden = true; target = null; document.body.classList.remove('touring'); }
  addEventListener('scroll', place, true); addEventListener('resize', place);
  addEventListener('keydown', e => {
    if (!card || card.hidden) return;
    if (busy && e.key !== 'Escape') { e.preventDefault(); return; }   // ignore extra presses while a step is still loading
    if (e.key === 'ArrowRight') { e.preventDefault(); i >= STEPS.length - 1 ? stop() : go(i + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(i - 1); }
    if (e.key === 'Escape') stop();
  });
  document.addEventListener('click', e => { if (e.target.closest('[data-tour-start], #ig-tour-start')) start(); });
})();
</script>
"""
TOUR_CSS = """
.tour{position:fixed;z-index:1000;width:min(360px,calc(100vw - 32px));background:#12161c;border:1px solid #2f3846;border-radius:14px;
      padding:16px 18px 14px;box-shadow:0 24px 60px rgba(0,0,0,.55);color:#eceff4;font-family:'Geist',system-ui,sans-serif;
      transition:opacity .2s cubic-bezier(0.22,1,0.36,1), transform .2s cubic-bezier(0.22,1,0.36,1)}
.tour.out{opacity:0;transform:translateY(4px);pointer-events:none}
.tour[hidden], .tour-ring[hidden]{display:none}
.tour-bar{height:3px;background:#222935;border-radius:999px;overflow:hidden;margin-bottom:12px}
.tour-bar i{display:block;height:100%;background:#f5a524;transform-origin:left;transition:transform .35s cubic-bezier(0.22,1,0.36,1)}
.tour-top{display:flex;justify-content:space-between;align-items:center;font-family:'Geist Mono',monospace;font-size:12px;color:#7f8a9a}
.tour-h{font-family:'Fraunces','Instrument Serif',Georgia,serif;font-weight:600;font-size:22px;line-height:1.15;margin:8px 0 6px;color:#eceff4}
.tour-p{font-size:14.5px;line-height:1.55;color:#b6bfcc;margin:0 0 14px}
.tour-nav{display:flex;justify-content:space-between;gap:8px}
.tour button{font:inherit;font-size:13.5px;font-weight:600;border-radius:9px;padding:7px 14px;cursor:pointer;border:1px solid #2f3846;background:transparent;color:#eceff4}
.tour .tour-f{background:#f5a524;border-color:#f5a524;color:#1b1204}
.tour .tour-x{padding:3px 10px;font-size:12.5px;color:#b6bfcc}
.tour button:disabled{opacity:.35;cursor:default}
.tour-ring{position:fixed;z-index:999;pointer-events:none;border:2px solid #f5a524;border-radius:14px;
           box-shadow:0 0 0 9999px rgba(5,6,8,.45);transition:opacity .25s cubic-bezier(0.22,1,0.36,1);opacity:0}
.tour-ring.snap{transition:none !important}
.top-r{display:flex;align-items:center;gap:12px}
.tour-go{font:inherit;font-size:13.5px;font-weight:600;border-radius:10px;padding:7px 14px;cursor:pointer;border:1px solid #f5a524;
         background:#f5a524;color:#1b1204}
@media (prefers-reduced-motion: reduce){.tour, .tour-ring, .tour-bar i{transition:none !important}.tour.out{transform:none}}
/* the active tab's outline was clipped by its row: give the row room */
.nav, .nav > div, .nav .tab-wrapper, .nav [role=tablist], .nav .tab-container{overflow:visible !important}
.nav [role=tablist], .nav .tab-container{padding:4px 2px 2px 3px !important}
"""

# terrain-style lines across the whole window: stronger behind the Overview headline, only at the edges on the data pages
TOPO_CSS = f"""
.gradio-container::before{{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background:url('{TOPO_URL}') center/cover no-repeat;opacity:.55;transition:opacity .8s ease;
  -webkit-mask-image:radial-gradient(ellipse 75% 75% at 50% 50%,transparent 35%,#000 85%);
  mask-image:radial-gradient(ellipse 75% 75% at 50% 50%,transparent 35%,#000 85%)}}
body.ov-on .gradio-container::before{{opacity:1;
  -webkit-mask-image:radial-gradient(ellipse 70% 80% at 25% 50%,#000 20%,transparent 75%);
  mask-image:radial-gradient(ellipse 70% 80% at 25% 50%,#000 20%,transparent 75%)}}
.gradio-container > *{{position:relative;z-index:1}}
"""
TOPO_JS = ("<script>setInterval(function(){var t=document.querySelector('[role=tab]');"
           "document.body.classList.toggle('ov-on',!!t&&t.getAttribute('aria-selected')==='true');},250);</script>")
TOUR_HEAD = TOUR_JS.replace("__STEPS__", json.dumps(tour_steps()).replace("</", "<\\/"))

with gr.Blocks(title="IGNIS Earth", theme=THEME, css=CSS + TOUR_CSS + TOPO_CSS, js=FORCE_DARK, head=MOTION_JS + TOUR_HEAD + TOPO_JS) as demo:
    gr.HTML("<div class='top'><div class='logo'><span class='mark'><i></i><i></i><i></i></span>IGNIS EARTH</div>"
            "<div class='top-r'><span class='state s-muted'>Student prototype, not an official fire warning</span>"
            "<button class='tour-go' type='button' data-tour-start>Guided demo</button></div></div>")
    with gr.Tabs(elem_classes="nav", selected="overview") as tabs:
        with gr.Tab("Overview", id="overview"):
            with gr.Row(elem_classes="ov"):
                with gr.Column(scale=5, min_width=360):
                    gr.HTML("<span class='ov-pill'><i></i>Veritas 2026 &nbsp;/&nbsp; AI + Climate</span>"
                            "<h1 class='ov-h'>Wildfires don't wait for morning.</h1>"
                            "<p class='ov-sub'>IGNIS Earth shows where a fire is likely to spread <b>tomorrow</b>, "
                            "and says when not to trust it.</p>"
                            "<p class='ov-p'>Every evening at 6 PM it scores the 5 km squares around each Greek wildfire, "
                            "explains each score in plain words, and flags days too unusual to rely on.</p>")
                    with gr.Row(elem_classes="ov-cta"):
                        go_brief = gr.Button("Open tonight's briefing", variant="primary", size="lg", scale=0, min_width=230)
                        go_res = gr.Button("See the results", variant="secondary", size="lg", scale=0, min_width=170)
                        gr.Button("Guided demo", variant="secondary", size="lg", scale=0, min_width=150, elem_id="ig-tour-start")
                    gr.HTML("<p class='ov-note'>Trained on 2014-2022, tested once on the 2024-2025 seasons, "
                            "built only on free NASA satellite and open weather data.</p>")
                with gr.Column(scale=6, min_width=360):
                    gr.HTML(OVERVIEW_MAP)
            gr.HTML(OVERVIEW_RESULT)
        with gr.Tab("Forecast", id="forecast"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes="side"):
                    fire = gr.Dropdown(FIRES, value=START, elem_id="ig-fire", label=f"Pick a real fire ({SEASON_LABEL.split(' (')[0]} seasons)")
                    gr.HTML("<div class='side-note'>Each one is a real wildfire from seasons the model never saw. "
                            "Pick one to load its 6 PM forecast.</div>"
                            "<div class='side-t'>Examples worth seeing</div>"
                            "<div class='side-note'>Our best day, the biggest spread, a clear miss and a false alarm.</div>")
                    ex_btns, ex_presets = [], []
                    for p in PRESETS:
                        if p.get("fire_id") not in FIRE_IDS:
                            continue
                        t, d = EXAMPLES.get(p["label"], (p["label"], ""))
                        ex_btns.append(gr.Button(t, size="sm", elem_classes="chip")); ex_presets.append(p)
                        gr.HTML(f"<div class='chip-d'>{d}</div>")
                    gr.HTML("<div class='side-t'>How to read the map</div><div class='legend'>"
                            "<div><span class='dot' style='background:#ff6a2b'></span>Burning tonight</div>"
                            "<div><span class='dot' style='background:#8a94a6'></span>Burned in the days before</div>"
                            "<div><span class='sw' style='background:rgba(217,56,30,.8)'></span>Highest risk tomorrow</div>"
                            "<div><span class='sw' style='background:rgba(245,165,36,.6)'></span>Medium risk</div>"
                            "<div><span class='sw' style='background:rgba(251,191,36,.35)'></span>Lower risk</div>"
                            "<div><span class='ln'></span>Tonight's wind</div>"
                            "<div><span class='sw' style='border:2px solid #f1ebe1'></span>Really burned (after you reveal)</div>"
                            "</div><div class='side-sub'>Hover the map for each spot's chance and reasons.</div>")
                with gr.Column(scale=4):
                    gr.HTML("<div class='vh'><h1>Where will this fire spread tomorrow?</h1>"
                            "<p>Each evening at 6 PM, IGNIS Earth scores every 5 km square around a fire for tomorrow. "
                            "These are real fires from seasons the model never saw.</p>" + HERO_STATS + "</div>")
                    gate = gr.HTML(elem_id="ig-gate")
                    brief = gr.HTML(elem_id="ig-brief")
                    revealed = gr.State(False)
                    with gr.Row():
                        with gr.Column(scale=3):
                            fmap = gr.HTML(elem_id="ig-map")
                            reveal_btn = gr.Button("Show what really happened", variant="primary", size="lg")
                            score = gr.HTML()
                        with gr.Column(scale=2):
                            villages = gr.Markdown()
                            top5 = gr.Dataframe(label="The 5 squares most likely to burn tomorrow", interactive=False, wrap=True, elem_id="ig-top5")
                            rank = gr.Radio([1, 2, 3, 4, 5], value=1, label="Why is this square at risk? Pick one from the table")
                            why_plot = gr.HTML(elem_id="ig-why")
        with gr.Tab("How it works") as tab2:
            gr.HTML("<div class='vh'><h1>How does IGNIS Earth make a forecast?</h1>"
                    "<p>A simple, readable model. Every forecast comes with its reasons and a check on whether to trust it.</p>"
                    "<div class='statline'><span><b class='mono'>2014-2022</b> learned from</span>"
                    "<span><b class='mono'>2023</b> tuned on</span><span><b class='mono'>2024-2025</b> tested once</span></div></div>")
            gr.HTML("<div class='steps'>"
                    "<div><b>1</b>NASA satellites spot active fires</div>"
                    "<div><b>2</b>Greece is cut into 5 km squares</div>"
                    "<div><b>3</b>Each square near a fire gets distance, wind, slope and weather</div>"
                    "<div><b>4</b>A simple model scores tomorrow's risk, with reasons</div>"
                    "<div><b>5</b>A trust check flags unusual days</div></div>")
            with gr.Row():
                with gr.Column(elem_classes="panel"):
                    explain_line("Red = raises the risk when it goes up. Grey = lowers it.")
                    gr.HTML(chart(coef_figure(), "coef"))
                with gr.Column(elem_classes="panel", elem_id="ig-trust"):
                    explain_line("Lower is better. On normal days (left) the model makes smaller errors than on unusual days (right).")
                    cov = gr.Slider(10, 100, value=50, step=5, label="Only trust forecasts on days similar to past fires (%)")
                    rc_plot = gr.HTML()
                    rc_txt = gr.Markdown()
            gr.Markdown("<div class='explain'><b>Limits:</b> smoke and clouds can hide fires, field burning can look like wildfire, "
                        "and the model doesn't know where firefighters went. It ranks risk; it doesn't predict the fire's exact path.</div>")
        with gr.Tab("Results", id="results") as tab3:
            gr.HTML("<div class='vh'><h1>Does it beat the simple wind rule?</h1>"
                    "<p>Tested once on the 2024-2025 fire seasons, never seen while building the model. "
                    "The <b>wind rule</b> warns every square just downwind of the fire.</p>" + HERO_STATS + "</div>")
            metric = gr.Radio(list(METRICS), value="Caught in the top 10", label="Show")
            with gr.Row():
                with gr.Column():
                    metric_help = gr.Markdown(f"<div class='explain'>{METRIC_HELP['Caught in the top 10']}</div>")
                    sc_plot = gr.HTML(score_figure("spread", "Caught in the top 10"), elem_id="ig-bars")
                with gr.Column(elem_classes="panel"):
                    explain_line("Dark grey = new fire that day. Amber = warned by our model. Light grey = warned by the wind rule.")
                    gr.HTML(chart(timeline_figure(), "timeline"))
            gr.Markdown("<div class='explain'>Data: NASA fire satellites (FIRMS), Open-Meteo weather, Copernicus elevation, GeoNames places. "
                        "Map tiles © OpenStreetMap contributors, © CARTO. Code drafted with help from Claude (Anthropic), reviewed by the author.</div>")

    outs = [fmap, gate, brief, top5, villages, score, why_plot]
    HIDE, SHOW = "Hide what really happened", "Show what really happened"

    def pick(fid):
        return (*show(fid, False), False, gr.update(value=SHOW), 1)

    def toggle(fid, rev):
        # only the map and the result card change, so the briefing, table and chart stay still instead of re-animating
        rev = not rev
        m, _, _, _, _, sc, _ = show(fid, rev)
        return (m, sc, rev, gr.update(value=HIDE if rev else SHOW))
    fire.change(pick, fire, outs + [revealed, reveal_btn, rank], show_progress="minimal")
    # only the map, the result line and the button change; no loading overlay, so nothing else dims or flickers
    reveal_btn.click(toggle, [fire, revealed], [fmap, score, revealed, reveal_btn], show_progress="hidden")
    rank.change(explain, [fire, rank], why_plot, show_progress="hidden")
    cov.change(riskcov, cov, [rc_plot, rc_txt], show_progress="hidden")
    metric.change(lambda m: score_figure("spread", m), metric, sc_plot, show_progress="hidden")
    metric.change(lambda m: f"<div class='explain'>{METRIC_HELP[m]}</div>", metric, metric_help, show_progress="hidden")
    demo.load(lambda fid: show(fid, False), fire, outs)
    demo.load(riskcov, cov, [rc_plot, rc_txt])
    # charts in a hidden tab are drawn too narrow; a resize once the tab is shown lets them fill their box
    for t in (tab2, tab3):
        t.select(None, None, None, js="() => { setTimeout(() => window.dispatchEvent(new Event('resize')), 60); }")
    go_brief.click(lambda: gr.Tabs(selected="forecast"), None, tabs)
    go_res.click(lambda: gr.Tabs(selected="results"), None, tabs)
    for b, p in zip(ex_btns, ex_presets):
        b.click(lambda p=p: p["fire_id"], None, fire)

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
