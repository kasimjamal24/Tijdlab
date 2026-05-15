# IMPORTS
import os
from glob import glob
import joblib  # voor opslaan/laden van het getrainde model
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
import geopandas as gpd
from shapely.geometry import shape, LineString
import cv2
from skimage.morphology import skeletonize as skimage_skeletonize
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# INPUT PADEN
# ============================================================

# Map met kaarten waarvan het model leert (trainingskaarten)
TRAINING_DIR = r"D:\QGIS\Tijdlab\Python\trainingskaarten"

# Map met kaarten die automatisch gedetecteerd moeten worden
DETECTIE_DIR = r"D:\QGIS\Tijdlab\Python\detectiekaarten"

# .GPKG bestanden ophalen
GT_GPKG    = r"D:\QGIS\Tijdlab\Data\Ground_Truth\Elementen.gpkg"
SOIL_GPKG  = r"D:\QGIS\Tijdlab\Data\Bodem\BRO-SGM-Bodemkaart-V2025-01_1.gpkg"
GEOL_VLAK  = r"D:\QGIS\Tijdlab\Data\Geologie\GKNederlandGeolVlak.gpkg" 
OUTPUT_ROOT = r"D:\QGIS\Tijdlab\Python\outputs"
GEOL_LIJN  = r"D:\QGIS\Tijdlab\Data\Geologie\GKNederlandGeolLijn.gpkg"  
MODEL_PATH  = os.path.join(OUTPUT_ROOT, "rf_model.joblib")

# Optionele laagnamen voor de Ground Truth GPKG
GT_LAYER_ROADS = "Wegen"
GT_LAYER_WATER = "Water"
GT_LAYER_PERCS = "percelen_fixed"

# Optionele configuratie voor attribuutkolommen (laat op None om eerste niet-geometry kolom te gebruiken)
SOIL_KEY_COL: str | None = None
GEOV_KEY_COL: str | None = None
GEOL_KEY_COL: str | None = None
MAX_SIZE = 2500  # downsample grote kaarten voor snelheid/consistentie

# ============================================================
# MODEL + PIPELINE INSTELLINGEN
# ============================================================
RANDOM_STATE = 42
N_ESTIMATORS = 250

# Two-stage: line gating threshold (hoe streng "is dit een lijn?")
LINE_THR = 0.38

# Type thresholds binnen line pixels (per klasse)
TYPE_THR = {
    1: 0.62,  # wegen
    2: 0.78,  # water (strenger tegen false positives)
    3: 0.45,  # percelen — verhoogd om false positives te verminderen
}

# Percelen edge-gating
PERC_EDGE_THR = 0.045

# Sampling
SAMPLE_BG = 30000
SAMPLE_ROAD = 35000
SAMPLE_WATER = 20000
SAMPLE_PERC = 70000    
# Postprocessing per klasse
POST_ROAD  = dict(close_ksize=3, open_ksize=3, min_pixels=60,  bridge_len_px=11)
POST_WATER = dict(close_ksize=5, open_ksize=3, min_pixels=70,  bridge_len_px=9)
POST_PERC  = dict(close_ksize=3, open_ksize=3, min_pixels=35,  bridge_len_px=7)

# Vectorisatie filters
MIN_AREA_POLY = 25
MIN_LEN_SKELETON_PX = 30

# Feature-indices in X voor gating.
# Volgorde in build_features_from_rgb: base(7), extra_color(4), extra_geo(2), multiscale(3), line_density(2), gabor(4) edge_strength zit in extra_geo op positie 1 => absolute index 12.
EDGE_FEAT_IDX = 12

# ============================================================
# IMAGE HELPERS
# ============================================================
def to_uint8(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Schaal willekeurige numerieke array naar uint8 [0..255].

    Voor 3D arrays (bijv. RGB) wordt per kanaal geschaald zodat
    kleurverhoudingen stabieler blijven voor feature-extractie.
    Optionele vmin/vmax overschrijven de automatische schaling.
    """
    if arr.dtype == np.uint8 and vmin is None and vmax is None:
        return arr
    a = arr.astype(np.float32)

    # Per kanaal schalen voor 3D input (bijv. H x W x 3 RGB)
    if a.ndim == 3 and vmin is None and vmax is None:
        out = np.empty_like(a)
        for c in range(a.shape[2]):
            ch = a[..., c]
            lo = float(np.nanmin(ch))
            hi = float(np.nanmax(ch))
            mx = hi - lo
            if mx <= 0 or np.isnan(mx):
                mx = 1.0
            out[..., c] = np.clip((ch - lo) / mx * 255.0, 0, 255)
        return out.astype(np.uint8)

    # Standaard: globale schaling (voor 2D arrays of als vmin/vmax opgegeven)
    lo = float(np.nanmin(a)) if vmin is None else float(vmin)
    hi = float(np.nanmax(a)) if vmax is None else float(vmax)
    a -= lo
    mx = hi - lo
    if mx <= 0 or np.isnan(mx):
        mx = 1.0
    a = (a / mx) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)

def preprocess(gray_u8: np.ndarray,
               denoise_strength: int = 3,
               clahe_clip: float = 2.0,
               blur_ksize: int = 3) -> np.ndarray:
    """Stabiele preprocessing voor historische scans.

    BELANGRIJK voor perceelgrenzen: de originele parameters (h=7, blur=3)
    verwijderden dunne potloodlijnen (1-2px) vóór feature-extractie.
    Oplossing: zwakkere denoise, fijnere CLAHE-tegel, geen blur.
    """
    # Voor dunne perceellijnen houdt een lage h-details beter overeind. Parameter blijft nu expliciet en voorspelbaar (zonder stille override).
    h_val = max(1, int(denoise_strength))
    g = cv2.fastNlMeansDenoising(gray_u8, h=h_val)
    # Fijnere CLAHE-tegels: 16x16 ipv 8x8 → meer lokaal contrast bij dunne lijnen
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(16, 16))
    g = clahe.apply(g)
    # Geen Gaussian blur: smeert dunne perceellijnen onherstelbaar weg (blur_ksize parameter genegeerd voor perceellijnen behoud)
    return g

def build_base_features(gray_u8: np.ndarray) -> np.ndarray:
    """Bouw basisfeatures op grijsbeeld (intensiteit, randen, lokale statistiek)."""
    g = gray_u8.astype(np.float32) / 255.0
    gx  = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    mean = cv2.blur(g, (7, 7))
    sq   = cv2.blur(g * g, (7, 7))
    std  = np.sqrt(np.maximum(sq - mean * mean, 0))
    feats = np.stack([g, gx, gy, mag, lap, mean, std], axis=-1)
    return feats.reshape(-1, feats.shape[-1]).astype(np.float32)

def make_ink_mask(gray_u8: np.ndarray, is_preprocessed: bool = False) -> np.ndarray:
    """Maak een ruwe binaire inktmasker voor lijndikte-features.

    Parameters
    ----------
    is_preprocessed : bool
        Als True wordt preprocessing overgeslagen omdat de input al
        bewerkt is. Voorkomt dubbele filtering wanneer de grijslaag
        al eerder gepreprocessed is (bijv. in build_features_from_rgb).
    """
    g = gray_u8 if is_preprocessed else preprocess(gray_u8, denoise_strength=3, clahe_clip=2.0, blur_ksize=3)
    bw = cv2.adaptiveThreshold(
        g, 255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,  
        C=3             
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return (bw > 0).astype(np.uint8)

def edge_strength(gray_u8: np.ndarray, is_preprocessed: bool = False) -> np.ndarray:
    """Bepaal randsterkte (0..1) met Canny; goed voor perceel-randdetectie.

    Parameters
    ----------
    is_preprocessed : bool
        Als True wordt preprocessing overgeslagen omdat de input al
        bewerkt is. Voorkomt dubbele filtering wanneer de grijslaag
        al eerder gepreprocessed is (bijv. in build_features_from_rgb).
    """
    g = gray_u8 if is_preprocessed else preprocess(gray_u8, denoise_strength=3, clahe_clip=2.0, blur_ksize=3)
    edges = cv2.Canny(g, 15, 60)   
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges.astype(np.float32) / 255.0

def gabor_features(gray_f32: np.ndarray) -> np.ndarray:
    """Gabor-filters op 4 richtingen — detecteert rechte perceelgrenzen gericht."""
    responses = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kern = cv2.getGaborKernel(
            (15, 15), sigma=2.0, theta=theta,
            lambd=8.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(gray_f32, cv2.CV_32F, kern)
        responses.append(np.abs(resp))
    return np.stack(responses, axis=-1)  # H x W x 4

def multiscale_edge_features(gray_u8: np.ndarray) -> np.ndarray:
    """Sobel-randsterkte op 3 schalen — kleine schaal vangt dunne perceellijnen,
    grote schaal vangt wegen/water. Geeft het model schaalinformatie mee."""
    g = gray_u8.astype(np.float32) / 255.0
    results = []
    for ksize in [3, 5, 9]:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = np.sqrt(gx * gx + gy * gy)
        results.append(mag)
    return np.stack(results, axis=-1)  # H x W x 3

def line_density_features(gray_u8: np.ndarray) -> np.ndarray:
    """Lokale lijnpixeldichtheid op 2 venstergroottes.

    Perceelgrenzen vormen een regelmatig raster: in een 15x15 venster rond
    een perceelpixel zijn er altijd meerdere andere lijnpixels. Dit onderscheidt
    ze van geïsoleerde vlekken/ruis op de kaart.
    """
    edges_low = cv2.Canny(gray_u8, 10, 40).astype(np.float32) / 255.0
    density_15 = cv2.blur(edges_low, (15, 15))
    density_31 = cv2.blur(edges_low, (31, 31))
    return np.stack([density_15, density_31], axis=-1)  # H x W x 2

def build_features_from_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """Bouw de volledige featurevector per pixel op basis van RGB en afgeleiden.

    Feature-groepen:
      - base (7):          intensiteit, Sobel x/y, gradient, Laplacian, mean, std
      - extra_color (4):   HSV saturatie + helderheid, lokale gemiddeldes
      - extra_geo (2):     inkt-afstandstransformatie, Canny randsterkte
      - multiscale (3):    Sobel-magnitude op schalen 3/5/9px
      - line_density (2):  lokale randpixeldichtheid in 15x15 en 31x31 venster
      - gabor (4):         richtingsfilters 0/45/90/135 graden
    Totaal: 22 features per pixel.
    """
    gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray_pre = preprocess(gray_u8)

    base = build_base_features(gray_pre).reshape(gray_pre.shape[0], gray_pre.shape[1], -1)

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sat = hsv[..., 1]
    val = hsv[..., 2]
    sat_mean = cv2.blur(sat, (11, 11))
    val_mean = cv2.blur(val, (11, 11))
    extra_color = np.stack([sat, val, sat_mean, val_mean], axis=-1)

    ink = make_ink_mask(gray_pre, is_preprocessed=True)
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dist_norm = dist / (dist.max() + 1e-6)
    ed = edge_strength(gray_pre, is_preprocessed=True)
    extra_geo = np.stack([dist_norm, ed], axis=-1)

    # Nieuwe features voor betere perceeldetectie
    ms = multiscale_edge_features(gray_pre)          # H x W x 3
    ld = line_density_features(gray_pre)              # H x W x 2
    gray_f32 = gray_pre.astype(np.float32) / 255.0
    gabor = gabor_features(gray_f32)                  # H x W x 4

    feats = np.concatenate([base, extra_color, extra_geo, ms, ld, gabor], axis=-1)
    return feats.reshape(-1, feats.shape[-1]).astype(np.float32)

# ============================================================
# MORPHOLOGIE + CLEANUP
# ============================================================
def remove_small_components(binmask_u8: np.ndarray, min_pixels: int = 120) -> np.ndarray:
    """Verwijder kleine verbonden componenten onder een gegeven drempel."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binmask_u8 > 0).astype(np.uint8), connectivity=8
    )
    out = np.zeros_like(binmask_u8, dtype=np.uint8)
    for lab in range(1, num):
        if stats[lab, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == lab] = 255
    return out

def bridge_linear_gaps(binmask_u8: np.ndarray, length_px: int) -> np.ndarray:
    """Brug kleine gaten in horizontale/verticale richting."""
    if length_px < 3:
        return binmask_u8
    L = max(3, int(length_px))
    if L % 2 == 0:
        L += 1
    out = binmask_u8.copy()
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)),
    ]
    for k in kernels:
        out = cv2.max(out, cv2.morphologyEx(binmask_u8, cv2.MORPH_CLOSE, k, iterations=1))
    return out

def clean_class_mask(binmask_u8: np.ndarray,
                     close_ksize: int,
                     open_ksize: int,
                     min_pixels: int,
                     bridge_len_px: int = 0) -> np.ndarray:
    """Opschonen van een binaire mask voor een bepaalde klasse."""
    out = (binmask_u8 > 0).astype(np.uint8) * 255
    if bridge_len_px > 0:
        out = bridge_linear_gaps(out, bridge_len_px)
    if close_ksize >= 3:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    if open_ksize >= 3:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=1)
    return remove_small_components(out, min_pixels=min_pixels)

def refine_prediction(pred: np.ndarray) -> np.ndarray:
    """Combineer ruwe klasses tot opgeschoonde klasses met vaste prioriteit."""
    roads = clean_class_mask((pred == 1).astype(np.uint8) * 255, **POST_ROAD)
    water = clean_class_mask((pred == 2).astype(np.uint8) * 255, **POST_WATER)
    percs = clean_class_mask((pred == 3).astype(np.uint8) * 255, **POST_PERC)
    refined = np.zeros_like(pred, dtype=np.uint8)
    refined[percs > 0] = 3
    refined[water > 0] = 2
    refined[roads > 0] = 1
    return refined

# ============================================================
# METRICS
# ============================================================
def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """Bereken per-klasse precision/recall/f1 + confusion matrix."""
    K = len(labels)
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    t = np.vectorize(lab_to_i.get)(y_true)
    p = np.vectorize(lab_to_i.get)(y_pred)
    cm = np.bincount(K * t + p, minlength=K * K).reshape(K, K)
    rows = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        rows.append((lab, prec, rec, f1, int(cm[i, :].sum())))
    df = pd.DataFrame(rows, columns=["class_id", "precision", "recall", "f1", "support_px"])
    return df, cm

# ============================================================
# FOUTKAART
# ============================================================
def schrijf_foutkaart(pred: np.ndarray,
                      mask: np.ndarray,
                      out_path: str,
                      profile: dict,
                      crs,
                      transform) -> None:
    """Maak en sla een visuele foutkaart op als GeoTIFF."""
    fout = np.zeros_like(pred, dtype=np.uint8)
    fout[(pred > 0) & (mask > 0)] = 1
    fout[(pred > 0) & (mask == 0)] = 2
    fout[(pred == 0) & (mask > 0)] = 3
    write_singleband_tif(out_path, fout, profile, crs, transform)

def schrijf_validatierapport(dfm: pd.DataFrame,
                             cm: np.ndarray,
                             out_dir: str,
                             kaartnaam: str,
                             is_validatie: bool) -> None:
    """Sla metrics op als leesbaar tekstrapport voor gebruik in thesis."""
    rol = "VALIDATIE (testkaart)" if is_validatie else "TRAINING (kaart ook gebruikt voor training)"
    rapport_pad = os.path.join(out_dir, f"{kaartnaam}_validatierapport.txt")
    klasse_namen = {0: "Achtergrond", 1: "Wegen/paden", 2: "Watergangen", 3: "Perceelgrenzen"}
    with open(rapport_pad, "w", encoding="utf-8") as f:
        f.write(f"VALIDATIERAPPORT — {kaartnaam}\n")
        f.write(f"Rol van deze kaart: {rol}\n")
        f.write("=" * 60 + "\n\n")
        f.write("PER-KLASSE STATISTIEKEN\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Klasse':<20} {'Precisie':>10} {'Recall':>10} {'F1-score':>10} {'Pixels':>10}\n")
        f.write("-" * 60 + "\n")
        for _, row in dfm.iterrows():
            naam = klasse_namen.get(int(row["class_id"]), str(int(row["class_id"])))
            f.write(
                f"{naam:<20} {row['precision']:>10.3f} {row['recall']:>10.3f} "
                f"{row['f1']:>10.3f} {int(row['support_px']):>10,}\n"
            )
        f.write("\nCONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        labels = ["Achtergrond", "Wegen", "Water", "Percelen"]
        header = f"{'':>12}" + "".join(f"{l:>12}" for l in labels)
        f.write(header + "\n")
        for i, label in enumerate(labels):
            rij = f"{label:>12}" + "".join(f"{cm[i, j]:>12}" for j in range(len(labels)))
            f.write(rij + "\n")
        f.write("\nTOELICHTING\n")
        f.write("-" * 60 + "\n")
        f.write("Precisie : van alle pixels die het model als lijn aanmerkt,\n")
        f.write("           welk deel was ook echt een lijn? (laag = veel false positives)\n")
        f.write("Recall   : van alle echte lijn-pixels, welk deel heeft het model\n")
        f.write("           gevonden? (laag = veel gemiste structuren)\n")
        f.write("F1-score : harmonisch gemiddelde van precisie en recall.\n")
        f.write("           Hogere score = beter overall.\n")
        f.write(f"\nFoutkaart is opgeslagen als: {kaartnaam}_foutkaart.tif\n")
    print(f"  Validatierapport opgeslagen: {rapport_pad}")

# ============================================================
# SKELETON -> LIJNEN (perceelgrenzen)
# ============================================================
def skeletonize_mask(binmask_u8: np.ndarray) -> np.ndarray:
    """Maak een dun skeleton (1 pixel) van een dikke binaire lijnmasker."""
    bool_mask = (binmask_u8 > 0)
    skel = skimage_skeletonize(bool_mask)
    return skel.astype(np.uint8) * 255

def trace_skeleton_to_lines(skel_u8: np.ndarray, transform, min_len_px: int = 60) -> list[LineString]:
    """Volg skeleton-pixels en bouw er LineString-geometrieën van.

    Ondersteunt zowel open lijnen (met endpoints) als gesloten lussen
    zonder endpoints.
    """
    sk = (skel_u8 > 0)
    H, W = sk.shape
    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0,  -1),           (0,  1),
            (1,  -1),  (1, 0),  (1,  1)]

    def neighbors(r, c):
        return [
            (r + dr, c + dc)
            for dr, dc in nbrs
            if 0 <= r + dr < H and 0 <= c + dc < W and sk[r + dr, c + dc]
        ]
    coords = np.argwhere(sk)
    deg = {(r, c): len(neighbors(r, c)) for r, c in coords}
    endpoints = [p for p, d in deg.items() if d == 1]
    all_nodes = [tuple(rc) for rc in coords]
    visited = set()
    lines = []

    def pix_to_xy(r, c):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        return (x, y)
    starts = endpoints + [p for p in all_nodes if p not in set(endpoints)]
    for start in starts:
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        cur = start
        prev = None
        while True:
            nbs = neighbors(*cur)
            nxt = None
            for nb in nbs:
                if nb == prev:
                    continue
                if nb not in visited or deg.get(nb, 0) >= 3:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev = cur
            cur = nxt
            path.append(cur)
            if cur == start and len(path) > 2:
                # Gesloten lus geraakt; stop zodat dit segment eenmaal wordt opgeslagen.
                break
            visited.add(cur)
            if deg.get(cur, 0) == 1 or deg.get(cur, 0) >= 3:
                break
        if len(path) >= min_len_px:
            coords_xy = [pix_to_xy(r, c) for (r, c) in path]
            coords_xy2 = [coords_xy[0]]
            for p in coords_xy[1:]:
                if p != coords_xy2[-1]:
                    coords_xy2.append(p)
            if len(coords_xy2) >= 2:
                lines.append(LineString(coords_xy2))
    return lines

def sample_training_pixels(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Neem per kaart een gebalanceerde subsample voor modeltraining."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    idx2 = np.where(y == 2)[0]
    idx3 = np.where(y == 3)[0]

    def take(idx: np.ndarray, nmax: int) -> np.ndarray:
        n = min(nmax, idx.size)
        return rng.choice(idx, size=n, replace=False) if n > 0 else np.array([], dtype=int)

    sel = np.concatenate([
        take(idx0, SAMPLE_BG),
        take(idx1, SAMPLE_ROAD),
        take(idx2, SAMPLE_WATER),
        take(idx3, SAMPLE_PERC),
    ])
    if sel.size == 0:
        return np.empty((0, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    return X[sel], y[sel].astype(np.uint8, copy=False)

# ============================================================
# RAPPORTAGE HELPER
# ============================================================
def maak_nette_tabel(df: pd.DataFrame, element_naam: str, categorie_kolom: str) -> pd.DataFrame:
    """Maak een nette frequentietabel met aantallen en percentages."""
    if categorie_kolom not in df.columns:
        return pd.DataFrame(columns=["element", "categorie", "aantal", "percentage"])
    df = df.dropna(subset=[categorie_kolom]).copy()
    df = df.rename(columns={"class": "element", categorie_kolom: "categorie", "count": "aantal"})
    df["element"] = element_naam
    df["aantal"] = pd.to_numeric(df["aantal"], errors="coerce").fillna(0).astype(int)
    totaal = int(df["aantal"].sum())
    df["percentage"] = np.where(
        totaal > 0,
        (df["aantal"] / totaal * 100).round(1),
        0.0
    )
    df = df.sort_values("aantal", ascending=False)
    return df[["element", "categorie", "aantal", "percentage"]]

# ============================================================
# GROUND TRUTH RASTERISATIE
# ============================================================
def rasteriseer_met_buffer(gdf: gpd.GeoDataFrame,
                           out_shape: tuple[int, int],
                           transform,
                           buf: float,
                           burn_val: int) -> np.ndarray:
    """Rasteriseer vectorgeometrie met een buffer naar een binaire rasterlaag."""
    geoms = [
        (g.buffer(buf, cap_style=2, join_style=2), burn_val)
        for g in gdf.geometry
        if g is not None
    ]
    if not geoms:
        return np.zeros(out_shape, dtype=np.uint8)
    return rasterize(geoms, out_shape, transform=transform, fill=0, dtype=np.uint8)

def build_ground_truth_mask(roads: gpd.GeoDataFrame,
                            water: gpd.GeoDataFrame,
                            percs: gpd.GeoDataFrame,
                            crs,
                            transform,
                            H: int,
                            W: int) -> np.ndarray:
    """Bouw ground-truth mask op (0 = achtergrond, 1/2/3 = lijnklasses)."""
    roads_crs = roads.to_crs(crs) if roads.crs != crs else roads
    water_crs = water.to_crs(crs) if water.crs != crs else water
    percs_crs = percs.to_crs(crs) if percs.crs != crs else percs
    px = abs(transform.a)
    m1 = rasteriseer_met_buffer(roads_crs, (H, W), transform, px * 1.2, 1)
    m2 = rasteriseer_met_buffer(water_crs, (H, W), transform, px * 1.8, 2)
    m3 = rasteriseer_met_buffer(percs_crs, (H, W), transform, px * 1.2, 3)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[m3 > 0] = 3
    mask[m2 > 0] = 2
    mask[m1 > 0] = 1
    return mask

# ============================================================
# FEATURE + LABEL EXTRACTIE
# ============================================================
def extract_features_and_labels(
    tif_pad: str,
    roads: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    percs: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lees één GeoTIFF en geef features (X), labels (y) en metadata terug."""
    with rasterio.open(tif_pad) as src:
        h, w = src.height, src.width
        if max(h, w) > MAX_SIZE:
            scale = MAX_SIZE / float(max(h, w))
            new_h, new_w = int(h * scale), int(w * scale)
            data = src.read(out_shape=(src.count, new_h, new_w), resampling=Resampling.bilinear)
            transform = src.transform * src.transform.scale((w / new_w), (h / new_h))
        else:
            data = src.read()
            transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
    if crs is None:
        raise ValueError(f"Kaart heeft geen CRS/georeferentie: {tif_pad}")
    rgb = np.stack([data[0]] * 3, axis=-1) if data.shape[0] < 3 else np.stack(data[:3], axis=-1)
    rgb_u8 = to_uint8(rgb)
    H, W = rgb_u8.shape[:2]
    mask = build_ground_truth_mask(roads, water, percs, crs, transform, H, W)
    X = build_features_from_rgb(rgb_u8)
    y = mask.reshape(-1)
    meta = {
        "crs": crs,
        "transform": transform,
        "profile": profile,
        "H": H,
        "W": W,
        "rgb_u8": rgb_u8,
    }
    return X, y, meta

# ============================================================
# CLASSIFICATIE: TWO-STAGE (LINE GATE + TYPE)
# ============================================================
def predict_two_stage(rf: RandomForestClassifier, X: np.ndarray, H: int, W: int) -> np.ndarray:
    """Eerst beslissen of een pixel een lijn is, daarna welk type lijn."""
    proba = rf.predict_proba(X)
    classes = rf.classes_.astype(np.uint8)
    p = {int(c): proba[:, i] for i, c in enumerate(classes)}
    p1 = p.get(1, np.zeros(proba.shape[0], dtype=np.float32))
    p2 = p.get(2, np.zeros(proba.shape[0], dtype=np.float32))
    p3 = p.get(3, np.zeros(proba.shape[0], dtype=np.float32))
    p_line = p1 + p2 + p3
    is_line = p_line >= LINE_THR
    scores   = np.stack([p1, p2, p3], axis=1)
    best_idx = np.argmax(scores, axis=1)
    best_class = np.array([1, 2, 3], dtype=np.uint8)[best_idx]
    best_score = scores[np.arange(scores.shape[0]), best_idx]
    ok = np.zeros_like(is_line, dtype=bool)
    for cls in (1, 2, 3):
        cls_mask = (best_class == cls)
        ok |= (cls_mask & (best_score >= TYPE_THR[cls]))
    edge_feat = X[:, EDGE_FEAT_IDX]
    ok &= ~((best_class == 3) & (edge_feat < PERC_EDGE_THR))
    pred_flat = np.zeros(proba.shape[0], dtype=np.uint8)
    keep = is_line & ok
    pred_flat[keep] = best_class[keep]
    return pred_flat.reshape(H, W).astype(np.uint8)

# ============================================================
# TRAINING (SAMPLING)
# ============================================================
def fit_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train RandomForest op een reeds gesamplede trainingsset."""
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X, y)
    return rf

# ============================================================
# EXPORT HELPERS
# ============================================================
def write_singleband_tif(path: str, arr_u8: np.ndarray, profile: dict, crs, transform):
    """Schrijf een enkelbands GeoTIFF met de juiste georeferentie."""
    prof = profile.copy()
    prof.update(count=1, dtype=rasterio.uint8,
                height=arr_u8.shape[0], width=arr_u8.shape[1],
                transform=transform, crs=crs)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr_u8.astype(np.uint8), 1)

def vectorize_class(pred: np.ndarray, cid: int, transform, crs, min_area: float) -> list:
    """Vectoriseer een bepaalde klasse uit het predictieraster naar polygonen."""
    binmask = (pred == cid).astype(np.uint8)
    geoms = []
    for geom, val in shapes(binmask, mask=binmask.astype(bool), transform=transform):
        if val != 1:
            continue
        s = shape(geom)
        if s.area >= min_area:
            geoms.append(s)
    return geoms

# ============================================================
# EXPORT + ANALYSE VAN ÉÉN KAART
# ============================================================
def exporteer_minuutplan(
    tif_pad: str,
    rf: RandomForestClassifier,
    roads: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    percs: gpd.GeoDataFrame,
    soil: gpd.GeoDataFrame,
    geov: gpd.GeoDataFrame,
    geol: gpd.GeoDataFrame,
    is_validatie: bool = False,
) -> None:
    """Volledige predict + export pipeline voor één GeoTIFF."""
    kaartnaam = os.path.splitext(os.path.basename(tif_pad))[0]
    out_dir = os.path.join(OUTPUT_ROOT, kaartnaam)
    os.makedirs(out_dir, exist_ok=True)
    print("\n==============================")
    rol_label = "[VALIDATIE]" if is_validatie else "[TRAINING]"
    print(f"Bezig met kaart: {kaartnaam}  {rol_label}")
    try:
        X, y, meta = extract_features_and_labels(tif_pad, roads, water, percs)
    except ValueError as e:
        print("OVERGESLAGEN:", e)
        return
    crs       = meta["crs"]
    transform = meta["transform"]
    profile   = meta["profile"]
    H, W      = meta["H"], meta["W"]
    mask      = y.reshape(H, W)

    # Predict
    pred_raw = predict_two_stage(rf, X, H, W)

# Sla ruwe modeloutput op voor procesanalyse / thesis
    write_singleband_tif(
        os.path.join(out_dir, "pred_raw.tif"),
        pred_raw.astype(np.uint8),
        profile, crs, transform
    )

# Postprocessing
    pred = refine_prediction(pred_raw)

# Sla opgeschoonde voorspelling op als tussenstap
    write_singleband_tif(
        os.path.join(out_dir, "pred_postprocessed.tif"),
        pred.astype(np.uint8),
        profile, crs, transform
    )

    # Bestandsnamen met kaartnaam
    metrics_path = os.path.join(out_dir, f"{kaartnaam}_metrics.csv")
    cm_path = os.path.join(out_dir, f"{kaartnaam}_confusion_matrix.txt")
    foutkaart_path = os.path.join(out_dir, f"{kaartnaam}_foutkaart.tif")
    gt_mask_path = os.path.join(out_dir, f"{kaartnaam}_ground_truth_mask.tif")
    pred_classes_path = os.path.join(out_dir, f"{kaartnaam}_pred_classes.tif")
    pred_wegen_path = os.path.join(out_dir, f"{kaartnaam}_pred_wegen_mask.tif")
    pred_water_path = os.path.join(out_dir, f"{kaartnaam}_pred_water_mask.tif")
    pred_percelen_path = os.path.join(out_dir, f"{kaartnaam}_pred_percelen_mask.tif")
    skeleton_path = os.path.join(out_dir, f"{kaartnaam}_perceel_skeleton.tif")
    gpkg_out = os.path.join(out_dir, f"{kaartnaam}_detected_vectors.gpkg")
    shp_wegen = os.path.join(out_dir, f"{kaartnaam}_wegen_paden.shp")
    shp_water = os.path.join(out_dir, f"{kaartnaam}_watergangen.shp")
    shp_percelen = os.path.join(out_dir, f"{kaartnaam}_perceelgrenzen_lijnen.shp")

    # Metrics
    dfm, cm = fast_metrics(y, pred.reshape(-1), [0, 1, 2, 3])
    dfm.to_csv(metrics_path, sep=";", encoding="utf-8-sig", index=False)
    np.savetxt(cm_path, cm.astype(int), fmt="%d")

    # Foutkaart
    schrijf_foutkaart(
        pred, mask,
        out_path=foutkaart_path,
        profile=profile, crs=crs, transform=transform
    )

    # Validatierapport
    schrijf_validatierapport(dfm, cm, out_dir, kaartnaam, is_validatie)

    # Raster exports
    write_singleband_tif(gt_mask_path, mask.astype(np.uint8), profile, crs, transform)
    write_singleband_tif(pred_classes_path, pred.astype(np.uint8), profile, crs, transform)
    debug_exports = [
        (1, pred_wegen_path),
        (2, pred_water_path),
        (3, pred_percelen_path),
    ]
    for cid, dbg_path in debug_exports:
        write_singleband_tif(dbg_path, (pred == cid).astype(np.uint8), profile, crs, transform)

    # Vectorisatie naar GeoPackage + shapefile
    if os.path.exists(gpkg_out):
        os.remove(gpkg_out)
    for cid, layer_name, shp_path in [
        (1, "wegen_paden", shp_wegen),
        (2, "watergangen", shp_water),
    ]:
        geoms = vectorize_class(pred, cid, transform, crs, min_area=MIN_AREA_POLY)
        if geoms:
            gdf = gpd.GeoDataFrame({"class": [layer_name] * len(geoms)}, geometry=geoms, crs=crs)
            gdf.to_file(gpkg_out, layer=layer_name, driver="GPKG")
            gdf.to_file(shp_path, driver="ESRI Shapefile")

    # Percelen als echte lijnen via skeleton
    bin3   = (pred == 3).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    bin3   = cv2.morphologyEx(bin3, cv2.MORPH_CLOSE, kernel, iterations=1)
    bin3   = cv2.morphologyEx(bin3, cv2.MORPH_OPEN,  kernel, iterations=1)
    bin3   = remove_small_components(bin3, min_pixels=35)
    skel3 = skeletonize_mask(bin3)
    write_singleband_tif(skeleton_path, (skel3 > 0).astype(np.uint8), profile, crs, transform)
    lines = trace_skeleton_to_lines(skel3, transform, min_len_px=MIN_LEN_SKELETON_PX)
    if lines:
        gdf_lines = gpd.GeoDataFrame({"type": ["perceelgrens"] * len(lines)}, geometry=lines, crs=crs)
        gdf_lines.to_file(gpkg_out, layer="perceelgrenzen_lijnen", driver="GPKG")
        gdf_lines.to_file(shp_percelen, driver="ESRI Shapefile")

    # Koppeling shapefiles aan bodem, geovlak en geolijn
    try:
        shp_layers = [
            ("wegen_paden", shp_wegen),
            ("watergangen", shp_water),
            ("perceelgrenzen_lijnen", shp_percelen),
        ]
        for layer, shp_path in shp_layers:
            if not os.path.exists(shp_path):
                continue

            det = gpd.read_file(shp_path)
            if det.empty:
                continue

            soil_ = soil.to_crs(det.crs) if soil.crs != det.crs else soil
            geov_ = geov.to_crs(det.crs) if geov.crs != det.crs else geov
            geol_ = geol.to_crs(det.crs) if geol.crs != det.crs else geol

            soil_cols = [c for c in soil_.columns if c != "geometry"]
            geov_cols = [c for c in geov_.columns if c != "geometry"]
            geol_cols = [c for c in geol_.columns if c != "geometry"]

            if not soil_cols:
                print(f"  Geen attribuutkolommen in bodemkaart, laag '{layer}' overgeslagen.")
                continue
            if not geov_cols:
                print(f"  Geen attribuutkolommen in geovlak, laag '{layer}' overgeslagen.")
                continue
            if not geol_cols:
                print(f"  Geen attribuutkolommen in geolijn, laag '{layer}' overgeslagen.")
                continue

            soil_key = SOIL_KEY_COL if (SOIL_KEY_COL and SOIL_KEY_COL in soil_cols) else soil_cols[0]
            geov_key = GEOV_KEY_COL if (GEOV_KEY_COL and GEOV_KEY_COL in geov_cols) else geov_cols[0]
            geol_key = GEOL_KEY_COL if (GEOL_KEY_COL and GEOL_KEY_COL in geol_cols) else geol_cols[0]

            det_soil = gpd.sjoin(det, soil_, how="left", predicate="intersects")
            det_geov = gpd.sjoin(det, geov_, how="left", predicate="intersects")
            det_geol = gpd.sjoin(det, geol_, how="left", predicate="intersects")

            group_col = "type" if "type" in det.columns else "class"

            freq_soil = det_soil.groupby([group_col, soil_key]).size().reset_index(name="count")
            freq_geov = det_geov.groupby([group_col, geov_key]).size().reset_index(name="count")
            freq_geol = det_geol.groupby([group_col, geol_key]).size().reset_index(name="count")

            freq_soil = freq_soil.rename(columns={group_col: "class"})
            freq_geov = freq_geov.rename(columns={group_col: "class"})
            freq_geol = freq_geol.rename(columns={group_col: "class"})

            maak_nette_tabel(freq_soil, layer, soil_key).to_csv(
                os.path.join(out_dir, f"{kaartnaam}_freq_{layer}_soil.csv"),
                sep=";", encoding="utf-8-sig", index=False
            )
            maak_nette_tabel(freq_geov, layer, geov_key).to_csv(
                os.path.join(out_dir, f"{kaartnaam}_freq_{layer}_geovlak.csv"),
                sep=";", encoding="utf-8-sig", index=False
            )
            maak_nette_tabel(freq_geol, layer, geol_key).to_csv(
                os.path.join(out_dir, f"{kaartnaam}_freq_{layer}_geolijn.csv"),
                sep=";", encoding="utf-8-sig", index=False
            )
    except Exception as e:
        print("Bodem/geologie koppeling overgeslagen:", e)
    print("Klaar voor", kaartnaam)
    print("Output bestanden:", os.listdir(out_dir))

# ============================================================
# MAIN
# ============================================================
def main(hergebruik_model: bool = False) -> None:
    """Hoofdfunctie: verzamel data, train model, verwerk kaarten."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Lees statische datasets één keer in
    print("--- 1: Inlezen statische datasets... ---")
    roads = gpd.read_file(GT_GPKG, layer=GT_LAYER_ROADS)
    water = gpd.read_file(GT_GPKG, layer=GT_LAYER_WATER)
    percs = gpd.read_file(GT_GPKG, layer=GT_LAYER_PERCS)
    soil  = gpd.read_file(SOIL_GPKG)
    geov  = gpd.read_file(GEOL_VLAK)
    geol  = gpd.read_file(GEOL_LIJN)

    # Trainingskaarten
    train_tifs = (
        glob(os.path.join(TRAINING_DIR, "*.tif")) +
        glob(os.path.join(TRAINING_DIR, "*.tiff"))
    )

    # Detectiekaarten
    val_tifs = (
        glob(os.path.join(DETECTIE_DIR, "*.tif")) +
        glob(os.path.join(DETECTIE_DIR, "*.tiff"))
    )
    print(f"\nTrainingskaarten  ({len(train_tifs)}): {[os.path.basename(t) for t in train_tifs]}")
    print(f"Detectiekaarten   ({len(val_tifs)}):   {[os.path.basename(t) for t in val_tifs]}")
    if not train_tifs:
        print("FOUT: geen trainingskaarten gevonden in TRAINING_DIR. Controleer het pad in de config.")
        return

    # MODEL LADEN OF TRAINEN
    if hergebruik_model and os.path.exists(MODEL_PATH):
        print(f"\nBestaand model laden uit: {MODEL_PATH}")
        rf = joblib.load(MODEL_PATH)
    else:
        print("\n--- 2: features verzamelen + direct samplen van trainingskaarten ---")
        sampled_X, sampled_y = [], []
        rng = np.random.default_rng(RANDOM_STATE)
        for tif in train_tifs:
            naam = os.path.basename(tif)
            print(f"  Features laden: {naam}")
            try:
                X, y, _ = extract_features_and_labels(tif, roads, water, percs)
                Xs, ys = sample_training_pixels(X, y, rng)
                if Xs.size == 0:
                    print("  OVERGESLAGEN: geen trainbare pixels na sampling.")
                    continue
                sampled_X.append(Xs)
                sampled_y.append(ys)
            except ValueError as e:
                print(f"  OVERGESLAGEN: {e}")
        if not sampled_X:
            print("Geen geldige trainingskaarten. Script gestopt.")
            return
        X_all = np.concatenate(sampled_X, axis=0)
        y_all = np.concatenate(sampled_y, axis=0)
        print(f"Totaal gesampled: {X_all.shape[0]:,} pixels uit {len(sampled_X)} kaarten")
        print("\n--- 3: model trainen ---")
        rf = fit_rf(X_all, y_all)
        joblib.dump(rf, MODEL_PATH)
        print(f"Model opgeslagen: {MODEL_PATH}")

    # Predict + exporteer alle kaarten
    print("\n--- 4: predicties + export per kaart ---")
    for tif in train_tifs:
        exporteer_minuutplan(tif, rf, roads, water, percs, soil, geov, geol, is_validatie=False)
    if val_tifs:
        print("\n--- Detectiekaarten ---")
        for tif in val_tifs:
            exporteer_minuutplan(tif, rf, roads, water, percs, soil, geov, geol, is_validatie=True)
        print("\nDetectie voltooid. Zie '*_validatierapport.txt' per kaartmap voor de scores.")
    print("\nAlle kaarten verwerkt.")
    
if __name__ == "__main__":
    main(hergebruik_model=False)