# -*- coding: utf-8 -*-
import json
import os
import pickle
import random
import re
import string
import threading
import time
import urllib.request
from datetime import datetime
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGIN", "http://localhost:5000").split(",")]
CORS(app, supports_credentials=True, origins=_allowed_origins)

# Supabase (optional — fill .env to enable)
try:
    from supabase import create_client as _sb_create
    _sb_url = os.getenv("SUPABASE_URL", "")
    _sb_key = os.getenv("SUPABASE_KEY", "")
    if _sb_url and _sb_key and not _sb_url.startswith("https://xxxx"):
        supabase = _sb_create(_sb_url, _sb_key)
        SUPABASE_READY = True
        print("[SUPABASE] Baglanti kuruldu.")
    else:
        supabase = None
        SUPABASE_READY = False
        print("[UYARI] .env dosyasinda SUPABASE_URL/KEY eksik — auth devre disi.")
except ImportError:
    supabase = None
    SUPABASE_READY = False
    print("[UYARI] 'supabase' paketi eksik — pip install supabase")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YENIDEN_EGITIM_ESIGI = 500
YENI_CSV = "yeni_ratingler.csv"

SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", os.getenv("SUPABASE_KEY", ""))

# Bellekte oda kaydı (uygulama yeniden başlayınca sıfırlanır)
ROOMS: dict = {}

def oda_kodu_uret() -> str:
    while True:
        kod = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if kod not in ROOMS:
            return kod

# ---------------------------------------------------------------------------
# Startup: load heavy assets once
# ---------------------------------------------------------------------------
MOVIES = pd.read_csv("movies.csv")

with open("avg_rating.pkl", "rb") as f:
    _ar = pickle.load(f)
AVG_RATING          = _ar["avg_rating"]
TOPLAM_RATING_COUNT = _ar["toplam_count"]
del _ar

with open("knn_model.pkl", "rb") as f:
    _knn = pickle.load(f)

# Load matrix — support both legacy dense DataFrame and new (sparse, movie_ids) tuple
with open("matrix_df.pkl", "rb") as f:
    _raw = pickle.load(f)
if isinstance(_raw, tuple):
    _sparse, _movie_ids = _raw
else:
    # Legacy dense DataFrame: convert to sparse and free the ~3.4 GB
    _movie_ids = list(_raw.index)
    _sparse = csr_matrix(_raw.values)
    del _raw

# Thread-safe model state: all mutable objects live in this dict
_model_lock = threading.Lock()
_STATE = {
    "knn":        _knn,
    "sparse":     _sparse,
    "movie_ids":  _movie_ids,
    "id_to_idx":  {mid: i for i, mid in enumerate(_movie_ids)},
    "valid_ids":  set(_movie_ids),
}
del _knn, _sparse, _movie_ids

# Decision Tree model (optional — run dt_egit.py first to generate pkl files)
try:
    with open("dt_model.pkl", "rb") as f:
        DT_MODEL = pickle.load(f)
    with open("dt_features.pkl", "rb") as f:
        DT_FEATURES = pickle.load(f)
    with open("film_havuzu.pkl", "rb") as f:
        FILM_HAVUZU = pickle.load(f)
    DT_READY = True
    print("[DT] Model yuklendi: %d film" % len(FILM_HAVUZU))
except FileNotFoundError:
    DT_MODEL = DT_FEATURES = FILM_HAVUZU = None
    DT_READY = False
    print("[UYARI] DT modeli bulunamadi. dt_egit.py'yi calistirin.")

# Naive Bayes model (optional — run nb_egit.py first to generate pkl files)
try:
    with open("nb_model.pkl", "rb") as f:
        NB_MODEL = pickle.load(f)
    with open("nb_features.pkl", "rb") as f:
        NB_FEATURES = pickle.load(f)
    NB_READY = True
    print("[NB] Model yuklendi.")
except FileNotFoundError:
    NB_MODEL = NB_FEATURES = None
    NB_READY = False
    print("[UYARI] NB modeli bulunamadi. nb_egit.py'yi calistirin.")

_csv_lock = threading.Lock()
_egitim_durumu = {"son_egitim": None, "egitiliyor": False}

# ---------------------------------------------------------------------------
# TMDB poster lookup
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "f9ff3f80297a9760b16c9c30f6a2feac")
TMDB_BASE    = "https://api.themoviedb.org/3/movie/"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"
POSTER_YOK   = "/static/poster_yok.svg"
print("[TMDB] API key:", "YUKLENDI ✓" if TMDB_API_KEY else "EKSIK — .env dosyasini kontrol et")

try:
    _links = pd.read_csv("links.csv", usecols=["movieId", "imdbId", "tmdbId"])
    _lt = _links.dropna(subset=["tmdbId"]).copy()
    _lt["tmdbId"] = _lt["tmdbId"].astype(int)
    TMDB_LOOKUP = dict(zip(_lt["movieId"], _lt["tmdbId"]))
    _li = _links.dropna(subset=["imdbId"]).copy()
    _li["imdbId"] = _li["imdbId"].astype(int).apply(lambda x: "tt%07d" % x)
    IMDB_LOOKUP = dict(zip(_li["movieId"], _li["imdbId"]))
    print("[TMDB] links.csv yuklendi: %d tmdb, %d imdb" % (len(TMDB_LOOKUP), len(IMDB_LOOKUP)))
    del _links, _lt, _li
except FileNotFoundError:
    TMDB_LOOKUP = {}
    IMDB_LOOKUP = {}
    print("[UYARI] links.csv bulunamadi — afis/link devre disi.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=5000)
def tmdb_detay_getir(movie_id):
    tmdb_id = TMDB_LOOKUP.get(movie_id)
    imdb_id = IMDB_LOOKUP.get(movie_id, "")
    detay = {
        "poster":   POSTER_YOK,
        "overview": "",
        "imdb_url": ("https://www.imdb.com/title/" + imdb_id + "/") if imdb_id else "",
        "tmdb_url": ("https://www.themoviedb.org/movie/" + str(tmdb_id)) if tmdb_id else "",
        "play_url": ("https://www.playimdb.com/title/" + imdb_id + "/") if imdb_id else "",
    }
    if not tmdb_id:
        return detay
    try:
        url = TMDB_BASE + str(tmdb_id) + "?api_key=" + TMDB_API_KEY
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        path = data.get("poster_path")
        if path:
            detay["poster"] = TMDB_IMG + path
        detay["overview"] = data.get("overview", "")
        time.sleep(0.05)
    except Exception as e:
        print("[DETAY] movieId=%s -> HATA: %s" % (movie_id, e))
    return detay


def _year_from_title(title):
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


def _yeni_rating_sayisi():
    if not os.path.exists(YENI_CSV):
        return 0
    try:
        return len(pd.read_csv(YENI_CSV))
    except Exception:
        return 0


def aktif_kullanici():
    return session.get("user_id")


def gecmise_kaydet(results, yontem):
    uid = aktif_kullanici()
    if not SUPABASE_READY:
        print("[GECMIS] Supabase hazir degil — atlanıyor.")
        return
    if not uid:
        print("[GECMIS] Kullanici giris yapmamis — atlanıyor.")
        return
    try:
        rows = [
            {"user_id": uid, "film_adi": f["title"],
             "movie_id": f["movie_id"], "algoritma": yontem}
            for f in results
        ]
        supabase.table("oneri_gecmisi").insert(rows).execute()
        print("[GECMIS] %d film kaydedildi (user=%s, algo=%s)" % (len(rows), uid[:8], yontem))
    except Exception as e:
        print("[GECMIS] Kayit hatasi:", e)


# ---------------------------------------------------------------------------
# Background retraining
# ---------------------------------------------------------------------------
def modeli_yeniden_egit():
    _egitim_durumu["egitiliyor"] = True
    try:
        eski = pd.read_csv("temiz_orneklem.csv")
        yeni = pd.read_csv(YENI_CSV)
        birlesik = pd.concat([eski, yeni], ignore_index=True)
        birlesik = birlesik.drop_duplicates(
            subset=["userId", "movieId"], keep="last"
        )

        matrix_df = birlesik.pivot_table(
            index="movieId", columns="userId", values="rating"
        ).fillna(0)
        movie_ids = list(matrix_df.index)
        sparse = csr_matrix(matrix_df.values)
        del matrix_df  # free the dense DataFrame immediately

        model = NearestNeighbors(
            n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=-1
        )
        model.fit(sparse)

        with open("knn_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("matrix_df.pkl", "wb") as f:
            pickle.dump((sparse, movie_ids), f)

        with _model_lock:
            _STATE["knn"]       = model
            _STATE["sparse"]    = sparse
            _STATE["movie_ids"] = movie_ids
            _STATE["id_to_idx"] = {mid: i for i, mid in enumerate(movie_ids)}
            _STATE["valid_ids"] = set(movie_ids)

        _egitim_durumu["son_egitim"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            "[YENIDEN EGITIM] Tamamlandi — %d satir ile egitildi." % len(birlesik)
        )
    except Exception as e:
        print("[YENIDEN EGITIM] Hata:", e)
    finally:
        _egitim_durumu["egitiliyor"] = False


def birikim_kontrol(total_new):
    if (total_new > 0
            and total_new % YENIDEN_EGITIM_ESIGI == 0
            and not _egitim_durumu["egitiliyor"]):
        t = threading.Thread(target=modeli_yeniden_egit, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Decision Tree öneri
# ---------------------------------------------------------------------------
def dt_oneri(tur, min_puan_raw, yil_baslangic, yil_bitis, oneri_sayisi):
    filtreli = FILM_HAVUZU.copy()

    if tur != "Tümü":
        filtreli = filtreli[
            filtreli["genres"].str.contains(tur, na=False, regex=False)
        ]
    if min_puan_raw != "Tümü":
        cutoff = float(min_puan_raw.replace("+", ""))
        filtreli = filtreli[filtreli["avg_rating"] >= cutoff]
    if yil_baslangic:
        filtreli = filtreli[filtreli["year"] >= int(yil_baslangic)]
    if yil_bitis:
        filtreli = filtreli[filtreli["year"] <= int(yil_bitis)]

    if filtreli.empty:
        return []

    X = filtreli[DT_FEATURES]
    proba = DT_MODEL.predict_proba(X)[:, 1]
    filtreli = filtreli.copy()
    filtreli["dt_score"] = proba

    sonuclar = (
        filtreli[filtreli["dt_score"] >= 0.5]
        .sort_values("dt_score", ascending=False)
        .head(oneri_sayisi)
    )

    results = []
    for _, row in sonuclar.iterrows():
        genres_list = [g for g in str(row["genres"]).split("|")
                       if g != "(no genres listed)"]
        score_pct = round(float(row["dt_score"]) * 100, 1)
        year_val  = int(row["year"]) if pd.notna(row["year"]) else None
        detay = tmdb_detay_getir(int(row["movieId"]))
        results.append(dict(
            movie_id=int(row["movieId"]),
            title=row["title"],
            genres=genres_list,
            year=year_val,
            avg_rating=round(float(row["avg_rating"]), 2),
            similarity=score_pct,
            reason="Sectiginiz kriterlere %" + str(score_pct) + " oraninda uyuyor",
            poster=detay["poster"],
            overview=detay["overview"],
            imdb_url=detay["imdb_url"],
            tmdb_url=detay["tmdb_url"],
            play_url=detay["play_url"],
        ))
    return results


# ---------------------------------------------------------------------------
# Naive Bayes öneri
# ---------------------------------------------------------------------------
def nb_oneri(tur, min_puan_raw, yil_baslangic, yil_bitis, oneri_sayisi):
    filtreli = FILM_HAVUZU.copy()

    if tur != "Tümü":
        filtreli = filtreli[
            filtreli["genres"].str.contains(tur, na=False, regex=False)
        ]
    if min_puan_raw != "Tümü":
        cutoff = float(min_puan_raw.replace("+", ""))
        filtreli = filtreli[filtreli["avg_rating"] >= cutoff]
    if yil_baslangic:
        filtreli = filtreli[filtreli["year"] >= int(yil_baslangic)]
    if yil_bitis:
        filtreli = filtreli[filtreli["year"] <= int(yil_bitis)]

    if filtreli.empty:
        return []

    X = filtreli[NB_FEATURES]
    proba = NB_MODEL.predict_proba(X)[:, 2]
    filtreli = filtreli.copy()
    filtreli["nb_score"] = proba

    sonuclar = (
        filtreli.sort_values("nb_score", ascending=False)
        .head(oneri_sayisi)
    )

    results = []
    for _, row in sonuclar.iterrows():
        genres_list = [g for g in str(row["genres"]).split("|")
                       if g != "(no genres listed)"]
        score_pct = round(float(row["nb_score"]) * 100, 1)
        year_val  = int(row["year"]) if pd.notna(row["year"]) else None
        detay = tmdb_detay_getir(int(row["movieId"]))
        results.append(dict(
            movie_id=int(row["movieId"]),
            title=row["title"],
            genres=genres_list,
            year=year_val,
            avg_rating=round(float(row["avg_rating"]), 2),
            similarity=score_pct,
            reason="Sectiginiz turde yuksek puanli olma olasiligi %" + str(score_pct),
            poster=detay["poster"],
            overview=detay["overview"],
            imdb_url=detay["imdb_url"],
            tmdb_url=detay["tmdb_url"],
            play_url=detay["play_url"],
        ))
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/kayit", methods=["GET", "POST"])
def kayit():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        sifre = request.form.get("sifre", "")
        if not SUPABASE_READY:
            return render_template("kayit.html", hata="Supabase baglantisi yapilamadi.")
        try:
            supabase.auth.sign_up({"email": email, "password": sifre})
            return render_template("kayit.html",
                mesaj="Dogrulama maili gonderildi — e-postani kontrol et.")
        except Exception as e:
            return render_template("kayit.html", hata=str(e))
    return render_template("kayit.html")


@app.route("/giris", methods=["GET", "POST"])
def giris():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        sifre = request.form.get("sifre", "")
        if not SUPABASE_READY:
            return render_template("giris.html", hata="Supabase baglantisi yapilamadi.")
        try:
            res = supabase.auth.sign_in_with_password({"email": email, "password": sifre})
            session["user_id"]    = res.user.id
            session["user_email"] = res.user.email
            session["token"]      = res.session.access_token
            return redirect(url_for("oneri"))
        except Exception:
            return render_template("giris.html", hata="E-posta veya sifre hatali.")
    return render_template("giris.html")


@app.route("/cikis")
def cikis():
    if SUPABASE_READY:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    session.clear()
    return redirect(url_for("index"))


@app.route("/sifre-sifirla", methods=["GET", "POST"])
def sifre_sifirla():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not SUPABASE_READY:
            return render_template("sifre_sifirla.html", hata="Supabase baglantisi yapilamadi.")
        try:
            supabase.auth.reset_password_for_email(email)
            return render_template("sifre_sifirla.html",
                mesaj="Sifre sifirlama maili gonderildi.")
        except Exception as e:
            return render_template("sifre_sifirla.html", hata=str(e))
    return render_template("sifre_sifirla.html")


@app.route("/gecmis")
def gecmis():
    if not aktif_kullanici():
        return redirect(url_for("giris"))
    gecmis_data = []
    if SUPABASE_READY:
        try:
            res = (supabase.table("oneri_gecmisi")
                   .select("*")
                   .eq("user_id", aktif_kullanici())
                   .order("olusturma", desc=True)
                   .limit(50)
                   .execute())
            gecmis_data = res.data
        except Exception as e:
            print("[GECMIS] Okuma hatasi:", e)
    return render_template("gecmis.html", gecmis=gecmis_data)


@app.route("/analiz")
def analiz():
    yeni_puan = _yeni_rating_sayisi()
    return render_template("analiz.html",
        toplam_rating=TOPLAM_RATING_COUNT,
        yeni_puan=yeni_puan,
    )


@app.route("/oneri", methods=["GET", "POST"])
def oneri():
    if request.method == "GET":
        film_adi_param = request.args.get("film_adi", "").strip()
        auto = bool(film_adi_param and request.args.get("auto"))
        form_data = {
            "yontem": "knn", "film_adi": film_adi_param,
            "tur": "Tümü", "min_puan": "Tümü",
            "yil_baslangic": "", "yil_bitis": "", "oneri_sayisi": 10,
        }
        return render_template("oneri.html", results=None, error=None,
                               form=form_data, query_title=None, auto_submit=auto)

    yontem        = request.form.get("yontem", "knn")
    film_adi      = request.form.get("film_adi", "").strip()
    tur           = request.form.get("tur", "Tümü").strip()
    min_puan_raw  = request.form.get("min_puan", "Tümü").strip()
    yil_baslangic = request.form.get("yil_baslangic", "").strip()
    yil_bitis     = request.form.get("yil_bitis", "").strip()
    _VALID_ONERI_SAYISI = {5, 10, 15, 20}
    try:
        oneri_sayisi = int(request.form.get("oneri_sayisi", 10))
    except (ValueError, TypeError):
        oneri_sayisi = 10
    if oneri_sayisi not in _VALID_ONERI_SAYISI:
        oneri_sayisi = 10

    form = dict(
        yontem=yontem, film_adi=film_adi, tur=tur,
        min_puan=min_puan_raw, yil_baslangic=yil_baslangic,
        yil_bitis=yil_bitis, oneri_sayisi=oneri_sayisi,
    )

    # ── Naive Bayes path ──────────────────────────────────────────────────
    if yontem == "nb":
        if not NB_READY:
            msg = ("Naive Bayes modeli hazir degil. "
                   "Lutfen once nb_egit.py dosyasini calistirin.")
            return render_template("oneri.html", results=None, error=msg,
                                   form=form, query_title=None)
        results = nb_oneri(tur, min_puan_raw, yil_baslangic, yil_bitis, oneri_sayisi)
        if not results:
            msg = ("Secilen filtrelerle eslesen Naive Bayes onerisi bulunamadi. "
                   "Kisitlamalari gevseterek tekrar deneyin.")
            return render_template("oneri.html", results=None, error=msg,
                                   form=form, query_title=None)
        gecmise_kaydet(results, "nb")
        return render_template("oneri.html", results=results, query_title=None,
                               error=None, form=form)

    # ── Decision Tree path ────────────────────────────────────────────────
    if yontem == "dt":
        if not DT_READY:
            msg = ("Decision Tree modeli hazir degil. "
                   "Lutfen once dt_egit.py dosyasini calistirin.")
            return render_template("oneri.html", results=None, error=msg,
                                   form=form, query_title=None)

        results = dt_oneri(tur, min_puan_raw, yil_baslangic, yil_bitis, oneri_sayisi)
        if not results:
            msg = ("Secilen filtrelerle eslesen Decision Tree onerisi bulunamadi. "
                   "Kisitlamalari gevseterek tekrar deneyin.")
            return render_template("oneri.html", results=None, error=msg,
                                   form=form, query_title=None)
        gecmise_kaydet(results, "dt")
        return render_template("oneri.html", results=results, query_title=None,
                               error=None, form=form)

    # ── KNN path ──────────────────────────────────────────────────────────
    if not film_adi:
        return render_template("oneri.html", results=None,
                               error="Lütfen bir film adı girin.",
                               form=form, query_title=None)

    # Snapshot model state so a concurrent retrain doesn't affect this request
    with _model_lock:
        knn       = _STATE["knn"]
        sparse    = _STATE["sparse"]
        movie_ids = _STATE["movie_ids"]
        id_to_idx = _STATE["id_to_idx"]
        valid_ids = _STATE["valid_ids"]

    mask = MOVIES["title"].str.contains(film_adi, case=False, na=False, regex=False)
    candidates = MOVIES[mask & MOVIES["movieId"].isin(valid_ids)]

    if candidates.empty:
        return render_template("oneri.html", results=None, error=None,
                               not_in_dataset=True, searched_title=film_adi,
                               form=form, query_title=None)

    best_row = (
        candidates.assign(cnt=candidates["movieId"].map(AVG_RATING))
        .sort_values("cnt", ascending=False)
        .iloc[0]
    )
    query_movie_id = int(best_row["movieId"])
    query_title    = best_row["title"]

    row_idx = id_to_idx[query_movie_id]
    row_vec = sparse[row_idx]  # sparse row slice, sklearn handles it directly

    n_req = min(51, knn.n_samples_fit_)
    distances, indices = knn.kneighbors(row_vec, n_neighbors=n_req)
    distances = distances[0]
    indices   = indices[0]

    results = []
    for dist, idx in zip(distances, indices):
        mid = int(movie_ids[idx])
        if mid == query_movie_id:
            continue

        movie_rows = MOVIES[MOVIES["movieId"] == mid]
        if movie_rows.empty:
            continue
        movie_row = movie_rows.iloc[0]

        title  = movie_row["title"]
        genres = [g for g in movie_row["genres"].split("|")
                  if g != "(no genres listed)"]
        year   = _year_from_title(title)
        avg    = round(float(AVG_RATING.get(mid, 0)), 2)
        sim    = round((1 - float(dist)) * 100, 1)

        if tur != "Tümü" and tur not in genres:
            continue
        if min_puan_raw != "Tümü":
            cutoff = float(min_puan_raw.replace("+", ""))
            if avg < cutoff:
                continue
        if yil_baslangic and year is not None:
            if year < int(yil_baslangic):
                continue
        if yil_bitis and year is not None:
            if year > int(yil_bitis):
                continue

        detay = tmdb_detay_getir(mid)
        reason = "Bu filmi izleyenler " + query_title + " filmini de sevdi"
        results.append(dict(
            movie_id=mid,
            title=title,
            genres=genres,
            year=year,
            avg_rating=avg,
            similarity=sim,
            reason=reason,
            poster=detay["poster"],
            overview=detay["overview"],
            imdb_url=detay["imdb_url"],
            tmdb_url=detay["tmdb_url"],
            play_url=detay["play_url"],
        ))

        if len(results) >= oneri_sayisi:
            break

    if not results:
        msg = ("Seçilen filtrelerle eşleşen öneri bulunamadı. "
               "Kısıtlamaları gevşetmeyi deneyin.")
        return render_template("oneri.html", results=None, error=msg,
                               form=form, query_title=None)

    gecmise_kaydet(results, "knn")
    return render_template(
        "oneri.html",
        results=results,
        query_title=query_title,
        error=None,
        form=form,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "JSON bekleniyor"}), 400

    movie_id   = data.get("movie_id")
    rating     = data.get("rating")
    session_id = data.get("session_id", "anon")

    if movie_id is None or rating is None:
        return jsonify({"status": "error",
                        "message": "movie_id ve rating zorunlu"}), 400
    try:
        movie_id = int(movie_id)
        rating   = float(rating)
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Gecersiz deger"}), 400
    if not (0.5 <= rating <= 5.0):
        return jsonify({"status": "error", "message": "Rating 0.5-5.0 arasında olmalı"}), 400

    new_row = pd.DataFrame([{
        "userId":    str(session_id),
        "movieId":   movie_id,
        "rating":    rating,
        "timestamp": int(time.time()),
    }])

    with _csv_lock:
        if os.path.exists(YENI_CSV):
            existing = pd.read_csv(YENI_CSV)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        combined.to_csv(YENI_CSV, index=False)
        total = len(combined)

    birikim_kontrol(total)

    uid = aktif_kullanici()
    if uid and SUPABASE_READY:
        try:
            supabase.table("begeni_bildirimleri").upsert({
                "user_id":  uid,
                "movie_id": movie_id,
                "rating":   rating,
            }).execute()
        except Exception as e:
            print("[BEGENI] Kayit hatasi:", e)

    return jsonify({"status": "ok", "total_new": total})


def _izleme_context(movie_id, oda_kodu="", is_host=True):
    movie_row = MOVIES[MOVIES["movieId"] == movie_id]
    if movie_row.empty:
        return None
    title   = movie_row.iloc[0]["title"]
    imdb_id = IMDB_LOOKUP.get(movie_id, "")
    if not imdb_id:
        return None
    detay    = tmdb_detay_getir(movie_id)
    play_url = "https://www.playimdb.com/title/" + imdb_id + "/"
    return dict(
        movie_id=movie_id,
        oda_kodu=oda_kodu,
        is_host=is_host,
        title=title,
        play_url=play_url,
        poster=detay["poster"],
        imdb_url=detay["imdb_url"],
        tmdb_url=detay["tmdb_url"],
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_anon_key=SUPABASE_ANON_KEY,
    )


@app.route("/izle/<int:movie_id>")
def izle(movie_id):
    # Otomatik oda oluştur ve oraya yönlendir
    kod = oda_kodu_uret()
    ROOMS[kod] = {"film_id": movie_id, "host_id": aktif_kullanici()}
    return redirect("/oda/%s/%d" % (kod, movie_id))


@app.route("/oda-olustur", methods=["POST"])
def oda_olustur():
    data    = request.get_json(silent=True) or {}
    film_id = data.get("film_id")
    if not film_id:
        return jsonify({"status": "hata"}), 400
    kod = oda_kodu_uret()
    ROOMS[kod] = {"film_id": int(film_id), "host_id": aktif_kullanici()}
    url = "/oda/%s/%d" % (kod, int(film_id))
    return jsonify({"status": "ok", "kod": kod, "url": url})


@app.route("/oda/<kod>/<int:movie_id>")
def oda(kod, movie_id):
    kod = kod.upper().strip()
    ROOMS.setdefault(kod, {"film_id": movie_id, "host_id": None})
    is_host = ROOMS[kod].get("host_id") == aktif_kullanici()
    ctx = _izleme_context(movie_id, oda_kodu=kod, is_host=is_host)
    if not ctx:
        return redirect(url_for("oneri"))
    return render_template("izleme.html", **ctx)


@app.route("/detay/<int:movie_id>")
def film_detay(movie_id):
    tmdb_id = TMDB_LOOKUP.get(movie_id)
    imdb_id = IMDB_LOOKUP.get(movie_id, "")
    if not tmdb_id:
        return jsonify({"hata": "Film bulunamadi"}), 404
    try:
        url = "%s%d?api_key=%s&language=en-US" % (TMDB_BASE, tmdb_id, TMDB_API_KEY)
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        curl = "%s%d/credits?api_key=%s&language=en-US" % (TMDB_BASE, tmdb_id, TMDB_API_KEY)
        with urllib.request.urlopen(curl, timeout=5) as resp:
            credits = json.loads(resp.read())

        directors = [c["name"] for c in credits.get("crew", []) if c["job"] == "Director"]
        cast      = [c["name"] for c in credits.get("cast", [])[:7]]
        poster_p  = data.get("poster_path")
        backdrop_p = data.get("backdrop_path")

        return jsonify({
            "title":          data.get("title", ""),
            "original_title": data.get("original_title", ""),
            "overview":       data.get("overview", ""),
            "poster":         (TMDB_IMG + poster_p) if poster_p else POSTER_YOK,
            "backdrop":       ("https://image.tmdb.org/t/p/w1280" + backdrop_p) if backdrop_p else "",
            "release_date":   data.get("release_date", ""),
            "runtime":        data.get("runtime"),
            "vote_average":   round(float(data.get("vote_average") or 0), 1),
            "vote_count":     data.get("vote_count", 0),
            "genres":         [g["name"] for g in data.get("genres", [])],
            "directors":      directors,
            "cast":           cast,
            "imdb_url":       ("https://www.imdb.com/title/" + imdb_id + "/") if imdb_id else "",
            "play_url":       ("https://www.playimdb.com/title/" + imdb_id + "/") if imdb_id else "",
            "tmdb_url":       "https://www.themoviedb.org/movie/" + str(tmdb_id),
        })
    except Exception as e:
        print("[FILM_DETAY] movieId=%s -> HATA: %s" % (movie_id, e))
        return jsonify({"hata": str(e)}), 500


@app.route("/trailer/<int:movie_id>")
def trailer(movie_id):
    tmdb_id = TMDB_LOOKUP.get(movie_id)
    if not tmdb_id:
        return jsonify({"key": None})
    try:
        url = TMDB_BASE + str(tmdb_id) + "/videos?api_key=" + TMDB_API_KEY + "&language=en-US"
        with urllib.request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read())
        videos = data.get("results", [])
        for v in videos:
            if v.get("site") == "YouTube" and v.get("type") == "Trailer":
                return jsonify({"key": v["key"]})
        for v in videos:
            if v.get("site") == "YouTube":
                return jsonify({"key": v["key"]})
        return jsonify({"key": None})
    except Exception as e:
        print("[TRAILER] movieId=%s -> HATA: %s" % (movie_id, e))
        return jsonify({"key": None})


@app.route("/ara")
def ara():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    mask = MOVIES["title"].str.contains(q, case=False, na=False, regex=False)
    hits = MOVIES[mask].head(8)
    results = []
    for _, row in hits.iterrows():
        mid = int(row["movieId"])
        results.append({
            "title": row["title"],
            "movie_id": mid,
            "genres": row["genres"].replace("|", " · ") if pd.notna(row["genres"]) else "",
        })
    return jsonify(results)


@app.route("/kaydet", methods=["POST"])
def kaydet():
    if not aktif_kullanici():
        return jsonify({"status": "giris_gerekli"}), 401
    if not SUPABASE_READY:
        return jsonify({"status": "error"}), 503
    data = request.get_json(silent=True) or {}
    year_raw = data.get("year")
    try:
        year_val = int(year_raw) if year_raw not in (None, "None", "", "null") else None
    except (ValueError, TypeError):
        year_val = None
    try:
        supabase.table("kaydedilen_filmler").upsert({
            "user_id"   : aktif_kullanici(),
            "movie_id"  : int(data["movie_id"]),
            "title"     : data.get("title", ""),
            "genres"    : data.get("genres", ""),
            "year"      : year_val,
            "avg_rating": float(data.get("avg_rating", 0)),
            "poster_url": data.get("poster_url", ""),
            "tmdb_url"  : data.get("tmdb_url", ""),
        }).execute()
        return jsonify({"status": "ok"})
    except Exception as e:
        print("[KAYDET] Hata:", e)
        return jsonify({"status": "zaten_var"})


@app.route("/gecmis-temizle", methods=["POST"])
def gecmis_temizle():
    if not aktif_kullanici():
        return jsonify({"status": "giris_gerekli"}), 401
    if not SUPABASE_READY:
        return jsonify({"status": "error"}), 503
    try:
        supabase.table("oneri_gecmisi").delete().eq("user_id", aktif_kullanici()).execute()
        return jsonify({"status": "ok"})
    except Exception as e:
        print("[GECMIS-TEMIZLE] Hata:", e)
        return jsonify({"status": "error"}), 500


@app.route("/kaydet-kaldir", methods=["POST"])
def kaydet_kaldir():
    if not aktif_kullanici():
        return jsonify({"status": "giris_gerekli"}), 401
    if not SUPABASE_READY:
        return jsonify({"status": "error"}), 503
    data = request.get_json(silent=True) or {}
    try:
        supabase.table("kaydedilen_filmler")\
            .delete()\
            .eq("user_id", aktif_kullanici())\
            .eq("movie_id", int(data.get("movie_id", 0)))\
            .execute()
    except Exception as e:
        print("[KAYDET-KALDIR] Hata:", e)
    return jsonify({"status": "kaldirildi"})


@app.route("/dashboard")
def dashboard():
    if not aktif_kullanici():
        return redirect(url_for("giris"))
    kaydedilenler = []
    gecmis_data = []
    if SUPABASE_READY:
        try:
            kaydedilenler = supabase.table("kaydedilen_filmler")\
                .select("*")\
                .eq("user_id", aktif_kullanici())\
                .order("kaydedilme", desc=True)\
                .execute().data
        except Exception as e:
            print("[DASHBOARD] Kaydedilenler hatasi:", e)
        try:
            gecmis_data = supabase.table("oneri_gecmisi")\
                .select("*")\
                .eq("user_id", aktif_kullanici())\
                .order("olusturma", desc=True)\
                .limit(50)\
                .execute().data
        except Exception as e:
            print("[DASHBOARD] Gecmis hatasi:", e)
    return render_template("dashboard.html",
        kaydedilenler=kaydedilenler,
        gecmis=gecmis_data,
        kullanici=session.get("user_email"),
    )


@app.route("/profil")
def profil():
    if not aktif_kullanici():
        return redirect(url_for("giris"))
    saved_count   = 0
    history_count = 0
    algo_counts   = {"knn": 0, "dt": 0, "nb": 0}
    if SUPABASE_READY:
        try:
            saved_count = len(supabase.table("kaydedilen_filmler")
                              .select("movie_id")
                              .eq("user_id", aktif_kullanici())
                              .execute().data)
        except Exception:
            pass
        try:
            rows = supabase.table("oneri_gecmisi")\
                .select("algoritma")\
                .eq("user_id", aktif_kullanici())\
                .execute().data
            history_count = len(rows)
            for r in rows:
                algo = r.get("algoritma", "")
                if algo in algo_counts:
                    algo_counts[algo] += 1
        except Exception:
            pass
    return render_template("profil.html",
        kullanici=session.get("user_email"),
        saved_count=saved_count,
        history_count=history_count,
        algo_counts=algo_counts,
    )


@app.route("/sistem-durumu")
def sistem_durumu():
    yeni = _yeni_rating_sayisi()
    kalan = YENIDEN_EGITIM_ESIGI - (yeni % YENIDEN_EGITIM_ESIGI)
    if yeni == 0:
        kalan = YENIDEN_EGITIM_ESIGI
    return jsonify({
        "toplam_rating":      TOPLAM_RATING_COUNT,
        "yeni_rating":        yeni,
        "sonraki_egitim_kac": kalan,
        "son_egitim":         _egitim_durumu["son_egitim"],
        "egitiliyor":         _egitim_durumu["egitiliyor"],
    })


if __name__ == "__main__":
    # use_reloader=False: reloader spawns a child process which breaks
    # background threads (they'd only live in the child, not persist)
    app.run(debug=True, port=5000, use_reloader=False)
