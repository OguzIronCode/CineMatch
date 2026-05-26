---
name: project-cinematic
description: CineMatch — Flask tabanlı MovieLens ml-32m film öneri sistemi, 3 ML algoritması
metadata:
  type: project
---

**CineMatch** — Veri Madenciliği dersi bitirme projesi.

**Mimari:**
- Backend: Python/Flask (`app.py`)
- Frontend: Jinja2 HTML şablonları + özel CSS (`static/style.css`)
- Veritabanı: Supabase (kullanıcı auth, öneri geçmişi, kaydedilen filmler)
- Poster API: TMDB (api key `.env`'de)

**ML Modelleri:**
- KNN (`knn_model.pkl` + `matrix_df.pkl`): Film benzerliğine göre cosine similarity
- Decision Tree (`dt_model.pkl` + `dt_features.pkl` + `film_havuzu.pkl`)
- Naive Bayes (`nb_model.pkl` + `nb_features.pkl`)

**Veri Pipeline:**
- Ham veri: MovieLens ml-32m → 32M rating
- %15 stratified sampling → 4.7M satır
- Cold-start filtresi: ≥20 rating/kullanıcı, ≥10 rating/film
- KS testi ile doğrulama: D=0.0037, p=1.0000
- Yeni feedback: `yeni_ratingler.csv` → 500 feedback sonrası arka planda yeniden eğitim

**Sayfalar:** `/` (index), `/oneri`, `/analiz`, `/giris`, `/kayit`, `/dashboard`, `/gecmis`, `/profil`

**Why:** Veri Madenciliği dersi proje ödevi. **How to apply:** Öneri algoritmaları, veri pipeline ve Supabase entegrasyonu hakkında sorularda bu bağlamı kullan.
