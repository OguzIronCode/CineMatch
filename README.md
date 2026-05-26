# CineMatch — Akıllı Film Öneri Sistemi

CineMatch, MovieLens ml-32m veri seti üzerinde eğitilmiş üç farklı makine öğrenmesi algoritmasını kullanan kişiselleştirilmiş film öneri platformudur. Flask tabanlı web arayüzü ve Supabase kimlik doğrulama altyapısıyla tam yığın bir uygulama olarak geliştirilmiştir.

---

## İçindekiler

- [Özellikler](#özellikler)
- [Teknoloji Yığını](#teknoloji-yığını)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Öneri Algoritmaları](#öneri-algoritmaları)
- [Veri Pipeline'ı](#veri-pipelineı)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [API Referansı](#api-referansı)
- [Veritabanı Şeması](#veritabanı-şeması)
- [Proje Yapısı](#proje-yapısı)
- [Dağıtım](#dağıtım)

---

## Özellikler

- **3 Farklı Öneri Algoritması** — KNN (cosine similarity), Karar Ağacı ve Naive Bayes ile farklı yaklaşımlar
- **Kullanıcı Kimlik Doğrulama** — Supabase üzerinden kayıt, giriş ve şifre sıfırlama
- **Film Kütüphanesi** — Beğenilen filmleri kaydet ve kişisel koleksiyon oluştur
- **Öneri Geçmişi** — Yapılan tüm öneri sorgularının kaydı ve istatistikleri
- **Canlı Arama** — Film veritabanında anlık otomatik tamamlama
- **TMDB Entegrasyonu** — Film afişleri, açıklamalar ve YouTube fragmanları
- **Dinamik Yeniden Eğitim** — 500 yeni kullanıcı puanı biriktikçe arka planda model güncelleme
- **Veri Analizi Sayfası** — Veri seti üzerinde keşifsel veri analizi ve interaktif grafikler
- **Karanlık Tema** — Özel tasarım sistemi, Bootstrap kullanılmayan sıfırdan CSS

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Backend | Python 3.x, Flask 3.0+ |
| ML / Veri | scikit-learn 1.3+, pandas 2.1+, SciPy 1.11+ |
| Frontend | HTML5, Jinja2, Vanilla JS, Chart.js 4.4.3 |
| Veritabanı | Supabase (PostgreSQL) |
| Kimlik Doğrulama | Supabase Auth |
| Harici API | TMDB (The Movie Database) |
| Dağıtım | Render.com, Gunicorn 21.2+ |

---

## Sistem Mimarisi

```
Kullanıcı İsteği
      │
      ▼
┌─────────────────────────────────────┐
│          Flask Uygulaması           │
│  app.py — 842 satır, 18 endpoint    │
│                                     │
│  ┌──────────┐  ┌──────────────────┐ │
│  │  Session │  │  Threading Lock  │ │
│  │ (Supabase│  │ _model_lock      │ │
│  │   Auth)  │  │ _csv_lock        │ │
│  └──────────┘  └──────────────────┘ │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌────────┐     ┌────────────────────────┐
│Supabase│     │    ML Model Katmanı    │
│  Auth  │     │                        │
│  DB    │     │  knn_model.pkl (54 MB) │
└────────┘     │  matrix_df.pkl (54 MB) │
               │  dt_model.pkl          │
               │  nb_model.pkl          │
               │  film_havuzu.pkl       │
               └────────────┬───────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │      TMDB API          │
               │  Afiş · Açıklama      │
               │  Fragman (YouTube)     │
               └────────────────────────┘
```

**Eşzamanlılık Tasarımı:**
- `_model_lock` — KNN durum güncellemelerinde thread-safe erişim
- `_csv_lock` — `yeni_ratingler.csv` yazım kilidi
- Arka plan daemon thread'i — eşik aşıldığında otomatik yeniden eğitim
- `@lru_cache(maxsize=5000)` — TMDB API yanıtları için bellek içi önbellekleme

---

## Öneri Algoritmaları

### 1. KNN — En Yakın Komşular (Cosine Similarity)

Kullanıcı-film derecelendirme matrisi üzerinde cosine similarity kullanarak en yakın 20 komşuyu bulur.

```
Giriş filmi → Seyredilme geçmişi olan kullanıcılar →
Bu kullanıcıların izlediği diğer filmler → Benzerlik sıralaması
```

- **Matris Formatı:** Bellek verimliliği için Seyrek CSR (yaklaşık 3,4 GB yoğun matris → küçük pickle)
- **Model Dosyaları:** `knn_model.pkl`, `matrix_df.pkl`

### 2. Karar Ağacı (Decision Tree)

Film öznitelikleri üzerinde eğitilmiş sınıflandırıcı. Kullanıcı kriter seçer, model her film için tercih olasılığı tahmin eder.

- **Giriş Öznitelikleri:** Tür, yayın yılı, ortalama puan
- **Model Dosyaları:** `dt_model.pkl`, `dt_features.pkl`, `film_havuzu.pkl`

### 3. Naive Bayes — Olasılıksal Sınıflandırma

Benzer öznitelik tabanlı yaklaşım; Bayesçi olasılık ile her filmi puanlar.

- **Model Dosyaları:** `nb_model.pkl`, `nb_features.pkl`

### Dinamik Yeniden Eğitim

```python
# 500 yeni puan biriktikçe tetiklenir
if new_ratings_count >= RETRAIN_THRESHOLD:
    threading.Thread(target=retrain_knn, daemon=True).start()
```

Yeni puanlar `yeni_ratingler.csv` içinde toplanır, mevcut eğitim verisine eklenir, tekrarlananlar kaldırılır ve KNN modeli arka planda yeniden eğitilir.

---

## Veri Pipeline'ı

**Kaynak Veri:** MovieLens ml-32m
- 32.000.204 derecelendirme
- 200.948 kullanıcı
- 84.432 film

**Örnekleme ve Temizleme (`temiz_orneklem.csv` → 120 MB)**

```
Ham Veri (32M puan)
        │
        ▼
%15 Tabakalı Rastgele Örnekleme
        │
        ▼
Soğuk Başlangıç Filtresi:
  · Kullanıcı başına ≥20 puan
  · Film başına ≥10 puan
        │
        ▼
~4,7 Milyon Temiz Puan
        │
        ▼
İstatistiksel Doğrulama (Kolmogorov-Smirnov):
  D = 0,0037 · p = 1,0000 → Dağılım eşleşmesi onaylandı
```

**Model Eğitim Çıktıları:**

| Dosya | Boyut | Açıklama |
|-------|-------|----------|
| `knn_model.pkl` | 54 MB | KNN en yakın komşu modeli |
| `matrix_df.pkl` | 54 MB | Kullanıcı-film seyrek matrisi |
| `avg_rating.pkl` | 348 KB | Film başına ortalama puan önbelleği |
| `dt_model.pkl` | 16 KB | Karar Ağacı sınıflandırıcısı |
| `film_havuzu.pkl` | 3,1 MB | Karar Ağacı meta verisi olan filmler |
| `nb_model.pkl` | 1,8 KB | Naive Bayes sınıflandırıcısı |

---

## Kurulum

### Gereksinimler

- Python 3.10+
- pip
- Supabase hesabı
- TMDB API anahtarı

### Adımlar

```bash
# 1. Depoyu klonla
git clone https://github.com/OguzHAN/CineMatch.git
cd CineMatch

# 2. Sanal ortam oluştur ve etkinleştir
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Ortam değişkenlerini yapılandır (aşağıya bak)
cp .env.example .env

# 5. Ortalama puanları ön hesapla (opsiyonel, avg_rating.pkl yoksa)
python precompute.py

# 6. Uygulamayı başlat
python app.py
```

Uygulama `http://localhost:5000` adresinde çalışır.

---

## Yapılandırma

Proje kök dizininde `.env` dosyası oluştur:

```env
FLASK_SECRET_KEY=guclu-ve-rastgele-bir-anahtar
SUPABASE_URL=https://<proje-id>.supabase.co
SUPABASE_KEY=<supabase-anon-public-key>
```

| Değişken | Açıklama |
|----------|----------|
| `FLASK_SECRET_KEY` | Flask oturum şifrelemesi için gizli anahtar |
| `SUPABASE_URL` | Supabase proje URL'si |
| `SUPABASE_KEY` | Supabase anonim/public anahtar |

**TMDB API Anahtarı** `app.py` içinde `TMDB_API_KEY` sabitiyle tanımlanmıştır. Kendi anahtarınla değiştirmek için [themoviedb.org](https://www.themoviedb.org/settings/api) adresinden ücretsiz hesap oluştur.

---

## API Referansı

### Sayfalar

| Metod | Endpoint | Açıklama |
|-------|----------|----------|
| GET | `/` | Ana sayfa |
| GET/POST | `/oneri` | Film öneri sayfası ve öneri işleme |
| GET | `/analiz` | Veri analizi ve EDA grafikleri |
| GET | `/dashboard` | Kullanıcı panosu (auth gerekli) |
| GET | `/gecmis` | Öneri geçmişi (auth gerekli) |
| GET | `/profil` | Kullanıcı profili (auth gerekli) |
| POST | `/kayit` | Kullanıcı kaydı |
| POST | `/giris` | Kullanıcı girişi |
| GET | `/cikis` | Çıkış yap |
| POST | `/sifre-sifirla` | Şifre sıfırlama isteği |

### JSON API Uç Noktaları

| Metod | Endpoint | Gövde / Parametreler | Yanıt |
|-------|----------|----------------------|-------|
| GET | `/ara?q=<sorgu>` | `q`: min 2 karakter | `[{movieId, title, year}, ...]` |
| GET | `/trailer/<movie_id>` | — | `{key: "youtube_video_key"}` |
| POST | `/feedback` | `{movie_id, rating, session_id}` | `{success, retrain_triggered}` |
| POST | `/kaydet` | `{movie_id, title, genres, ...}` | `{success, message}` |
| POST | `/kaydet-kaldir` | `{movie_id}` | `{success, message}` |
| POST | `/gecmis-temizle` | — | `{success}` |
| GET | `/sistem-durumu` | — | `{yeni_puan_sayisi, egitim_durumu, ...}` |

---

## Veritabanı Şeması

Supabase (PostgreSQL) üzerinde üç tablo:

### `oneri_gecmisi` — Öneri Geçmişi

```sql
CREATE TABLE oneri_gecmisi (
  id          BIGSERIAL PRIMARY KEY,
  user_id     UUID REFERENCES auth.users,
  film_adi    TEXT,
  movie_id    INTEGER,
  algoritma   TEXT CHECK (algoritma IN ('knn', 'dt', 'nb')),
  olusturma   TIMESTAMPTZ DEFAULT NOW()
);
```

### `kaydedilen_filmler` — Kaydedilen Filmler

```sql
CREATE TABLE kaydedilen_filmler (
  id          BIGSERIAL PRIMARY KEY,
  user_id     UUID REFERENCES auth.users,
  movie_id    INTEGER,
  title       TEXT,
  genres      TEXT,
  year        INTEGER,
  avg_rating  FLOAT,
  poster_url  TEXT,
  tmdb_url    TEXT,
  kaydedilme  TIMESTAMPTZ DEFAULT NOW()
);
```

### `begeni_bildirimleri` — Kullanıcı Puanları

```sql
CREATE TABLE begeni_bildirimleri (
  id          BIGSERIAL PRIMARY KEY,
  user_id     UUID,
  movie_id    INTEGER,
  rating      FLOAT
);
```

---

## Proje Yapısı

```
CineMatch/
├── app.py                  # Ana Flask uygulaması (18 endpoint, 842 satır)
├── precompute.py           # Ortalama puan ön hesaplama yardımcısı
├── requirements.txt        # Python bağımlılıkları
├── render.yaml             # Render.com dağıtım yapılandırması
├── .env                    # Ortam değişkenleri (commit edilmez)
│
├── static/
│   ├── style.css           # Özel karanlık tema tasarım sistemi (1200+ satır)
│   └── poster_yok.svg      # Afiş bulunamadığında yer tutucu görseli
│
├── templates/
│   ├── _nav.html           # Navigasyon bileşeni
│   ├── _footer.html        # Alt bilgi bileşeni
│   ├── index.html          # Ana sayfa — hero ve özellikler
│   ├── oneri.html          # Öneri sayfası — ana özellik
│   ├── dashboard.html      # Kullanıcı panosu — sekmeli arayüz
│   ├── analiz.html         # EDA ve veri görselleştirme
│   ├── gecmis.html         # Öneri geçmişi listesi
│   ├── profil.html         # Profil yönetimi
│   ├── giris.html          # Giriş formu
│   ├── kayit.html          # Kayıt formu
│   └── sifre_sifirla.html  # Şifre sıfırlama formu
│
└── [Model Dosyaları]       # Pickle ile serileştirilmiş ML modelleri
    ├── movies.csv              # 87.585 film meta verisi
    ├── links.csv               # TMDB/IMDb ID eşleştirmeleri
    ├── temiz_orneklem.csv      # Eğitim verisi (120 MB, .gitignore'da)
    ├── knn_model.pkl           # KNN modeli (54 MB)
    ├── matrix_df.pkl           # Kullanıcı-film matrisi (54 MB)
    ├── avg_rating.pkl          # Puan ortalamaları önbelleği
    ├── dt_model.pkl            # Karar Ağacı modeli
    ├── dt_features.pkl         # Karar Ağacı öznitelikleri
    ├── film_havuzu.pkl         # Karar Ağacı için film meta verisi
    ├── nb_model.pkl            # Naive Bayes modeli
    └── nb_features.pkl         # Naive Bayes öznitelikleri
```

---

## Dağıtım

Proje Render.com üzerinde dağıtılmak üzere yapılandırılmıştır (`render.yaml`).

```yaml
services:
  - type: web
    name: cinematic
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
    plan: free
    envVars:
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
```

**Ücretsiz Katman Kısıtlamaları:**
- Tek Gunicorn worker — sınırlı eşzamanlı istek
- Geçici dosya sistemi — yeniden dağıtımda `yeni_ratingler.csv` sıfırlanır
- Model dosyaları (`.pkl`) boyutu nedeniyle Git'e commit edilmez; dağıtımda mevcut olmaları gerekir

---

## Katkıda Bulunma

1. Depoyu fork edin
2. Özellik dalı oluşturun (`git checkout -b ozellik/yeni-algoritma`)
3. Değişiklikleri commit edin (`git commit -m 'feat: yeni algoritma eklendi'`)
4. Dala push edin (`git push origin ozellik/yeni-algoritma`)
5. Pull Request açın

---

## Lisans

Bu proje MIT lisansı altında dağıtılmaktadır.

---

*Veri Madenciliği Bitirme Projesi — MovieLens ml-32m veri seti kullanılarak geliştirilmiştir.*
