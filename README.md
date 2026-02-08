# 📰 News4You → An Instagram Automation Pipeline

An end-to-end AI system that:

**Fetches news → Clusters into stories → Picks best article → Designs Instagram carousel → Auto-posts**

Built for fully automated news storytelling!
Check out our instagram handle [@news4you2026](https://www.instagram.com/news4you2026)

---

## 🚀 What This Project Does

1. **Fetches news** from multiple APIs (NewsAPI, GNews, Mediastack)
2. **Understands content using NLP**

   * Entity extraction (people, places, orgs)
   * Synonym (entity) normalization
3. **Groups similar articles into STORIES**

   * Sentence embeddings (MiniLM)
   * Entity-weighted embeddings
   * HDBSCAN clustering
4. **Selects best article per story**

   * Source credibility
   * Content richness
   * Headline signals
5. **Applies timestamp filter logic**
   * Allows topic cooldown to stop spam
   * Clears older posts 
6. **Generates Instagram carousel slides**

   * Auto text fitting
   * Branded templates
   * Clean styles using Pillow
7. **Uploads to Cloudinary**
8. **Publishes automatically to Instagram**

---


## 🧠 Core AI Techniques

| Feature              | Method Used                       |
| -------------------- | --------------------------------- |
| Text Understanding   | **spaCy NER**                     |
| Entity Normalization | Fuzzy match + WordNet             |
| Semantic Embeddings  | **SentenceTransformers (MiniLM)** |
| Story Clustering     | **HDBSCAN**                       |
| Story De-duplication | Entity hashing                    |
| Article Ranking      | Rule-based scoring                |
| Visual Generation    | PIL dynamic layout engine         |

---


## ⚙️ Setup (Run This Yourself)

### 1️⃣ Install Dependencies (requirements.txt is provided)

```bash
pip install spacy sentence-transformers hdbscan rapidfuzz nltk pillow cloudinary requests scikit-learn
python -m spacy download en_core_web_sm
```

---

### 2️⃣ Add API Keys

Create a **.env** file or set environment variables:

```bash
NEWSAPI_KEY=your_key
GNEWS_KEY=your_key
MEDIASTACK_KEY=your_key

CLOUDINARY_CLOUD_NAME=xxx
CLOUDINARY_API_KEY=xxx
CLOUDINARY_API_SECRET=xxx

INSTAGRAM_USER_ID=xxx
INSTAGRAM_ACCESS_TOKEN=xxx
```

---

### 3️⃣ Run the Bot

```bash
python main.py
```

Pipeline will:

✔ Fetch news
✔ Cluster stories
✔ Pick top article
✔ Generate slides
✔ Upload images
✔ Post to Instagram

---

## 📌 Output Example

Each run produces:

```
slide_1_x.png
slide_content_1_x.png
slide_content_2_x.png
```

Then publishes a **carousel post** automatically.

---


## 📂 Important Files

| File                   | Purpose                       |
| ---------------------- | ----------------------------- |
| `news_pipeline.py`     | AI story detection + ranking  |
| `carousel_renderer.py` | Slide generation engine       |
| `cloudinary_upload.py` | Image hosting                 |
| `insta_publish.py`     | Instagram posting             |
| `queue_db.json`        | Story memory + cooldown logic |

---

## 🔁 Smart Features

* Topic cooldown prevents spam posts
* Duplicate story detection
* Auto font scaling
* Works even if some APIs fail
* Entity normalization
* Entity Weighted similarity scouting
* Sentence embedding + clustering = better story grouping

---

## 🧩 Use Cases

* AI news pages
* Automated media accounts
* Research on story clustering
* NLP + CV integration project

---

If something fails: check API keys first 🔑
**Made with ♡**
