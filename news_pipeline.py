# news_pipeline.py

import os, json, requests, hashlib
import spacy
from datetime import datetime, timezone, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "queue_db.json"

newsapi_key = os.getenv("NEWSAPI_KEY")
gnews_key = os.getenv("GNEWS_KEY")
mstack_key = os.getenv("MEDIASTACK_KEY")

nlp = spacy.load("en_core_web_sm")

# ---------------- DB ----------------
def load_db():
    if not os.path.exists(DB_FILE):
        return {"queue": [], "posted": [], "recent_topics": {}}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)

# ---------------- SCORING ----------------
def score_article(article):
    score = 0
    if article.get("image"): score += 2
    if len(article.get("desc","")) > 120: score += 1

    trusted = ["reuters","bbc","the hindu","indian express","ndtv"]
    if any(t in article.get("source","").lower() for t in trusted):
        score += 3

    if ":" in article.get("title",""): score += 1
    if any(w in article["title"].lower()
           for w in ["breaking","announces","launches","wins"]):
        score += 2

    return score

# ---------------- FETCH ----------------
def fetch_master_news(query):
    articles, seen = [], set()

    # NewsAPI
    url = f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={newsapi_key}"
    for a in requests.get(url).json().get("articles", []):
        if a["url"] not in seen:
            articles.append({
                "title": a["title"],
                "url": a["url"],
                "desc": a.get("description",""),
                "source": a["source"]["name"],
                "image": a.get("urlToImage")
            })
            seen.add(a["url"])

    # GNews
    url = f"https://gnews.io/api/v4/top-headlines?q={query}&token={gnews_key}&lang=en"
    for a in requests.get(url).json().get("articles", []):
        if a["url"] not in seen:
            articles.append({
                "title": a["title"],
                "url": a["url"],
                "desc": a.get("description",""),
                "source": a["source"]["name"],
                "image": a.get("image")
            })
            seen.add(a["url"])

    # Mediastack
    if mstack_key:
        url = f"http://api.mediastack.com/v1/news?access_key={mstack_key}&keywords={query}&languages=en"
        for a in requests.get(url).json().get("data", []):
            if a.get("url") and a["url"] not in seen:
                articles.append({
                    "title": a.get("title",""),
                    "url": a["url"],
                    "desc": a.get("description",""),
                    "source": a.get("source","Mediastack"),
                    "image": a.get("image")
                })
                seen.add(a["url"])

    return articles

# ---------------- NLP + SIMILARITY ----------------
def normalize_topic(name):
    aliases = {
        "modi": "Narendra Modi",
        "pm modi": "Narendra Modi",
        "joe biden": "Joe Biden"
    }
    return aliases.get(name.lower(), name.title())

def cluster_articles(articles, threshold=0.55):
    texts = [(a["title"] + " " + a["desc"]).lower() for a in articles]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    clusters, used = [], set()
    for i in range(len(articles)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(articles)):
            if similarity_matrix[i][j] >= threshold:
                group.append(j)
                used.add(j)
        clusters.append(group)

    return clusters, similarity_matrix, tfidf_matrix

# ---------------- MAIN ENTRY ----------------
def get_next_article(query="technology india"):
    raw = fetch_master_news(query)

    for a in raw:
        doc = nlp(a["title"] + " " + a["desc"])
        a["entities"] = [normalize_topic(e.text) for e in doc.ents]

    clusters, sim_matrix, tfidf_matrix = cluster_articles(raw)

    stories = []
    for c in clusters:
        group = [raw[i] for i in c]
        stories.append(max(group, key=score_article))

    if not stories:
        return None

    chosen = max(stories, key=score_article)

    return {
        "article": chosen,
        "clusters": clusters,
        "similarity_matrix": sim_matrix,
        "tfidf_matrix": tfidf_matrix,
        "raw_articles": raw
    }
