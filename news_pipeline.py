# news_pipeline.py

import os, json, requests, hashlib
import spacy
from datetime import datetime, timezone, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime, timedelta

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
    try:
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
    except: print("NewsAPI error")

    # GNews
    try:
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
    except: print("GNews error")

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
def normalize_topic(name, label):
    """
    Cleans and standardizes entity names based on spaCy labels.
    """
    name = name.strip()

    acronym_map = {
        "US": "United States",
        "U.S.": "United States",
        "UK": "United Kingdom",
        "U.K.": "United Kingdom",
        "PMO": "PMO India",
        "BJP": "BJP"
    }

    if name.upper() in acronym_map:
        return acronym_map[name.upper()]

    aliases = {
        "narendra modi": "Narendra Modi",
        "pm modi": "Narendra Modi",
        "modi": "Narendra Modi",
        "amit shah": "Amit Shah",
        "joe biden": "Joe Biden",
        "biden": "Joe Biden"
    }

    lower_name = name.lower()
    if lower_name in aliases:
        return aliases[lower_name]

    if name.isupper() and len(name) <= 5:
        return name

    return name.title()


def cluster_articles(articles, threshold=0.55):
    texts = [
        (a.get("title", "") + " " + a.get("desc", "")).lower()
        for a in articles
    ]

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
            if j not in used and similarity_matrix[i][j] >= threshold:
                group.append(j)
                used.add(j)
        clusters.append(group)

    return clusters, similarity_matrix, tfidf_matrix

def generate_story_id(entities, title):
    base = " ".join(sorted(entities)) + title[:60]
    return hashlib.md5(base.encode()).hexdigest()

def build_story_from_cluster(cluster, articles):
    from collections import Counter

    story_articles = [articles[i] for i in cluster]

    all_entities = []
    for a in story_articles:
        all_entities.extend(a.get("entities", []))

    entity_counts = Counter(all_entities)

    main_entities = [e for e, c in entity_counts.most_common(3)]

    story_id = generate_story_id(main_entities, story_articles[0]["title"])

    return {
        "story_id": story_id,
        "size": len(cluster),
        "entities": main_entities,
        "representative_title": story_articles[0]["title"],
        "articles": story_articles
    }

def add_trends_to_queue(db, trends):

    existing_story_ids = {
        x.get('story_id') for x in db['queue'] + db['posted']
        if x.get('story_id')
    }


    existing_urls = {x['url'] for x in db['queue']} | {x['url'] for x in db['posted']}

    for t in trends:

        if t["story_id"] in existing_story_ids:
            continue

        art = t["best_article"]
        topic_name = t["topic"]

        if art['url'] in existing_urls:
            continue

        if not topic_allowed(db, topic_name):
            continue

        db['queue'].append({
            "story_id": t["story_id"],
            "topic": t["topic"],
            "subtopics": t["subtopics"],
            "title": art["title"],
            "desc": art["desc"],
            "url": art["url"],
            "image": art["image"],
            "source": art["source"],
            "score": score_article(art),
            "added_time": datetime.now(timezone.utc).isoformat()
        })

        existing_urls.add(art['url'])

from datetime import timedelta

def topic_allowed(db, topic, cooldown_hours=6):
    last_time = db["recent_topics"].get(topic)
    if not last_time:
        return True
    return datetime.now() - datetime.fromisoformat(last_time) > timedelta(hours=cooldown_hours)

COOLDOWN_HOURS = 36

def clean_recent_topics(db):
    now = datetime.now(timezone.utc)
    new_recent = {}

    for topic, ts in db["recent_topics"].items():
        last_time = datetime.fromisoformat(ts)
        if now - last_time < timedelta(hours=COOLDOWN_HOURS):
            new_recent[topic] = ts  # still cooling
        else:
            print(f"♻️ Cooldown expired for {topic}")

    db["recent_topics"] = new_recent

def get_next_post(db):
    if not db["queue"]:
        return None
    # Sort by score + recency
    sorted_queue = sorted(db["queue"], key=lambda x: (-x["score"], x["added_time"]))

    for item in sorted_queue:
        if topic_allowed(db, item["topic"]):
            return item
    return None

def mark_posted(db, item):
    db["queue"].remove(item)
    db["posted"].append(item)
    db["recent_topics"][item["topic"]] = datetime.now().isoformat()

    # Keep last 50 posts only
    db["posted"] = db["posted"][-50:]

# ---------------- MAIN ENTRY ----------------
def get_next_article(query="technology india"):
    db = load_db()
    clean_recent_topics(db)
    raw = fetch_master_news(query)

    for a in raw:
        doc = nlp(a["title"] + " " + a["desc"])
        a["entities"] = [normalize_topic(e.text, e.label_) for e in doc.ents
                        if e.label_ in ["ORG", "PERSON", "GPE"]]

    clusters, sim_matrix, tfidf_matrix = cluster_articles(raw)

    stories = [
        build_story_from_cluster(cluster, raw)
        for cluster in clusters
    ]

    stories.sort(key=lambda x: x["size"], reverse=True)

    trends_for_queue = []
    for s in stories:
            best = max(s['articles'], key=score_article)
            trends_for_queue.append({
                "story_id": s["story_id"],
                "topic": s["entities"][0] if s["entities"] else "General",
                "best_article": best,
                "subtopics": [a['title'] for a in s['articles']]
            })

    add_trends_to_queue(db, trends_for_queue)

    chosen=get_next_post(db)

    if chosen:
        mark_posted(db, chosen)

    save_db(db)

    return {
        "article": chosen,
        "clusters": clusters,
        "similarity_matrix": sim_matrix,
        "tfidf_matrix": tfidf_matrix,
        "raw_articles": raw
    }
