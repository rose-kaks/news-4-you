# news_pipeline.py

import os, json, requests, hashlib
import spacy
from datetime import datetime, timezone, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from rapidfuzz import process, fuzz


DB_FILE = "queue_db.json"

newsapi_key = os.getenv("NEWSAPI_KEY")
gnews_key = os.getenv("GNEWS_KEY")
mstack_key = os.getenv("MEDIASTACK_KEY")

nlp = spacy.load("en_core_web_sm")

# #---------summarize text -----------------
# def summarize_text(text, max_sentences=2):
#     if not text or len(text) <= 200:
#         return text
    
#     doc = nlp(text)
#     sentences = [sent.text.strip() for sent in doc.sents]
    
#     summary = " ".join(sentences[:max_sentences])
    
#     if len(summary) > 200:
#         summary = summary[:200] + "..."
        
#     return summary

# ---------------- DB ----------------
def load_db():
    if not os.path.exists(DB_FILE):
        return {"queue": [], "posted": [], "recent_topics": {}}

    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Corrupted queue_db.json detected, resetting DB")
        return {"queue": [], "posted": [], "recent_topics": {}}


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

lemmatizer = WordNetLemmatizer()

KNOWN_ENTITIES = [
    "Narendra Modi",
    "Joe Biden",
    "Amit Shah"
]

ACRONYMS = {
    "govt": "Government",
    "us": "United States",
    "u.s.": "United States",
    "pm": "Prime Minister"
}

def resolve_entity(name):
    for e in KNOWN_ENTITIES:
        if fuzz.partial_ratio(name.lower(), e.lower()) > 80:
            return e
    return None


def canonical_synonym(word):
    try:
        word = word.lower()
        lemma = lemmatizer.lemmatize(word)
        synsets = wordnet.synsets(lemma, pos=wordnet.NOUN)
        if not synsets:
            return None

        best = max(synsets[0].lemmas(), key=lambda l: l.count())
        return best.name().replace("_", " ").title()
    except:
        return None


def normalize_topic(name, label=None):
    name = name.strip()

    # Acronym expansion
    if name.lower() in ACRONYMS:
        return ACRONYMS.get(name.lower())

    # Person/entity fuzzy match
    entity = resolve_entity(name)
    if entity:
        return entity

    # WordNet lexical normalization (ONLY for common nouns)
    if label not in ["PERSON", "ORG", "GPE"]:
        synonym = canonical_synonym(name)
        if synonym:
            return synonym

    # Fallback formatting
    return name.title()

def cluster_articles(articles, threshold=0.40):
    # texts = [
    #     (a.get("title", "") + " " + a.get("title", "") + " " + a.get("desc", "")).lower()
    #     for a in articles
    # ]

    texts = []
    for a in articles:
        entity_blob = " ".join(a.get("entities", []))
        combined = f"{a.get('title','')} {a.get('desc','')} {entity_blob} {entity_blob}"
        texts.append(combined.lower())

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
            entity_overlap = len(set(articles[i]["entities"]) & set(articles[j]["entities"]))
            if j not in used and (similarity_matrix[i][j] >= threshold or entity_overlap >= 2):
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
            "topic": topic_name,
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
        print("🧾 Queue size before posting:", len(db["queue"]))

from datetime import timedelta

def topic_allowed(db, topic, cooldown_hours=6):
    last_time_iso = db["recent_topics"].get(topic)
    if not last_time_iso:
        return True

    last_time = datetime.fromisoformat(last_time_iso)
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - last_time > timedelta(hours=cooldown_hours)


    # return datetime.now() - datetime.fromisoformat(last_time) > timedelta(hours=cooldown_hours)


COOLDOWN_HOURS = 36

def clean_recent_topics(db):
    now = datetime.now(timezone.utc)
    new_recent = {}

    for topic, ts in db["recent_topics"].items():
        last_time = datetime.fromisoformat(ts)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
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
    db["recent_topics"][item["topic"]] = datetime.now(timezone.utc).isoformat()

    # Keep last 50 posts only
    db["posted"] = db["posted"][-50:]

# ---------------- MAIN ENTRY ----------------


def get_next_article(query="technology india"):
    db = load_db()
    clean_recent_topics(db)
    raw = fetch_master_news(query)
    
    for a in raw:
        doc = nlp(a["title"] + " " + a["desc"])
        entities = []
        for e in doc.ents:
            if e.label_ in ["ORG", "PERSON", "GPE"]:
                clean = normalize_topic(e.text, e.label_)
                if clean:
                    entities.append(clean)

        a["entities"] = list(set(entities))  # remove duplicates

    clusters, sim_matrix, tfidf_matrix = cluster_articles(raw)

    stories = [
        build_story_from_cluster(cluster, raw)
        for cluster in clusters
    ]

    stories.sort(key=lambda x: x["size"], reverse=True)

    trends_for_queue = []
    for s in stories:
            best = max(s['articles'], key=score_article)
            if s["entities"]:
                topic_name = max(s["entities"], key=lambda e: sum(e in a["entities"] for a in s["articles"]))
            else:
                topic_name = "General"
            trends_for_queue.append({
                "story_id": s["story_id"],
                "topic": topic_name,
                "best_article": best,
                "subtopics": [a['title'] for a in s['articles']]
            })

    add_trends_to_queue(db, trends_for_queue)

    chosen=get_next_post(db)

    if chosen:
        mark_posted(db, chosen)  # This handles the 'pop' from queue to posted

    save_db(db)

    return {
        "article": chosen,
        "clusters": clusters,
        "similarity_matrix": sim_matrix,
        "tfidf_matrix": tfidf_matrix,
        "raw_articles": raw
    }
