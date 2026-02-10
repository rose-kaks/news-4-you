# news_pipeline.py
# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import os, json, requests, hashlib
import spacy # NLP library for named entity recognition
from datetime import datetime, timezone, timedelta
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer # For text similarity 
from sklearn.metrics.pairwise import cosine_similarity  # For comparing article similarity
import time
import nltk # Natural Language Toolkit for WordNet synonym lookups
# Download required NLTK data if not present
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
from nltk.corpus import wordnet  # For finding word synonyms
from nltk.stem import WordNetLemmatizer # For word normalization
from rapidfuzz import process, fuzz # For fuzzy string matching (entity resolution)
from sentence_transformers import SentenceTransformer # For generating semantic embeddings
import numpy as np
import hdbscan # Hierarchical clustering algorithm for grouping similar articles
from collections import defaultdict


from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model_summary = T5ForConditionalGeneration.from_pretrained(model_name)

#----SUMMARIZATION---------------
"""
    Generates a concise, abstractive summary of news text using a 
    Transformer-based Seq2Seq model. Optimized for Instagram carousel slides.
    """

    # 1. ENCODING & TOKENIZATION
    # Convert raw text into numerical tensors. 
    # 'summarize: ' is the specific task prefix required by T5 models.
    # 'max_length=1024' ensures we stay within the model's positional embedding limits.
    # 2. MODEL INFERENCE (GENERATION)
    # length_penalty=2.0: Higher value encourages longer, more descriptive summaries 
    # (better for filling 1080x1080 carousel slides than short snippets).
    # num_beams=4: Uses Beam Search to evaluate the top 4 word sequences for better grammar.
    # 3. DECODING
    # Converts token IDs back to human-readable text.
    # skip_special_tokens=True: Removes T5-specific markers like </s> and <pad>.
def summarize(text):
    if len(text) < 70:
        return text
    
    inputs = tokenizer.encode("headline: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_summary.generate(inputs, max_new_tokens=40, length_penalty=1, num_beams=5, do_sample=False, early_stopping=False, eos_token_id=tokenizer.eos_token_id, length_penalty=1.0 )     
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DB_FILE = "queue_db.json" # Local database file to store queue and posted articles

# Load API keys from environment variables
newsapi_key = os.getenv("NEWSAPI_KEY")  # NewsAPI.org key
gnews_key = os.getenv("GNEWS_KEY") # GNews.io key
mstack_key = os.getenv("MEDIASTACK_KEY") # Mediastack key

# Load spaCy language model for NLP tasks (entity extraction)
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

# --------------------------------------------------
# DATABASE MANAGEMENT
# --------------------------------------------------
# ---------------- DB ----------------
def load_db():
    """
    Loads the article database from disk.
    Database structure:
    - queue: Articles waiting to be posted
    - posted: Articles that have been published
    - recent_topics: Topic names with their last post timestamp (for cooldown)
    
    Returns:
        Dictionary with queue, posted, and recent_topics
    """
    # Create new database if file doesn't exist
    if not os.path.exists(DB_FILE):
        return {"queue": [], "posted": [], "recent_topics": {}}

    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If file is corrupted, reset to empty database
        print("⚠️ Corrupted queue_db.json detected, resetting DB")
        return {"queue": [], "posted": [], "recent_topics": {}}


def save_db(db):
    """
    Persists the database to disk as JSON.
    
    Args:
        db: Database dictionary to save
    """
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=4)


# --------------------------------------------------
# ARTICLE SCORING
# --------------------------------------------------
# ---------------- SCORING ----------------
def score_article(article):
    """
    Assigns a quality score to an article based on various criteria.
    Higher scores indicate better articles to post.
    
    Scoring factors:
    - Has image: +2 points
    - Long description (>120 chars): +1 point
    - Trusted source (Reuters, BBC, etc.): +3 points
    - Title has colon (indicates detail): +1 point
    - Breaking news keywords: +2 points
    
    Args:
        article: Article dictionary
    
    Returns:
        Integer score (typically 0-9)
    """
    score = 0
    if article.get("image"): score += 2
    if len(article.get("desc","")) > 120: score += 1

    # Bonus for trusted news sources
    trusted = ["reuters","bbc","the hindu","indian express","ndtv"]
    if any(t in article.get("source","").lower() for t in trusted):
        score += 3
    # Detailed titles often indicate quality
    if ":" in article.get("title",""): score += 1
    # Breaking news and announcements are high-value
    if any(w in article["title"].lower()
           for w in ["breaking","announces","launches","wins"]):
        score += 2

    return score

#--------------------------------------------------
# NEWS FETCHING FROM MULTIPLE APIS
# --------------------------------------------------
# ---------------- FETCH ----------------
def fetch_master_news(query):
    """
    Fetches news articles from multiple news APIs and deduplicates them.
    Uses three sources: NewsAPI, GNews, and Mediastack.
    
    Args:
        query: Search query string (e.g., "technology india")
    
    Returns:
        List of article dictionaries with standardized format:
        {title, url, desc, source, image}
    """
    articles, seen = [], set()  # seen tracks URLs to prevent duplicates

    # --------------------------------------------------
    # SOURCE 1: NewsAPI
    # --------------------------------------------------
    # NewsAPI
    try:
        url = f"https://newsapi.org/v2/top-headlines?q={query}&apiKey={newsapi_key}"
        for a in requests.get(url).json().get("articles", []):
            # Skip if we've already seen this URL
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

    # --------------------------------------------------
    # SOURCE 2: GNews
    # --------------------------------------------------
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

    # --------------------------------------------------
    # SOURCE 3: Mediastack
    # --------------------------------------------------
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

# --------------------------------------------------
# NLP: ENTITY NORMALIZATION
# --------------------------------------------------
# ---------------- NLP + SIMILARITY ----------------

# Entity Normalization

lemmatizer = WordNetLemmatizer()  # For converting words to their base form

# Known entities for fuzzy matching (helps normalize similar spellings)
KNOWN_ENTITIES = [
    "Narendra Modi",
    "Joe Biden",
    "Amit Shah"
]

# Common acronym expansions
ACRONYMS = {
    "govt": "Government",
    "us": "United States",
    "u.s.": "United States",
    "pm": "Prime Minister"
}

def resolve_entity(name):
    """
    Attempts to match a name to a known entity using fuzzy string matching.
    Helps normalize variations like "N. Modi" -> "Narendra Modi"
    
    Args:
        name: Entity name to resolve
    
    Returns:
        Canonical entity name if match found (>80% similarity), else None
    """
    for e in KNOWN_ENTITIES:
        if fuzz.partial_ratio(name.lower(), e.lower()) > 80:
            return e
    return None


def canonical_synonym(word):
    """
    Finds the most common synonym for a word using WordNet.
    Example: "car" and "automobile" both map to "Car"
    
    Args:
        word: Word to find canonical form for
    
    Returns:
        Most common synonym in title case, or None if not found
    """
    try:
        word = word.lower()
        lemma = lemmatizer.lemmatize(word)   # Get base form (e.g., "running" -> "run")
        synsets = wordnet.synsets(lemma, pos=wordnet.NOUN) # Get synonym sets
        if not synsets:
            return None
        # Pick the most commonly used synonym from the first synset
        best = max(synsets[0].lemmas(), key=lambda l: l.count())
        return best.name().replace("_", " ").title()
    except:
        return None


def normalize_topic(name, label=None):
    """
    Normalizes topic names to a canonical form for better grouping.
    
    Process:
    1. Expand acronyms (PM -> Prime Minister)
    2. Match to known entities (fuzzy matching)
    3. Find WordNet synonyms (only for common nouns, not people/orgs)
    4. Fall back to title case
    
    Args:
        name: Topic/entity name to normalize
        label: spaCy entity label (PERSON, ORG, GPE, etc.)
    
    Returns:
        Normalized topic name
    """
    name = name.strip()

    # Acronym expansion
    if name.lower() in ACRONYMS:
        return ACRONYMS.get(name.lower())

    # Person/entity fuzzy match
    entity = resolve_entity(name)
    if entity:
        return entity

    # WordNet lexical normalization (ONLY for common nouns)
    # Don't apply to proper nouns (people, organizations, places)
    if label not in ["PERSON", "ORG", "GPE"]:
        synonym = canonical_synonym(name)
        if synonym:
            return synonym

    # Fallback formatting
    return name.title()

# --------------------------------------------------
# ARTICLE CLUSTERING: SEMANTIC EMBEDDINGS
# --------------------------------------------------
### Article Clustering into Stories

# 1. Initialize Model
# This model converts text into 384-dimensional vectors that capture semantic meaning
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Extract Sentence Embedding Logic
def get_weighted_embeddings(articles, entity_weight=0.8):
    """
    Generates semantic embeddings for articles with extra weight on entities.
    This helps cluster articles about the same topic even if worded differently.
    
    Process:
    1. Encode article text (title + description)
    2. Encode extracted entities separately
    3. Combine with weighted sum (entities get 0.8x weight)
    4. Re-normalize to unit length
    
    Args:
        articles: List of article dictionaries with entities
        entity_weight: How much to weight entity matches (default 0.8)
    
    Returns:
        Numpy array of normalized embeddings (one per article)
    """
    texts, entity_texts = [], []

    for a in articles:
        # Combine Title + Desc
        texts.append((a["title"] + " " + a["desc"]).lower())
        # Use entities to add their weight to embeddings
        entity_texts.append(" ".join(a.get("entities", [])).lower())
    # Generate embeddings for text and entities
    text_emb = model.encode(texts, normalize_embeddings=True)
    entity_emb = model.encode(entity_texts, normalize_embeddings=True)

    # Weighted combination + Re-normalization
    # Articles sharing entities will be closer in embedding space
    final_emb = text_emb + entity_weight * entity_emb
    final_emb = final_emb / np.linalg.norm(final_emb, axis=1, keepdims=True)
    return final_emb

# 3. Utility: Convert Noise (-1) to individual clusters
def convert_noise_to_clusters(labels):
    """
    HDBSCAN labels outliers as -1 (noise). This converts each noise point
    to its own unique cluster so we don't lose articles.
    
    Args:
        labels: Array of cluster labels from HDBSCAN
    
    Returns:
        New labels array with noise points assigned unique IDs
    """
    new_labels = labels.copy()
    # Start new IDs from the current max label + 1
    max_label = max(labels) if len(labels) > 0 else 0
    # Give each noise point its own cluster ID
    for i, label in enumerate(labels):
        if label == -1:
            max_label += 1
            new_labels[i] = max_label
    return new_labels

# 4. Use HDBSCAN method
def cluster_hdbscan_emb(articles):
    """
    Clusters articles into stories using HDBSCAN on semantic embeddings.
    Articles about the same event/topic will be grouped together.
    
    HDBSCAN advantages:
    - Automatically determines number of clusters
    - Handles varying cluster sizes
    - More robust than k-means for news articles
    
    Args:
        articles: List of article dictionaries
    
    Returns:
        List of clusters, where each cluster is a list of article indices
        Example: [[0,3,5], [1,2], [4]] means articles 0,3,5 are about the same story
    """
    # Get semantic embeddings that capture meaning
    embeddings = get_weighted_embeddings(articles)

    # Configure HDBSCAN for news clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2, # At least 2 articles per cluster
        min_samples=2, # Minimum samples for core point
        metric='euclidean', # Distance metric in embedding space
        cluster_selection_method='leaf' # 'leaf' is tighter for news
    )
    # Perform clustering
    labels = clusterer.fit_predict(embeddings)
    # Convert outliers (-1) to individual clusters
    labels = convert_noise_to_clusters(labels)
    
    # Re-grouping into list of indices
    # Convert from label array to grouped indices
    clusters_dict = defaultdict(list)
    for i, label in enumerate(labels):
        clusters_dict[label].append(i)
    
    return list(clusters_dict.values())

# def cluster_articles(articles, threshold=0.40):
#     texts = []
#     for a in articles:
#         entity_blob = " ".join(a.get("entities", []))
#         combined = f"{a.get('title','')} {a.get('desc','')} {entity_blob} {entity_blob}"
#         texts.append(combined.lower())

#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform(texts)

#     similarity_matrix = cosine_similarity(tfidf_matrix)

#     clusters, used = [], set()
#     for i in range(len(articles)):
#         if i in used:
#             continue
#         group = [i]
#         used.add(i)
#         for j in range(i + 1, len(articles)):
#             entity_overlap = len(set(articles[i]["entities"]) & set(articles[j]["entities"]))
#             if j not in used and (similarity_matrix[i][j] >= threshold or entity_overlap >= 2):
#                 group.append(j)
#                 used.add(j)
#         clusters.append(group)

#     return clusters, similarity_matrix, tfidf_matrix


# story id is added in queue db and used to filter out stories with same id
# Best article to represent that story is selected via 
# scoring system constituing if-then rules, refer score function


def generate_story_id(entities, title):
    """
    Creates a unique hash ID for a story based on its main entities and title.
    Same story from different sources will have the same ID.
    
    Args:
        entities: List of main entities in the story
        title: Title of representative article
    
    Returns:
        MD5 hash string as story ID
    """
    # Sort entities for consistency, truncate title to avoid minor variations
    base = " ".join(sorted(entities)) + title[:60]
    return hashlib.md5(base.encode()).hexdigest()

def build_story_from_cluster(cluster, articles):
    """
    Builds a story object from a cluster of related articles.
    
    Process:
    1. Extract articles in this cluster
    2. Find most common entities across all articles
    3. Generate unique story ID
    4. Package as story object
    
    Args:
        cluster: List of article indices in this cluster
        articles: Full list of all articles
    
    Returns:
        Story dictionary with: story_id, size, entities, representative_title, articles
    """
    from collections import Counter
    # Get all articles in this cluster
    story_articles = [articles[i] for i in cluster]
    # Collect all entities mentioned across all articles in cluster
    all_entities = []
    for a in story_articles:
        all_entities.extend(a.get("entities", []))
    # Count entity frequency
    entity_counts = Counter(all_entities)
    # Top 3 most common entities define this story's topic
    main_entities = [e for e, c in entity_counts.most_common(3)]
    # Generate unique ID for this story
    story_id = generate_story_id(main_entities, story_articles[0]["title"])

    return {
        "story_id": story_id, 
        "size": len(cluster), # How many articles about this story
        "entities": main_entities, # Key people/orgs/places
        "representative_title": story_articles[0]["title"],
        "articles": story_articles # All articles in this story
    }

def add_trends_to_queue(db, trends):
    """
    Adds new story trends to the posting queue, filtering out duplicates
    and articles that don't pass topic cooldown.
    
    Args:
        db: Database dictionary
        trends: List of story trends to potentially add
    """
    # Build sets of existing story IDs and URLs to avoid duplicates
    existing_story_ids = {
        x.get('story_id') for x in db['queue'] + db['posted']
        if x.get('story_id')
    }

    existing_urls = {x['url'] for x in db['queue']} | {x['url'] for x in db['posted']}

    for t in trends:
        # Skip if we've already queued/posted this story
        if t["story_id"] in existing_story_ids:
            continue

        art = t["best_article"]
        topic_name = t["topic"]
        # Skip if we've already used this exact article
        if art['url'] in existing_urls:
            continue
        # Check topic cooldown - don't post same topic too frequently
        if not topic_allowed(db, topic_name):
            continue
        # Add to queue
        db['queue'].append({
            "story_id": t["story_id"],
            "topic": topic_name,
            "subtopics": t["subtopics"],
            "title": summarize(art["title"]),
            "desc": art["desc"],
            "url": art["url"],
            "image": art["image"],
            "source": art["source"],
            "score": score_article(art),
            "added_time": datetime.now(timezone.utc).isoformat()
        })

        existing_urls.add(art['url'])
        print("🧾 Queue size before posting:", len(db["queue"]))

# --------------------------------------------------
# TOPIC COOLDOWN MANAGEMENT
# --------------------------------------------------
from datetime import timedelta

def topic_allowed(db, topic, cooldown_hours=6):
    """
    Checks if enough time has passed since we last posted about this topic.
    Prevents flooding the feed with the same topic.
    
    Args:
        db: Database dictionary
        topic: Topic name to check
        cooldown_hours: Hours to wait between posts on same topic (default 6)
    
    Returns:
        True if topic is allowed (cooldown expired or never posted), False otherwise
    """
    last_time_iso = db["recent_topics"].get(topic)
    if not last_time_iso:
        return True  # Never posted about this topic before
    # Parse timestamp and ensure it's timezone-aware
    last_time = datetime.fromisoformat(last_time_iso)
    if last_time.tzinfo is None:
        last_time = last_time.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - last_time > timedelta(hours=cooldown_hours)


    # return datetime.now() - datetime.fromisoformat(last_time) > timedelta(hours=cooldown_hours)


COOLDOWN_HOURS = 36  # Global cooldown period for all topics

def clean_recent_topics(db):
    """
    Removes topics from the cooldown tracker if their cooldown has expired.
    This keeps the database clean and allows topics to be posted again.
    
    Args:
        db: Database dictionary (modified in place)
    """
    now = datetime.now(timezone.utc)
    new_recent = {}

    for topic, ts in db["recent_topics"].items():
        # Parse and normalize timestamp
        last_time = datetime.fromisoformat(ts)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        # Keep in recent_topics only if still within cooldown period
        if now - last_time < timedelta(hours=COOLDOWN_HOURS):
            new_recent[topic] = ts  # still cooling
        else:
            print(f"♻️ Cooldown expired for {topic}")

    db["recent_topics"] = new_recent

#--------------------------------------------------
# QUEUE MAINTENANCE
# --------------------------------------------------
QUEUE_TTL_HOURS = 24   # Auto-delete queue items older than this

def clean_old_queue_items(db):
    """
    Removes stale articles from the queue that are too old to be relevant.
    News loses value quickly, so we auto-delete after 24 hours.
    
    Args:
        db: Database dictionary (modified in place)
    """
    now = datetime.now(timezone.utc)
    new_queue = []

    for item in db["queue"]:
        try:
            # Parse when this article was added
            added_time = datetime.fromisoformat(item["added_time"])
            if added_time.tzinfo is None:
                added_time = added_time.replace(tzinfo=timezone.utc)

            age_hours = (now - added_time).total_seconds() / 3600
            # Keep if still fresh
            if age_hours <= QUEUE_TTL_HOURS:
                new_queue.append(item)
            else:
                print(f"🗑️ Removing stale queue item: {item['title'][:60]}")

        except Exception as e:
            # Remove items with invalid timestamps
            print("⚠️ Invalid timestamp, removing:", item.get("title"))
    
    db["queue"] = new_queue

#--------------------------------------------------
# QUEUE RETRIEVAL
# --------------------------------------------------
def get_next_post(db):
    """
    Selects the next article to post from the queue.
    
    Selection criteria:
    1. Sort by score (quality) descending
    2. Then by recency (newer first)
    3. Filter out topics still in cooldown
    
    Args:
        db: Database dictionary
    
    Returns:
        Article dictionary to post, or None if queue is empty or all topics in cooldown
    """
    if not db["queue"]:
        return None
    # Sort by score + recency
    # sorted_queue = sorted(db["queue"], key=lambda x: (-x["score"], x["added_time"]))
    sorted_queue = sorted(
        db["queue"],
        key=lambda x: (-x["score"], datetime.fromisoformat(x["added_time"]))
    )

    # Find first article whose topic is not in cooldown
    for item in sorted_queue:
        if topic_allowed(db, item["topic"]):
            return item
    return None

def mark_posted(db, item):
    """
    Marks an article as posted by moving it from queue to posted list
    and recording the topic in recent_topics for cooldown tracking.
    
    Args:
        db: Database dictionary (modified in place)
        item: Article that was just posted
    """
    db["queue"].remove(item)
    db["posted"].append(item)
    # Record timestamp for topic cooldown
    db["recent_topics"][item["topic"]] = datetime.now(timezone.utc).isoformat()

    # Keep last 50 posts only
    db["posted"] = db["posted"][-50:]

#  --------------------------------------------------
# MAIN PIPELINE ENTRY POINT
# --------------------------------------------------
# ---------------- MAIN ENTRY ----------------


def get_next_article(query="technology india"):
    """
    Main pipeline function that orchestrates the entire news processing workflow.
    
    Complete workflow:
    1. Load database and clean old data
    2. Fetch fresh news from multiple APIs
    3. Extract entities using NLP
    4. Cluster articles into stories using semantic embeddings
    5. Build story objects and score articles
    6. Add new stories to queue (if not duplicates and pass cooldown)
    7. Select best article to post next
    8. Mark as posted and save database
    
    Args:
        query: Search query for news APIs (default: "technology india")
    
    Returns:
        Dictionary with:
        - article: Next article to post (or None)
        - clusters: Article clusters (grouped by story)
        - raw_articles: All fetched articles before processing
    """
    # --------------------------------------------------
    # STEP 1: LOAD AND CLEAN DATABASE
    # --------------------------------------------------
    db = load_db()
    clean_recent_topics(db)
    clean_old_queue_items(db)

    # --------------------------------------------------
    # STEP 2: FETCH FRESH NEWS
    # --------------------------------------------------
    raw = fetch_master_news(query)

    if not raw:
        return {"article": None, "clusters": [], "raw_articles": []}

    # --------------------------------------------------
    # STEP 3: EXTRACT ENTITIES WITH NLP
    # --------------------------------------------------
    for a in raw:
        # Use spaCy to extract named entities (people, organizations, places)
        doc = nlp(a["title"] + " " + a["desc"])
        entities = []
        for e in doc.ents:
            if e.label_ in ["ORG", "PERSON", "GPE"]:
                # Normalize entity name (fuzzy matching, synonym expansion, etc.)
                clean = normalize_topic(e.text, e.label_)
                if clean:
                    entities.append(clean)

        a["entities"] = list(set(entities))  # remove duplicates

    # clusters, sim_matrix, tfidf_matrix = cluster_articles(raw)
    clusters = cluster_hdbscan_emb(raw)
    # --------------------------------------------------
    # STEP 5: BUILD STORY OBJECTS
    # --------------------------------------------------
    stories = [
        build_story_from_cluster(cluster, raw)
        for cluster in clusters
    ]
    # Sort stories by size (bigger clusters = more coverage = more important)
    stories.sort(key=lambda x: x["size"], reverse=True)
    # --------------------------------------------------
    # STEP 6: PREPARE STORIES FOR QUEUE
    # --------------------------------------------------
    trends_for_queue = []
    for s in stories:
        # Pick best article from cluster based on quality score
        best = max(s["articles"], key=score_article)

        # Determine main topic: entity mentioned most across all articles in cluster
        if s["entities"]:
            topic_name = max(
                s["entities"],
                key=lambda e: sum(e in a["entities"] for a in s["articles"])
            )
        else:
            topic_name = "General"

        trends_for_queue.append({
            "story_id": s["story_id"],
            "topic": topic_name,
            "best_article": best,
            "subtopics": [a["title"] for a in s["articles"]],
        })

    # --------------------------------------------------
    # STEP 7: ADD TO QUEUE (WITH DEDUPLICATION)
    # --------------------------------------------------
    add_trends_to_queue(db, trends_for_queue)
    # --------------------------------------------------
    # STEP 8: SELECT NEXT ARTICLE TO POST
    # --------------------------------------------------
    chosen=get_next_post(db)

    if chosen:
        mark_posted(db, chosen)  # This handles the 'pop' from queue to posted
    # --------------------------------------------------
    # STEP 9: SAVE DATABASE
    # --------------------------------------------------
    save_db(db)

    return {
        "article": chosen,
        "clusters": clusters,
        # "similarity_matrix": sim_matrix,
        # "tfidf_matrix": tfidf_matrix,
        "raw_articles": raw
    }
