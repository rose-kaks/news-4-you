# insta_publish.py

import requests
import time
import os

IG_USER_ID = os.getenv("INSTAGRAM_USER_ID")
ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
GRAPH_URL = "https://graph.facebook.com/v19.0"


def create_image_container(image_url, caption=None):
    payload = {
        "image_url": image_url,
        "is_carousel_item": True,
        "access_token": ACCESS_TOKEN
    }
    if caption:
        payload["caption"] = caption

    r = requests.post(
        f"{GRAPH_URL}/{IG_USER_ID}/media",
        data=payload
    )
    r.raise_for_status()
    return r.json()["id"]


def create_carousel_container(children_ids, caption):
    payload = {
        "media_type": "CAROUSEL",
        "children": ",".join(children_ids),
        "caption": caption,
        "access_token": ACCESS_TOKEN
    }

    r = requests.post(
        f"{GRAPH_URL}/{IG_USER_ID}/media",
        data=payload
    )
    r.raise_for_status()
    return r.json()["id"]


def publish_container(container_id):
    payload = {
        "creation_id": container_id,
        "access_token": ACCESS_TOKEN
    }

    r = requests.post(
        f"{GRAPH_URL}/{IG_USER_ID}/media_publish",
        data=payload
    )
    r.raise_for_status()
    return r.json()


def post_carousel(image_urls, caption):
    child_ids = []

    for url in image_urls:
        cid = create_image_container(url)
        child_ids.append(cid)
        time.sleep(2)  # Instagram rate safety

    carousel_id = create_carousel_container(child_ids, caption)
    time.sleep(5)

    return publish_container(carousel_id)
