# data_retrieval.py
"""
            +----------------+
            | Configuration  |
            +----------------+
                    |
            +----------------+
            | main()         |
            | - Load config  |
            | - Load .env    |
            +----------------+
                    |
                +--------+
                |Reddit  |
                +--------+
                    |
               run_reddit()
                    |
        write CSV: text,published_at,post_id,title,upvotes,category
"""

import os
import json
import re
import requests
import pandas as pd
import requests.auth
from tqdm import tqdm
from typing import List, Any, Dict
from dotenv import load_dotenv
from datetime import datetime, timezone
from dateutil.parser import parse as parse_date


REQUIRED_COLUMNS = ["text", "published_at", "post_id", "title", "upvotes", "category"]
OUTPUT_DIR = "Data_MLReady"
OUTPUT_FILE = "comments_by_category.csv"


def _strip_timestamp_filter(q: str) -> str:
    """
    Rimuove eventuali pattern tipo: timestamp:1710720000..1711411199
    perché il filtro temporale lo applichiamo via config (start/end).
    """
    return re.sub(r'\s*timestamp:\d+\.\.\d+\s*', ' ', q).strip()


def extract_reddit_comments(children: List[Dict], min_score: int = 1) -> List[Dict]:
    results: List[Dict] = []
    for child in children:
        if child.get('kind') != 't1':
            continue

        data = child.get('data', {}) or {}
        score = data.get('score', 0)
        body = data.get('body', '') or ''

        if score >= min_score and not body.startswith('[deleted]') and body.strip():
            results.append({
                'id': data.get('id'),
                'author': data.get('author'),
                'score': score,
                'created_utc': datetime.fromtimestamp(
                    data.get('created_utc', 0), timezone.utc
                ).isoformat(),
                'body': body
            })

        replies = data.get('replies', {})
        if replies and isinstance(replies, dict):
            results.extend(
                extract_reddit_comments(replies.get('data', {}).get('children', []), min_score)
            )
    return results


def write_reddit_comments(comments: List[Dict[str, Any]]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ffpath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # Creiamo sempre un CSV pulito con le colonne richieste e nell’ordine richiesto
    df = pd.DataFrame(comments, columns=REQUIRED_COLUMNS)
    df.to_csv(ffpath, index=False)
    print(f"Wrote {len(df)} rows to {ffpath}.")


def run_reddit(
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    user_agent: str,
    subreddits: List[str],
    categories: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> None:
    # Auth
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {'grant_type': 'password', 'username': username, 'password': password}
    headers = {'User-Agent': user_agent}
    token_res = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth,
        data=data,
        headers=headers,
        timeout=30,
    )
    token_res.raise_for_status()
    headers['Authorization'] = f"bearer {token_res.json().get('access_token', '')}"

    start_time: datetime = config['start_time']
    end_time: datetime = config['end_time']
    limit: int = config.get('limit', 100)
    min_score: int = config.get('comment_score_min', 1)

    all_rows: List[Dict[str, Any]] = []

    # Loop per subreddit → categoria → query
    for subreddit in subreddits:
        for cat in categories:
            cat_name = cat.get("name", "Uncategorized")
            queries = cat.get("queries", []) or []

            for raw_q in queries:
                q = _strip_timestamp_filter(raw_q)

                # Cerca i post
                resp = requests.get(
                    f'https://oauth.reddit.com/r/{subreddit}/search',
                    headers=headers,
                    params={'q': q, 'limit': limit, 'sort': 'relevance', 'restrict_sr': True},
                    timeout=30,
                )
                posts = resp.json().get('data', {}).get('children', [])

                # Estrai ID e titoli post
                post_tasks: List[Dict[str, Any]] = [
                    p.get('data', {}) for p in posts if isinstance(p, dict)
                ]

                # Scarica e filtra i commenti
                for post_data in tqdm(post_tasks, desc=f"Reddit comments [{subreddit} | {cat_name}]"):
                    post_id = post_data.get('id')
                    title = post_data.get('title', '') or ''

                    # comment tree
                    resp_c = requests.get(
                        f'https://oauth.reddit.com/comments/{post_id}',
                        headers=headers,
                        params={'depth': 10, 'limit': 500},
                        timeout=30,
                    )

                    try:
                        json_data = resp_c.json()
                        if (
                            isinstance(json_data, list)
                            and len(json_data) > 1
                            and isinstance(json_data[1], dict)
                            and 'data' in json_data[1]
                        ):
                            comment_mass = json_data[1]['data'].get('children', [])
                        else:
                            comment_mass = []
                    except ValueError:
                        comment_mass = []

                    high_comments = extract_reddit_comments(comment_mass, min_score=min_score)

                    for comment in high_comments:
                        pub = parse_date(comment['created_utc'])
                        if start_time <= pub <= end_time:
                            all_rows.append({
                                'text': comment.get('body', ''),
                                'published_at': comment.get('created_utc', ''),
                                'post_id': post_id,
                                'title': title,
                                'upvotes': comment.get('score', 0),
                                'category': cat_name
                            })

    # Scrivi CSV finale con schema fisso
    write_reddit_comments(all_rows)


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    start_time = parse_date(cfg["time_window"]["start"])
    end_time = parse_date(cfg["time_window"]["end"])

    reddit_subreddits = cfg["reddit_subreddits"]
    reddit_categories = cfg["reddit_categories"]

    load_dotenv()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, username, password, user_agent]):
        raise ValueError("Reddit API credentials not found. Please set the Reddit environment variables.")

    scraping_config = {
        'REDDIT': {
            'limit': cfg.get('limit', 1000),
            'comment_score_min': cfg.get('reddit_min_upvotes', 1),
            'start_time': start_time,
            'end_time': end_time
        }
    }

    try:
        print("Starting Reddit data retrieval...")
        run_reddit(
            client_id,
            client_secret,
            username,
            password,
            user_agent,
            reddit_subreddits,
            reddit_categories,
            scraping_config['REDDIT']
        )
        print("Reddit data retrieval complete.")
    except Exception as e:
        print(f"An error occurred during data retrieval: {e}")
        raise
    finally:
        print("\nData retrieval process completed.\n")


if __name__ == "__main__":
    main()
