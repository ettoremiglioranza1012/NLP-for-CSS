# yt_data.py
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
            /                \
      +---------+        +----------+
      | YouTube |        |  Reddit  |
      +---------+        +----------+
           |                  |
     get_top_videos()   run_reddit()
           |                  |
  get_all_comments()   extract_reddit_comments()
           |                  |
     run_youtube()     save_reddit_comments()
           |                  |
     merge_all_data()     merge to comments_category1.csv
           |
     save as CSV
"""

import os
import time
import json
import requests
import pandas as pd
import requests.auth
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from datetime import datetime, timezone
from dateutil.parser import parse as parse_date

def get_top_videos(topic: str, api_key: str, max_results: int) -> List[str]:
    search_url = "https://www.googleapis.com/youtube/v3/search"
    videos: List[str] = []

    params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "maxResults": max_results,
        "order": "relevance",
        "key": api_key,
    }

    res = requests.get(search_url, params=params).json()
    items = res.get("items", [])
    videos.extend(item['id']['videoId'] for item in items if 'videoId' in item['id'])

    return videos

def get_all_comments(video_id: str, api_key: str, max_results: int) -> List[dict[str, str]]:
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet",
        "id": video_id,
        "key": api_key
    }

    video_response = requests.get(video_url, params=video_params).json()
    items = video_response.get("items", [])
    if not items:
        return []

    video_title = items[0]["snippet"]["title"]

    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "textFormat": "plainText",
        "key": api_key
    }

    response = requests.get(url, params=params).json()
    items = response.get("items", [])

    for item in items:
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "text": snippet["textDisplay"],
            "published_at": snippet["publishedAt"],
            "post_id": video_id,
            "title": video_title,
            "upvotes": snippet.get("likeCount", 0)
        })

    return comments

def merge_all_data(dfs: list[pd.DataFrame]) -> None:
    ffname = "comments_category1.csv"
    db_dir_name = "Data_MLReady"
    os.makedirs(db_dir_name, exist_ok=True)
    ffpath = os.path.join(db_dir_name, ffname)

    try:
        if os.path.exists(ffpath):
            ff = pd.read_csv(ffpath)
        else:
            ff = pd.DataFrame()

        for curr_df in dfs:
            ff = pd.concat([ff, curr_df], ignore_index=True)

        ff.to_csv(ffpath, index=False)
        print("Data correctly concatenated to final db")
    except Exception as e:
        print(f"An error occurred while merging data: {e}")

def run_youtube(api_key: str, topics: list[str], config: dict[str, any]) -> None:
    # collect all videos across topics
    video_ids: List[str] = []
    for topic in topics:
        video_ids.extend(get_top_videos(topic, api_key, config["max_results_video"]))

    all_comments: List[dict[str, any]] = []
    for video_id in tqdm(video_ids, desc="YouTube comments"):
        video_comments = get_all_comments(video_id, api_key, config["max_results_comments"])
        for comment in video_comments:
            published = parse_date(comment["published_at"])
            if (
                config["start_time"] <= published <= config["end_time"]
                and comment.get("upvotes", 0) >= config.get("min_upvotes", 0)
            ):
                all_comments.append(comment)

    if all_comments:
        df = pd.DataFrame(all_comments)
        merge_all_data([df])

def extract_reddit_comments(children: List[dict], min_score: int = 5) -> List[dict]:
    results = []
    for child in children:
        kind = child.get('kind')
        data = child.get('data', {})
        if kind != 't1':
            continue
        score = data.get('score', 0)
        body = data.get('body', '')
        if score >= min_score and not body.startswith('[deleted]') and not body.startswith('[removed]') and body.strip():
            results.append({
                'id': data.get('id'),
                'author': data.get('author'),
                'score': score,
                'created_utc': datetime.fromtimestamp(data.get('created_utc', 0), timezone.utc).isoformat(),
                'body': body
            })
        replies = data.get('replies', {})
        if replies and isinstance(replies, dict):
            results.extend(
                extract_reddit_comments(replies.get('data', {}).get('children', []), min_score)
            )
    return results

def save_reddit_comments(comments: list[dict[str, any]]) -> None:
    db_dir_name = "Data_MLReady"
    os.makedirs(db_dir_name, exist_ok=True)
    ffname = "reddit_comments.csv"
    ffpath = os.path.join(db_dir_name, ffname)
    df = pd.DataFrame(comments)
    if os.path.exists(ffpath):
        origin = pd.read_csv(ffpath)
        origin = pd.concat([origin, df], ignore_index=True)
        origin.to_csv(ffpath, index=False)
    else:
        df.to_csv(ffpath, index=False)
    print("Table correctly updated.")

def run_reddit(
    client_id: str,
    client_secret: str,
    username: str,
    password: str,
    user_agent: str,
    subreddits: list[str],
    queries: list[str],
    config: dict[str, any]
) -> None:
    # authenticate once
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {'grant_type': 'password', 'username': username, 'password': password}
    headers = {'User-Agent': user_agent}
    token_res = requests.post(
        'https://www.reddit.com/api/v1/access_token',
        auth=auth,
        data=data,
        headers=headers
    )
    headers['Authorization'] = f"bearer {token_res.json().get('access_token', '')}"


    # collect all posts
    post_tasks: list[dict[str, any]] = []
    for subreddit in subreddits:
        for query in queries:
            params = {
                'q': query,
                'limit': config['limit'],
                'sort': 'new',
                'restrict_sr': True
            }
            # add after if specified
            if 'after' in config and config['after']:
                params['after'] = config['after']

            resp = requests.get(
                f'https://oauth.reddit.com/r/{subreddit}/search',
                headers=headers,
                params=params
            )
            posts = resp.json().get('data', {}).get('children', [])
            for post in posts:
                post_tasks.append(post.get('data', {}))

    all_comments: list[dict[str, any]] = []
    for post_data in tqdm(post_tasks, desc="Reddit comments"):
        post_id = post_data.get('id')
        title = post_data.get('title', '')

        # fetch comment tree
        resp = requests.get(
            f'https://oauth.reddit.com/comments/{post_id}',
            headers=headers,
            params={'depth': 10, 'limit': 500}
        )

        # safely parse JSON and extract comment list
        try:
            json_data = resp.json()
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
            # non-JSON response
            comment_mass = []

        # filter & flatten
        high_comments = extract_reddit_comments(comment_mass, min_score=config.get('comment_score_min', 0))
        for comment in high_comments:
            pub = parse_date(comment['created_utc'])
            if config['start_time'] <= pub <= config['end_time']:
                all_comments.append({
                    'text': comment.get('body', ''),
                    'published_at': comment.get('created_utc', ''),
                    'post_id': post_id,
                    'title': title,
                    'upvotes': comment.get('score', 0)
                })

    if all_comments:
        save_reddit_comments(all_comments)


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    start_time = parse_date(cfg["time_window"]["start"])
    end_time = parse_date(cfg["time_window"]["end"])

    youtube_topics = cfg["youtube_topics"]
    reddit_subreddits = cfg["reddit_subreddits"]
    reddit_queries = cfg["reddit_queries"]

    load_dotenv()
    API_KEY = os.getenv("YT_API_KEY")
    if not API_KEY:
        raise ValueError("YouTube API key not found. Please set the YT_API_KEY environment variable.")

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, username, password, user_agent]):
        raise ValueError("Reddit API credentials not found. Please set the Reddit environment variables.")

    scraping_config = {
        'YT': {
            'max_results_video': cfg.get('max_results_video', 50),
            'max_results_comments': cfg.get('max_results_comments', 100),
            'start_time': start_time,
            'end_time': end_time,
            'min_upvotes': cfg.get('youtube_min_upvotes', 0)
        },
        'REDDIT': {
            'limit': cfg.get('limit', 1000),
            'comment_score_min': cfg.get('reddit_min_upvotes', 5),
            'start_time': start_time,
            'end_time': end_time,
            'after': cfg.get('REDDIT', {}).get('after') 
        }
    }

    try:
        print("Starting YouTube data retrieval...")
        # run_youtube(API_KEY, youtube_topics, scraping_config['YT'])
        print("YouTube data retrieval complete.")

        print("Starting Reddit data retrieval...")
        run_reddit(
            client_id,
            client_secret,
            username,
            password,
            user_agent,
            reddit_subreddits,
            reddit_queries,
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
