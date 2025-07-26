# YouTube & Reddit Data Scraping Script

This Python script retrieves data from YouTube and Reddit APIs for research purposes. It searches for videos and posts related to specified topics and extracts comments with configurable score thresholds.

## Features

- **YouTube**: Retrieves top videos for a given topic and extracts all comments
- **Reddit**: Searches posts in specified subreddits and extracts high-scoring comments
- **Configurable**: Enable/disable each platform via flags in the main function
- **Data Export**: Saves YouTube comments to CSV and Reddit data to JSON

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install requests pandas python-dotenv
```

### 3. Create .env File

Create a `.env` file in the project root with your API credentials:

```
# YouTube API
YT_API_KEY=your_youtube_api_key_here

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password
REDDIT_USER_AGENT=your_app_name/1.0
```

### 4. Get API Credentials

**YouTube API:**
- Go to Google Cloud Console
- Enable YouTube Data API v3
- Create credentials and get API key

**Reddit API:**
- Go to reddit.com/prefs/apps
- Create a new app (script type)
- Get client ID and secret

## Usage

Run the script:

```bash
python yt_data.py
```

Configure platforms in the `flags` dictionary within the `main()` function:

```python
flags = {
    'YT': True,      # Enable YouTube scraping
    'REDDIT': True   # Enable Reddit scraping
}
```

## Output

- **fftext_data.csv**: CSV file in `Data_MLReady/` directory