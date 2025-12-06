import os
import time
import argparse
import pandas as pd
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditScraper:
    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service()
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def scrape_subreddit(self, subreddit, limit=1000, sort_by="hot"):
        url = f"https://www.reddit.com/r/{subreddit}/{sort_by}/"
        logger.info(f"Scraping subreddit: {subreddit} | Sort: {sort_by}")
        self.driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        posts = soup.find_all('div', {'data-testid': 'post-container'})
        data = []
        for post in posts:
            title_elem = post.find('h3')
            if title_elem:
                title = title_elem.text
            else:
                continue
            score_elem = post.find('div', {'data-click-id': 'score'})
            score = score_elem.text if score_elem else '0'
            comments_elem = post.find('span', string=lambda x: x and 'comment' in x.lower())
            comments = comments_elem.text if comments_elem else '0 comments'
            data.append({
                'title': title,
                'score': score,
                'comments': comments,
                'subreddit': subreddit
            })
        df = pd.DataFrame(data)
        logger.info(f"Scraped {len(df)} posts from r/{subreddit}")
        return df

    def scrape_multiple_subreddits(self, subreddits, limit_per_sub=1000):
        all_data = []
        for sub in subreddits:
            df = self.scrape_subreddit(sub, limit=limit_per_sub)
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def save_data(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")

        stats = {
            'total_posts': len(df),
            'columns': df.columns.tolist()
        }
        logger.info(f"Scraping Statistics: {stats}")
        return stats

    def close(self):
        self.driver.quit()
        logger.info("WebDriver closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    parser = argparse.ArgumentParser(description='Reddit Web Scraper (Selenium + BeautifulSoup)')
    parser.add_argument('--subreddit', type=str, default='artificialintelligence', help='Subreddit name to scrape')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum number of posts to scrape')
    parser.add_argument('--sort', type=str, default='hot', choices=['hot', 'new', 'top', 'rising'], help='Sorting method')
    parser.add_argument('--output', type=str, default='data/raw/reddit_data1.csv', help='Output file path')
    parser.add_argument('--multiple', type=str, nargs='+', help='Multiple subreddits to scrape')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)

    with RedditScraper(headless=args.headless) as scraper:
        if args.multiple:
            df = scraper.scrape_multiple_subreddits(args.multiple, limit_per_sub=args.limit)
        else:
            df = scraper.scrape_subreddit(args.subreddit, limit=args.limit, sort_by=args.sort)
        scraper.save_data(df, args.output)
        print(f"\nScraping completed successfully!")
        print(f"Total posts: {len(df)}")
        if df.empty:
            print("⚠️ No data scraped. The DataFrame is empty.")
        else:
            print(f"Columns: {df.columns.tolist()}")
            sample_cols = [col for col in ['title', 'score', 'comments', 'subreddit'] if col in df.columns]
            if sample_cols:
                print(df.head(3)[sample_cols])
            else:
                print("⚠️ Expected columns not found in DataFrame.")
        print(f"Data saved to: {args.output}")

if __name__ == "__main__":
    main()
