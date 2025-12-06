"""
Reddit Web Scraper using Selenium and BeautifulSoup
Automated web scraping to collect structured data from Reddit including:
- Post titles, bodies, upvotes, comments, timestamps, authors
- No API keys required - pure web scraping approach
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import time
import re

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditScraper:
    """Reddit data scraper using Selenium and BeautifulSoup"""
    
    def __init__(self, headless=True):
        """
        Initialize Selenium WebDriver for web scraping
        
        Args:
            headless (bool): Run browser in headless mode
        """
        chrome_options = Options()
        
        if headless:
            chrome_options.add_argument('--headless')
        
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
        logger.info("Selenium WebDriver initialized for web scraping")
        logger.info(f"Headless mode: {headless}")
    
    def scroll_page(self, scroll_pause_time=2, max_scrolls=10):
        """
        Scroll page to load more content dynamically
        
        Args:
            scroll_pause_time (float): Time to wait between scrolls
            max_scrolls (int): Maximum number of scrolls
        """
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        
        while scrolls < max_scrolls:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            
            # Calculate new scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                break
            
            last_height = new_height
            scrolls += 1
            logger.info(f"Scrolled {scrolls}/{max_scrolls} times")
    
    def scrape_subreddit(self, subreddit_name, limit=1000, sort_by='hot'):
        """
        Scrape posts from a specific subreddit using Selenium + BeautifulSoup
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Maximum number of posts to scrape
            sort_by (str): Sorting method ('hot', 'new', 'top', 'rising')
        
        Returns:
            pd.DataFrame: Scraped data
        """
        url = f"https://old.reddit.com/r/{subreddit_name}/{sort_by}/"
        logger.info(f"Starting to scrape r/{subreddit_name} (limit={limit}, sort={sort_by})")
        logger.info(f"Navigating to {url}")
        
        self.driver.get(url)
        time.sleep(3)  # Wait for page to load
        
        # Scroll to load more posts
        num_scrolls = max(5, limit // 25)
        self.scroll_page(scroll_pause_time=2, max_scrolls=num_scrolls)
        
        # Parse page content with BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        posts = soup.find_all('div', class_='thing')
        
        posts_data = []
        
        logger.info(f"Found {len(posts)} posts on page, extracting up to {limit}...")
        
        for idx, post in enumerate(tqdm(posts[:limit], desc="Scraping posts")):
            try:
                post_data = self._extract_post_data(post, subreddit_name)
                posts_data.append(post_data)
                
            except Exception as e:
                logger.error(f"Error scraping post {idx}: {str(e)}")
                continue
        
        df = pd.DataFrame(posts_data)
        logger.info(f"Successfully scraped {len(df)} posts from r/{subreddit_name}")
        
        return df
    
    def _extract_post_data(self, post_element, subreddit_name):
        """
        Extract data from a post element using BeautifulSoup
        
        Args:
            post_element: BeautifulSoup element containing post data
            subreddit_name (str): Name of subreddit
        
        Returns:
            dict: Extracted post data
        """
        try:
            # Extract post ID
            post_id = post_element.get('data-fullname', '').replace('t3_', '')
            
            # Extract title
            title_elem = post_element.find('a', class_='title')
            title = title_elem.text.strip() if title_elem else 'N/A'
            
            # Extract author
            author_elem = post_element.find('a', class_='author')
            author = author_elem.text if author_elem else '[deleted]'
            
            # Extract score/upvotes
            score_elem = post_element.find('div', class_='score unvoted')
            if not score_elem:
                score_elem = post_element.find('div', class_='score')
            score_text = score_elem.text if score_elem else '0'
            
            # Parse score (handle 'k' notation)
            try:
                if 'k' in score_text.lower():
                    upvotes = int(float(score_text.lower().replace('k', '')) * 1000)
                else:
                    upvotes = int(score_text) if score_text.isdigit() else 0
            except:
                upvotes = 0
            
            # Extract comment count
            comments_elem = post_element.find('a', class_='comments')
            comments_text = comments_elem.text if comments_elem else '0 comments'
            num_comments = int(''.join(filter(str.isdigit, comments_text))) if comments_text else 0
            
            # Extract timestamp
            time_elem = post_element.find('time')
            created_utc = time_elem.get('datetime') if time_elem else None
            
            # Extract URL/permalink
            permalink = post_element.get('data-permalink', '')
            url = f"https://reddit.com{permalink}" if permalink else 'N/A'
            
            # Extract flair
            flair_elem = post_element.find('span', class_='linkflairlabel')
            flair = flair_elem.text.strip() if flair_elem else 'None'
            
            # Extract domain
            domain_elem = post_element.find('span', class_='domain')
            domain = domain_elem.text.strip() if domain_elem else 'self'
            
            # Extract upvote ratio if available
            upvote_ratio_elem = post_element.find('div', class_='score likes')
            upvote_ratio = 1.0  # Default
            
            # Check if it's a self post
            is_self = 'self.' in domain
            
            # Extract body text for self posts
            body = ''
            if is_self:
                expando_elem = post_element.find('div', class_='expando')
                if expando_elem:
                    body_elem = expando_elem.find('div', class_='md')
                    if body_elem:
                        body = body_elem.get_text(separator=' ', strip=True)
            
            # Extract gilded count
            gilded_elem = post_element.find('span', class_='gilded-icon')
            gilded = 1 if gilded_elem else 0
            
            post_data = {
                'post_id': post_id,
                'title': title,
                'body': body,
                'author': author,
                'upvotes': upvotes,
                'upvote_ratio': upvote_ratio,
                'num_comments': num_comments,
                'created_utc': created_utc,
                'url': url,
                'is_self': is_self,
                'subreddit': subreddit_name,
                'flair': flair,
                'domain': domain,
                'gilded': gilded,
                'permalink': f"https://reddit.com{permalink}",
                'scraped_at': datetime.now()
            }
            
            return post_data
            
        except Exception as e:
            logger.error(f"Error in _extract_post_data: {str(e)}")
            raise
    
    def scrape_comments(self, post_url, limit=100):
        """
        Scrape comments from a specific post using BeautifulSoup
        
        Args:
            post_url (str): URL of the post
            limit (int): Maximum number of comments to scrape
        
        Returns:
            list: List of comment dictionaries
        """
        logger.info(f"Scraping comments from {post_url}")
        
        self.driver.get(post_url)
        time.sleep(2)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        # Find all comments
        comments = soup.find_all('div', class_='comment')
        
        comments_data = []
        
        for idx, comment in enumerate(comments[:limit]):
            try:
                # Extract comment ID
                comment_id = comment.get('data-fullname', '').replace('t1_', '')
                
                # Extract author
                author_elem = comment.find('a', class_='author')
                author = author_elem.text if author_elem else '[deleted]'
                
                # Extract comment body
                body_elem = comment.find('div', class_='md')
                body = body_elem.get_text(separator=' ', strip=True) if body_elem else ''
                
                # Extract score
                score_elem = comment.find('span', class_='score unvoted')
                if not score_elem:
                    score_elem = comment.find('span', class_='score')
                score_text = score_elem.text if score_elem else '0'
                score = int(''.join(filter(str.isdigit, score_text))) if score_text else 0
                
                # Extract timestamp
                time_elem = comment.find('time')
                created_utc = time_elem.get('datetime') if time_elem else None
                
                # Check if submitter
                is_submitter = 'submitter' in comment.get('class', [])
                
                comment_data = {
                    'comment_id': comment_id,
                    'post_url': post_url,
                    'body': body,
                    'author': author,
                    'score': score,
                    'created_utc': created_utc,
                    'is_submitter': is_submitter
                }
                
                comments_data.append(comment_data)
                
            except Exception as e:
                logger.error(f"Error scraping comment: {str(e)}")
                continue
        
        logger.info(f"Scraped {len(comments_data)} comments")
        return comments_data
    
    def scrape_multiple_subreddits(self, subreddit_list, limit_per_sub=500):
        """
        Scrape posts from multiple subreddits
        
        Args:
            subreddit_list (list): List of subreddit names
            limit_per_sub (int): Limit per subreddit
        
        Returns:
            pd.DataFrame: Combined scraped data
        """
        all_data = []
        
        for subreddit_name in subreddit_list:
            try:
                df = self.scrape_subreddit(subreddit_name, limit=limit_per_sub)
                all_data.append(df)
                logger.info(f"Completed scraping r/{subreddit_name}")
                time.sleep(3)  # Be polite between subreddits
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total posts scraped: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_data(self, df, output_path):
        """Save scraped data to CSV"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Data saved to {output_path}")
        
        # Save basic statistics
               # Save basic statistics safely
        stats = {
            'total_posts': len(df),
            'unique_authors': df['author'].nunique() if 'author' in df.columns else 0,
            'avg_upvotes': df['upvotes'].mean() if 'upvotes' in df.columns else 0,
            'avg_comments': df['num_comments'].mean() if 'num_comments' in df.columns else 0,
            'subreddits': df['subreddit'].unique().tolist() if 'subreddit' in df.columns else []
        }

        logger.info(f"Scraping Statistics: {stats}")
        return stats

    
    def close(self):
        """Close the WebDriver"""
        self.driver.quit()
        logger.info("WebDriver closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Reddit Web Scraper (Selenium + BeautifulSoup)')
    parser.add_argument('--subreddit', type=str, default='artificialintelligence',
                        help='Subreddit name to scrape')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of posts to scrape')
    parser.add_argument('--sort', type=str, default='hot',
                        choices=['hot', 'new', 'top', 'rising'],
                        help='Sorting method')
    parser.add_argument('--output', type=str, default='data/raw/reddit_data1.csv',
                        help='Output file path')
    parser.add_argument('--multiple', type=str, nargs='+',
                        help='Multiple subreddits to scrape')
    parser.add_argument('--headless', action='store_true', default=True,
                        help='Run browser in headless mode')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize scraper
    with RedditScraper(headless=args.headless) as scraper:
        # Scrape data
        if args.multiple:
            df = scraper.scrape_multiple_subreddits(args.multiple, limit_per_sub=args.limit)
        else:
            df = scraper.scrape_subreddit(args.subreddit, limit=args.limit, sort_by=args.sort)
        
                # Save data
        scraper.save_data(df, args.output)

        print(f"\nScraping completed successfully!")
        print(f"Total posts: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data saved to: {args.output}")

        # Safely print a preview
        if not df.empty:
            cols_to_show = [c for c in ['title', 'author', 'upvotes', 'num_comments'] if c in df.columns]
            print("\nSample data:")
            print(df.head(3)[cols_to_show] if cols_to_show else df.head(3))
        else:
            print("\n⚠️ No data scraped. The DataFrame is empty.")



if __name__ == "__main__":
    main()