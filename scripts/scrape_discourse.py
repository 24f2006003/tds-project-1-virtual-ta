from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from selenium.common.exceptions import TimeoutException, WebDriverException

# Discourse URL and cookies
DISCOURSE_URL = "https://discourse.onlinedegree.iitm.ac.in"
COOKIES = {
    "_t": "GlC309NtbWSxLHfS0txXhil3EzQYnypO3JVSgVpE5dCr4PnnwOmVP%2B%2FbFbKgOCaELUdXjhvKGtuZl0Dp7CJjIw2siVIuK1zOxkFp3UQdUniw21%2BRUTRl29NUgx3R8oYXjoJtKVlXt4I4tmkz0msDDHAN%2FaX9%2F%2BMPnB7F4USd6dE90jYIUIBf1WepIbQxCtk3PQq%2BMJ%2BvwISJuO4busfVxcXsdmfjgxpdp6jRfPGy3r3MTm%2Bp%2BCCfd4UON7EG9KELEdgzM%2BZjfbCP%2Bs0iPr4ayJiQDYsXgHMy1ZH9urcOA1A4O6ZIvQYh%2Fnpvyu6UO7zV--fZWRulBjt16jmrGr--blolbRRad5q9CBLZ%2Bd705g%3D%3D",
    "_forum_session": "tJ6jlT2l5TocJ5TP5EiKmMm3%2FSIsKNXGD1JOGbsgpXrRG4o482159Sezjl85GmIV6Pxx01auwCEfje6oUXIq9%2BBcb9H73Y0T4T31RqyMEa%2BDWHKOBewoceoRAkkJuiJM9uuYNCkt53N0L%2FxghT%2FLPfaJ51Om9S47Z1qBBV8zxiBXIf%2Foi%2FAGBj2g3DhnEC7e9%2FRrHSvTpeAYR5mkxqwbQIP4UeQ6vKvaYpxl6TIqNUNQaVKXLN1oi3T6UvsUeqmrUTjdqgaPDyuwrQh5oxZv27j%2F9jNH8ZzdcHyb2AdHbAPJ3xMewsTziiWfx4%2FMA56WDRYb%2Fz3UapC5ZpdzO7UIBhWyhcMyVtSNGI8B3nMQ99mP5mHPrMK2znB4DWRpBIqmxx34mzj1sW695AcYrdpr3VQNYPwFqljg4ldrNHaxXg8X3NY5%2BqjcydfHxfoiJM4KP0DR5Oc8ReJof94YWiSJ0OlEdvjnXA%2Bsry1xtGyP5lKDQsaCLOLSPiGvyvwvzg%3D%3D--73Wl3IANKO3aQjqX--kogTOmMoloFGz3PKBnRdMQ%3D%3D"
}

def scrape_discourse_posts(start_date, end_date):
    # Set up Selenium with Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    
    try:
        # Navigate to Discourse and apply cookies
        driver.get(DISCOURSE_URL)
        for name, value in COOKIES.items():
            driver.add_cookie({"name": name, "value": value})
        driver.refresh()
        
        # Wait for page to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception as e:
            print(f"Error loading initial page: {e}")
            return []
        
        # Ensure data directory exists
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "../data")
        os.makedirs(data_dir, exist_ok=True)
        
        posts = []
        page = 1
        max_pages = 10  # Limited for debugging
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        retries = 3
        
        while page <= max_pages:
            print(f"Fetching page {page}...")
            for attempt in range(retries):
                try:
                    driver.get(f"{DISCOURSE_URL}/latest?page={page}")
                    
                    # Wait for posts to load
                    WebDriverWait(driver, 15).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr.topic-list-item"))
                    )
                    break  # Success, exit retry loop
                except (TimeoutException, WebDriverException) as e:
                    print(f"Attempt {attempt + 1} failed for page {page}: {e}")
                    if attempt == retries - 1:
                        print(f"Skipping page {page} after {retries} attempts.")
                        page += 1
                        continue
                    time.sleep(2)  # Wait before retrying
            
            # Check for login page
            if driver.find_elements(By.CSS_SELECTOR, "form#login-form, .login-modal"):
                print("Error: Redirected to login page. Check cookies.")
                break
            
            # Parse page source
            soup = BeautifulSoup(driver.page_source, "html.parser")
            post_elements = soup.select("tr.topic-list-item")
            
            if not post_elements:
                print("No posts found on page. Possible selector mismatch.")
                print(f"Page source sample: {driver.page_source[:1000]}")
                page += 1
                continue
            
            for post in post_elements:
                # Debug: Print raw post HTML
                print(f"Post HTML sample: {str(post)[:200]}...")
                
                # Try extracting date
                date_elem = post.select_one("time, .relative-date, [data-time], [title*='ago'], .post-date, .topic-timestamp")
                date_str = None
                post_date = None
                
                if date_elem:
                    date_str = date_elem.get("data-time") or date_elem.get("title") or date_elem.text.strip()
                    try:
                        if date_elem.get("data-time"):
                            post_date = datetime.fromtimestamp(int(date_str) / 1000)
                        else:
                            for fmt in ["%Y-%m-%d", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%Y-%m-%d %H:%M:%S"]:
                                try:
                                    post_date = datetime.strptime(date_str, fmt)
                                    break
                                except ValueError:
                                    continue
                    except Exception as e:
                        print(f"Error parsing date '{date_str}': {e}")
                
                # Extract title
                title_elem = post.select_one("span.link-top-line a.raw-topic-link")
                print(f"Title element: {title_elem}")
                
                # Extract excerpt or fetch content
                content_elem = post.select_one(".topic-excerpt")
                content = content_elem.text.strip() if content_elem else None
                
                if title_elem:
                    title = title_elem.text.strip()
                    url = DISCOURSE_URL + title_elem["href"] if title_elem["href"].startswith("/") else title_elem["href"]
                    
                    # Fetch full post content if no excerpt
                    if not content:
                        try:
                            driver.get(url)
                            WebDriverWait(driver, 5).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, ".cooked"))
                            )
                            post_soup = BeautifulSoup(driver.page_source, "html.parser")
                            content_elem = post_soup.select_one(".cooked")
                            content = content_elem.text.strip() if content_elem else "No content available"
                            driver.back()
                        except Exception as e:
                            print(f"Error fetching content for {url}: {e}")
                            content = "No content available"
                    
                    # Collect post
                    if post_date and start_dt <= post_date <= end_dt:
                        posts.append({"text": f"{title} {content}", "url": url})
                        print(f"Scraped post: {title} (Date: {post_date})")
                    else:
                        # Fallback: Collect post without date filtering
                        posts.append({"text": f"{title} {content}", "url": url})
                        print(f"Scraped post (no/invalid date): {title}")
                
                else:
                    print("Warning: Missing title for a post. Skipping.")
            
            page += 1
            time.sleep(1)
        
        # Save posts to file
        output_path = os.path.join(data_dir, "discourse_posts.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(posts)} posts to {output_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
        
        return posts
    
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_discourse_posts("2025-01-01", "2025-04-14")