from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import os
import time

URL = "https://tds.s-anand.net/#/2025-01/"

def scrape_course_content():
    # Set up Selenium with Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode for efficiency
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")  # For Vercel compatibility
    driver = webdriver.Chrome(options=options)
    
    try:
        # Navigate to course page
        driver.get(URL)
        
        # Wait for content to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)  # Additional wait for JavaScript rendering
        
        # Extract content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Try specific selectors first, fall back to broader ones
        content_elements = soup.select('.content, .main-content, article, section, div, p, h1, h2, h3, li')
        content = " ".join([elem.get_text(strip=True) for elem in content_elements if elem.get_text(strip=True)])
        
        # Fallback: Extract all text from body if no content found
        if not content:
            content = soup.body.get_text(strip=True, separator=" ") if soup.body else ""
            print("Warning: Used fallback extraction. Check page structure.")
        
        # Debug: Print content length and sample
        print(f"Extracted content length: {len(content)} characters")
        if len(content) < 50:
            print("Warning: Content is very short. Page may not have loaded correctly.")
            print(f"Page source sample: {driver.page_source[:500]}")
        
        # Ensure data directory exists (use absolute path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "../data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save content to JSON
        output_path = os.path.join(data_dir, "course_content.json")
        data = [{"text": content, "url": URL}]
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved course content to {output_path}")
        except Exception as e:
            print(f"Error writing to file: {e}")
        
        return data
    
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_course_content()