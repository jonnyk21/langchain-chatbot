import os
import requests
from bs4 import BeautifulSoup
import time
import json

def scrape_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, and nav elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup
        
        # Get text content
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

def save_to_file(content, filename):
    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", filename), "w", encoding="utf-8") as f:
        f.write(content)

def main():
    # URLs to scrape (both English and German versions)
    urls = {
        # Study Programs
        "study_programs_en": "https://www.hnu.de/en/programmes",
        "study_programs_de": "https://www.hnu.de/studiengaenge",
        
        # International
        "international_en": "https://www.hnu.de/en/international",
        "international_de": "https://www.hnu.de/international",
        
        # Research
        "research_en": "https://www.hnu.de/en/research",
        "research_de": "https://www.hnu.de/forschung",
        
        # Library
        "library_de": "https://www.hnu.de/bibliothek",
        
        # Campus
        "campus_de": "https://www.hnu.de/campus",
        
        # News
        "news_de": "https://www.hnu.de/aktuelles",
        
        # Student Life
        "student_life_de": "https://www.hnu.de/studierendenleben",
        
        # Application
        "application_de": "https://www.hnu.de/bewerbung"
    }
    
    for page_name, url in urls.items():
        print(f"Scraping {page_name}...")
        content = scrape_page(url)
        if content:
            save_to_file(content, f"hnu_{page_name}.txt")
            print(f"Successfully saved {page_name}")
        time.sleep(2)  # Be nice to the server

if __name__ == "__main__":
    main()
