#!/usr/bin/env python3

"""
Web Scraper for AI Training Data
Scrapes website content and saves to CSV format using pandas
"""

import argparse
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import sys
from urllib.parse import urlparse, urljoin
import logging
from typing import List, Dict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncWebScraper:
    def __init__(self, base_url: str, delay: float = 1.0, max_concurrent: int = 20):
        """
        Initialize the async web scraper

        Args:
            base_url: The base URL to scrape
            delay: Delay between requests in seconds
            max_concurrent: Maximum number of concurrent requests
        """
        self.base_url = base_url
        self.delay = delay
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might cause CSV issues
        text = re.sub(r'[^\w\s\.,!?;:()\-\'""]', ' ', text)
        return text

    def extract_sections(self, soup: BeautifulSoup, url: str, title: str) -> List[Dict[str, str]]:
        """
        Extract multiple sections/chunks from a webpage for better training data

        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            title: Page title

        Returns:
            List of dictionaries with section data
        """
        sections = []
        scraped_at = pd.Timestamp.now().isoformat()

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag:
            meta_desc = meta_tag.get('content', '')

        # Method 1: Extract by headings and their content
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for i, heading in enumerate(headings):
            heading_text = self.clean_text(heading.get_text())
            if not heading_text or len(heading_text) < 3:
                continue

            # Get content between this heading and the next heading of same or higher level
            content_elements = []
            current_level = int(heading.name[1])  # h1 -> 1, h2 -> 2, etc.

            # Find all siblings after this heading
            for sibling in heading.find_next_siblings():
                # Stop if we hit another heading of same or higher level
                if sibling.name and sibling.name.startswith('h'):
                    sibling_level = int(sibling.name[1])
                    if sibling_level <= current_level:
                        break

                # Collect paragraphs, lists, and other content
                if sibling.name in ['p', 'ul', 'ol', 'div', 'section']:
                    text = self.clean_text(sibling.get_text())
                    if text and len(text) > 20:
                        content_elements.append(text)

            # Create section entry
            section_content = ' '.join(content_elements)
            if section_content and len(section_content.split()) >= 10:  # At least 10 words
                sections.append({
                    'url': url,
                    'page_title': self.clean_text(title),
                    'section_title': heading_text,
                    'section_content': section_content,
                    'content_type': 'section',
                    'word_count': len(section_content.split()),
                    'meta_description': self.clean_text(meta_desc),
                    'scraped_at': scraped_at
                })

        # Method 2: Extract individual paragraphs as separate entries
        paragraphs = soup.find_all('p')
        for para in paragraphs:
            para_text = self.clean_text(para.get_text())
            if para_text and len(para_text.split()) >= 15:  # At least 15 words
                # Try to find the nearest heading for context
                context_heading = ""
                for heading in para.find_all_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    context_heading = self.clean_text(heading.get_text())
                    if context_heading:
                        break

                sections.append({
                    'url': url,
                    'page_title': self.clean_text(title),
                    'section_title': context_heading or 'Introduction',
                    'section_content': para_text,
                    'content_type': 'paragraph',
                    'word_count': len(para_text.split()),
                    'meta_description': self.clean_text(meta_desc),
                    'scraped_at': scraped_at
                })

        # Method 3: Extract list items as separate entries
        lists = soup.find_all(['ul', 'ol'])
        for list_elem in lists:
            items = list_elem.find_all('li')
            if len(items) >= 3:  # Only process lists with multiple items
                # Find context heading
                context_heading = ""
                for heading in list_elem.find_all_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    context_heading = self.clean_text(heading.get_text())
                    if context_heading:
                        break

                # Combine list items into chunks
                list_content = []
                for item in items:
                    item_text = self.clean_text(item.get_text())
                    if item_text and len(item_text) > 5:
                        list_content.append(f"• {item_text}")

                if list_content:
                    combined_list = '\n'.join(list_content)
                    if len(combined_list.split()) >= 10:
                        sections.append({
                            'url': url,
                            'page_title': self.clean_text(title),
                            'section_title': context_heading or 'List',
                            'section_content': combined_list,
                            'content_type': 'list',
                            'word_count': len(combined_list.split()),
                            'meta_description': self.clean_text(meta_desc),
                            'scraped_at': scraped_at
                        })

        return sections

    async def scrape_page(self, url: str) -> List[Dict[str, str]]:
        """
        Scrape a single page asynchronously

        Args:
            url: URL to scrape

        Returns:
            List of dictionaries with scraped data sections
        """
        async with self.semaphore:
            try:
                logger.info(f"Scraping: {url}")

                async with self.session.get(url) as response:
                    response.raise_for_status()
                    content = await response.read()

                soup = BeautifulSoup(content, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""

                # Extract multiple sections from the page
                sections = self.extract_sections(soup, url, title_text)

                if sections:
                    logger.info(f"Extracted {len(sections)} sections from {url}")

                # Add delay to be respectful
                await asyncio.sleep(self.delay)
                return sections

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return []

    def find_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find relevant links on the page to scrape

        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links

        Returns:
            List of URLs to scrape
        """
        links = []
        base_domain = urlparse(base_url).netloc

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Skip empty links, anchors, and external protocols
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue

            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)

            # Only include links from the same domain
            if parsed_url.netloc == base_domain:
                # Skip common non-content pages
                skip_patterns = [
                    '/edit', '/talk:', '/special:', '/user:', '/file:',
                    '/category:', '/template:', '/help:', '/wikipedia:',
                    '.pdf', '.jpg', '.png', '.gif', '.css', '.js',
                    'action=', 'printable=', 'oldid='
                ]

                if not any(pattern in full_url.lower() for pattern in skip_patterns):
                    links.append(full_url)

        return list(set(links))  # Remove duplicates

    async def scrape_website(self, max_pages: int = 10, follow_links: bool = True) -> List[Dict[str, str]]:
        """
        Scrape the website starting from base_url using async/await

        Args:
            max_pages: Maximum number of pages to scrape
            follow_links: Whether to follow links to scrape more pages

        Returns:
            List of scraped data dictionaries
        """
        all_sections = []
        urls_to_scrape = [self.base_url]
        scraped_urls = set()

        while urls_to_scrape and len(scraped_urls) < max_pages:
            # Take a batch of URLs to scrape concurrently
            batch_size = min(self.max_concurrent, len(urls_to_scrape), max_pages - len(scraped_urls))
            current_batch = []

            for _ in range(batch_size):
                if urls_to_scrape:
                    url = urls_to_scrape.pop(0)
                    if url not in scraped_urls:
                        current_batch.append(url)
                        scraped_urls.add(url)

            if not current_batch:
                break

            logger.info(f"Processing batch of {len(current_batch)} URLs (Total scraped: {len(scraped_urls)}/{max_pages})")

            # Create tasks for concurrent scraping
            tasks = []
            for url in current_batch:
                task = asyncio.create_task(self.scrape_page_with_links(url, follow_links))
                tasks.append(task)

            # Wait for all tasks in the batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed: {result}")
                    continue

                sections, new_links = result
                if sections:
                    all_sections.extend(sections)

                # Add new links for future batches
                if follow_links and len(scraped_urls) < max_pages:
                    for link in new_links:
                        if link not in scraped_urls and link not in urls_to_scrape:
                            urls_to_scrape.append(link)

        logger.info(f"Scraping completed. Pages scraped: {len(scraped_urls)}, Total sections: {len(all_sections)}")
        return all_sections

    async def scrape_page_with_links(self, url: str, follow_links: bool) -> tuple:
        """
        Scrape a page and return both sections and links

        Returns:
            Tuple of (sections, links)
        """
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    content = await response.read()

                soup = BeautifulSoup(content, 'html.parser')

                # Remove unwanted elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else ""

                # Extract sections
                sections = self.extract_sections(soup, url, title_text)

                # Extract links if needed
                links = []
                if follow_links:
                    links = self.find_links(soup, url)

                await asyncio.sleep(self.delay)
                return sections, links

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return [], []

async def main():
    parser = argparse.ArgumentParser(description='Async Web Scraper for AI Training Data')
    parser.add_argument('url', help='Website URL to scrape')
    parser.add_argument('-o', '--output', default='scraped_data.csv',
                       help='Output CSV file name (default: scraped_data.csv)')
    parser.add_argument('-p', '--pages', type=int, default=1,
                       help='Maximum number of pages to scrape (default: 1)')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('-c', '--concurrent', type=int, default=20,
                       help='Maximum concurrent requests (default: 20)')
    parser.add_argument('--no-follow-links', action='store_true',
                       help='Don\'t follow links to scrape additional pages')

    args = parser.parse_args()

    # Validate URL
    parsed_url = urlparse(args.url)
    if not parsed_url.scheme:
        args.url = 'https://' + args.url

    follow_links = not args.no_follow_links and args.pages > 1

    logger.info(f"Starting async scraping: {args.url}")
    logger.info(f"Max concurrent requests: {args.concurrent}")
    logger.info(f"Max pages: {args.pages}")

    # Use async context manager
    async with AsyncWebScraper(args.url, args.delay, args.concurrent) as scraper:
        try:
            scraped_sections = await scraper.scrape_website(args.pages, follow_links)

            if not scraped_sections:
                logger.error("No data was scraped successfully")
                sys.exit(1)

            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(scraped_sections)
            df.to_csv(args.output, index=False, encoding='utf-8')

            # Statistics
            unique_pages = df['url'].nunique()
            total_sections = len(df)
            avg_words_per_section = df['word_count'].mean()
            total_words = df['word_count'].sum()

            print(f"\n📊 Async Scraping Results:")
            print(f"   Pages scraped: {unique_pages}")
            print(f"   Total sections: {total_sections}")
            print(f"   Average words per section: {avg_words_per_section:.0f}")
            print(f"   Total words: {total_words:,}")

        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
