import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests

def get_links_from_url(url, company_name):
    """Get all links from a given URL."""
    link_list = []
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        for anchor in soup.find_all('a'):
            href = anchor.get('href')
            if href:
                if href.startswith('http') or href.startswith('https'):
                    link_list.append(href)
                else:
                    # Resolve relative URLs
                    new_link = urljoin(url, href)
                    link_list.append(new_link)

    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")

    # Deduplicate links
    return list(set(link_list))


def get_text_from_links(links):
    """Scrape text content from a list of links."""
    all_content = []
    for link in links:
        try:
            response = requests.get(link)
            response.raise_for_status()  # Ensure the request was successful
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text(strip=True)  # Clean and strip text
            all_content.append(content)
        except requests.RequestException as e:
            print(f"Error fetching {link}: {e}")
    return "\n".join(all_content)  # Combine all content


def save_to_file(content, filename="scraped_content.txt"):
    """Save scraped content to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


# Main Script
url = "https://www.nhs.uk/conditions/#A"
company_name = "nhs"

# Step 1: Get all links
result_links = get_links_from_url(url, company_name)

# Step 2: Scrape text from links
context = get_text_from_links(result_links)

# Step 3: Save text to a file
save_to_file(context)

# Output results
print(f"Scraped content saved. Total length: {len(context)} characters.")
