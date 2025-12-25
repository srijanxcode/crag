import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def web_search(query, max_results=5):
    """
    Performs a DuckDuckGo search and returns concatenated snippets.
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": query}

    response = requests.post(url, data=params, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    snippets = []
    results = soup.select(".result__snippet")

    for r in results[:max_results]:
        snippets.append(r.get_text())

    return "\n".join(snippets)
