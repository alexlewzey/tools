"""Save the urls and webpage titles of the current Chrome tabs to the clipboard
in a dictionary format."""
import logging
import re
import time

import pyperclip
import requests
from lxml import html
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from clmac.helpers import core
from clmac.helpers.core import Optional, whitespacer
from clmac.helpers.typer import Typer

logger = logging.getLogger(__name__)

PAGE_LOAD_TIME = 0.2
JS_LOAD_TIME = 1


class DriverChrome:
    def __init__(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("## incognito")
        self.driver = webdriver.Chrome(ChromeDriverManager().install())

    @core.sleep_after(PAGE_LOAD_TIME)
    def load_page(self, url: str):
        """Url must have http protocol."""
        self.driver.get(url)

    @core.sleep_after(JS_LOAD_TIME)
    def scroll_bottom(self):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def render_full_page(self, url: str):
        self.load_page(url)
        self.scroll_bottom()

    def get_page_content(self, url: str):
        self.render_full_page(url)
        return self.driver.page_source

    def __getattr__(self, attr):
        return getattr(self.driver, attr)


def enforce_url_protocol(url: str):
    protocol = "https://www."
    if not re.search(r"^https?://(www\.)?", url):
        return protocol + url
    else:
        return url


def render_page_source(driver, url: str, wait: float = 2) -> str:
    """Load page with driver wait for a few seconds and retrieve the rendered
    html returning it as a string."""
    url = enforce_url_protocol(url)
    driver.get(url)
    time.sleep(wait)
    driver.quit()
    return driver.page_source


def request_page_source(url: str) -> str:
    """Return the webpage title corresponding to the url that is passed into
    the function."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/61.0.3163.100 Safari/537.36"
        )
    }
    url = enforce_url_protocol(url)
    return requests.get(url, headers=headers).text


def parse_page_title(page_source: str) -> str:
    page = html.fromstring(page_source)
    title = page.xpath(".//title")[0].text
    return title if isinstance(title, str) else "no title found"


def link_urls_to_page_titles(urls: list[str]) -> list[tuple[str, str]]:
    """Take a list of urls and find the corresponding webpage title for each
    and them as a key value pair to a dictionary."""
    url_pairs = []
    for url in urls:
        title = request_page_source(url)
        logger.debug(f"page_title: {title}")
        url_pairs.append(
            (title, url),
        )
    return url_pairs


def render_and_parse_pages(urls: list[str]) -> list[tuple[str, str]]:
    """"""
    driver = DriverChrome()
    url_pairs = []
    for url in urls:
        page_source = driver.get_page_content(url)
        title = parse_page_title(page_source)
        logger.debug(f"page_title: {title}")
        url_pairs.append(
            (whitespacer(title), whitespacer(url)),
        )
    driver.driver.quit()
    return url_pairs


def dict2string(url_pairs: dict) -> str:
    """Convert a dict into a string where each key pair value is a line."""
    output: str = ""
    for k, v in url_pairs.items():
        output += f"{k}: {v}\n"
    return output


def tuples2string(url_pairs: list[tuple[str, str]]) -> str:
    """Convert a list of tuples into a string where each tuple is a line."""
    output: str = ""
    for title, url in url_pairs:
        output += f"{title}: {url}\n"
    return output


def clip_urls(n_urls: Optional[int] = None) -> None:
    typer = Typer()
    typer.alt_tab()
    urls = typer.get_urls(n_urls=n_urls)

    url_pairs = render_and_parse_pages(urls)

    logger.debug(f"url_pairs: {url_pairs}")
    output_str = tuples2string(url_pairs)
    logger.debug(f"output_str: {output_str}")
    pyperclip.copy(output_str.strip())


if __name__ == "__main__":
    clip_urls()
