"""Uv run python -m src.houses.houses."""

import os
import re
import time
import urllib.parse
from pathlib import Path

import polars as pl
import requests
from playwright.sync_api import sync_playwright

dir_project = Path(__file__).parent
dir_tmp = dir_project / "tmp"
dir_tmp.mkdir(exist_ok=True)
user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
rightmove_pw: str = os.environ["RIGHTMOVE_PW"]
gmail: str = os.environ["GMAIL"]

link: str = "https://www.rightmove.co.uk/properties/173060330#/?channel=RES_BUY"
postcode: str = "SK6 1SB"
house_number: str = "10"


def get_hauscope_estimate(link):
    url: str = "https://hauscope.com/"
    path_state = dir_tmp / "hauscope_state.json"
    query = urllib.parse.quote(link, safe="")
    url += f"result?url={query}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = (
                browser.new_context(storage_state=path_state, user_agent=user_agent)
                if path_state.exists()
                else browser.new_context(user_agent=user_agent)
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")

            range_pattern = re.compile(r"£\d{1,3},\d{3}\s–\s£\d{1,3},\d{3}")
            range_element = page.get_by_text(range_pattern)
            range_element.wait_for(state="visible")
            estimate = range_element.inner_text().replace(" range", "")
        return estimate
    except Exception:
        return None


def get_zoopla_estimate(postcode: str, house_number: str) -> str | None:
    path_zoopla_state = dir_tmp / "zoopla_state.json"
    # data-testid="sale-estimate"
    url: str = "https://www.zoopla.co.uk/home-values/"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = (
                browser.new_context(
                    storage_state=path_zoopla_state, user_agent=user_agent
                )
                if path_zoopla_state.exists()
                else browser.new_context(user_agent=user_agent)
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")

            print("accept cookies")
            if not path_zoopla_state.exists():
                accept_button = page.get_by_role(
                    "button", name=re.compile(r"accept|agree", re.IGNORECASE)
                ).first
                accept_button.wait_for(state="visible", timeout=3000)
                accept_button.click()
                context.storage_state(path=path_zoopla_state)

            print("auth")
            time.sleep(5)
            page.screenshot(path=dir_tmp / "zoopla.png")

            page.locator(
                "#address-selection-postcode-input-homeowner-intent-wizard"
            ).fill(postcode)
            page.get_by_role("button", name="Look up postcode").click()

            context.storage_state(path=path_zoopla_state)

            time.sleep(60)
        # return estimate
        return None
    except Exception:
        return None


def get_rightmove_estimate(postcode: str, house_number: str) -> str | None:
    path_rightmove_state = dir_tmp / "rightmove_state.json"
    url: str = "https://www.rightmove.co.uk/house-value.html"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False)
            context = (
                browser.new_context(
                    storage_state=path_rightmove_state, user_agent=user_agent
                )
                if path_rightmove_state.exists()
                else browser.new_context(user_agent=user_agent)
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")

            print("accept cookies")
            if not path_rightmove_state.exists():
                accept_button = page.get_by_role(
                    "button", name=re.compile(r"accept|agree", re.IGNORECASE)
                ).first
                accept_button.wait_for(state="visible", timeout=3000)
                accept_button.click()
                context.storage_state(path=path_rightmove_state)

            print("select address")
            # time.sleep(0.5)
            page.fill('input[name="postcodeInput"]', postcode)
            # time.sleep(0.5)
            page.click('button[name="submitSearchButton"]')
            # time.sleep(0.5)

            options: list[str] = page.locator(
                'select[name="addressSelect"] option'
            ).all_inner_texts()
            for option in options:
                if option.startswith(f"{house_number},"):
                    selector = option
                    break
            page.select_option('select[name="addressSelect"]', label=selector)
            time.sleep(1)

            print("radios")
            radio = page.get_by_role("radio", name="No")
            if radio.count() > 0:
                radio.check()
                time.sleep(0.5)
                page.get_by_role("radio", name="Prefer not to say").check()
                time.sleep(0.5)
                page.locator("#emailQuestionNo").check()
                time.sleep(0.5)
                page.get_by_text("Continue").click()
                time.sleep(0.5)

            print("login")
            email_input = page.locator("#rma-email-input")
            if email_input.count() > 0:
                email_input.fill(gmail)
                time.sleep(0.5)
                page.locator("#rma-emailSubmit").click()
                time.sleep(0.5)
                page.locator("#rma-password-input").fill(rightmove_pw)
                time.sleep(0.5)
                page.locator("#rma-submit")

            print("parse estimate")
            card = page.get_by_test_id("propertyInsights-component")
            card.wait_for(state="visible")
            blob = card.inner_text()
            m = re.search(r"£[\d.]+k\s*-\s*£[\d.]+k", blob)
            if m:
                estimate = m.group()
        return estimate
    except Exception:
        return None


def get_sqm(postcode: str, house_number: str):
    epc_token = os.environ["EPC_TOKEN"]
    try:
        r = requests.get(
            "https://api.get-energy-performance-data.communities.gov.uk/api/domestic/search",
            params={"postcode": postcode},
            headers={
                "Authorization": f"Bearer {epc_token}",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        data = r.json()["data"]
        df = pl.DataFrame(data)
        matches = df.filter(pl.col("addressLine1").str.contains(f"^{house_number},?"))
        if matches.height == 0:
            raise RuntimeError("No matches!")
        cert_id = matches[0, "certificateNumber"]
        r = requests.get(
            "https://api.get-energy-performance-data.communities.gov.uk/api/certificate",
            params={"certificate_number": cert_id},
            headers={
                "Authorization": f"Bearer {epc_token}",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        spm = r.json()["data"]["total_floor_area"]
        return spm
    except Exception:
        return None


def get_sold_price(postcode: str, house_number: str) -> tuple:
    try:
        r = requests.get(
            "http://landregistry.data.gov.uk/data/ppi/transaction-record.json",
            params={"propertyAddress.postcode": postcode, "_pageSize": 200},
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        prices = (
            pl.DataFrame(r.json()["result"]["items"])
            .with_columns(paon=pl.col("propertyAddress").struct["paon"])
            .filter(pl.col("paon") == house_number)
            .with_columns(year=pl.col("transactionDate").str.split(" ").list[-1])
            .filter(
                pl.col("transactionCategory").struct["label"].list[0].struct["_value"]
                == "Standard price paid transaction"
            )
            .sort("year", descending=True)
        )
        price = prices[0, "pricePaid"]
        year = prices[0, "year"]
        return price, year
    except Exception:
        return (None, None)


def main():
    # zoopla_estimate = get_zoopla_estimate(postcode=postcode, house_number=house_number)
    # print(zoopla_estimate)

    # hauscope_estimate = get_hauscope_estimate(link=link)
    # rightmove_estimate = get_rightmove_estimate(postcode=postcode, house_number=house_number)
    sqm = get_sqm(postcode=postcode, house_number=house_number)
    # price, year = get_sold_price(postcode=postcode, house_number=house_number)
    print()
    print("RESULTS")
    print("-------")
    print(f"postcode: {postcode}")
    print(f"house_number: {house_number}")
    # print(f'rightmove_estimate: {rightmove_estimate}')
    # print(f'hauscope_estimate: {hauscope_estimate}')
    print(f"sqm: {sqm}")
    # print(f'price: {price:,}')
    # print(f'year: {year}')


if __name__ == "__main__":
    main()
