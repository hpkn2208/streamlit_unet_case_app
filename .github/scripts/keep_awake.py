"""Visits the deployed Streamlit app like a real browser so its inactivity
timer resets, and clicks the "wake up" button if Streamlit Cloud already put
it to sleep. A bare curl/HTTP GET doesn't reliably do either — Streamlit only
counts a real script run (over the app's WebSocket connection) as activity.
"""

import os
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ["APP_URL"]


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="networkidle", timeout=60_000)
        time.sleep(3)

        wake_button = page.get_by_text("get this app back up", exact=False)
        if wake_button.count() > 0:
            print("App was asleep — clicking wake-up button.")
            wake_button.first.click()
            time.sleep(15)
        else:
            print("App was already awake — visit refreshed its inactivity timer.")

        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to ping {APP_URL}: {exc}", file=sys.stderr)
        raise
