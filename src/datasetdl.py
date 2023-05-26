#chromium webdriver needs to be set up in order for this script to function
#selenium is required, this script may not run on servers without gui
#(I don't know)
#the keywords present here may not be the right ones
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.parse import unquote, urlparse, parse_qs
from selenium.webdriver.common.by import By
import os
import requests
import time

# Delay between actions to avoid overwhelming the server
DELAY = 0.1
# Number of page scrolls to perform for each search term
NUM_SCROLLS = 5


def download_imgs(keyword, save_path, max_num, driver):
    # Make directory if it does not exist
    os.makedirs(save_path, exist_ok=True)
    
    # Navigate to the DuckDuckGo image search page for the search term
    driver.get(f"https://duckduckgo.com/?q={keyword}&t=h_&iax=images&ia=images")

    # Scroll down to load more images
    for _ in range(NUM_SCROLLS):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(DELAY+0.5)

    # Select image elements and extract the 'src' attribute
    images = driver.find_elements(By.CSS_SELECTOR,"#zci-images > div > div.tile-wrap img")
    image_urls = [img.get_attribute('src') for img in images]

    # Download and save each image
    for i, url in enumerate(image_urls[:max_num]):
        # Parse the URL and extract the 'u' parameter
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        original_url = unquote(params['u'][0])

        # Download the image
        try:
            response = requests.get(original_url)
            response.raise_for_status()
        except (requests.RequestException, ValueError):
            print(f"Failed to download image {url}")
            continue

        # Save the image
        filename = os.path.join(save_path, f"{keyword}_{i}.jpg".replace('"',''))
        with open(filename, 'wb') as f:
            f.write(response.content)

        time.sleep(DELAY)


def main():
    # Define search terms and directories
    #cat_terms = ["cats","wild cat","ocelot animal","tuxedo cat","sphynx cat","long haired cat"]
    #not_cat_terms = ["pizza", "pasta", "umbrella", "bear"]
    cat_terms = ["\"running\" \"cat\"","sleepy cat","cat rolling","cat ears animal","cat rough sketch","upside down cat","funny cat positions","cat in the shadow",]
    not_cat_terms = ["dolphin","shark","water","underwater","tree","flower and grass field","simple car", "umbrella","moon","soccer ball","guitar","river","clock","clothes","dresses"]
    # Combine the terms with their respective directories
    search_terms = {term: "dset/cat" for term in cat_terms}
    search_terms.update({term: "dset/not_cat" for term in not_cat_terms})
    # Number of images to download per category
    num_images_per_category = 400

    # Create a new Chrome browser instance
    driver = webdriver.Chrome()

    try:
        for term, dir in search_terms.items():
            download_imgs(term, dir, num_images_per_category, driver)
    finally:
        # Close the browser
        driver.quit()


if __name__ == "__main__":
    main()
