from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import chromedriver_autoinstaller
import requests
import io
from PIL import Image

chromedriver_autoinstaller.install()
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
chrome_options.add_experimental_option("detach", True)
web_driver = webdriver.Chrome(options=chrome_options)

web_driver.get("https://google.com/")

def get_images(web_driver, delay, max_images):
    def scroll_down(web_driver):
        web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    search_query = "çanta"
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_query}"
    web_driver.get(url)

    image_urls = set()
    while len(image_urls) < max_images:
        scroll_down(web_driver)

        thumbnail = web_driver.find_elements(By.CLASS_NAME, "YQ4gaf")

        for image in thumbnail[len(image_urls): max_images]:
            try:
                image.click()
                time.sleep(delay)
            except:
                continue

            images_larger = web_driver.find_elements(By.CLASS_NAME, "sFlh5c FyHeAf iPVvYb")
            for image in images_larger:
                if image.get_attribute('src') and "http" in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    print(f"Found {len(image_urls)}")
    return image_urls


def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = download_path + file_name
    except Exception as e:
        print("Fail", e)

    with open(file_path, "wb") as f:
        image.save(f, "PNG")

    print("Success")

urls = get_images(web_driver, 2, 5)
print(urls)
web_driver.quit()


