from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
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

def get_images(web_driver, delay, max_images):
    def scroll_down(web_driver):
        """Sayfayı aşağı kaydırarak yeni resimlerin yüklenmesini sağla."""
        web_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    # Google Görseller arama URL'si
    search_query = "çanta"
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_query}"
    web_driver.get(url)

    image_urls = set()
    thumbnails = web_driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")  # Thumbnail seçici

    for thumbnail in thumbnails[:max_images]:  # Maksimum resim sayısı kadar işlem yap
        try:
            thumbnail.click()
            time.sleep(delay)

            # Büyük resim URL'sini bul
            large_image = web_driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")
            src_url = large_image.get_attribute("src")

            if src_url and "http" in src_url:
                image_urls.add(src_url)
                print(f"Found {len(image_urls)} images.")
        except Exception as e:
            print("Error:", e)

        scroll_down(web_driver)

    return image_urls


def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = download_path + file_name
        image.save(file_path, "PNG")
        print(f"Image saved to {file_path}")
    except Exception as e:
        print("Error downloading image:", e)


# Resim URL'lerini al
urls = get_images(web_driver, 2, 5)
print(urls)

web_driver.quit()
