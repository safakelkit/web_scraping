import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
import chromedriver_autoinstaller

chromedriver_autoinstaller.install()

search_query = "laptop çantası"  
output_folder = "google_images"
os.makedirs(output_folder, exist_ok=True)

driver = webdriver.Chrome()
driver.get(f"https://www.google.com/search?tbm=isch&q={search_query}")
time.sleep(3)

downloaded_count = 0
max_images = 3500  

while downloaded_count < max_images:
    images = driver.find_elements(By.XPATH, '//img[contains(@class, "YQ4gaf") or contains(@class, "rg_i Q4LuWd")]')
    print(f"Bulunan görsel sayısı: {len(images)}")

    for i, image in enumerate(images):
        try:
            img_url = image.get_attribute("src") or image.get_attribute("data-src")
            if img_url and img_url.startswith("http"):
                img_path = os.path.join(output_folder, f"image_{downloaded_count + 1}.jpg")
                urllib.request.urlretrieve(img_url, img_path)
                print(f"{img_path} indirildi.")
                downloaded_count += 1
            if downloaded_count >= max_images: 
                break
        except Exception as e:
            print(f"Görsel indirilemedi: {e}")

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    try:
        more_button = driver.find_element(By.XPATH, '//input[@value="Daha fazla göster"]')
        more_button.click()
        time.sleep(3)
    except Exception:
        print("Hata")

driver.quit()
print(f"Toplam {downloaded_count} görsel indirildi.")