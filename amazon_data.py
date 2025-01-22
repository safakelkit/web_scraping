import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib.request
import chromedriver_autoinstaller
from selenium.common.exceptions import NoSuchElementException

url = "https://www.amazon.com.tr/bavul-valiz/s?k=bavul+valiz"

output_folder = "amazon_valiz_images"
os.makedirs(output_folder, exist_ok=True)

chromedriver_autoinstaller.install()

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
driver.get(url)
time.sleep(3) 

try:
    cookie_button = driver.find_element(By.ID, "sp-cc-accept")
    cookie_button.click()
    print("Çerez onayı kabul edildi.")
    time.sleep(2)
except NoSuchElementException:
    print("Çerez onay düğmesi bulunamadı, devam ediliyor.")

images_collected = set()
max_images = 1500 

while len(images_collected) < max_images:
    try:
        images = driver.find_elements(By.XPATH, '//img[contains(@class, "s-image")]')
        for image in images:
            img_url = image.get_attribute("src")
            if img_url and img_url not in images_collected: 
                images_collected.add(img_url)
                img_path = os.path.join(output_folder, f"image_{len(images_collected)}.jpg")
                urllib.request.urlretrieve(img_url, img_path) 
                print(f"{img_path} indirildi.")

            if len(images_collected) >= max_images:
                break

        try:
            next_button = driver.find_element(By.XPATH, '//a[contains(@class, "s-pagination-next")]')
            driver.execute_script("arguments[0].scrollIntoView(true);", next_button)  
            next_button.click() 
            time.sleep(3)  
            print("Sonraki sayfaya geçiliyor.")
        except NoSuchElementException:
            print("Son sayfa.")
            break

    except Exception as e:
        print(f"Hata: {e}")
        break

driver.quit()
print(f"Toplam {len(images_collected)} görsel indirildi.")