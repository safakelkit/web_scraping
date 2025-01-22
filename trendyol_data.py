import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller
import urllib.request

chromedriver_autoinstaller.install()

categories = {
"Erkek_Sirt_Cantasi": "https://www.trendyol.com/erkek-sirt-cantasi-x-g2-c115"
}

output_folder = "product_images"
os.makedirs(output_folder, exist_ok=True)

for category_name, url in categories.items():
    category_folder = os.path.join(output_folder, category_name)
    os.makedirs(category_folder, exist_ok=True)

    driver = webdriver.Chrome() 
    driver.get(url)
    time.sleep(5) 

    print(f"'{category_name}' kategorisi için veri çekiliyor...")
    products_collected = set()
    scroll_attempts = 0 
    max_scroll_attempts = 10  
    scroll_pause_time = 2  

    while len(products_collected) < 1500:  
        try:
            products = driver.find_elements(By.XPATH, '//img[contains(@class, "p-card-img")]')

            for product in products:
                try:
                    img_url = product.get_attribute("src")
                    if img_url and img_url not in products_collected:  
                        products_collected.add(img_url)
                        img_path = os.path.join(category_folder, f"{category_name}_{len(products_collected)}.jpg")
                        urllib.request.urlretrieve(img_url, img_path)  
                        print(f"Görsel kaydedildi: {img_path}")

                        if len(products_collected) >= 1000:
                            break
                except Exception as e:
                    print(f"Görsel indirilemedi: {e}")

            if len(products_collected) >= 1500:
                break

            driver.execute_script("window.scrollBy(0, 1000);") 
            time.sleep(scroll_pause_time)  

            new_products = driver.find_elements(By.XPATH, '//img[contains(@class, "p-card-img")]')
            if len(new_products) == len(products): 
                scroll_attempts += 1
                if scroll_attempts >= max_scroll_attempts: 
                    print(f"'{category_name}' kategorisinde daha fazla görsel bulunamadı.")
                    break
            else:
                scroll_attempts = 0 

        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            break

    driver.quit() 
    print(f"'{category_name}' kategorisinden {len(products_collected)} görsel indirildi.")

print("Tüm görseller başarıyla indirildi!")