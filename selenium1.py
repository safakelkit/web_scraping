from requests import options
from selenium import webdriver
from selenium.webdriver.common.by import By
import selenium.webdriver.chrome.options
import chromedriver_autoinstaller
import requests
import time

chromedriver_autoinstaller.install()
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")
chrome_options.add_experimental_option("detach", True)

web_driver = webdriver.Chrome(chrome_options)
web_driver.delete_all_cookies()
web_driver.get('https://www.hepsiburada.com/lenovo-ideapad-slim-3-15iah8-intel-core-i5-12450h-8gb-512gb-ssd-freedos-15-6-fhd-tasinabilir-bilgisayar-83er000wtr-pm-HBC000059XX67')
time.sleep(5)

img = web_driver.find_element(by=By.XPATH, value="/html/body/div[2]/div/div/main/div/div/div[2]/section[2]/div[1]/div/div/ol/li[1]/div/div/div/div[2]/div/picture/img")
src = img.get_attribute("src")
url = src

