import pip

pip.main(['install', "selenium"])

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_experimental_option("debuggerAddress", "localhost:9222")

driver = webdriver.Chrome(options=options)

driver.execute_script("window.open('about:blank', '_blank');")
driver.switch_to.window(driver.window_handles[-1])

print("Current URL (new tab):", driver.current_url)

driver.get("https://redbox-trial.uktrade.digital/admin/redbox_core/chathistory/?")

print("Current URL after navigation:", driver.current_url)

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    select_all_checkbox = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "action-toggle"))
    )
    select_all_checkbox.click()
    print("selected all the items")

    action_dropdown = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.NAME, "action"))
    )
    action_dropdown.click()

    link = driver.find_element(By.XPATH, "//a[@title='Click here to select the objects across all pages']")
    link.click()
    
    export_option = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//option[text()='Export Selected']"))
    )
    export_option.click()
    print("Export selected items option chosen")

    go_button = driver.find_element(By.XPATH, "//button[text()='Go']")
    go_button.click()

finally:
    driver.quit()
