import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Khởi tạo webdriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.w3schools.com/html/html_forms.asp")  # Trang test

actions = ['click_try_it_yourself', 'click_submit', 'scroll_down']
q_table = np.zeros((1, len(actions)))  # đơn giản, 1 state

# RL hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 100

def do_action(action_name):
    try:
        if action_name == 'click_try_it_yourself':
            btn = driver.find_element(By.LINK_TEXT, "Try it Yourself »")
            btn.click()
            return 10
        elif action_name == 'click_submit':
            btn = driver.find_element(By.XPATH, "//button[contains(text(),'Submit')]")
            btn.click()
            return 5
        elif action_name == 'scroll_down':
            driver.execute_script("window.scrollBy(0, 300);")
            return 1
    except Exception:
        return -5
    return 0

for ep in range(episodes):
    driver.get("https://www.w3schools.com/html/html_forms.asp")
    time.sleep(1)
    
    done = False
    total_reward = 0
    
    for _ in range(3):  # mỗi tập chỉ làm 3 hành động
        if np.random.rand() < epsilon:
            action_index = np.random.randint(len(actions))
        else:
            action_index = np.argmax(q_table[0])
        
        action = actions[action_index]
        reward = do_action(action)
        total_reward += reward
        
        q_table[0, action_index] += alpha * (reward + gamma * np.max(q_table[0]) - q_table[0, action_index])
        time.sleep(0.5)
    
    print(f"Episode {ep+1}, Total reward: {total_reward}")

driver.quit()
print("🏁 Học xong. Q-table:", q_table)
