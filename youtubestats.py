import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Setup WebDriver
service = Service(r'C:\\Users\\hudso\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe')
driver = webdriver.Chrome(service=service)

# Target URL and YouTube channel
channel_name = 'mrbeast'
base_url = f'https://www.viewstats.com/@{channel_name}'

# Function to extract views and subscribers from the channelytics page
def get_channelytics_data(driver, base_url):
    channelytics_url = f"{base_url}/channelytics"
    driver.get(channelytics_url)
    time.sleep(3)  # Wait for the page to load
    
    # Extract views and subscribers from the graph or text
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # This part may need manual inspection for exact HTML structure
    views = soup.find('div', text='Views').find_next_sibling().text
    subscribers = soup.find('div', text='Subscribers').find_next_sibling().text
    
    return views, subscribers

# Function to extract video data
def get_video_data(driver, base_url):
    video_url = f"{base_url}/videos"
    driver.get(video_url)
    time.sleep(3)  # Wait for the page to load
    
    # Extract video links
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    video_links = [a['href'] for a in soup.select('a.video-link-selector')]  # Update the selector based on the actual HTML
    
    video_data = []
    
    for link in video_links:
        video_page_url = f"https://www.viewstats.com{link}"
        driver.get(video_page_url)
        time.sleep(2)  # Wait for the page to load
        
        # Extract data from the video page
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        title = soup.find('h1', class_='title-class').text
        thumbnail = soup.find('img', class_='thumbnail-class')['src']
        est_revenue = soup.find('div', class_='revenue-class').text
        views_per_hour = soup.find('div', class_='views-hour-class').text
        thumbnail_changes = soup.find('div', class_='thumbnail-changes-class').text
        title_changes = soup.find('div', class_='title-changes-class').text
        
        video_data.append({
            'Title': title,
            'Thumbnail': thumbnail,
            'Est. Revenue': est_revenue,
            'Views Changed per Hour': views_per_hour,
            'Thumbnail Changes': thumbnail_changes,
            'Title Changes': title_changes
        })
        
    return video_data

# Function to extract similar channels
def get_similar_channels(driver, base_url):
    similar_channels_url = f"{base_url}/similarChannels"
    driver.get(similar_channels_url)
    time.sleep(3)  # Wait for the page to load
    
    # Extract similar channels data
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    channels = []
    for row in soup.select('table.similar-channels-table-selector tr'):
        cols = row.find_all('td')
        if len(cols) > 1:
            channel_name = cols[0].text.strip()
            similarity_score = cols[1].text.strip()
            channels.append({'Channel Name': channel_name, 'Similarity Score': similarity_score})
    
    return channels

# Get channelytics data
views, subscribers = get_channelytics_data(driver, base_url)
print(f"Views: {views}, Subscribers: {subscribers}")

# Get video data
video_data = get_video_data(driver, base_url)

# Get similar channels data
similar_channels = get_similar_channels(driver, base_url)

# Saving data to CSV
df_video = pd.DataFrame(video_data)
df_channels = pd.DataFrame(similar_channels)

df_video.to_csv(f'{channel_name}_videos.csv', index=False)
df_channels.to_csv(f'{channel_name}_similar_channels.csv', index=False)

# Close the driver
driver.quit()
