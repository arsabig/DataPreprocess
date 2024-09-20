from googleapiclient.discovery import build
import pandas as pd
import datetime

# You need to set up your API key from Google Developers Console
API_KEY = ''
# API_KEY = ''
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
country = 'FR'
lang = 'fr'

def get_youtube_service():
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

def get_channels_by_category(service, category, country, lang):
    request = service.search().list(
        part='snippet',
        type='channel',
        q=category,
        regionCode=country,
        relevanceLanguage = lang,
        maxResults=50
    )
    response = request.execute()
    return response['items']

def get_channel_details(service, channel_id):
    request = service.channels().list(
        part='snippet,statistics,contentDetails',
        id=channel_id
    )
    response = request.execute()
    return response['items'][0]

def get_recent_video(service, channel_id):
    request = service.search().list(
        part='snippet',
        channelId=channel_id,
        order='date',
        maxResults=1
    )
    response = request.execute()
    return response['items'][0]

def get_popular_video(service, channel_id):
    request = service.search().list(
        part='snippet',
        channelId=channel_id,
        order='viewCount',
        maxResults=1
    )
    response = request.execute()
    return response['items'][0]

def main():
    service = get_youtube_service()
    # categories = ['food|cooking', 'beauty', 'fitness', 'fashion', 'tech']
    categories = ['tech']
    channels_data = []

    for category in categories:
        channels = get_channels_by_category(service, category, country, lang)
        for channel in channels:
            channel_details = get_channel_details(service, channel['id']['channelId'])
            recent_video = get_recent_video(service, channel['id']['channelId'])
            popular_video = get_popular_video(service, channel['id']['channelId'])

            subscriber_count = int(channel_details['statistics']['subscriberCount'])
            last_video_date = recent_video['snippet']['publishedAt']
            last_video_date = datetime.datetime.strptime(last_video_date, "%Y-%m-%dT%H:%M:%SZ")

            if subscriber_count > 100000 and (datetime.datetime.now() - last_video_date).days <= 30:
                channels_data.append([
                    country,
                    channel_details['snippet']['title'],
                    channel_details['snippet'].get('email', 'N/A'),
                    category,
                    f"https://www.youtube.com/watch?v={popular_video['id']['videoId']}"
                ])
                

    df = pd.DataFrame(channels_data, columns=['Country', 'Channel Name', 'E-mail', 'Genre', 'Video Link'])
    df.to_csv('litsofchannels.csv', sep='\t', encoding='utf-8')
    print(df)

if __name__ == '__main__':
    main()
