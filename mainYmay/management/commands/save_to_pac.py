import os
import sqlite3

import pytube
from pytube import YouTube

# Connect to the SQLite database
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

# Select all videos from the video table
c.execute("SELECT id, url, title FROM mainYmay_video")
videos = c.fetchall()

# Loop through the videos and download them using pytube
for id, url, title in videos:
    print(f"Downloading video {id} ({title}) from {url}")
    try:
        # Create a pytube video object
        video = pytube.YouTube(url)

        # Download the video
        video.streams.get_highest_resolution().download(output_path='data/videos', filename=f"{title}.mp4")

        print(f"Downloaded video {id} ({title})")
    except Exception as e:
        print(f"Failed to download video {id} ({title}): {e}")

# Close the database connection
conn.close()
