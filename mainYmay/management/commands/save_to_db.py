import os
import sys
import django

import googleapiclient.discovery
from django.core.management.base import BaseCommand

from mainYmay.models import Video


class Command(BaseCommand):
    help = 'Save videos from a YouTube playlist to the database'

    def handle(self, *args, **options):
        youtube = googleapiclient.discovery.build('youtube', 'v3',
                                                  developerKey="AIzaSyCn_pUz94R8tsGkXB9pfaKH5TESzXL4ctg")

        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=38,  # Устанавливаем максимальное количество видео для сохранения
            playlistId="PLMNYxpfNIsWzMjY7uMjZhbrhX5ll6_ss0"
        )
        response = request.execute()

        for item in response['items']:
            video_title = item['snippet']['title'].upper()  # Преобразуем название видео в верхний регистр
            video_url = f'https://www.youtube.com/watch?v={item["snippet"]["resourceId"]["videoId"]}'

            video, created = Video.objects.get_or_create(title=video_title, url=video_url)
            if created:
                video.save()
