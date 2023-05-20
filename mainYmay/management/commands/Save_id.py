from django.core.management.base import BaseCommand
from mainYmay.models import Video


class Command(BaseCommand):
    help = 'Save videos from a YouTube playlist to the database'

    def handle(self, *args, **options):
        videos = Video.objects.all()
        for video in videos:
            url = video.url
            if 'youtube.com/watch?v=' in url:
                video_id = url.split('youtube.com/watch?v=')[1]
                if '&' in video_id:
                    video_id = video_id.split('&')[0]
                video.video_id = video_id
                video.save()
