from django.db import models
from embed_video.fields import EmbedVideoField

from django.db import models


class User(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField(null=True)
    email = models.EmailField(null=True)
    password = models.CharField(max_length=100, null=True)
    last_login = models.DateTimeField(auto_now=True)  # Добавляем поле last_login
    is_active = models.BooleanField(default=True)

    # Методы

    def __str__(self):
        return self.name


# Model for Video links
class Video(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    video_id = models.CharField(max_length=50)
    embed = EmbedVideoField()

    def __str__(self):
        return self.title


class Survey(models.Model):
    native_language = models.CharField(max_length=100)
    how_did_you_find_us = models.CharField(max_length=100)
    level_of_interest = models.IntegerField()
    reason_for_learning = models.TextField()
    time_to_dedicate = models.CharField(max_length=100)
