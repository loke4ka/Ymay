from django.db import models
from embed_video.fields import EmbedVideoField
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin

from django.db import models


class User(models.Model):
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField(null=True)
    email = models.EmailField(null=True)
    password = models.CharField(max_length=100, null=True)
    last_login = models.DateTimeField(auto_now=True)  # Добавляем поле last_login
    is_active = models.BooleanField(default=True)

    # Методы
    def is_authenticated(self):
        return True

    def __str__(self):
        return self.name


class AdminUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    is_staff = models.BooleanField(default=False)

    #

    def __str__(self):
        return self.user.name


# Model for Video links
class Video(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    video_id = models.CharField(max_length=50)
    embed = EmbedVideoField()
    order = models.IntegerField(default=0)

    def __str__(self):
        return self.title


class Survey(models.Model):
    native_language = models.CharField(max_length=100)
    how_did_you_find_us = models.CharField(max_length=100)
    level_of_interest = models.IntegerField()
    reason_for_learning = models.TextField()
    time_to_dedicate = models.CharField(max_length=100)


class Quiz(models.Model):
    title = models.CharField(max_length=100)
    videos = models.ManyToManyField(Video)

    def __str__(self):
        return self.title


class Question(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)
    text = models.CharField(max_length=200)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    order = models.IntegerField(default=0)

    def __str__(self):
        return self.text


class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    text = models.CharField(max_length=100)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return self.text


class UserProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    is_correct = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'quiz', 'question')
