from django.contrib import admin
from mainYmay.models import *
from embed_video.admin import AdminVideoMixin
from .models import Quiz, Question, Answer, UserProgress

admin.site.register(User)


class AdminVideo(AdminVideoMixin, admin.ModelAdmin):
    pass


admin.site.register(Video, AdminVideo)
admin.site.register(Survey)


class AnswerInline(admin.TabularInline):
    model = Answer
    extra = 4


class QuestionAdmin(admin.ModelAdmin):
    inlines = [AnswerInline]
    list_display = ('id', 'text', 'quiz_title')  # Добавлено поле 'id' в список отображаемых полей
    list_filter = ('quiz',)  # Фильтрация по полю 'quiz'

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        queryset = queryset.select_related(
            'quiz')  # Используется select_related для эффективного получения связанного куиза
        return queryset

    def quiz_title(self, obj):
        return obj.quiz.title  # Возвращает титулку связанного куиза

    quiz_title.short_description = 'Quiz'  # Заголовок для столбца 'quiz'


class QuizAdmin(admin.ModelAdmin):
    list_display = ('title', 'get_videos')  # Используйте 'get_videos' вместо 'video'
    list_filter = ('videos',)  # Используйте 'videos' вместо 'video'

    def get_videos(self, obj):
        return ", ".join([str(video) for video in obj.videos.all()])


admin.site.register(Quiz, QuizAdmin)
admin.site.register(Question, QuestionAdmin)
admin.site.register(Answer)
admin.site.register(UserProgress)
