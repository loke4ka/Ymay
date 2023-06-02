import random
from django.utils.text import slugify
from mainYmay.models import Video, Question


def populate_questions():
    videos = Video.objects.all()

    for video in videos:
        title = video.title
        slug = slugify(title)

        # Создаем вопрос для каждого видео
        question_text = "What is the translation of this letter ?"

        # Создаем вопрос в базе данных
        question = Question.objects.create(
            text=question_text,
            correct_answer=title,
            video=video
        )

        print(f"Created question: {question_text}")

    print("Question population completed.")


# Запуск скрипта для заполнения базы данных вопросами
populate_questions()
