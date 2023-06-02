import googleapiclient.discovery
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages, auth
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.core.mail import send_mail
import random
import requests
from django.core.cache import cache
from django.http import JsonResponse
from embed_video.fields import EmbedVideoField
import embed_video
from embed_video.backends import detect_backend
import tensorflow as tf
from django.conf import settings
from .models import Question, Answer
from random import sample
from random import shuffle
from django.db.models import F
from .models import Quiz, Question, Answer, UserProgress
from django.contrib.auth import logout
from django.shortcuts import get_object_or_404

import os
import numpy as np
from keras.models import load_model

from .backends import UserBackend
from .models import User
from .models import Video

from PIL import Image


# from keras.models import load_model
# import tensorflow


def welcome(request):
    # Ваша логика обработки запроса здесь

    # Возвращаем рендеринг страницы "welcome.html" с контекстом данных, если необходимо
    return render(request, 'welcome.html', context={})


def select_lang(request):
    return render(request, 'select_language.html', context={})


def hdyk(request):
    return render(request, 'HDYK.html')


def hmedy(request):
    return render(request, 'HMEDY.html')


def wryse(request):
    return render(request, 'WRYSE.html')


def tiwyca(request):
    return render(request, 'TIWYCA.html')


def wiydst(request):
    return render(request, 'WIYDST.html')


def onboard_complete(request):
    return render(request, 'onboard-complete.html')


def register_name(request):
    if request.method == 'POST':
        name = request.POST['full_name']
        # Проверка на пустое значение
        if not name:
            messages.error(request, 'Please enter your full name.')
            return redirect('register_name')

        # Создаем нового пользователя с указанным именем
        user = User(name=name)
        user.save()

        # Перенаправляем на следующую страницу
        return redirect('register_age')

    return render(request, 'WIYN.html')


def register_age(request):
    if request.method == 'POST':
        age = request.POST['age']
        # Проверка на пустое значение
        if not age:
            messages.error(request, 'Please enter your age.')
            return redirect('register_age')

        # Обновляем возраст пользователя
        user = User.objects.latest('id')
        user.age = age
        user.save()

        # Перенаправляем на следующую страницу
        return redirect('register_email')

    return render(request, 'HORY.html')  # Отправляем HTML-страницу с формой на регистрацию возраста


def register_email(request):
    if request.method == 'POST':
        email = request.POST['email']  # Получаем значение email из POST-запроса

        if not email:
            messages.error(request, 'Please enter your email.')
            return redirect('register_email')

        # Обновляем
        user = User.objects.latest('id')
        user.email = email
        user.save()
        return redirect('register_password')  # Измените на URL вашей следующей страницы
    return render(request, 'WIYEA.html')


def register_password(request):
    if request.method == 'POST':
        password = request.POST['password']
        # Дополнительная логика обработки пароля, например, проверка на сложность и сохранение в базе данных
        user = User.objects.latest('id')
        user.password = password
        user.save()
        # Перенаправляем на следующую страницу после успешной обработки пароля
        return redirect('register_success')

    return render(request, 'CAP.html')


def register_success(request):
    return render(request, 'CPC.html')


def home_page(request):
    return render(request, 'homepage.html')


def sign_in(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        # Очистка данных аутентификации перед входом
        auth.logout(request)

        # Аутентификация пользователя с использованием вашего бэкенда
        user = UserBackend().authenticate(request, email=email, password=password)

        # Проверка, что функция authenticate() вернула пользователя
        if user is not None:
            # Пользователь аутентифицирован успешно
            auth.login(request, user, backend='mainYmay.backends.UserBackend')
            return redirect('homepage')
        else:
            # Пользователь не аутентифицирован
            return redirect('sign_in')

    return render(request, 'sign_in_form.html')


def logout_view(request):
    if request.method == 'POST':
        auth.logout(request)  # Очистка сессии пользователя
        return redirect('homepage')
    else:
        return


def send_email(api_key, subject, body, to):
    response = requests.post(
        'https://api.elasticemail.com/v2/email/send',
        data={
            'apikey': api_key,
            'subject': subject,
            'from': 'nitr1248@gmail.com',
            'fromName': 'Geka',
            'to': to,
            'bodyHtml': body,
        }
    )

    if response.status_code == 200:
        return True
    else:
        return False


# Данные для функции send_email
api_key = '5E8C08782F3DBC157E2A2E9802D629F20A16F2793D69311F8F64F4767072F5AE19BD6E0B614E9857FBF8B56744571859'


def password_reset(request):
    if request.method == 'POST':
        email = request.POST['email']

        try:
            # Генерируем 4-значный код
            code = ''.join(random.choices('0123456789', k=4))
            # Отправляем код на указанный адрес электронной почты
            subject = 'Password Reset Code'
            body = 'Your password reset code is: {}'.format(code)
            to = email

            if send_email(api_key, subject, body, to):
                print('Email was sent successfully.')
            else:
                print('Email sending failed.')

            # Сохраняем код сброса пароля и адрес электронной почты в кэше на 5 минут
            cache.set('password_reset_code', code, 300)
            cache.set('password_reset_email', email, 300)
            print(code)
            return redirect('password_reset_confirm')  # Редирект на страницу подтверждения ввода кода сброса пароля
        except User.DoesNotExist:
            print('User with email {} does not exist.'.format(email))
    return render(request, 'forget_password.html')  # Отправляем пользователю страницу с формой ввода электронной почты


def password_reset_confirm(request):
    if request.method == 'POST':
        otp_input1 = request.POST['opt-input-1']
        otp_input2 = request.POST['opt-input-2']
        otp_input3 = request.POST['opt-input-3']
        otp_input4 = request.POST['opt-input-4']

        otp_code = otp_input1 + otp_input2 + otp_input3 + otp_input4

        if cache.get('password_reset_code') == otp_code:

            # Получение адреса электронной почты пользователя из кэша
            email = cache.get('password_reset_email')
            # Перенаправление на страницу смены пароля
            return redirect('create_new_password')
        else:
            # Если код неверный, возвращаем ошибку
            return render(request, 'confirm-otp-code.html', {'error': 'Неверный код подтверждения'})
    else:
        # Если запрос не методом POST, отображаем страницу с формой ввода кода
        return render(request, 'confirm-otp-code.html')


def create_new_password(request):
    if request.method == 'POST':
        # Получаем введенные пользователем пароли
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        # Проверяем, что пароли совпадают
        if password1 == password2:
            # Получаем код сброса пароля и адрес электронной почты из кэша
            code = cache.get('password_reset_code')
            email = cache.get('password_reset_email')

            # Проверяем, что код сброса пароля и адрес электронной почты есть в кэше
            if code and email:
                try:
                    # Получаем пользователя по адресу электронной почты
                    user = User.objects.get(email=email)

                    # Устанавливаем новый пароль
                    user.password = password1
                    user.save()

                    # Входим в систему
                    login(request, user)

                    # Очищаем кэш от кода сброса пароля и адреса электронной почты
                    cache.delete('password_reset_code')
                    cache.delete('password_reset_email')

                    # Редирект на страницу приветствия
                    return redirect('reset_successfully')
                except User.DoesNotExist:
                    print('User with email {} does not exist.'.format(email))
        else:
            print('Passwords do not match.')
    return render(request, 'create-new-password.html')


def reset_successfully(request):
    return render(request, 'reset-successfully.html')


"""
def get_video(request):
    if request.method == 'GET':
        letter = request.GET.get('letter')  # Получаем выбранную букву из GET-параметров
        # Используем выбранную букву для получения видео из базы данных
        video = Video.objects.filter(title=letter).first()
        if video:
            video_url = video.video_file.url
            # Возвращаем URL видео в формате JSON
            return JsonResponse({'video_url': video_url})
        else:
            # Если видео не найдено, возвращаем ошибку
            return JsonResponse({'error': 'Video not found'})
    else:
        # Если запрос не GET, возвращаем ошибку
        return JsonResponse({'error': 'Invalid request method'})
"""
model_path = os.path.join(settings.BASE_DIR, "AI", "Model", "keras_model.h5")
labels_path = os.path.join(settings.BASE_DIR, "AI", "Model", "labels.txt")

# Загрузка модели и меток классов
model = None
class_labels = []


def load_model_and_labels():
    global model, class_labels
    model = load_model(model_path)
    with open(labels_path, 'r', encoding='utf-8') as f:
        class_labels = f.read().splitlines()


def preprocess_image(image):
    # Пример предварительной обработки изображения
    resized_image = image.resize((224, 224))
    normalized_image = np.array(resized_image) / 255.0
    return normalized_image


def predict_gesture(image):
    preprocessed_image = preprocess_image(image)
    input_data = np.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(input_data)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class_label, confidence


def home_language(request):
    letter = 'А'

    if request.method == 'POST':
        letter = request.POST.get('letter')

    video = Video.objects.filter(title=letter).first()
    video_url = video.url if video else None

    predicted_class_label, confidence = None, None

    if 'videoCapture' in request.FILES:
        if not model:
            load_model_and_labels()

        video_capture = request.FILES['videoCapture']
        image = Image.open(video_capture)
        predicted_class_label, confidence = predict_gesture(image)

    return render(request, 'home-language.html', {'video_url': video_url, 'letter': letter,
                                                  'predicted_class_label': predicted_class_label,
                                                  'confidence': confidence})


def account_page(request):
    return render(request, 'account-fullpage.html')


def settings_page(request):
    return render(request, 'settings.html')


@login_required
def quiz_page(request, quiz_title, question_order):
    # Получите текущего пользователя
    user = request.user

    if request.method == 'POST':

        print(question_order)

        # Обработка отправки ответа
        order = question_order
        selected_answer_id = request.POST['selected_answer_id']

        # Получение данных вопроса, квиза и выбранного ответа из базы данных
        question = Question.objects.get(order=order)
        quiz = Quiz.objects.get(title=quiz_title)
        selected_answer = Answer.objects.get(id=selected_answer_id)

        # Проверка правильности ответа
        is_correct = selected_answer.is_correct and selected_answer.question == question

        try:
            # Попытайтесь обновить существующую запись в UserProgress для текущего пользователя, квиза и вопроса
            user_progress = UserProgress.objects.get(user=user, quiz=quiz, question=question)
            user_progress.is_correct = is_correct
            user_progress.save()
        except UserProgress.DoesNotExist:
            # Если записи не существует, создайте новую запись в UserProgress
            UserProgress.objects.create(user=user, quiz=quiz, question=question, is_correct=is_correct)

        # Получение следующего вопроса
        next_question = Question.objects.filter(quiz=quiz, order__gt=question.order).order_by('order').first()

        if next_question:
            # Следующий вопрос существует, переходим к следующему вопросу и видео

            next_video = next_question.video
            answers = Answer.objects.filter(question=next_question)
            return redirect('quiz', quiz_title=quiz.title, question_order=next_question.order)

        else:
            # Следующего вопроса нет, получаем следующее видео
            next_video = None
            if next_video is None:
                # Больше видео нет, квиз завершен
                return redirect('quiz_completed')

            next_question = Question.objects.filter(quiz=quiz, order=1).first()
            answers = Answer.objects.filter(question=next_question)

        # Получение всех ответов для данного вопроса
        answers = Answer.objects.filter(question=question)

        # Установка стилей для секций Correct и Incorrect
        correct_section_style = 'display: block;' if is_correct else 'display: none;'
        wrong_section_style = 'display: block;' if not is_correct else 'display: none;'

        # Передача данных в контекст шаблона
        context = {
            'question': next_question,
            'quiz': quiz,
            'video': next_video,
            'answers': answers,
            'correct_section_style': correct_section_style,
            'wrong_section_style': wrong_section_style
        }

        return render(request, 'translate-this-sentence.html', context)

    # Отображение страницы с новым вопросом и видео (при получении GET запроса)
    quiz = Quiz.objects.get(title=quiz_title)

    # Получение текущего вопроса и видео
    question = Question.objects.get(order=question_order)
    video = question.video

    # Получите прогресс пользователя для предыдущего вопроса
    previous_question_order = question_order - 1
    previous_question = Question.objects.filter(quiz=quiz, order=previous_question_order).first()

    user_progress = UserProgress.objects.filter(user=user, quiz=quiz, question=previous_question).first()

    # Получение всех ответов для текущего вопроса
    answers = Answer.objects.filter(question=question)

    # Установка начальных стилей для секций Correct и Incorrect
    correct_section_style = 'display: none;'
    wrong_section_style = 'display: none;'

    if user_progress:
        # Пользователь уже ответил на предыдущий вопрос
        if user_progress.is_correct:
            correct_section_style = 'display: block;'
            wrong_section_style = 'display: none;'
        else:
            correct_section_style = 'display: none;'
            wrong_section_style = 'display: block;'

    # Передача данных в контекст шаблона
    context = {
        'question': question,
        'quiz': quiz,
        'video': video,
        'answers': answers,
        'correct_section_style': correct_section_style,
        'wrong_section_style': wrong_section_style
    }

    return render(request, 'translate-this-sentence.html', context)


def quiz_complete(request):
    return render(request, 'lesson-completed.html')


def leaderboard(request):
    return render(request, 'leaderboard.html')


def people_profile(request):
    return render(request, 'people-profile.html')


def premium_go(request):
    return render(request, 'light-premium-version-fit.html')


def subscribe_plan(request):
    return render(request, 'subscription-plan.html')


def select_payment_method(request):
    return render(request, 'select-subscription-payment-method.html')


def sub_success(request):
    return render(request, 'subscription-successfull.html')
