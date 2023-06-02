"""
URL configuration for Qsign3 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.views import LogoutView, LoginView

from mainYmay.views import *

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', welcome, name='welcome'),
                  path('select_language', select_lang, name='select_language'),
                  path('hdyk', hdyk, name='hdyk'),
                  path('hmedy', hmedy, name='hmedy'),
                  path('wryse', wryse, name='wryse'),

                  path('tiwyca', tiwyca, name='tiwyca'),
                  path('wiydst', wiydst, name='wiydst'),
                  path('onboard_complete', onboard_complete, name='onboard_complete'),

                  path('register/name/', register_name, name='register_name'),
                  path('register/age/', register_age, name='register_age'),
                  path('register/email/', register_email, name='register_email'),
                  path('register/password/', register_password, name='register_password'),
                  path('register/success/', register_success, name='register_success'),

                  path('homepage', home_page, name='homepage'),

                  path('sign_in', sign_in, name='sign_in'),

                  path('password_reset', password_reset, name='password_reset'),
                  path('password_reset_confirm', password_reset_confirm, name='password_reset_confirm'),
                  path('create_new_password', create_new_password, name='create_new_password'),
                  path('reset_successfully', reset_successfully, name='reset_successfully'),

                  # path('get_video/', get_video, name='get_video'),
                  path('home_language/', home_language, name='home_language'),
                  path('account_page/', account_page, name='account_page'),
                  path('settings_page/', settings_page, name='settings_page'),

                  path('logout/', logout_view, name='logout'),

                  path('quiz/<str:quiz_title>/<int:question_order>/', quiz_page, name='quiz'),

                  path('leaderboard/', leaderboard, name='leaderboard'),

                  path('people_profile/', people_profile, name='people_profile'),
                  path('premium_go/', premium_go, name='premium_go'),
                  path('subscribe_plan/', subscribe_plan, name='subscribe_plan'),
                  path('select_payment_method/', select_payment_method, name='select_payment_method'),
                  path('sub_success/', sub_success, name='sub_success'),

                  path('quiz_completed', quiz_complete, name='quiz_completed'),

              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
