from mainYmay.models import User
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.backends import ModelBackend


class UserBackend(ModelBackend):
    def logout(self, request):
        # Очистка данных сессии
        request.session.flush()

        # Удаляем идентификатор сессии из cookie
        auth_logout(request)

    def authenticate(self, request, email=None, password=None, **kwargs):
        try:
            user = User.objects.filter(email=email).first()
        except User.DoesNotExist:
            return None

        if user and user.password == password:
            return user

    def get_user(self, user_id):
        if user_id is None:
            return None

        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None
