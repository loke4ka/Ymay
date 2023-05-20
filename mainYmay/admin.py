from django.contrib import admin
from mainYmay.models import *
from embed_video.admin import AdminVideoMixin

admin.site.register(User)


class AdminVideo(AdminVideoMixin, admin.ModelAdmin):
    pass


admin.site.register(Video, AdminVideo)
admin.site.register(Survey)
