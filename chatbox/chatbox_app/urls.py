from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path("", views.home, name="home"),
    path("chatboxroom/", views.room, name="room"),
    path("close_instance/", views.close_instance, name="close_instance")
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT, show_indexes=True)