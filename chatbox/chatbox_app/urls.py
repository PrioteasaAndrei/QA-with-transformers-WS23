from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("chatboxroom/", views.room, name="room"),
    path("close_instance/", views.close_instance, name="close_instance")
]