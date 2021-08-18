from django.contrib import admin
from django.urls import path
from django.urls.conf import include
from .views import main, motion_blur, high_pass, grayscale, test


urlpatterns = [
    path('', main),
    path('test', test),
    path('api/motion/', motion_blur),
    path('api/high-pass/', high_pass),
    path('api/grayscale/', grayscale),
]
