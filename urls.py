# game/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_game_popularity, name='predict_game_popularity'),
    path('signup/', views.signup, name='signup'),
    path('dashboard/', views.dashboard_view, name='dashboard'), # <<< ADD THIS LINE
]