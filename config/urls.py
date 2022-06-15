from django.contrib import admin
from django.conf.urls import include
from django.urls import path


from diary import views
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    
    
    path('rest-auth/', include('rest_auth.urls')),
    path('rest-auth/registration/', include('rest_auth.registration.urls')),
    
    path('app_login/', views.app_login),
    
    path('post/', views.PostList.as_view()),
    path('post/<int:pk>/', views.PostDetail.as_view()),
    
    path('graph/', views.GraphList.as_view()),
    path('graphMonth/', views.GraphListMonth.as_view()), # 월간그래프 추가
    
    # recommendsong 수정
    path('recommendsong/', views.RecommendSong.as_view()),

    path('relaxsong/', views.RelaxSong.as_view()),
   
    
    
    
    

    
    # path('api-auth/', include('rest_framework.urls')),
    

]