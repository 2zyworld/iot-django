from requests import request
from .models import POST,Graph,Recom,Relax,GraphMonth
from rest_framework import serializers
from django.contrib.auth.models import User


class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = POST
        fields = [
            "id",
            "author",
            "title",
            "content",
            "dt_created",
            "dt_modified",
            "color",
            "angry",
            "sadness",
            "surprise",
            "fear",
            "trust",
            "joy",
            "anticipation",
            "disgust"
            
            
        ]

   

class PostDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = POST
        fields = [
            "id",
            "author_email",
            "title",
            "content",
            "dt_created",
            "dt_modified",
        ]
        
class GraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = Graph
        fields = [
            "id",
            # "author",
            "angry",
            "anticipation",
            "joy",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "trust",
            "date",
            # "color",
        ]

class RecomSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recom
        fields = [
            "title",
            "link",
        ]
    
class RelaxSerializer(serializers.ModelSerializer):
    class Meta:
        model = Relax
        fields = [
            "title",
            "link",
        ]

class GraphMonthSerializer(serializers.ModelSerializer): # 월간그래프 추가
    class Meta:
        model = GraphMonth
        fields = [
            "id",
            "month",
            "angry",
            "anticipation",
            "joy",
            "fear",
            "surprise",
            "sadness",
            "disgust",
            "trust",
            "date",
        ]
    
    
