from django.db import models
from django.contrib.auth.models import User




    
class POST(models.Model):
    # 글의 제목, 내용, 작성일, 마지막 수정일
    author = models.ForeignKey(User, on_delete=models.CASCADE, to_field = "username")
    title = models.CharField("제목", max_length=50, null=False)
    content = models.TextField("내용", null=False)
    dt_created = models.DateTimeField("작성일", auto_now_add=True, null=False)
    dt_modified = models.DateTimeField("수정일", auto_now=True, null=False)
    color = models.TextField(max_length=50,null=True, blank=True)
    angry = models.FloatField(null=True, blank=True)
    sadness = models.FloatField(null=True, blank=True)
    surprise = models.FloatField(null=True, blank=True)
    fear = models.FloatField(null=True, blank=True)
    trust = models.FloatField(null=True, blank=True)
    joy = models.FloatField(null=True, blank=True)
    anticipation = models.FloatField(null=True, blank=True)
    disgust = models.FloatField(null=True, blank=True)
    
    
    

    def __str__(self):
        return self.title
    
    
    
class Comment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name= 'comment' )
    
class   Graph(models.Model):
    # author = models.ForeignKey(User, on_delete=models.CASCADE, to_field = "username")
    angry = models.FloatField(null=True, blank=True)
    sadness = models.FloatField(null=True, blank=True)
    surprise = models.FloatField(null=True, blank=True)
    fear = models.FloatField(null=True, blank=True)
    trust = models.FloatField(null=True, blank=True)
    joy = models.FloatField(null=True, blank=True)
    anticipation = models.FloatField(null=True, blank=True)
    disgust = models.FloatField(null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True, null=False)
    color = models.TextField(max_length=50,null=True, blank=True)

class Recom(models.Model):
    title = models.CharField(null=True, blank=True, max_length=1000)
    link = models.CharField(null=True, blank=True, max_length=1000)

class Relax(models.Model):
    title = models.CharField(null=True, blank=True, max_length=1000)
    link = models.CharField(null=True, blank=True, max_length=1000)

class GraphMonth(models.Model): # 월간그래프 추가
    angry = models.FloatField(null=True, blank=True)
    sadness = models.FloatField(null=True, blank=True)
    surprise = models.FloatField(null=True, blank=True)
    fear = models.FloatField(null=True, blank=True)
    trust = models.FloatField(null=True, blank=True)
    joy = models.FloatField(null=True, blank=True)
    anticipation = models.FloatField(null=True, blank=True)
    disgust = models.FloatField(null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True, null=False)
    color = models.TextField(max_length=50,null=True, blank=True)
    month = models.TextField(max_length=50,null=True, blank=True)

    

    
