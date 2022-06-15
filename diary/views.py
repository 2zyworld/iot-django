from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_200_OK

from django.contrib.auth import authenticate
import json

from django.shortcuts import render
from rest_framework import generics

from django.db.models import Avg, Sum
from datetime import date, timedelta
import sqlite3
# 데이터 처리
from .models import POST,Graph,Recom,Relax,GraphMonth
from .serializers import PostSerializer, GraphSerializer, RecomSerializer, RelaxSerializer, GraphMonthSerializer

# APIView를 사용하기 위해 import
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
# AI Package
from diary.AI.kobert_main.inference import *
import pandas as pd

# 노래추천 유사도 계산
from sklearn.metrics.pairwise import cosine_similarity


# AI package
bert_model, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bert_model, dr_rate=0.5)
model.load_state_dict(torch.load("/Users/qbae/Desktop/portfolio/Iot_project3/IOT/restful_server/diary/encorder_data/model1.pt",map_location=torch.device("cpu")))
model.eval()

tokenizer = get_tokenizer()
token = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
device = torch.device("cpu")

#mqtt
# import paho.mqtt.client as mqtt

def inference_ai(sentence):
    dataset = BERTDatasetInference(sentence, token, 256, True, False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    output = []
    for token_ids, valid_length, segment_ids in loader:
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        output.append(model(token_ids, valid_length, segment_ids))
    softmax = torch.nn.Softmax(dim=1)
    output = softmax(output[0])[0].tolist()
    return output

@api_view(['POST'])
@permission_classes((AllowAny,))
def app_login(request):
    #username = request.data.get('username')
    username = request.data.get('username')
    #email = request.data.get('email')
    password = request.data.get('password')

    print('===> username ', username)
    print('===> password ', password)

    if username is None or password is None:
        return Response({'error': 'Please provide both username and password'},
                        status=HTTP_400_BAD_REQUEST)

    # 여기서 authenticate로 유저 validate
    user = authenticate(username=username, password=password)
    print('>>>> user ', user)
    if user:
        print("로그인 성공!")
        return JsonResponse({'code': '0000', 'msg': '로그인 성공입니다.'}, status=200)
        
    if not user:
        print("실패")
        return JsonResponse({'code': '1001', 'msg': '로그인 실패입니다.'}, status=200)
        # return Response({'error': 'Invalid credentials'}, status=HTTP_404_NOT_FOUND)

    # user 로 토큰 발행
    token, _ = Token.objects.get_or_create(user=user)

    return Response({'token': token.key}, status=HTTP_200_OK)



# Blog의 목록을 보여주는 역할
class PostList(APIView):

    # Blog list를 보여줄 때
    def get(self, request):
        posts = POST.objects.all()
        # 여러 개의 객체를 serialization하기 위해 many=True로 설정
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    # 새로운 Blog 글을 작성할 때
    def post(self, request):
        # # db 초기화 
        # con = sqlite3.connect('IOT/restful_server/db.sqlite3')
        # cursor = con.cursor()

        # cursor.execute("DELETE FROM diary_recom")
        # con.commit()

        # cursor.close()
        # con.close()

        # request.data는 사용자의 입력 데이터
        color = ""
        sentiment = [0,0,0,0,0,0,0,0]
        
        data = request.data
        content = data['content']
        
        temp = pd.read_csv("/Users/qbae/Desktop/portfolio/Iot_project3/IOT/restful_server/diary/encorder_data/encoder.csv", encoding="utf-8")
        encoder = temp.sort_values(by=['code']).reset_index(drop=True)
        
        sentence = f"{content}"
        output = inference_ai(sentence)
        
        

        color = (colorize(output, encoder))# RGB 코드 255곱해서 정수로 변환해야 할 수 있다.
        color_hex = "".join([hex(int(255 * x))[-2:] for x in color])
        print(color_hex)

        sentiment= list(emotion(output,encoder)) # 8가지 숫자.
        summation = sum(sentiment)
        sentiment = [x / summation for x in sentiment]
        print(sentiment)
        

        result = [color_hex, sentiment[0], sentiment[1], sentiment[2], sentiment[3], sentiment[4], sentiment[5], sentiment[6], sentiment[7]]
        
        
        # 영상별 감정값 불러오기 # 최종 저장되는 영상별 감정데이터 
        video_senti_value = pd.read_json("/Users/qbae/Desktop/portfolio/Iot_project3/IOT/restful_server/diary/recom_song_data/video_list_senti_value_df_2.json", orient='index') 
        # print('영상감정',video_senti_value)
        # 감정값 코사인 유사도 인덱스 불러오기 #기존 영상감정에대한 유사도인덱스(내림차순으로 정렬)
        senti_sim_sorted_ind = np.load("/Users/qbae/Desktop/portfolio/Iot_project3/IOT/restful_server/diary/recom_song_data/video_list_senti_value_sim_index.npy") 
        # print('유사도',senti_sim_sorted_ind)
        
        user_senti = pd.DataFrame(sentiment).T # 8가지 감정값
        user_senti.columns = ['anger','anticipation','joy','trust','fear','surprise','sadness','disgust']
        # user_senti = user_senti.iloc[:,1:] # color_hex제외
        # print('사용자감정',user_senti, type(user_senti))

        # 영상별 감정데이터
        video_data = video_senti_value.iloc[:, :2]  # 영상제목, 링크만 추출
        senti_data = video_senti_value.iloc[:, 2:]  # 감정값만 추출

        # 사용자 감정데이터와 결합
        user_add = senti_data.append(user_senti) # 감정값에 사용자 감정 추가
        # print('영상감정+사용자감정',user_add)

        # 사용자 감정값 추가하여 유사도 구하기
        user_add_sim = cosine_similarity(user_add, user_add)
        user_add_sim = user_add_sim.argsort()[:, ::-1]
        user_high_sim = user_add_sim[-1][1]  # 가장 마지막 인덱스 유사도 배열에서 자기자신(0) 다음으로 높은 인덱스 추출

        # 기존 유사도배열에서 user_high_sim의 영상과 가장 유사한 인덱스 추출하기
        rec_ind_high = senti_sim_sorted_ind[user_high_sim, :10].reshape(-1)  # index를 사용하기위해 1차원array로 변경, 유사도가 가장 높은 5개

        recommended_ind = np.concatenate([rec_ind_high], 0)
        result_data = video_data.iloc[recommended_ind]
        title = result_data['title'].tolist()
        link = result_data['link'].tolist()
        link = ['https://www.youtube.com'+link[n] for n in range(0,len(link))]
        ctx = {}
    
        for i in range(1,9):
            result[i] = result[i] * 100
            
        data.update(color=result[0])
        data.update(angry=result[1])
        data.update(anticipation=result[2])
        data.update(joy=result[3])
        data.update(fear=result[4])
        data.update(surprise=result[5])
        data.update(sadness=result[6])
        data.update(disgust=result[7])
        data.update(trust=result[8])



        post_serializer = PostSerializer(data=data)
        graph_serializer = GraphSerializer(data=data)
        graph_month_serializer = GraphMonthSerializer(data=data) # 월간그래프
        for i in range(len(title)):
            ctx['title'] = title[i]
            ctx['link'] = link[i]
            recom_serializer = RecomSerializer(data=ctx)
            if recom_serializer.is_valid():
                recom_serializer.save()
        print('post_serializer', data)
        print('recom_serializer', )
       
        angry = int(result[1])
        anticipation=int(result[2])
        joy=int(result[3])
        fear=int(result[4])
        surprise=int(result[5])
        sadness=int(result[6])
        disgust=int(result[7])
        trust=int(result[8])
       
        
        if post_serializer.is_valid(): #유효성 검사
            if graph_serializer.is_valid():
                graph_serializer.save() # 저장
                if graph_month_serializer.is_valid(): # 월간그래프
                    graph_month_serializer.save() # 저장
            post_serializer.save() # 저장
            return JsonResponse({'code': '0000', 'msg': '일기가 성공적으로 저장되었습니다.','color':f'{color_hex}','angry':f'{angry}','anticipation':f'{anticipation}','joy':f'{joy}','fear':f'{fear}','surprise':f'{surprise}','sadness':f'{sadness}','disgust':f'{disgust}','trust':f'{trust}'}, status=status.HTTP_201_CREATED)
        return JsonResponse({'code': '1001', 'msg': '일기 저장에 실패했습니다.'}, status=status.HTTP_400_BAD_REQUEST)
       
        

# post의 detail을 보여주는 역할
class PostDetail(APIView):
    # Blog 객체 가져오기
    def get_object(self, pk):
        try:
            return POST.objects.get(pk=pk)
        except POST.DoesNotExist:
            raise Http404
    
    # post의 detail 보기
    def get(self, request, pk, format=None):
        post = self.get_object(pk)
        serializer = PostSerializer(post)
        return Response(serializer.data)

    # post 수정하기
    def put(self, request, pk, format=None):
        post = self.get_object(pk)
        serializer = PostSerializer(post, data=request.data) 
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data) 
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # post 삭제하기
    def delete(self, request, pk, format=None):
        post = self.get_object(pk)
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)   
    
    
    # Blog의 목록을 보여주는 역할
class GraphList(APIView):
    # Blog list를 보여줄 때
    def post(self, request):
            # request.data는 사용자의 입력 데이터
            serializer = GraphSerializer(data=request.data)
            if serializer.is_valid(): #유효성 검사
                serializer.save() # 저장
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    def get(self, request):

        #graphs = Graph.objects.filter().values('id')

        list=[]
        emotion = {}
        emotion['id'] = 1
        # id = Graph.objects.filter(date__range = ["2022-05-29", "2022-06-01"]).values('id')
        # emotion.update(id)
        # 실제 조건 : 일주일 동안의 감정
        # today = datetime.today()
        # enddate = startdate - timedelta(days=6)
        # date__range = [enddate, today]
        angry = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(angry=Avg('angry'))
        emotion.update(angry)

        anticipation = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(anticipation=Avg('anticipation'))
        emotion.update(anticipation)

        joy = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(joy=Avg('joy'))
        emotion.update(joy)

        fear = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(fear=Avg('fear'))
        emotion.update(fear)

        surprise = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(surprise=Avg('surprise'))
        emotion.update(surprise)

        sadness = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(sadness=Avg('sadness'))
        emotion.update(sadness)

        disgust = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(disgust=Avg('disgust'))
        emotion.update(disgust)

        trust = Graph.objects.filter(date__range = ["2022-06-12", "2022-06-20"]).aggregate(trust=Avg('trust'))
        emotion.update(trust)
        list.append(emotion)
        print(list)
        
        # 여러 개의 객체를 serialization하기 위해 many=True로 설정
        # serializer = GraphSerializer(graphs, many = True)
        
        serializer = GraphSerializer(list, many = True) 
        
        # graph = Graph.objects.all()
        # serializer = GraphSerializer(graph, many = True) # json 형식으로
        return Response(serializer.data)

    # # 새로운 Blog 글을 작성할 때
    # def post(self, request):
    #     # request.data는 사용자의 입력 데이터
    #     serializer = GraphSerializer(data=request.data)
    #     if serializer.is_valid(): #유효성 검사
    #         serializer.save() # 저장
    #         return JsonResponse({'code': '0000', 'msg': '일기가 성공적으로 저장되었습니다.'}, status=status.HTTP_201_CREATED)
    #     return JsonResponse({'code': '1001', 'msg': '일기 저장에 실패했습니다.'}, status=status.HTTP_400_BAD_REQUEST)

class RecommendSong(APIView):
    
    def get(self, request): 
        Song = Recom.objects.all()
        serializer = RecomSerializer(Song, many = True)

        return Response(serializer.data)

class RelaxSong(APIView):
    def get(self, request):
            relaxs = Relax.objects.all()
            # 여러 개의 객체를 serialization하기 위해 many=True로 설정
            serializer = RelaxSerializer(relaxs, many=True)
            return Response(serializer.data)
        # 새로운 Blog 글을 작성할 때
    def post(self, request):
            # request.data는 사용자의 입력 데이터
            serializer = RelaxSerializer(data=request.data)
            if serializer.is_valid(): #유효성 검사
                serializer.save() # 저장
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GraphListMonth(APIView):
    # Blog list를 보여줄 때
    def get(self, request):
        
        api_list=[]
        
        month_list_start = ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01",
                            "2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01"]
        month_list_end = ["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31", "2022-06-30",
                        "2022-07-31", "2022-08-31", "2022-09-30", "2022-10-31", "2022-11-30", "2022-12-31"]
        
        for n in range(0, len(month_list_start)):
            emotion = []
            month = {}
            start_day = month_list_start[n]
            end_day = month_list_end[n]
            
            angry = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(angry=Sum('angry'))
            emotion.append(angry.get('angry'))
            anticipation = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(anticipation=Sum('anticipation'))
            emotion.append(anticipation.get('anticipation'))
            joy = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(joy=Sum('joy'))
            emotion.append(joy.get('joy'))
            fear = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(fear=Sum('fear'))
            emotion.append(fear.get('fear'))
            surprise = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(surprise=Sum('surprise'))
            emotion.append(surprise.get('surprise'))
            sadness = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(sadness=Sum('sadness'))
            emotion.append(sadness.get('sadness'))
            disgust = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(disgust=Sum('disgust'))
            emotion.append(disgust.get('disgust'))
            trust = GraphMonth.objects.filter(date__range = [start_day, end_day]).aggregate(trust=Sum('trust'))
            emotion.append(trust.get('trust'))
            
            emotion = [0 if emotion[n]==None else emotion[n] for n in range(0,len(emotion))]
            print('감정별 값',emotion)
            
            month_sum = sum(emotion)
            
            try : 
                angry = emotion[0] / month_sum * 100
                anticipation = emotion[1] / month_sum * 100
                joy = emotion[2] / month_sum * 100
                fear = emotion[3] / month_sum * 100
                surprise = emotion[4] / month_sum * 100
                sadness = emotion[5] / month_sum * 100
                disgust = emotion[6] / month_sum * 100
                trust = emotion[7] / month_sum * 100
            except :
                angry = 0
                anticipation = 0
                joy = 0
                fear = 0
                surprise = 0
                sadness = 0
                disgust = 0
                trust = 0
            
            month['id'] = 1
            month['month'] = str(n+1) + "월"
            month['angry'] = angry
            month['anticipation'] = anticipation
            month['joy'] = joy
            month['fear'] = fear
            month['surprise'] = surprise
            month['sadness'] = sadness
            month['disgust'] = disgust
            month['trust'] = trust
            print('월별:', month)
            api_list.append(month)
        
        print(api_list)
        serializer = GraphMonthSerializer(api_list, many = True) 
        return Response(serializer.data)          