from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import base64
import sys
import requests
import asyncio
import threading


def actual_analysis(interview_id, question_id, audio_bytes):
    print("thread started")
    mapper = ["anger", "disgust", "fear", "joy",
              "neutral", "other", "sadness", "surprise", "unknown"]
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
    
    try:
        rec_result = inference_pipeline(audio_bytes, output_dir="./outputs", granularity="utterance", extract_embedding=False)
        print("finished analysis")

        response = {}
        for idx in range(len(rec_result[0]["labels"])):
            if mapper[idx] != "other" and mapper[idx] != "unknown":
                response[mapper[idx]] = rec_result[0]["scores"][idx]
    
        sending_result = {'response': response, 'interview_id':interview_id, 'question_id':question_id}
        print(sending_result)
        url = 'https://18.221.19.88/post_speech_emotion_results/'
        req_response = requests.post(url, json=sending_result, verify=False)
        print("sent with response of :", req_response)

    except Exception as e:
        print(f"An error occurred: {e}")

@csrf_exempt
def speech_emotion_analysis(request):
    # Process a base64 encoded string sent as part of a json object with key: "audio"
    # csrf exempt since the certificate is self-signed 
    if request.method != 'POST':
        return HttpResponse(status=404)
    
    json_data = json.loads(request.body)
    base64_audio_text = json_data['audio']
    interview_id = json_data['interview_id']
    question_id = json_data['question_id']
    audio_bytes = base64.b64decode(base64_audio_text)
    
    if len(audio_bytes) % 2 != 0:
        padding_length = 2 - (len(audio_bytes) % 2)
        audio_bytes += b'\x00' * padding_length
    
    print("received input")

    # actual_analysis(interview_id, question_id, audio_bytes)
    thread = threading.Thread(target=actual_analysis, args=(interview_id, question_id, audio_bytes))
    thread.start()

    return JsonResponse({'status':'success'})

