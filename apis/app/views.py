from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import base64


@csrf_exempt
def speech_emotion_analysis(request):
    # Process a base64 encoded string sent as part of a json object with key: "audio"
    # csrf exempt since the certificate is self-signed 
    if request.method != 'POST':
        return HttpResponse(status=404)

    json_data = json.loads(request.body)
    base64_audio_text = json_data['audio']

    mapper = ["angry", "disgust", "fear", "happy",
              "neutral", "other", "sad", "surprised", "unknown"]
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
    audio_bytes = base64.b64decode(base64_audio_text)

    rec_result = inference_pipeline(
        audio_bytes, output_dir="./outputs", granularity="utterance", extract_embedding=False)
    max_emotion_score = np.argmax(rec_result[0]["scores"])

    results = {
        "emotion": mapper[max_emotion_score],
        "confidence":rec_result[0]["scores"][max_emotion_score]
    }

    return JsonResponse(results)