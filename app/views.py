from django.shortcuts import render
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from . import tasks

import json

import os

# Create your views here.
class Recommender(TemplateView):
    @csrf_exempt
    def post(self, request, **kwargs):
        try:
            if not os.path.isfile('cosine_sim.npy'):
                tasks.train()

            var = json.loads(request.POST)
            ret = tasks.predict(var['minimal'], var['maksimal'], var['lokasi'], var['kategori'])

            return HttpResponse(ret.to_json(), content_type='application/json')
        except:
            pass
        