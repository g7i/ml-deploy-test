from django.shortcuts import render
from django.http import JsonResponse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('crop.h5')


def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)


def test(img):
    classes = ["Early_blight",
               "Late_blight", "healthy"]
    result = model.predict_classes(
        [prepare(img)])
    return classes[int(result)]


def index(request):
    # b = test('/home/gourav-saini/Desktop/plant/Potato early blight.jfif')
    # a = test('/home/gourav-saini/Desktop/plant/xyz.JPG')
    # return HttpResponse(b)
    if request.method == 'POST':
        img = request.FILES['img']
        a = test(img)
        return JsonResponse({'class': a})
    return render(request, 'upload.html')
