from django.shortcuts import render
from django.http import HttpResponse
import cv2
from ultralytics import YOLO
import numpy as np



model = YOLO("D:\\Minor_Project_2\\model_project\\best.pt")


def predict_return(image):
 
    results = model.predict(image, save=False, save_txt=False)

    orig_img_array = results[0].orig_img.copy()

    per = results[0].boxes.conf.tolist()

    class_names = results[0].names

    for box, p in zip(results[0].boxes.xyxy, per):
        xmin, ymin, xmax, ymax = map(int, box)
        class_name = class_names[0]
        cv2.rectangle(orig_img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(orig_img_array, class_name+" "+str(p)[0:4], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return orig_img_array

from django.core.files.base import ContentFile
import base64


def index(request):
    return render(request, 'index.html')

def home(request):


    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        output_image = predict_return(img)
        _, img_encoded = cv2.imencode('.png', output_image)
        img_data = base64.b64encode(img_encoded).decode()
        return render(request, 'home.html', {'uploaded_image': img_data})
    
    return render(request, 'home.html')



