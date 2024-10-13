from flask import Flask, request, render_template, redirect, url_for
import os
from PIL import Image
import torch
from torchvision import transforms, models
import shutil
import cv2
from classify import detect_and_classify_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load models
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
AlexNetmodel = models.alexnet(pretrained=True)
AlexNetmodel.classifier = torch.nn.Sequential(
    torch.nn.Linear(9216, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.4),
    torch.nn.Linear(1024, 5),
    torch.nn.LogSoftmax(dim=1)
)
AlexNetmodel.load_state_dict(torch.load('alex_model2.pth'))
AlexNetmodel.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            detect_and_classify_images(filepath, yolo_model, AlexNetmodel, transform)
            return redirect(url_for('results', filename=file.filename))
    return render_template('index.html')

@app.route('/results/<filename>')
def results(filename):
    return render_template('results.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)