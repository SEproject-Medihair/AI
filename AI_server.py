from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def preprocess_image(image_file):
    # 이미지 전처리를 위한 변환 정의
    transform = transforms.Compose([
        transforms.Resize([int(600), int(600)], interpolation=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Lambda(lambda x: x.rotate(90)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 이미지 로드 및 전처리
    #image = Image.open(image_path).convert("RGB")
    image = Image.open(BytesIO(image_file.read())).convert("RGB")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict_image(model, input_tensor):
    # 모델을 평가 모드로 설정
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities

def predict_image2(model, input_tensor):
    # 모델을 평가 모드로 설정
    model2.eval()
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities

# 모델 로드
loaded_model = torch.load('aram_model6_architecture.pth')
state_dict = loaded_model.state_dict()

loaded_model2 = torch.load('aram_model1_architecture.pth')
state_dic2t2 = loaded_model2.state_dict()

model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
model2 = EfficientNet.from_name('efficientnet-b0', num_classes=4)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        input_tensor = preprocess_image(file)
        predicted_class, probabilities = predict_image(model, input_tensor)
        predicted_class2, probabilities2 = predict_image2(model2, input_tensor)
        return jsonify({'predicted_class': predicted_class,'predicted_class2': predicted_class2 })

@app.route('/')
def home():
    return '<h1>AI_page</h1>'



if __name__ == '__main__':
    print("LocalHost")
    app.run(host='localhost', port=8082)
