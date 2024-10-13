from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import torch

def detect_and_classify_images(img_path, yolo_model, AlexNetmodel, transform):
    output_folder = 'static/images/output/'
    class_labels = {0: 'a10', 1: 'c130', 2: 'f16', 3: 'f22', 4: 'v22'}
    
    img_cv2 = cv2.imread(img_path)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    
    results = yolo_model(img_cv2)
    results_df = results.pandas().xyxy[0]

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()

    for index, row in results_df.iterrows():
        if row['confidence'] > 0.5:
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cropped_image = img_pil.crop((xmin, ymin, xmax, ymax))
            cropped_image_tensor = transform(cropped_image).unsqueeze(0)
            with torch.no_grad():
                outputs = AlexNetmodel(cropped_image_tensor)
                _, predicted = torch.max(outputs, 1)
                class_idx = predicted.item()
            label = class_labels.get(class_idx, "Unknown")
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=2)
            draw.text((xmin, ymin - 10), label, fill="red", font=font)
    
    annotated_img_path = os.path.join(output_folder, os.path.basename(img_path))
    img_pil.save(annotated_img_path)