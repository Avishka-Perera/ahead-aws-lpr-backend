import os
import torch
import numpy as np
from PIL import Image
import boto3
from number_extractor import NumberResolver

base_path = os.environ["FUNCTION_DIR"]
ACCESS_KEY = os.environ["ACCESS_KEY"]
SECRET_KEY = os.environ["SECRET_KEY"]
yolo_model_class_path = f"{base_path}/yolov5"
yolo_model_weights_path = f"{base_path}/model-weights/yolo.pt"
char_model_weights_path = f"{base_path}/model-weights/char.pt"
data_path = f"{base_path}/images"

plate_detection_model = torch.hub.load(yolo_model_class_path, 'custom', yolo_model_weights_path, source="local")
device = "cuda" if torch.cuda.is_available() else "cpu" 
nr = NumberResolver(char_model_weights_path, device)

def handler(event, context):
    bucket, image_key = event["bucket"], event["key"]
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    local_path = f"/tmp/image.jpg"
    with open(local_path, 'wb') as f:
        s3.download_fileobj(bucket, image_key, f)

    img = Image.open(local_path)
    lp_locals = plate_detection_model(img)
    lp_locals = lp_locals.pandas().xyxy[0]
    annotations = lp_locals.apply(lambda row: (row["xmin"], row["xmax"], row["ymin"], row["ymax"]), axis=1)
    annotations = list(map(list, annotations.to_list()))
    
    # extract the numbers
    
    img_cv = np.array(img)
    for i, plate_stat in enumerate(annotations):
        xmin, xmax, ymin, ymax = plate_stat
        xmin,xmax,ymin,ymax = int(np.round(xmin)),int(np.round(xmax)),int(np.round(ymin)),int(np.round(ymax))
        plate = img_cv[ymin:ymax, xmin:xmax, :]
        number = nr(plate)
        annotations[i].insert(0, number)


    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": {
            "annotations ": annotations
        }
    }