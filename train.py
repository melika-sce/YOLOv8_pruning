from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
results = model.train(data='VisDrone.yaml', epochs=1, imgsz=1024, batch=1, bn_sparsity=True)
