import torch
import time
import json
import os
import numpy as np
from thop import profile, clever_format
from models.resnet import get_resnet18_teacher
from models.mobilenet import get_mobilenet_student

def measure_throughput(model, device, input_size=(3, 96, 96), batch_size=64, repetitions=50):
    model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    timings = []
    with torch.no_grad():
        for _ in range(repetitions):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda': torch.cuda.synchronize()
            timings.append(time.time() - start)
            
    avg_time = np.mean(timings)
    throughput = batch_size / avg_time
    return throughput

def measure_and_save():
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, 3, 96, 96).to(device)
    
    # Define models to measure
    models_to_measure = [
        ('ResNet-18 (Teacher)', get_resnet18_teacher()),
        ('MobileNetV3 (Student)', get_mobilenet_student())
    ]
    
    print(f"--- Measuring Models on {device} ---")
    
    for name, model in models_to_measure:
        model.to(device)
        model.eval()
        
        # FLOPs & Params
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
        
        # Throughput
        throughput = measure_throughput(model, device)
        
        print(f"{name}: {params/1e6:.2f}M Params, {flops/1e9:.2f}G FLOPs, {throughput:.0f} img/s")
        
        results[name] = {
            'params_M': round(params / 1e6, 2),
            'flops_G': round(flops / 1e9, 3),
            'throughput': round(throughput, 1)
        }

    # Save to JSON
    os.makedirs('logs', exist_ok=True)
    with open('logs/model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Metrics saved to logs/model_metrics.json")

if __name__ == '__main__':
    measure_and_save()