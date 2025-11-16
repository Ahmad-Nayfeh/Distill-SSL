import torch.nn as nn
import torchvision.models as models

def get_mobilenet_student():
    """
    Returns a MobileNet V3-Small model, which will serve as our student.
    """
    model = models.mobilenet_v3_small(weights=None)
    
    # STL-10 has 10 classes
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 10)
    
    return model

if __name__ == '__main__':
    # Test if the model builds
    model = get_mobilenet_student()
    print("MobileNet V3-Small Student built successfully.")
    print(model)