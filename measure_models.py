import torch
from thop import profile, clever_format

# Import our model-building functions
from models.resnet import get_resnet18_teacher
from models.mobilenet import get_mobilenet_student

def measure_model(model, model_name):
    """
    Uses thop to calculate and print the FLOPs and Params for a model.
    """
    # We need a dummy input tensor with the correct size (1 image, 3 channels, 96x96)
    dummy_input = torch.randn(1, 3, 96, 96)
    
    # Profile the model
    # inputs=() needs a tuple, so we add a comma
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False) 
    
    # Format the numbers to be human-readable (e.g., "1.2G" or "11.2M")
    flops_str, params_str = clever_format([flops, params], "%.2f")
    
    print(f"--- Model: {model_name} ---")
    print(f"Parameters: {params_str}")
    print(f"FLOPs: {flops_str}")
    print("------------------------")

if __name__ == '__main__':
    # Measure the Teacher
    teacher_model = get_resnet18_teacher()
    measure_model(teacher_model, "ResNet-18 (Teacher)")
    
    # Measure the Student
    student_model = get_mobilenet_student()
    measure_model(student_model, "MobileNetV3-Small (Student)")