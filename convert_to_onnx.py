import torch.onnx
from model import *

if __name__ == '__main__':
    ### Settings ###
    model_path = "/home/dataset/EH/project/Model/DenseDepth/ckpt_19_80.pth"
    model_type = "densenet"
    dataset_type = "nyu"
    batch_size = 1
    onnx_name = "test.onnx"
    
    print(f"Model type: {model_type}. Convert from pytorch model to onnx model.")
    model = DenseDepth(encoder_pretrained=False, type=model_type)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model_state_dict"])
    
    if dataset_type == "nyu":
        input = torch.randn((batch_size, 3, 480, 640))
        torch.onnx.export(
            model,
            input,
            onnx_name,
            input_names = ['modelInput'],
            output_names = ['modelOutput']
            )
    print("converting complete")