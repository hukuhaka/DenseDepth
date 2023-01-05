import argparse as arg
import time

import torch
from torchvision import transforms

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

import torch

# from scipy.io import savemat

from model import *
from data_nyu import *
from data_kitti import *

def colorize(value, vmin=0.1, vmax=10.0, cmap="binary"):

    value = value.cpu().numpy()[0, :, :]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def main():
    parser = arg.ArgumentParser(description="Test the model that has been trained")
    parser.add_argument("--image", type=str, default="/home/dataset/EH/DataSet/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png")
    parser.add_argument("--depth", type=str, default="/home/dataset/EH/DataSet/kitti/annotated_inpainting/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png")
    parser.add_argument("--model", type=str, default="/home/dataset/EH/project/Model/DenseDepth/1-4-2254ckpt_19_83.pth",
                        help="Model path")
    parser.add_argument("--type", type=str, default="kitti", choices=["nyu", "kitti"], help="Type of model. Choose nyu or kitti")
    parser.add_argument("--backbone", default="mobilevit", type=str, choices=["densenet", "efficientnet", "mobilevit"],
                        help="Choose backbone network densenet or efficientnet")
    parser.add_argument("--device", type=str, default="cuda", help="Choose device cuda or cpu")
    parser.add_argument("--matplotlib", type=str, default="false", choices=["true", "false"], help="Show image true or false")
    parser.add_argument("--cal_error", type=str, default="false", choices=["true", "false"], help="Calculate error true or false. It use depth image")
    parser.add_argument("--time_test", type=str, default="true", choices=["true", "false"], help="Calculate time testing true or false. It use depth image")
    
    parser.add_argument("--nyudata", default="/home/dataset/EH/DataSet/nyu_data.zip", type=str,
                        help="NYU DataSet Path")
    parser.add_argument("--kitticsv", default="/home/dataset/EH/DataSet/kitti/kitti_train.csv", type=str,
                        help="Kitti DataSet csv path")
    parser.add_argument("--kittidata", default="/home/dataset/EH/DataSet/kitti", type=str,
                        help="Kitti DataSet Path")
    
    args = parser.parse_args()
    
    
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

    model = DenseDepth(encoder_pretrained=False, type=args.backbone)
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt["model_state_dict"])
    if args.device == "cuda":
        model = model.to(device)
    
    print(f"Model backbone type: {args.backbone}")
    print(f"Model data type: {args.type}")
    
    model.eval()
    tf_toTensor = transforms.ToTensor()
    
    if args.type == "nyu":
        image = Image.open(args.image).resize((640, 480)).convert("RGB")
        if args.cal_error == "true": 
            depth = Image.open(args.depth).resize((640//2, 480//2))
            depth = np.clip(np.array(depth) / 1000.0, 0.1, 10.0)
    elif args.type == "kitti":
        image = Image.open(args.image).resize((640, 384)).convert("RGB")
        if args.cal_error == "true": 
            depth = Image.open(args.depth).resize((640//2, 384//2))
            depth = np.clip(np.array(depth) / 1000.0, 1.0, 80.0)
    image = np.clip(np.array(image) / 255, 0, 1)
    image = tf_toTensor(image)
    image = np.expand_dims(image, axis=0)
    image = torch.Tensor(image).float().to(device)
    
    preds = model(image)
    
    output = colorize((preds.data).squeeze(0), cmap="plasma")
    output = output.transpose((1,2,0))
    output = Image.fromarray(output)
    output.save("test.png")
    
    if args.cal_error == "true":
        depth = tf_toTensor(depth)
        depth = np.expand_dims(depth, axis=0)
        depth = torch.Tensor(depth).float()
        error = F.mse_loss(preds, depth)
        print(f"MSE loss: {error}")
    
    if args.time_test == "true" and args.type == "nyu":
        maxDepth = 10.0
        minDepth = 0.1
        TrainLoader, ValidationLoader, TestLoader = nyu_DataLoader(
            path=args.nyudata, batch_size=1, test=False, minDepth=minDepth, maxDepth=maxDepth)
        with torch.no_grad():
            start = time.time()
            for batch in TestLoader:
                image = torch.tensor(batch["image"]).to(device, dtype=torch.float)
                time_testing = model(image)
            print(f"Total test time: {time.time() - start}s. avg time: {(time.time() - start)/len(TestLoader):.4f}s.")
            
    elif args.time_test == "true" and args.type == "kitti":
        maxDepth = 80.0
        minDepth = 1.0
        TrainLoader, ValidationLoader, TestLoader = kitti_DataLoader(
            csvpath=args.kitticsv, datapath=args.kittidata,
            batch_size=1, test=False, minDepth=minDepth, maxDepth=maxDepth)
        with torch.no_grad():
            start = time.time()
            for batch in TestLoader:
                image = torch.tensor(batch["image"]).to(device, dtype=torch.float)
                time_testing = model(image)
            print(f"Total test time: {time.time() - start}s. avg time: {(time.time() - start)/len(TestLoader):.4f}s.")
    # preds = {"preds": np.array(preds.detach().squeeze(), dtype=np.float64)}
    # savemat("matrix_test.mat", preds)
    
    if args.matplotlib == "true" and args.cal_error == "false":
        plt.imshow(preds.detach().squeeze())
        plt.show()
    elif args.matplotlib == "true" and args.cal_error == "true":
        plt.subplot(2, 2, 1)
        plt.imshow(preds.detach().squeeze())
        plt.axis("off")
        plt.title("model pred")
        
        plt.subplot(2, 2, 2)
        plt.imshow(depth.detach().squeeze())
        plt.axis("off")
        plt.title("depth camera")
        
        plt.subplot(2, 2, 3)
        plt.imshow(Image.open(args.image).resize((640, 384)).convert("RGB"))
        plt.axis("off")
        plt.title("rgb image")
        
        plt.show()


if __name__ == '__main__':
    main()
