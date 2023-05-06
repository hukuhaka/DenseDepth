import argparse
import time
import os

import numpy as np
from tqdm import tqdm

import torch
from torch import optim, nn

from clearml import Task, Logger

from utils import *
from dataloader import DepthDataLoader

import warnings
warnings.filterwarnings('ignore')

class DenseDepthTrainer:
    def __init__(self, args):
        self.args = args
        lt = time.localtime()
        
        name = "Densedepth Train" + "-" + str(lt.tm_mon).zfill(2) + str(lt.tm_mday).zfill(2) + "-" +str(lt.tm_hour).zfill(2) + str(lt.tm_min).zfill(2)
        self.task = Task.init(project_name="Densedepth", task_name=name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        time_config = str(lt.tm_mon).zfill(2)+str(lt.tm_mday).zfill(2) + \
            "-"+str(lt.tm_hour).zfill(2)+str(lt.tm_min).zfill(2)
        project_name = "Densedepth"+"-"+time_config
        
        self.dir_name = self.args.save_path+"/"+time_config
        os.makedirs(self.dir_name, exist_ok=True)
        
        self.model, self.loss, self.optimizer = model_setting(self.args, self.device)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, epochs=self.args.epochs, steps_per_epoch=1, anneal_strategy ="cos", pct_start=0.25,
            cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, div_factor=self.args.lr*1e+04/1, final_div_factor=1/0.1)

        self.TrainLoader = DepthDataLoader(args, mode="train", base_data=args.dataset_type[0]).data
        self.TestLoader = DepthDataLoader(args, mode="online_eval", base_data=args.dataset_type[0]).data
        
        if args.dataset_type[0] == "nyu":
            self.min_depth = 1e-03
            self.max_depth = 10
        elif args.dataset_type[0] == "kitti":
            self.min_depth = 1e-03
            self.max_depth = 80
        elif args.dataset_type[0] == "diode":
            self.min_depth = 1e-03
            self.max_depth = 350.0
        
    
    def train(self):
        
        print("DenseDepth Training Start"
              f"Encoder: {self.args.encoder_model}. Optimizer: {self.args.optimizer}. Epoch: {self.args.epochs}.")
        print("Start Training...")
        
        for epoch in range(self.args.epochs):
            
            self.model.train()
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            Logger.current_logger().report_scalar(title="epoch", series="lr", iteration=epoch, value=epoch_lr)
            
            loss_tracking = Recording()
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs} | lr: {epoch_lr}")
            with tqdm(self.TrainLoader, unit="batch") as trainloader:
                for idx, batch in enumerate(trainloader):
                    self.optimizer.zero_grad()
                    
                    image = torch.tensor(batch["image"], dtype=torch.float32, device=self.device)
                    depth = torch.tensor(batch['depth'], dtype=torch.float32, device=self.device)
                    
                    pred = self.model(image)
                    
                    mask = depth > self.min_depth
                    loss = self.loss(pred, depth, mask.to(torch.bool))
                    loss.backward()
                    self.optimizer.step()
                    
                    loss_tracking.update(loss.item())
                    
                    Logger.current_logger().report_scalar(title="train", series="loss", iteration=idx+epoch*len(self.TrainLoader), value=loss.item())
                    Logger.current_logger().report_scalar(title="train", series="max depth", iteration=idx+epoch*len(self.TrainLoader), value=pred.max())
                    trainloader.set_postfix(loss=loss_tracking.data, loss_avg=loss_tracking.avg)
            
            self.scheduler.step()
            Logger.current_logger().report_scalar(title="epoch", series="avg loss", iteration=epoch, value=loss_tracking.avg)
            self.model.eval()
            
            with torch.no_grad():
                
                a1_tracking = Recording()
                rmse_tracking = Recording()
                
                for idx, batch in enumerate(tqdm(self.TestLoader)):

                    image = torch.tensor(batch["image"], dtype=torch.float32, device=self.device)
                    depth = torch.tensor(batch['depth'], dtype=torch.float32, device=self.device)
                    
                    pred = self.model(image)
                    
                    mask = depth > self.min_depth
                    test_error = compute_errors(depth[mask], pred[mask])
                    
                    a1_tracking.update(test_error["a1"])
                    rmse_tracking.update(test_error["rmse"])
            
            print(f"Train Loss: {loss_tracking.avg}, Test a1: {a1_tracking.avg}, Test rmse: {rmse_tracking.avg}\n")
            Logger.current_logger().report_scalar(title="epoch", series="a1", iteration=epoch, value=a1_tracking.avg)
            Logger.current_logger().report_scalar(title="epoch", series="rmse", iteration=epoch, value=rmse_tracking.avg)
            
            model_name = str(epoch).zfill(3)+"-"+self.args.encoder_model+"-"+str(loss_tracking.avg)+".pt"
            save_checkpoint(model=self.model, optimizer=self.optimizer,
                            epoch=epoch, filename=model_name, root=self.dir_name)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--encoder_model", type=str, default="mobilevitv2")
    
    # # KITTI Only
    # parser.add_argument("--dataset_type", default=["kitti"], type=list, action="store",
    #                     choices=["nyu", "kitti", "kitti_dense", "diode"], help="Choose Dataset Type")
    # parser.add_argument("--train_file", type=list, action="store",
    #                     default=["./database/kitti_eigen_train.txt"], help="Train List File")
    # parser.add_argument("--test_file", type=str,
    #                     default="./database/kitti_eigen_test.txt", help="Test List File")
    # parser.add_argument("--image_path", type=list, action="store",
    #                     default=["/home/dataset_disk/KITTI/KITTI_RGB_Image"], help="Image Path to train")
    # parser.add_argument("--depth_path", type=list, action="store",
    #                     default=["/home/dataset_disk/KITTI/KITTI_PointCloud"], help="Depth Image Path to train")
    # parser.add_argument("--test_image_path", type=str,
    #                     default="/home/dataset_disk/KITTI/KITTI_RGB_Image", help="Image Path to train")
    # parser.add_argument("--test_depth_path", type=str,
    #                     default="/home/dataset_disk/KITTI/KITTI_PointCloud", help="Depth Image Path to train")
    
    # DIODE Only
    parser.add_argument("--dataset_type", default=["diode", "kitti_dense"], type=list, action="store",
                        choices=["nyu", "kitti", "kitti_dense", "diode"], help="Choose Dataset Type")
    parser.add_argument("--train_file", type=list, action="store",
                        default=["./database/DIODE_outdoor_train.txt", "./database/kitti_eigen_train_numpy.txt"], help="Train List File")
    parser.add_argument("--test_file", type=str,
                        default="./database/DIODE_outdoor_test.txt", help="Test List File")
    parser.add_argument("--image_path", type=list, action="store",
                        default=["/home/dataset_disk/DIODE", "/home/dataset_disk/KITTI/KITTI_RGB_Image"], help="Image Path to train")
    parser.add_argument("--depth_path", type=list, action="store",
                        default=["/home/dataset_disk/DIODE", "/home/dataset_disk/KITTI/KITTI_DenseMap"], help="Depth Image Path to train")
    parser.add_argument("--test_image_path", type=str,
                        default="/home/dataset_disk/DIODE", help="Image Path to train")
    parser.add_argument("--test_depth_path", type=str,
                        default="/home/dataset_disk/DIODE", help="Depth Image Path to train")
    
    parser.add_argument("--save_path", default="/home/model_disk/DenseDepth",
                        type=str, help="Model Save Path")
    
    args = parser.parse_args()
    
    trainer = DenseDepthTrainer(args)
    trainer.train()
