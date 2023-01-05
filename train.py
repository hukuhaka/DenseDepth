import os
import time
import argparse as arg
import gc

import torch
from torch import nn
from torch.nn.functional import mse_loss

from torchvision import utils

from tqdm import tqdm

import aim
from aim.sdk.objects.image import convert_to_aim_image_list

from data_nyu import nyu_DataLoader
from data_kitti import kitti_DataLoader
from utils import *
from loss import loss_fn
from modify_loss_fn import modify_loss_fn
from model import DenseDepth


class DenseDepth_Training():
    def __init__(self):
        ### Save and Dataset path check
        if len(args.save) > 0 and not args.save.endswith("/"):
            raise ValueError("Invalid save path or add /")
        elif len(args.save) > 0 and not os.path.isdir((args.save)):
            raise ValueError(f"{args.save} is not avaliable directory")

        self.device = torch.device(
            "cuda" if args.device == "cuda" else "cpu")

        print('Count of using GPUs:', torch.cuda.device_count())
        print('Current cuda device:', torch.cuda.current_device())

        self.first_training = True

    def main(self):
        print("DenseDepth Model training...")

        ### Making Model
        print("Making DenseDepth Model...")
        if args.ckpt == None:
            model, optimizer, start_epoch = model_setting(
                depthmodel=DenseDepth,
                pretrained=True,
                epochs=args.epochs,
                lr=args.lr,
                ckpt=None,
                device=self.device,
                model=args.backbone
            )
        else:
            model, optimizer, start_epoch = model_setting(
                depthmodel=DenseDepth,
                pretrained=False,
                epochs=args.epochs,
                lr=args.lr,
                ckpt=args.ckpt,
                device=self.device,
            )

        ### Cuda Mode On
        model = model.cuda(self.device)
        if args.multigpu == True:
            model = nn.DataParallel(model)

        ### Dataset Load
        print(f"{args.type} Dataset Loading...")
        if args.type == "nyu":
            self.maxDepth = 10.0
            self.minDepth = 0.1
            TrainLoader, ValidationLoader, TestLoader = nyu_DataLoader(
                path=args.nyudata, batch_size=args.batch, test=args.test, minDepth=self.minDepth, maxDepth=self.maxDepth)
        elif args.type == "kitti":
            self.maxDepth = 80.0
            self.minDepth = 1.0
            TrainLoader, ValidationLoader, TestLoader = kitti_DataLoader(
                csvpath=args.kitticsv, datapath=args.kittidata,
                batch_size=args.batch, test=args.test, minDepth=self.minDepth, maxDepth=self.maxDepth)
        else:
            raise ValueError(f"{args.type} is not a correct dataset type")

        print(f"{args.type} DatLoader load complete...")
        num_trainloader = len(TrainLoader)

        ### aim Settings
        self.run = aim.Run(
            args.backbone,
            experiment="model comparison"
        )
        self.run["hparam"] = {
            "net": args.backbone,
            "model": "DenseDepth",
            "learning_rate": args.lr,
            "batch_size": args.batch,
            "data_type": args.type,
            "loss_fn": args.loss_fn,
            "dataset_size": num_trainloader,
        }

        lt = time.localtime()
        dir_name = args.save + \
            str(lt.tm_mon) + "-" + str(lt.tm_mday) + "-" + \
            str(lt.tm_hour).zfill(2) + str(lt.tm_min).zfill(2)
        os.makedirs(dir_name, exist_ok=True)
        f = open(dir_name+"/mode.txt", "w")
        f.write(
            f"net:{args.backbone}\nlearning_rate:{args.lr}\nbatch_size: {args.batch}\n"
            f"data_type: {args.type}\nloss_fn: {args.loss_fn}\ndataset_size: {num_trainloader}"
        )
        f.close()

        ### Starting training
        for epoch in range(start_epoch, args.epochs):

            model.train()

            loss_meter = AverageMeter()
            test_meter = AverageMeter()

            end = time.time()

            ### Training
            print("Starting training ... \n")
            with tqdm(TrainLoader, unit="batch") as tqdm_loader:
                for idx, batch in enumerate(tqdm_loader):

                    tqdm_loader.set_description(f"Epoch {epoch}")

                    optimizer.zero_grad()

                    image_x = torch.Tensor(batch["image"]).to(
                        self.device, dtype=torch.float)
                    depth_y = torch.Tensor(batch["depth"]).to(
                        self.device, dtype=torch.float)

                    preds = model(image_x)

                    # calculating the losses
                    if args.loss_fn == "depth_loss":
                        net_loss = loss_fn(
                            depth_y, preds, alpha=args.alpha, gamma=args.gamma,
                            maxDepth=self.maxDepth, device=self.device)
                    elif args.loss_fn == "modify_depth_loss":
                        net_loss = modify_loss_fn(
                            depth_y, preds, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                            maxDepth=self.maxDepth, device=self.device)

                    loss_meter.update(net_loss.data.item(), image_x.size(0))
                    net_loss.backward()
                    optimizer.step()

                    # Logging
                    num_iters = epoch * num_trainloader + idx
                    self.run.track(loss_meter.val, name="loss",
                                   step=num_iters, context={"subset": "train"})

                    tqdm_loader.set_postfix(
                        loss=loss_meter.val, loss_avg=loss_meter.avg)

                    del image_x
                    del depth_y
                    del preds
            gc.collect()
            torch.cuda.empty_cache()

            self.run.track(loss_meter.avg, name="avg_loss",
                           epoch=epoch, context={"subset": "train"})

            ### Testing
            print("\nStarting testing...\n")
            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(TestLoader)):

                    image_x = torch.Tensor(batch["image"]).to(
                        self.device, dtype=torch.float)
                    depth_y = torch.Tensor(batch["depth"]).to(
                        self.device, dtype=torch.float)

                    preds = model(image_x)

                    net_loss = torch.sqrt(mse_loss(preds, depth_y))
                    test_meter.update(net_loss.data.item(), image_x.size(0))

                    del image_x
                    del depth_y
                    del preds
            gc.collect()
            torch.cuda.empty_cache()

            self.run.track(test_meter.avg, name="test_loss",
                           epoch=epoch, context={"subset": "test"})

            ### Validation
            with torch.no_grad():
                for batch in ValidationLoader:

                    image = torch.Tensor(batch["image"]).to(
                        self.device).float()
                    depth = torch.Tensor(batch["depth"]).to(
                        self.device).float()

                    output = model(image)
                    output_convert = colorize(utils.make_grid(output.data, nrow=6, normalize=False), vmin=self.minDepth,
                                              vmax=self.maxDepth)
                    output_convert = convert_to_aim_image_list(output_convert)

                    if self.first_training:
                        image = convert_to_aim_image_list(image)
                        self.run.track(image, name="rgb", step=0,
                                       context={"subset": "RGB image"})
                        self.first_training = False
                    self.run.track(output_convert, name="predict",
                                   epoch=epoch, context={"subset": "Validation"})

                    del image
                    del depth
                    del output
            gc.collect()
            torch.cuda.empty_cache()

            print(
                f"Train Loss: {loss_meter.avg}, Test RMSE Loss: {test_meter.avg}")
            print("--------------------------------\n\n")

            ### Save Model
            if args.multigpu:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "loss": loss_meter.avg,
                    },
                    dir_name + "/" + 
                    "ckpt_{}_{}.pth".format(
                        epoch, int(loss_meter.avg * 100)),
                )
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "loss": loss_meter.avg,
                    },
                    dir_name + "/" + 
                    "ckpt_{}_{}.pth".format(
                        epoch, int(loss_meter.avg * 100)),
                )


if __name__ == "__main__":
    # Settings
    parser = arg.ArgumentParser(description="Training Densedepth Model")
    parser.add_argument("--backbone", default="mobilevit", type=str, choices=["densenet", "efficientnet", "mobilevit"],
                        help="Choose network densenet or efficientnet")
    parser.add_argument("--ckpt", default=None,
                        help="Model Checkpoint path")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of epochs to train")
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="Model learning rate")
    parser.add_argument("--batch", default=8, type=int,
                        help="Number of training batches")
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="L1 loss parameter")
    parser.add_argument("--beta", default=1.0, type=float,
                        help="L1 loss parameter")
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="L1 loss parameter")
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"],
                        help="Select device cuda or cpu")
    parser.add_argument("--multigpu", default=True, type=bool,
                        help="Turn on multi gpu mode training if computer use cuda")
    parser.add_argument("--type", default="kitti", type=str, choices=["nyu", "kitti"],
                        help="Dataset Type. Select nyu or kitti")
    parser.add_argument("--nyudata", default="/home/dataset/EH/DataSet/nyu_data.zip", type=str,
                        help="NYU DataSet Path")
    parser.add_argument("--kitticsv", default="/home/dataset/EH/DataSet/kitti/kitti_train.csv", type=str,
                        help="Kitti DataSet csv path")
    parser.add_argument("--kittidata", default="/home/dataset/EH/DataSet/kitti", type=str,
                        help="Kitti DataSet Path")
    parser.add_argument("--checkpoint", default="None", type=str,
                        help="Checkpoint Model path")
    parser.add_argument("--save", default="../Model/DenseDepth/", type=str,
                        help="Save Model path")
    parser.add_argument("--test", default=False, type=bool,
                        help="Check if the model is learning")
    parser.add_argument("--loss_fn", default="modify_depth_loss", type=str,
                        help="Loss function. Choose depth_loss or modify_depth_loss")
    args = parser.parse_args()

    Model = DenseDepth_Training()
    Model.main()
