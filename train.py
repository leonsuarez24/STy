import argparse
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from utils import (
    save_metrics,
    AverageMeter,
    save_npy_metric,
    save_reconstructed_images,
    set_seed,
    print_dict,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
from optics import OpticsSPC, hadamard
from torchvision import transforms
import logging


def main(args):
    set_seed(args.seed)

    path_name = f"s1_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_sd_{args.seed}_af1_{args.acceleration_factor_1}_bc_{args.base_channels}_rp_{args.reg_param}_im_{args.im_size}"

    args.save_path = args.save_path + path_name
    if os.path.exists(f"{args.save_path}/metrics/metrics.npy"):
        print("Experiment already done")
        exit()

    images_path, model_path, metrics_path = save_metrics(f"{args.save_path}")
    current_psnr = 0

    logging.basicConfig(
        filename=f"{metrics_path}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info(f"Starting training with parameters: {args}")

    loss_train_record = np.zeros(args.epochs)
    ssim_train_record = np.zeros(args.epochs)
    psnr_train_record = np.zeros(args.epochs)
    loss_val_record = np.zeros(args.epochs)
    ssim_val_record = np.zeros(args.epochs)
    psnr_val_record = np.zeros(args.epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    im_size = (args.im_size, args.im_size)

    dataset_MRI = FastMRI(
        batch_size=args.batch_size,
        split=0.1,
        workers=0,
        seed=args.seed,
        transforms=(
            transforms.Compose(
                [
                    transforms.Resize(
                        (args.im_size, args.im_size),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                ]
            )
            if args.im_size != 320
            else None
        ),
    )

    train_loader, val_loader, test_loader = dataset_MRI.get_fastmri_dataset()

    model = Model1(
        im_size=im_size,
        base_channel=args.base_channels,
        acc_factor_1=args.acceleration_factor_1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    reg_comp_1 = Compression_reg(
        parameter=args.reg_param,
        compression=((1 / (args.acceleration_factor_1))),
        epochs=args.epochs,
    )

    wandb.init(project=args.project_name, name=path_name, config=args)

    for epoch in range(args.epochs):
        model.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()
        train_loss_1 = AverageMeter()


        data_loop_train = tqdm(enumerate(train_loader), total=len(train_loader), colour="red")
        for _, train_data in data_loop_train:

            img = train_data
            img = img.to(device)

            pred, _ = model(img)

            loss_train_1 = criterion(pred, img)


            loss_train = loss_train_1 
            
            reg = reg_comp_1(model.e.get_mask()) 
            loss_train += reg
            

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            train_loss.update(loss_train.item())
            train_loss_1.update(loss_train_1.item())
            train_ssim.update(
                SSIM(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
            )
            train_psnr.update(
                PSNR(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
            )

            data_loop_train.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
            data_loop_train.set_postfix(
                loss=train_loss.avg, ssim=train_ssim.avg, psnr=train_psnr.avg
            )

        logging.info(
            f"Epoch: {epoch} - Train Loss: {train_loss.avg} - Train SSIM: {train_ssim.avg} - Train PSNR: {train_psnr.avg}"
        )

        val_loss = AverageMeter()
        val_ssim = AverageMeter()
        val_psnr = AverageMeter()
        val_loss_1 = AverageMeter()

        data_loop_val = tqdm(enumerate(val_loader), total=len(val_loader), colour="green")
        with torch.no_grad():
            model.eval()
            for _, val_data in data_loop_val:
                img = val_data
                img = img.to(device)

                pred, _ = model(img)

                loss_val_1 = criterion(pred, img)
                loss_val =  loss_val_1 
                reg = reg_comp_1(model.e.get_mask()) 
                loss_val += reg

                val_loss.update(loss_val.item())
                val_loss_1.update(loss_val_1.item())
                val_ssim.update(
                    SSIM(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
                )
                val_psnr.update(
                    PSNR(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
                )

                data_loop_val.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
                data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

        logging.info(
            f"Epoch: {epoch} - Val Loss: {val_loss.avg} - Val SSIM: {val_ssim.avg} - Val PSNR: {val_psnr.avg}"
        )

        if epoch == 200:
            reg_comp_1.update_regularization_paremeter(parameter=1e15)

        acc_f_1 = (
            torch.numel(model.e.get_mask())
            / (torch.sum(model.e.get_mask()) + 1e-10)
        ).item()

        if (
            val_psnr.avg > current_psnr
            and np.round(acc_f_1, 1) >= args.acceleration_factor_1):

            current_psnr = val_psnr.avg
            print(f"Saving model with PSNR: {current_psnr}\\Acceleration1: {acc_f_1}")
            torch.save(model.state_dict(), f"{model_path}/model.pth")


        mask_1 = save_coded_apertures(
            system_layer=model.e,
            row=8,
            pad=2,
            path=images_path,
            name=f"mask_1_{epoch}",
        )


        mask_imgs_1 = wandb.Image(mask_1, caption=f"Mask_1_{epoch}")

        recs_array, psnr_imgs, ssim_imgs = save_reconstructed_images(
            imgs=img[:, 0, :, :].unsqueeze(1),
            recons=pred[:, 0, :, :].unsqueeze(1),
            num_img=4,
            pad=2,
            path=images_path,
            name=f"reconstructed_{epoch}",
            PSNR=PSNR,
            SSIM=SSIM,
        )

        recs_imgs = wandb.Image(
            recs_array, caption=f"Epoch: {epoch}\nPSNR: {psnr_imgs}\nSSIM: {ssim_imgs}"
        )

        wandb.log(
            {
                "train_loss": train_loss.avg,
                "train_loss_1": train_loss_1.avg,
                "train_ssim": train_ssim.avg,
                "train_psnr": train_psnr.avg,
                "val_loss": val_loss.avg,
                "val_loss_1": val_loss_1.avg,
                "val_ssim": val_ssim.avg,
                "val_psnr": val_psnr.avg,
                "mask_1": mask_imgs_1,
                "acceleration_1": acc_f_1,
                "reg_param": reg_comp_1.parameter,
                "reconstructed": recs_imgs,
            }
        )

        loss_train_record[epoch] = train_loss.avg
        ssim_train_record[epoch] = train_ssim.avg
        psnr_train_record[epoch] = train_psnr.avg
        loss_val_record[epoch] = val_loss.avg
        ssim_val_record[epoch] = val_ssim.avg
        psnr_val_record[epoch] = val_psnr.avg

    # TEST BEST VAL MODEl

    test_loss = AverageMeter()
    test_ssim = AverageMeter()
    test_psnr = AverageMeter()
    test_loss_1 = AverageMeter()

    del model

    model = Model1(
        im_size=im_size,
        base_channel=args.base_channels,
        acc_factor_1=args.acceleration_factor_1,
    ).to(device)

    model.load_state_dict(torch.load(f"{model_path}/model.pth"))

    final_compression_1 = (
        torch.sum(model.e.get_mask()) / torch.numel(model.e.get_mask()) * 100
    ).item()

    print(
        f"Final Compression: {np.round(final_compression_1,3)}, Final acceleration: {np.round(1/(final_compression_1/100),3)}"
    )

    data_loop_test = tqdm(enumerate(test_loader), total=len(test_loader), colour="magenta")
    with torch.no_grad():
        model.eval()
        for _, test_data in data_loop_test:
            img = test_data
            img = img.to(device)

            pred, _ = model(img)

            loss_test_1 = criterion(pred, img)
            loss_test = loss_test_1

            reg = reg_comp_1(model.e.get_mask()) 
            loss_test += reg

            test_loss.update(loss_test.item())
            test_loss_1.update(loss_test_1.item())
            test_ssim.update(
                SSIM(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
            )
            test_psnr.update(
                PSNR(pred[:, 0, :, :].unsqueeze(1), img[:, 0, :, :].unsqueeze(1)).item()
            )

            data_loop_test.set_description("TEST")
            data_loop_test.set_postfix(loss=test_loss.avg, ssim=test_ssim.avg, psnr=test_psnr.avg)

    logging.info(
        f"Test Loss: {test_loss.avg} - Test SSIM: {test_ssim.avg} - Test PSNR: {test_psnr.avg}"
    )

    # save data
    save_npy_metric(
        dict(
            train_loss=loss_train_record,
            train_ssim=ssim_train_record,
            train_psnr=psnr_train_record,
            val_loss=loss_val_record,
            val_ssim=ssim_val_record,
            val_psnr=psnr_val_record,
            test_loss=test_loss.avg,
            test_ssim=test_ssim.avg,
            test_psnr=test_psnr.avg,
            test_acc_1=np.round(1 / (final_compression_1 / 100), 1),
        ),
        f"{metrics_path}/metrics",
    )

    wandb.log(
        {
            "test_loss": test_loss.avg,
            "test_ssim": test_ssim.avg,
            "test_psnr": test_psnr.avg,
            "test_acc_1": np.round(1 / (final_compression_1 / 100), 3),

        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default="5e-4")
    parser.add_argument("--epochs", type=int, default="500")
    parser.add_argument("--batch_size", type=int, default=2**5)
    parser.add_argument("--save_path", type=str, default="WEIGHTS/")
    parser.add_argument("--project_name", type=str, default="mri_2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--acceleration_factor_1", type=float, default=8)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--reg_param", type=float, default=1)
    parser.add_argument("--im_size", type=int, default=256)

    args = parser.parse_args()

    args_dict = vars(args)

    print_dict(args_dict)
    main(args)