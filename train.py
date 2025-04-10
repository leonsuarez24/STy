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
    get_dataset,
    save_reconstructed_images,
    set_seed,
    print_dict,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np

import logging
from models import UNet
from optics import OpticsSPC, hadamard


def main(args):
    set_seed(args.seed)

    path_name = f"S1_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_s1_{str(args.snapshots_1)}_s2_{args.snapshots_2}_{args.dataset}_sd_{args.seed}_bc_{args.base_channels}"

    args.save_path = args.save_path + path_name

    images_path, model_path, metrics_path = save_metrics(f"{args.save_path}")
    current_psnr = 0

    if os.path.exists(f"{args.save_path}/metrics/metrics.npy"):
        print("Experiment already done")
        exit()

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

    _, im_size, _, _, _, _, testloader, trainloader, valoader = get_dataset(
        args.dataset, "data", args.batch_size, seed=args.seed
    )
    im_size = (1, im_size[-2], im_size[-1])

    spc1 = OpticsSPC(
        input_size=im_size,
        snapshots=args.snapshots_1,
        matrix=hadamard(im_size[-2] * im_size[-1])[: args.snapshots_1],
    ).to(device)

    spc2 = OpticsSPC(
        input_size=im_size,
        snapshots=args.snapshots_2,
        matrix=hadamard(im_size[-2] * im_size[-1])[args.snapshots_1:args.snapshots_1 + args.snapshots_2],
    ).to(device)


    model = UNet(
        n_channels=1,
        base_channel=args.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    wandb.login(key="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    wandb.init(project=args.project_name, name="E2E_" + path_name, config=args)

    for epoch in range(args.epochs):
        model.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()

        data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour="red")
        for _, train_data in data_loop_train:

            img, _ = train_data
            img = img.to(device)

            meas_1 = spc1(img)
            meas_2 = spc2(img)

            pred = model(meas_1)

            loss_train = criterion(pred, meas_2)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            train_loss.update(loss_train.item())
            train_ssim.update(SSIM(pred, meas_2).item())
            train_psnr.update(PSNR(pred, meas_2).item())

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

        data_loop_val = tqdm(enumerate(valoader), total=len(valoader), colour="green")
        with torch.no_grad():
            model.eval()
            for _, val_data in data_loop_val:
                img, _ = val_data
                img = img.to(device)

                meas_1 = spc1(img)
                meas_2 = spc2(img)

                pred = model(meas_1)

                loss_val = criterion(pred, meas_2)

                val_loss.update(loss_val.item())
                val_ssim.update(SSIM(pred, meas_2).item())
                val_psnr.update(PSNR(pred, meas_2).item())

                data_loop_val.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
                data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

        logging.info(
            f"Epoch: {epoch} - Val Loss: {val_loss.avg} - Val SSIM: {val_ssim.avg} - Val PSNR: {val_psnr.avg}"
        )

        if val_psnr.avg > current_psnr:
            current_psnr = val_psnr.avg
            print(f"Saving model with PSNR: {current_psnr}")
            torch.save(model.state_dict(), f"{model_path}/model.pth")

        recs_array, psnr_imgs, ssim_imgs = save_reconstructed_images(
            imgs=meas_2,
            recons=pred,
            num_img=3,
            pad=2,
            path=images_path,
            name=f"reconstructed_images_{epoch}",
            PSNR=PSNR,
            SSIM=SSIM,
        )

        recs_images = wandb.Image(
            recs_array, caption=f"Epoch: {epoch}\nReal\nRec\nPSNRs: {psnr_imgs}\nSSIMs: {ssim_imgs}"
        )

        wandb.log(
            {
                "train_loss": train_loss.avg,
                "train_ssim": train_ssim.avg,
                "train_psnr": train_psnr.avg,
                "val_loss": val_loss.avg,
                "val_ssim": val_ssim.avg,
                "val_psnr": val_psnr.avg,
                "reconstructed_images": recs_images if epoch % 10 == 0 else None,
            }
        )

        loss_train_record[epoch] = train_loss.avg
        ssim_train_record[epoch] = train_ssim.avg
        psnr_train_record[epoch] = train_psnr.avg
        loss_val_record[epoch] = val_loss.avg
        ssim_val_record[epoch] = val_ssim.avg
        psnr_val_record[epoch] = val_psnr.avg

    # TEST BEST VAL MODEL

    test_loss = AverageMeter()
    test_ssim = AverageMeter()
    test_psnr = AverageMeter()

    del model

    model = UNet(
        n_channels=1,
        base_channel=args.base_channels,
    ).to(device)

    model.load_state_dict(torch.load(f"{model_path}/model.pth"))

    data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour="magenta")
    with torch.no_grad():
        model.eval()
        for _, test_data in data_loop_test:
            img, _ = test_data
            img = img.to(device)

            meas_1 = spc1(img)
            meas_2 = spc2(img)

            pred = model(meas_1)

            loss_test = criterion(pred, meas_2)

            test_loss.update(loss_test.item())
            test_ssim.update(SSIM(pred, meas_2).item())
            test_psnr.update(PSNR(pred, meas_2).item())

            data_loop_test.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
            data_loop_test.set_postfix(loss=test_loss.avg, ssim=test_ssim.avg, psnr=test_psnr.avg)

        logging.info(
            f"Test Loss: {test_loss.avg} - Test SSIM: {test_ssim.avg} - Test PSNR: {test_psnr.avg}"
        )

    # Save data
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
        ),
        f"{metrics_path}/metrics",
    )

    wandb.log({"test_loss": test_loss.avg, "test_ssim": test_ssim.avg, "test_psnr": test_psnr.avg})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2**5)
    parser.add_argument("--snapshots_1", type=int, default=int(0.4 * 32 * 32))
    parser.add_argument("--snapshots_2", type=int, default=int(0.4 * 32 * 32))
    parser.add_argument("--save_path", type=str, default="WEIGHTS/")
    parser.add_argument("--dataset", type=str, default="FMNIST")
    parser.add_argument("--project_name", type=str, default="S_t_y")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=64)
    args = parser.parse_args()

    args_dict = vars(args)

    print_dict(args_dict)
    main(args)