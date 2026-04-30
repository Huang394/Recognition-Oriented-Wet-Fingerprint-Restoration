import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torch.nn as nn
from pytorch_msssim import MS_SSIM

from warmup_scheduler import GradualWarmupScheduler

def crop(img, size, scale_factor=1.0):
    width, height = size
    width = int(width * scale_factor)
    height = int(height * scale_factor)
    _, _, w, h = img.size()
    left = (w - width) // 2
    top = (h - height) // 2
    right = (w + width) // 2
    bottom = (h + height) // 2
    return img[:, :, left:right, top:bottom]

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    ms_ssim_loss = MS_SSIM()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, args.data)
    max_iter = len(dataloader)
    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 480
    if args.resume:
        state = torch.load(args.resume)
        if 'epoch' in state.keys():
            epoch = state['epoch']
            optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        if 'epoch' in state.keys():
            print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_ms_ssim_adder = Adder()
    iter_pixel_adder = Adder()
    iter_ms_ssim_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1
    best_ssim=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            loss_ms_ssim = 1 - ms_ssim_loss(pred_img[2], label_img)

            loss = 0.15 * loss_content + 0.85 * loss_ms_ssim
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_ms_ssim_adder(loss_ms_ssim.item())

            epoch_pixel_adder(loss_content.item())
            epoch_ms_ssim_adder(loss_ms_ssim.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss ms_ssim: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                    iter_ms_ssim_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_ms_ssim_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_ms_ssim_adder.reset()     
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)
            
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch MS-SSIM Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_ms_ssim_adder.average()))
        epoch_ms_ssim_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            psnr_val, ssim_val = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB SSIM %.4f dB' % (epoch_idx, psnr_val, ssim_val))
            writer.add_scalar('PSNR', psnr_val, epoch_idx)
            writer.add_scalar('SSIM', ssim_val, epoch_idx)
            if psnr_val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best_PSNR.pkl'))
                best_psnr = psnr_val
            if ssim_val >= best_ssim:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best_SSIM.pkl'))
                best_ssim = ssim_val
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
