import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ssim
import torch.nn.functional as f

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

def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = valid_dataloader(args.data_dir, args.data, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(dataset):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred = crop(pred, (176, 36))
            label_img = crop(label_img, (176, 36))
            
            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)

            psnr_adder(psnr_val)
            ssim_adder(ssim_val)
            print('\r%03d'%idx, end=' ')
            
            if idx < 250 and idx % 50 == 0:
                if not os.path.exists(os.path.join('results', 'valid', '%d' % (ep))):
                    os.makedirs(os.path.join('results', 'valid', '%d' % (ep)))
                save_name = os.path.join('results', 'valid', '%d' % (ep), name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

    print('\n')
    model.train()
    return psnr_adder.average(), ssim_adder.average()
