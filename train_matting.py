import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from depthfm import UNetModel
from diffusers import AutoencoderKL
import torch.optim as optim
from depthfm.utils import *
from depthfm.dfm import *
from time import time
import argparse
from dataset.p3m10k import P3M10K
from torch.cuda.amp import autocast, GradScaler



def parser_args():
    parser = argparse.ArgumentParser("ddgan parameters")
    
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default='checkpoints/depthfm-v1.ckpt', help="Model ckpt to init from")
    parser.add_argument("--lr", default=3.0e-05)
    parser.add_argument("--num_epoch", default=50)
    parser.add_argument("--exp", default="exp/matting")
    parser.add_argument("--scale_factor", default=0.18215)
    parser.add_argument("--noising_step", default=400)
    parser.add_argument("--save_ckpt_every", default=1000)
    parser.add_argument("--image_size", default=512)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--vae_id", default="runwayml-df1.5/vae")
    parser.add_argument("--train_pkl", default="/data1/chenjun/MC/bit_cv/owncode/data_list/p3m_person.pkl")

    args = parser.parse_args()
    return args


def train(args):

    ## 模型
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32
    
    first_stage_model = AutoencoderKL.from_pretrained(args.vae_id, subfolder="vae").to(device)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    model = UNetModel(**ckpt['ldm_hparams'])
    model.load_state_dict(ckpt['state_dict'])
    empty_text_embed = ckpt['empty_text_embedding']

    accelerator.print("AutoKL size: {:.3f}MB".format(get_weight(first_stage_model)))
    accelerator.print("FM size: {:.3f}MB".format(get_weight(model)))


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    ## 数据
    dataset = P3M10K(args.train_pkl, 'train', size=args.image_size)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    data_loader, model, optimizer, scheduler = accelerator.prepare(data_loader, model, optimizer, scheduler)

    exp = args.exp
    os.makedirs(exp, exist_ok=True)

    if args.resume:
        init_epoch = 10
    else:
        init_epoch = 0
    
    log_steps = 0
    start_time = time()
    global_step = 0
    for epoch in range(init_epoch, args.num_epoch + 1):
        scheduler.step()

        for iteration, sample in enumerate(data_loader):
            img = sample['rgb_norm']
            alpha = sample['alpha']
            x_0 = img.to(device, dtype=dtype, non_blocking=True)
            y = alpha.to(device, non_blocking=True).repeat((1, 3, 1, 1))
            bs = x_0.shape[0]
            model.zero_grad()
            with torch.no_grad():
                z_0 = first_stage_model.encode(x_0).latent_dist.mode().mul_(args.scale_factor)
                alpha_l = first_stage_model.encode(y).latent_dist.mode().mul_(args.scale_factor)
            context = z_0
            text_embed = torch.tensor(empty_text_embed).to(device).repeat(bs, 1, 1)
            # sample t
            z_t, t = linear_sample(z_0, level=0.4)
            
            # estimate velocity
            u = alpha_l - z_t
            v = model(z_t, t.squeeze(), context=context, context_ca=text_embed)
            loss = F.l1_loss(v, u)
            accelerator.backward(loss)
            optimizer.step()
            log_steps += 1
            global_step += 1
            
            if accelerator.is_main_process:
                if global_step % 5 == 0:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    accelerator.print(
                        "epoch {} iteration {}, Loss: {:.4f}, Train Steps/Sec: {:.2f}".format(
                            epoch, global_step, loss.item(), steps_per_sec
                        )
                    )
                    # Reset monitoring variables:
                    log_steps = 0
                    start_time = time()

                    if global_step % args.save_ckpt_every == 0 and global_step > 0:
                        accelerator.print("Saving content.")
                        content = {
                            "iteration": global_step,
                            "noising_step": 400,
                            "empty_text_embedding":ckpt['empty_text_embedding'],
                            'ldm_hparams': ckpt['ldm_hparams'],
                            "state_dict": model.state_dict(),
                        }
                        name = 'ckpt_last.pth'
                        torch.save(content, os.path.join(exp, name))



if __name__ == '__main__':
    args = parser_args()
    train(args)