import importlib
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from NAFNet_arch import NAFNetLocal, NAFNet
from model_init import Model
from utils import imwrite, tensor2img
from quantizedNAFNet import QuantizedNAFNet, prepare_model_qat,convert_to_quantized
import torch.optim as optim
from ptflops import get_model_complexity_info
import numpy as np
from skimage.metrics import structural_similarity

class ModelAnalyzer:
    def is_quantized_model(network):
        has_quant = False
        has_float = False
        
        for module in network.modules():
            if isinstance(module, (torch.quantization.QuantStub, torch.quantization.DeQuantStub)):
                has_quant = True
            if isinstance(module, torch.nn.quantized.FloatFunctional):
                has_float = True
            
                
        return has_quant or has_float

    def count_operations(macs, is_quantized):
        if is_quantized:
            return macs * 2 * 8
        else:
            return macs * 2 * 32

    def count_flops(macs, is_quantized):
        if is_quantized:
            return macs/2
        else:
            return macs * 2

    def calculate_size(network, is_quantized):
        total = sum(p.numel() for p in network.parameters())
        return total * (1 if is_quantized else 4)

    def analyze(network, input_shape=(1, 3, 256, 256)):
        is_quantized = ModelAnalyzer.is_quantized_model(network)
        macs, params = get_model_complexity_info(
            network, input_shape[1:], 
            as_strings=False,
            print_per_layer_stat=False
        )
        
        bit_ops = ModelAnalyzer.count_operations(macs, is_quantized)
        size = ModelAnalyzer.calculate_size(network, is_quantized)
        flops = ModelAnalyzer.count_flops(macs, is_quantized)
        
        return {
            'type': 'Quantized (INT8)' if is_quantized else 'Full Precision (FP32)',
            'params': params,
            'size_mb': size / (1024 * 1024),
            'macs': macs,
            'bit_ops': bit_ops,
            'flops': flops,
        }

def print_analysis(network, input_shape=(1, 3, 256, 256)):
    results = ModelAnalyzer.analyze(network, input_shape)
    
    print("\nModel Analysis Results")
    print("-" * 50)
    print(f"Type: {results['type']}")
    print(f"Parameters: {results['params']:,}")
    print(f"Size: {results['size_mb']:.2f} MB")
    print(f"MACs: {results['macs']:,}")
    print(f"Ideal FLOPs: {results['flops']:,}")
    print(f"Bit Operations: {results['bit_ops']:,}")
    
    if "Quantized" in results['type']:
        print("\nQuantization Metrics:")
        print(f"Bits per parameter: 8")
        print(f"Bits per operation: 8")
    else:
        print("\nFull Precision Metrics:")
        print(f"Bits per parameter: 32")
        print(f"Bits per operation: 32")

class PSNRLoss(nn.Module):
    def __init__(self, weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.weight = weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            
        assert len(pred.size()) == 4
        return self.weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Invalid input_order {input_order}. Must be HWC or CHW')
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def calculate_psnr(img1, img2, input_order='HWC'):
    assert img1.shape == img2.shape, f'Image shapes differ: {img1.shape}, {img2.shape}.'
    
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Invalid input_order {input_order}. Must be HWC or CHW')
        
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order)
    img2 = reorder_image(img2, input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    def compute_psnr(img1, img2):
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        max_val = 1. if img1.max() <= 1 else 255.
        return 20. * np.log10(max_val / np.sqrt(mse))
    
    if img1.ndim == 3 and img1.shape[2] == 6:
        left1, right1 = img1[:,:,:3], img1[:,:,3:]
        left2, right2 = img2[:,:,:3], img2[:,:,3:]
        return (compute_psnr(left1, left2) + compute_psnr(right1, right2))/2
    else:
        return compute_psnr(img1, img2)

class DeblurringModel(Model):
    def __init__(self, training, load_path="intw32_quantized_model.pth"):
        super(DeblurringModel, self).__init__(training)
        channels = 3
        self.chkpt=load_path.split('.')[0]
        if load_path is not None and "intw32" in load_path:
            width = 30
            enc_blocks = [1, 1, 1, 28]
            mid_blocks = 1
            dec_blocks = [1, 1, 1, 1]
            
            self.network = QuantizedNAFNet(
                img_channel=channels,
                width=width,
                middle_blk_num=mid_blocks,
                enc_blk_nums=enc_blocks,
                dec_blk_nums=dec_blocks
            )
            self.load(self.network, load_path, True)
            
        elif load_path is not None and "fpw64" in load_path:
            width = 64
            enc_blocks = [1, 1, 1, 28]
            mid_blocks = 1
            dec_blocks = [1, 1, 1, 1]
            
            self.network = NAFNet(
                img_channel=channels,
                width=width,
                middle_blk_num=mid_blocks,
                enc_blk_nums=enc_blocks,
                dec_blk_nums=dec_blocks
            )
            self.load(self.network, load_path, False)
            self.network.load_state_dict(torch.load(load_path), strict=False)

        elif load_path is not None and "fpw32" in load_path:
            width = 32
            enc_blocks = [1, 1, 1, 28]
            mid_blocks = 1
            dec_blocks = [1, 1, 1, 1]
            
            self.network = NAFNet(
                img_channel=channels,
                width=width,
                middle_blk_num=mid_blocks,
                enc_blk_nums=enc_blocks,
                dec_blk_nums=dec_blocks
            )
            self.load(self.network, load_path, False)
            self.network.load_state_dict(torch.load(load_path), strict=False)

        print_analysis(self.network)
        self.network = self.network.to(self.device)

        if self.training=="train":
            self.setup_training()
            self.network=prepare_model_qat(self.network)
            self.network=convert_to_quantized(self.network)

        self.scale = 1

    def get_learning_rate(self):
        return [group['lr'] for group in self.optimizer_g.param_groups]

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        if 0 < current_iter < warmup_iter:
            for group in self.optimizer_g.param_groups:
                initial_lr = group.get('initial_lr', group['lr'])
                group['lr'] = initial_lr * (current_iter / warmup_iter)
        else:
            self.schedulers.step()

    def setup_training(self):
        self.network.train()
        self.loss = PSNRLoss().to(self.device)
        self.perceptual_loss = None
        self.optimizer_g = torch.optim.AdamW(
            params=self.network.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.9),
            eps=1e-8,
            weight_decay=0.001
        )
        self.schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, 
            T_max=400000, 
            eta_min=1e-7
        )

    def setup_optimizers(self):
        self.optimizer_g = torch.optim.AdamW(
            params=self.network.parameters(), 
            lr=0.001, 
            betas=(0.9, 0.9),
            eps=1e-8,
            weight_decay=0.001
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.input = data['lq'].to(self.device)
        if 'gt' in data:
            self.target = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        predictions = self.network(self.input)
        if not isinstance(predictions, list):
            predictions = [predictions]

        self.output = predictions[-1]

        total_loss = 0
        losses = OrderedDict()
        
        if self.loss:
            pixel_loss = 0.
            for pred in predictions:
                pixel_loss += self.loss(pred, self.target)

            total_loss += pixel_loss
            losses['pixel_loss'] = pixel_loss

        total_loss = total_loss + 0. * sum(p.sum() for p in self.network.parameters())

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.calculate_losses(losses)

    def test(self):
        self.network.eval()
        with torch.no_grad():
            n = len(self.input)
            outputs = []
            batch = n
            i = 0
            while i < n:
                j = min(i + batch, n)
                pred = self.network(self.input[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outputs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outputs, dim=0)
        self.network.train()

    def validate(self, dataloader, current_iter, rgb2bgr=True):
        self.metrics = {'psnr': 0}
        pbar = tqdm(total=len(dataloader), unit='image')
        count = 0

        for idx, data in enumerate(dataloader):
            name = osp.splitext(osp.basename(data['lq_path'][0]))[0]

            self.feed_data(data, is_val=True)
            self.test()

            visuals = self.get_current_visuals()
            output_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                target_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.target

            del self.input
            del self.output
            torch.cuda.empty_cache()

            self.metrics['psnr'] += calculate_psnr(output_img, target_img)
            count += 1
            pbar.update(1)
            pbar.set_description(f'Test {count}')
        pbar.close()

        results = OrderedDict()
        results['psnr'] = self.metrics['psnr']
        results['count'] = count

        self.results = results

        final_metrics = {}
        for key, metric in self.results.items():
            if key == 'count':
                count = metric
                continue
            final_metrics[key] = metric / count
        print("PSNR:", final_metrics['psnr'])

        return 0.

    def get_current_visuals(self):
        results = OrderedDict()
        results['lq'] = self.input.detach().cpu()
        results['result'] = self.output.detach().cpu()
        if hasattr(self, 'target'):
            results['gt'] = self.target.detach().cpu()
        return results

    def save(self, epoch, current_iter):
        self.save_model(self.network, self.chkpt, current_iter)
        self.save_checkpoint(epoch, current_iter)