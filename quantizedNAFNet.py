import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional
class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)
class LayerNormFunction(torch.autograd.Function):

    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
        self.float_op = FloatFunctional()

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_op = FloatFunctional()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.float_op.mul(x1, x2)

class QuantizedNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(dw_channel)  # Added for stable quantization
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True),
            nn.BatchNorm2d(dw_channel)  # Added for stable quantization
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(c)  # Added for stable quantization
        )
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(dw_channel // 2)  # Added for stable quantization
        )

        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(ffn_channel)  # Added for stable quantization
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(c)  # Added for stable quantization
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        
        self.float_op = FloatFunctional()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        
        # Quantization-friendly multiplication
        x = self.float_op.mul(x, self.sca(x))
        x = self.conv3(x)
        x = self.dropout1(x)
        
        # Quantization-friendly addition and multiplication
        y = self.float_op.add(inp, self.float_op.mul(x, self.beta))

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        # Quantization-friendly addition and multiplication
        return self.float_op.add(y, self.float_op.mul(x, self.gamma))

class QuantizedNAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        
        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self.intro = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(width)  # Added for stable quantization
        )
        
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(img_channel)  # Added for stable quantization
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        self.float_op = FloatFunctional()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[QuantizedNAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chan, 2*chan, 2, 2),
                    nn.BatchNorm2d(2*chan)  # Added for stable quantization
                )
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[QuantizedNAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.BatchNorm2d(chan * 2),  # Added for stable quantization
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[QuantizedNAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        # Quantize input
        x = self.quant(inp)
        
        x = self.intro(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = self.float_op.add(x, enc_skip)
            x = decoder(x)

        x = self.ending(x)
        x = self.float_op.add(x, inp)
        
        # Dequantize output
        x = self.dequant(x)
        
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def fuse_model(self):
        """Fuse conv/bn layers for inference"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                torch.quantization.fuse_modules(m, ['0', '1'], inplace=True)

class QuantizedNAFNetLocal(Local_Base, QuantizedNAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        QuantizedNAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

def prepare_model_qat(model):
    """Prepare the model for quantization-aware training"""
    model.train()
    
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    torch.quantization.prepare_qat(model)
    
    return model
def convert_to_quantized(model):
    """
    Converts a QAT NAFNet model to a quantized model for inference
    """
    model.eval()
    
    # Convert the model to a quantized model
    torch.quantization.convert(model)
    
    return model