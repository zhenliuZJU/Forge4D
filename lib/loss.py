import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from einops import rearrange
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path
import lpips
from .flow import unocclusion

def sequence_depth_loss(depth_pred, depth_gt, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    valid = (depth_gt != 0.0)
    assert not torch.isinf(depth_gt[valid.bool()]).any()
    bound = (torch.max(depth_gt)-torch.min(depth_gt))

    i_loss = (depth_pred[0] - depth_gt).abs()
    depth_loss =  i_loss[valid.bool()].sum()

    epe = torch.sum((depth_pred[-1] - depth_gt)**2, dim=2).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1pc': (epe < 0.01*bound).float().mean().item(),
        'train_3pc': (epe < 0.03*bound).float().mean().item()
    }

    return depth_loss, metrics

def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return ((torch.abs(network_output - gt))*mask).mean()
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt, mask=None):
    if mask is not None:
        return (((network_output - gt)**2)*mask).mean()
    return ((network_output - gt)**2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).contiguous().view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def discriminator_loss(real_output, fake_output):
    real_loss = -torch.mean(torch.log(real_output + 1e-12))
    fake_loss = -torch.mean(torch.log(1 - fake_output + 1e-12))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    g_loss = -torch.mean(torch.log(fake_output + 1e-12))
    return g_loss

def opacity_regular(data):
    loss = 0
    for view in ['lmain', 'rmain']:
        opacity = data[view]['opacity_maps'].permute(0, 2, 3, 1).view(-1, 1)
        loss = loss + (opacity*torch.log(opacity)).mean()
    return loss

def scale_regular(data):
    loss = 0
    for view in ['lmain', 'rmain']:
        scale = data[view]['scale_maps'].permute(0, 2, 3, 1).reshape(-1, 3)
        loss = loss + scale.mean()
    return loss

def anchor_loss(data):
    t1 = 0.1
    t2 = 0.05
    loss = torch.tensor(0.)
    bs =  data['lmain']['img'].shape[0]
    width = data['lmain']['img'].shape[3]
    for i in range(bs):
        if (data['lmain']['face'][i].sum() != 0) and (data['rmain']['face'][i].sum() != 0):
            for (anchor_left, anchor_right) in torch.stack((data['lmain']['face'][i], data['rmain']['face'][i]), dim=1):
                dist = data['lmain']['xyz'][i, int(anchor_left[0])+width*int(anchor_left[1]), :] - data['rmain']['xyz'][i, int(anchor_right[0])+width*int(anchor_right[1]), :]
                loss = loss + (torch.abs(torch.sum(dist**2).sqrt()-t2)+torch.abs(torch.sum(dist**2).sqrt()+t2))/2
        
    return loss


class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()
        
    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        
        return model.to(self.device).eval()
    
    def _load_weights(self):
        """Load pre-trained VGG weights. """
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)
        
        if not weight_file.exists():
            os.system(f'wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}')
        
        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]
        
        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        
        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)
                
                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(filter_sizes[i])
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)
    
    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()
        
        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(*list(self.vgg.features[output_indices[i]:output_indices[i+1]]))
            self.blocks.append(block.to(self.device).eval())
        
        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
    
    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean
    
    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))
    
    def forward(self, pred_img, target_img):
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)
        
        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)
        
        # Pixel-level error
        e0 = self._compute_error(target_img_p, pred_img_p)
        
        # Feature-level errors with scaling factors
        e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
        e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
        e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
        e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
        e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5
        
        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
        
        return total_loss


def flow_loss(pred_flow, target_flow, ref_flow, mask):
    B, V, C, H, W = pred_flow.shape
    occ_mask, _, uncertainty_fw, _ = unocclusion(target_flow.reshape(-1, C, H, W), ref_flow.reshape(-1, C, H, W))
    occ_mask = torch.ones_like(occ_mask)
    mask = mask.reshape(-1, 1, H, W)
    return F.l1_loss(pred_flow.reshape(-1, C, H, W) * occ_mask * uncertainty_fw * mask, target_flow.reshape(-1, C, H, W) * uncertainty_fw * occ_mask *mask), uncertainty_fw, occ_mask*mask
