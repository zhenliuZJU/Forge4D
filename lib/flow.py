import torch
import torch.nn.functional as F
import numpy as np
import math

def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.

    Args:
        flow: tensor of shape [num_batch, 2, height, width]
        mask: flow validity mask of shape [num_batch, 1, height, width]
    Returns:
        color_flow: color representation of flow [num_batch, 3, height, width]
    """
    n = 8
    num_batch, _, height, width = flow.size()
    
    # Handle mask
    if mask is None:
        mask = torch.ones((num_batch, 1, height, width), device=flow.device)
    
    # Move flow components to last dimension for calculations
    flow = flow.permute(0, 2, 3, 1)  # [B,H,W,2]
    mask = mask.permute(0, 2, 3, 1)  # [B,H,W,1]
    
    # Split flow into u and v components
    flow_u, flow_v = torch.unbind(flow, dim=3)
    
    # Compute max flow
    if max_flow is not None:
        max_flow = torch.maximum(max_flow, torch.tensor(1, device=flow.device))
    else:
        max_flow = torch.max(torch.abs(flow * mask))
    
    # Compute magnitude and angle
    mag = torch.sqrt(torch.sum(torch.square(flow), dim=3))
    angle = torch.atan2(flow_v, flow_u)
    
    # Convert to HSV color space
    im_h = torch.fmod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = torch.clamp(mag * n / max_flow, 0, 1)
    im_v = torch.clamp(n - im_s, 0, 1)
    
    # Stack HSV channels
    h, s, v = im_h, im_s, im_v  # Each is [B,H,W]
    
    # Convert HSV to RGB
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    i = i.to(dtype=torch.int32) % 6
    
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    
    # Initialize output tensor
    rgb = torch.zeros((num_batch, height, width, 3), device=flow.device)
    
    # Assign values based on hue sector
    # Sector 0: red > yellow
    mask0 = (i == 0).unsqueeze(-1)
    rgb0 = torch.stack([v, t, p], dim=-1)
    rgb = torch.where(mask0, rgb0, rgb)
    
    # Sector 1: yellow > green
    mask1 = (i == 1).unsqueeze(-1)
    rgb1 = torch.stack([q, v, p], dim=-1)
    rgb = torch.where(mask1, rgb1, rgb)
    
    # Sector 2: green > cyan
    mask2 = (i == 2).unsqueeze(-1)
    rgb2 = torch.stack([p, v, t], dim=-1)
    rgb = torch.where(mask2, rgb2, rgb)
    
    # Sector 3: cyan > blue
    mask3 = (i == 3).unsqueeze(-1)
    rgb3 = torch.stack([p, q, v], dim=-1)
    rgb = torch.where(mask3, rgb3, rgb)
    
    # Sector 4: blue > magenta
    mask4 = (i == 4).unsqueeze(-1)
    rgb4 = torch.stack([t, p, v], dim=-1)
    rgb = torch.where(mask4, rgb4, rgb)
    
    # Sector 5: magenta > red
    mask5 = (i == 5).unsqueeze(-1)
    rgb5 = torch.stack([v, p, q], dim=-1)
    rgb = torch.where(mask5, rgb5, rgb)
    
    # Apply mask and convert back to [B,C,H,W] format
    rgb = rgb * mask
    rgb = rgb.permute(0, 3, 1, 2)  # [B,3,H,W]
    
    return rgb

def image_warp(im, flow):
    """Performs a backward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, channels, height, width]
        flow: Batch of flow vectors. [num_batch, 2, height, width]
    Returns:
        warped: transformed image of the same shape as the input image.
    """
    # First, we need to convert the flow from [B, 2, H, W] to [B, H, W, 2] for grid_sample
    flow = flow.permute(0, 2, 3, 1)
    
    # Create grid from flow
    B, C, H, W = im.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1)
    yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 3).float()
    
    if im.is_cuda:
        grid = grid.cuda()
    
    vgrid = grid + flow
    
    # Scale grid to [-1,1] for grid_sample
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    
    # Perform the warp
    warped = F.grid_sample(im, vgrid, align_corners=True)
    
    return warped

def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)

def unocclusion(flow_fw, flow_bw):
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    # occ_thresh = 0.01 * mag_sq + 0.5
    occ_thresh = 0.1 * mag_sq + 0.5
    uncertainty_fw = torch.exp(-length_sq(flow_diff_fw/occ_thresh))
    uncertainty_bw = torch.exp(-length_sq(flow_diff_bw/occ_thresh))
    occ_fw = (length_sq(flow_diff_fw) < occ_thresh).float().to(flow_fw.device)
    occ_bw = (length_sq(flow_diff_bw) < occ_thresh).float().to(flow_bw.device)
    return occ_fw, occ_bw, uncertainty_fw, uncertainty_bw