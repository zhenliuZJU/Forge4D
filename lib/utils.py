import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov


def get_novel_calib(data, opt, ratio=0.5, intr_key='intr', extr_key='extr'):
    bs = data['lmain'][intr_key].shape[0]
    fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
    for i in range(bs):
        intr0 = data['lmain'][intr_key][i, ...].cpu().numpy()
        intr1 = data['rmain'][intr_key][i, ...].cpu().numpy()
        extr0 = data['lmain'][extr_key][i, ...].cpu().numpy()
        extr1 = data['rmain'][extr_key][i, ...].cpu().numpy()

        rot0 = extr0[:3, :3]
        rot1 = extr1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot0, rot1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        npose = np.diag([1.0, 1.0, 1.0, 1.0])
        npose = npose.astype(np.float32)
        npose[:3, :3] = rot.as_matrix()
        npose[:3, 3] = ((1.0 - ratio) * extr0 + ratio * extr1)[:3, 3]
        extr_new = npose[:3, :]
        intr_new = ((1.0 - ratio) * intr0 + ratio * intr1)

        if opt.use_hr_img:
            intr_new[:2] *= 2
        width, height = data['novel_view']['width'][i], data['novel_view']['height'][i]
        R = np.array(extr_new[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr_new[:3, 3], np.float32)

        FovX = focal2fov(intr_new[0, 0], width)
        FovY = focal2fov(intr_new[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=opt.znear, zfar=opt.zfar, K=intr_new, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(opt.trans), opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        fovx_list.append(FovX)
        fovy_list.append(FovY)
        world_view_transform_list.append(world_view_transform.unsqueeze(0))
        full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
        camera_center_list.append(camera_center.unsqueeze(0))

    data['novel_view']['FovX'] = torch.FloatTensor(np.array(fovx_list)).cuda()
    data['novel_view']['FovY'] = torch.FloatTensor(np.array(fovy_list)).cuda()
    data['novel_view']['world_view_transform'] = torch.concat(world_view_transform_list).cuda()
    data['novel_view']['full_proj_transform'] = torch.concat(full_proj_transform_list).cuda()
    data['novel_view']['camera_center'] = torch.concat(camera_center_list).cuda()
    return data


def depth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


def absolutedepth2pc(depth, extrinsic, intrinsic):
    B, C, S, S = depth.shape
    depth = depth[:, 0, :, :]
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device), torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
    pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)

    return pts.permute(0, 2, 1)


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics at unit directions using hardcoded SH polynomials.

    Args:
        deg: int SH deg. Currently 0-3 supported.
        sh: SH coeffs [..., C, (deg + 1) ** 2]
        dirs: unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * x * sh[..., 1] +
                  C1 * y * sh[..., 2] -
                  C1 * z * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xz * sh[..., 4] +
                      C2[1] * xy * sh[..., 5] +
                      C2[2] * (2.0 * yy - zz - xx) * sh[..., 6] +
                      C2[3] * yz * sh[..., 7] +
                      C2[4] * (zz - xx) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * x * (3 * zz - xx) * sh[..., 9] +
                          C3[1] * xz * y * sh[..., 10] +
                          C3[2] * x * (4 * yy - zz - xx) * sh[..., 11] +
                          C3[3] * y * (2 * yy - 3 * zz - 3 * xx) * sh[..., 12] +
                          C3[4] * z * (4 * yy - zz - xx) * sh[..., 13] +
                          C3[5] * z * (zz - xx) * sh[..., 14] +
                          C3[6] * z * (zz - 3 * xx) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xz * (zz - xx) * sh[..., 16] +
                              C4[1] * xy * (3 * zz - xx) * sh[..., 17] +
                              C4[2] * xz * (7 * yy - 1) * sh[..., 18] +
                              C4[3] * xy * (7 * yy - 3) * sh[..., 19] +
                              C4[4] * (yy * (35 * yy - 30) + 3) * sh[..., 20] +
                              C4[5] * yz * (7 * yy - 3) * sh[..., 21] +
                              C4[6] * (zz - xx) * (7 * yy - 1) * sh[..., 22] +
                              C4[7] * yz * (zz - 3 * xx) * sh[..., 23] +
                              C4[8] * (zz * (zz - 3 * xx) - xx * (3 * zz - xx)) * sh[..., 24])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def compute_norm(X, mask=None):
    b, n, _, s, _ = X.shape

    pixel_norms = torch.norm(X, p=2, dim=2, keepdim=True)

    if mask is None:
        mask = torch.ones_like(pixel_norms)

    masked_norms = pixel_norms * mask

    sum_per_image = torch.sum(masked_norms, dim=(3, 4))
    total_sum = torch.sum(sum_per_image, dim=1)

    if mask is None:
        norm_factor = n * s * s
    else:
        norm_factor = torch.sum(mask)

    norm_result = total_sum / norm_factor

    return norm_result.view(-1, 1, 1, 1, 1)


def batch_geometric_median(points, max_iter=1000, tol=1e-6, eps=1e-8):
    """Compute geometric median using the Weiszfeld algorithm (batched).

    Args:
        points: Tensor of shape (batch_size, n_samples).
        max_iter: Maximum number of iterations.
        tol: Convergence threshold.
        eps: Small constant to avoid division by zero.

    Returns:
        Geometric median, shape (batch_size, 1).
    """
    device = points.device
    batch_size, n_samples = points.shape

    y = torch.mean(points, dim=1, keepdim=True)

    for _ in range(max_iter):
        diff = points - y
        distances = torch.abs(diff)
        distances = torch.clamp(distances, min=eps)

        weights = 1.0 / distances
        y_new = torch.sum(points * weights, dim=1, keepdim=True) / torch.sum(weights, dim=1, keepdim=True)

        update_norm = torch.abs(y_new - y)
        if torch.all(update_norm < tol):
            break
        y = y_new

    return y.squeeze(dim=-1)


def get_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, warmup_steps=5000):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        if warmup_steps > 0 and step < warmup_steps:
            lr = np.interp([step], [0, warmup_steps], [0, lr_init])[0]
            return lr

        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        linear_lerp = lr_init * (1 - t) + lr_final * t

        return delay_rate * linear_lerp

    return helper


def save_3dgs_ply(filename, positions, colors, scales, rotations, opacities):
    """Save 3D gaussians to PLY file (compatible with 3DGS viewers like SIBR).

    Args:
        positions: [N, 3] float32
        colors: [N, 3] float32 (RGB 0-1)
        scales: [N, 3] float32
        rotations: [N, 4] float32 (unit quaternions)
        opacities: [N] float32 (0-1)
    """
    header = """ply
    format binary_little_endian 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3
    end_header
    """.format(len(positions))

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        data = np.zeros(len(positions), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ])
        data['x'], data['y'], data['z'] = positions.T
        data['f_dc_0'], data['f_dc_1'], data['f_dc_2'] = colors.T
        data['opacity'] = opacities
        data['scale_0'], data['scale_1'], data['scale_2'] = scales.T
        data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3'] = rotations.T
        data.tofile(f)


def gradient(depth):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3).repeat(1, depth.shape[-3], 1, 1).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=depth.device).view(1, 1, 3, 3).repeat(1, depth.shape[-3], 1, 1).cuda()

    grad_x = F.conv2d(depth, sobel_x, padding=1, stride=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1, stride=1)

    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude


def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{prefix} GPU Mem: Allocated={allocated:.2f} MB, Reserved={reserved:.2f} MB")


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=2)
    w2, x2, y2, z2 = q2.unbind(dim=2)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=2)
