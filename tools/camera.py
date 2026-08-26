import torch
import torch.nn.functional as F
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def generate_trajectory(P1: torch.Tensor, P2: torch.Tensor, num_frames: int=150) -> torch.Tensor:
    """
    生成从 P1 到 P2 的平滑相机轨迹
    Args:
        P1, P2: (4, 4) 或 (B, 4, 4) 的外参矩阵
        num_frames: 插值帧数
    Returns:
        trajectory: (num_frames, 4, 4) 的轨迹张量
    """
    alphas = torch.linspace(0, 1, num_frames, device=P1.device)
    return [interpolate_poses_torch(P1, P2, a.item()) for a in alphas]

def interpolate_poses_torch(P1: torch.Tensor, P2: torch.Tensor, alpha: float) -> torch.Tensor:
    P1.shape == P2.shape, "Input shapes must match!"
    # assert P1.shape[-2:] == (4, 4), "Input must be 4x4 matrices"
        
    # 提取旋转和平移
    R1, t1 = P1[..., :3, :3], P1[..., :3, 3]
    R2, t2 = P2[..., :3, :3], P2[..., :3, 3]
        
    # 旋转矩阵 → 四元数 (使用 scipy 的 Slerp)
    q1 = R.from_matrix(R1.cpu().numpy()).as_quat()  # (..., 4)
    q2 = R.from_matrix(R2.cpu().numpy()).as_quat()

    q1 = torch.from_numpy(q1).to(R1.device)  # (B, 4)
    q2 = torch.from_numpy(q2).to(R1.device)  # (B, 4)
        
    # Slerp 插值
    # print(q1.shape, q2.shape)
    # slerp = Slerp([0, 1], R.from_quat([q1, q2]))
    # R_interp = torch.tensor(
    #     slerp(alpha).as_matrix(),
    #     device=P1.device, dtype=P1.dtype
        # )  # (..., 3, 3)

    R_interp = batch_slerp(q1, q2, alpha)  # (..., 3, 3)
        
    # Lerp 插值平移
    t_interp = (1 - alpha) * t1 + alpha * t2  # (..., 3)
        
    # 构建插值后的外参矩阵
    P_interp = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], device=P1.device, dtype=P1.dtype).expand_as(P1).clone()
    P_interp[..., :3, :3] = R_interp
    P_interp[..., :3, 3] = t_interp
        
    return P_interp

def batch_slerp(q1: torch.Tensor, q2: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    批量四元数球面线性插值 (Slerp)
    Args:
        q1: (B, 4), 起始四元数 (需已归一化)
        q2: (B, 4), 终止四元数 (需已归一化)
        alpha: 插值系数 [0, 1]
    Returns:
        (B, 4), 插值后的四元数
    """
    # 确保输入是torch.Tensor且归一化
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)
        
    # 计算点积 (夹角余弦)
    dot = (q1 * q2).sum(dim=-1, keepdim=True)  # (B, 1)
        
    # 处理反向四元数 (保证走最短路径)
    mask = dot < 0
    q2 = torch.where(mask, -q2, q2)
    dot = torch.where(mask, -dot, dot)
        
    # 限制dot范围避免数值误差
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)  # (B, 1)
        
    # 插值计算
    sin_theta = torch.sin(theta)
    w1 = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(alpha * theta) / (sin_theta + 1e-8)
    q_new = w1 * q1 + w2 * q2  # (B, 4)

    rot_mat_np = R.from_quat(q_new.cpu().numpy()).as_matrix()  # (B, 3, 3)
    rot_mat = torch.from_numpy(rot_mat_np).to(q1.device)

    return rot_mat