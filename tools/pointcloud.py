import open3d as o3d
import numpy as np

def poisson_reconstruction(points, normals=None, depth=8):
    """
    使用泊松重建从点云生成网格
    points: (N, 3) 点云坐标
    normals: (N, 3) 法线向量（可选）
    depth: 重建深度（8-12，越大越精细）
    """
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法线（如果没有提供）
    if normals is None:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)
    else:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 泊松重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    
    # 移除低密度区域（可选）
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

def ball_pivoting_reconstruction(points, normals=None):
    """
    使用Ball-Pivoting算法重建网格
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if normals is None:
        pcd.estimate_normals()
    
    # 定义球体半径（需要根据点云密度调整）
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    
    return mesh

def alpha_shapes_reconstruction(points, alpha=0.1):
    """
    使用Alpha Shapes算法重建网格（适合简单形状）
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Alpha Shapes重建
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    
    return mesh

def pointcloud_to_mesh_workflow(points, method='poisson'):
    """
    完整的点云到网格转换工作流
    """
    # 1. 数据预处理
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. 去噪
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 3. 下采样（可选）
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    
    # 4. 法线估计
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)
    
    # 5. 选择重建方法
    if method == 'poisson':
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    elif method == 'ball_pivoting':
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    elif method == 'alpha_shapes':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
    
    # 6. 后处理
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh

if __name__ == '__main__':
    # 使用
    points = np.loadtxt('pointcloud.xyz')  # 加载点云
    mesh = pointcloud_to_mesh_workflow(points, method='poisson')

    # 保存网格
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)