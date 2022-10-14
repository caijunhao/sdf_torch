from sdf import TSDF, PSDF, GradientSDF
import pybullet as p
import numpy as np
import pybullet_data
import torch
import time
torch.set_default_dtype(torch.float32)
torch.manual_seed(777)
np.random.seed(777)


def random_sphere_sampling(r, theta_min=0.0, theta_max=np.pi/3, phi_min=0.0, phi_max=np.pi*2):
    theta = np.random.uniform(theta_min, theta_max)
    phi = np.random.uniform(phi_min, phi_max)
    # print('theta: {} | phi: {}'.format(theta / np.pi * 180, phi / np.pi * 180))
    x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)
    d_theta = np.array([r * np.cos(theta) * np.cos(phi),
                        r * np.cos(theta) * np.sin(phi),
                        -r * np.sin(theta)])  # correspond to y axis
    d_phi = np.array([-r * np.sin(theta) * np.sin(phi),
                      r * np.sin(theta) * np.cos(phi),
                      0.0])  # correspond to x axis
    view_direction = -np.array([x, y, z])  # correspond to z axis
    n_c2w = d_phi / np.linalg.norm(d_phi)
    o_c2w = d_theta / np.linalg.norm(d_theta)
    a_c2w = view_direction / np.linalg.norm(view_direction)
    transformation = np.eye(4)
    transformation[0:3, 0:3] = np.stack([n_c2w, o_c2w, a_c2w], axis=1)
    transformation[0:3, 3] = x, y, z
    return transformation.astype(np.float32)


mode = p.GUI  # p.DIRECT p.GUI
physics = p.connect(mode)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF('plane.urdf')

height, width = 480, 640
near, far = 0.01, 10.0
intrinsic = np.array([[450.0, 0.0, 320],
                      [0.0, 450.0, 240],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
focal_len = intrinsic[0][0]
aspect_ratio = 640 / 480
fov = (height / 2) / focal_len
fov = np.arctan(fov) * 2 / np.pi * 180
project_matrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)

num_fusion = 50
r = 0.5  # meter
half_range = 0.2
volume_bounds = np.array([[-half_range, half_range],
                          [-half_range, half_range],
                          [-half_range, half_range]])
resolution = np.array([200, 200, 200])

half_size = 0.05
box1 = p.createMultiBody(0,
                         p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, half_size, half_size]),
                         p.createVisualShape(p.GEOM_BOX, halfExtents=[half_size, half_size, half_size]),
                         basePosition=[0, 0, half_size])
box2 = p.createMultiBody(0,
                         p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size, half_size, half_size]),
                         p.createVisualShape(p.GEOM_BOX, halfExtents=[half_size, half_size, half_size]),
                         basePosition=[half_size*2, 0, half_size])
sphere = p.createMultiBody(0,
                           p.createCollisionShape(p.GEOM_SPHERE, radius=half_size),
                           p.createVisualShape(p.GEOM_SPHERE, radius=half_size),
                           basePosition=[0, half_size*2, half_size])
p.changeVisualShape(box1, -1, rgbaColor=np.random.uniform(size=3).tolist()+[1])
p.changeVisualShape(box2, -1, rgbaColor=np.random.uniform(size=3).tolist()+[1])
p.changeVisualShape(sphere, -1, rgbaColor=np.random.uniform(size=3).tolist()+[1])
p.stepSimulation()

sdf = GradientSDF(volume_bounds.T[0], resolution, 0.002, fuse_color=True)
total = 0
for i in range(num_fusion):
    t_c2w = random_sphere_sampling(r)
    view_matrix = p.computeViewMatrix(t_c2w[0:3, 3],
                                      t_c2w[0:3, 3]+t_c2w[0:3, 2],
                                      -t_c2w[0:3, 1])
    _, _, color, depth, _ = p.getCameraImage(width=width,
                                             height=height,
                                             viewMatrix=view_matrix,
                                             projectionMatrix=project_matrix,
                                             shadow=0,
                                             renderer=p.ER_TINY_RENDERER)
    p.stepSimulation()
    color = color[..., 0:-1].astype(np.float32)  # remove alpha channel
    depth = far * near / (far - (far - near) * depth)
    depth = depth.astype(np.float32)
    b = time.time()
    sdf.gsdf_integrate(depth, intrinsic, t_c2w, rgb=color)
    e = time.time()
    total += e - b
    print('{} frame(s) processed'.format(i))
    # sdf.sdf_integrate(depth, intrinsic, t_c2w, rgb=color)
print('fps: {}'.format(num_fusion / total))
sdf.write_pcl('tsdf_cube_pcl.ply', *sdf.compute_pcl(threshold=0.2))
sdf.write_mesh('tsdf_cube_mesh.ply', *sdf.marching_cubes())



