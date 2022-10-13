from skimage import measure
import numpy as np
import torch
import time


class TSDF(object):
    def __init__(self, origin, resolution, voxel_length, truncated_length=0.005,
                 fuse_color=False, device='cuda', dtype=torch.float):
        """
        Initialize SDF module
        :param origin: (ndarray) [3, ] the world coordinate of the voxel [0, 0, 0]
        :param resolution: (ndarray) [3, ] resolution over the total length of the volume
        :param voxel_length: (float) voxel size, where length = voxel_length * resolution
        :param truncated_length: (float) the length of the margin
        :param fuse_color: (bool) A flag specifying if fusing color information or not
        :param device: An object representing the device on which a torch.Tensor is or will be allocated.
        :param dtype: Data type for torch tensor.
        """
        self.origin = torch.from_numpy(origin).to(device).to(dtype)
        self.res = torch.from_numpy(resolution).to(torch.long).to(device)
        self.vox_len = torch.tensor(voxel_length, dtype=dtype, device=device)
        self.sdf_trunc = truncated_length
        self.fuse_color = fuse_color
        self.dev = device
        self.dt = dtype

        # Initialize volumes
        self.sdf_vol = torch.ones(*self.res).to(self.dt).to(self.dev)
        self.w_vol = torch.zeros(*self.res).to(self.dt).to(self.dev)
        if self.fuse_color:
            self.rgb_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)

        # get voxel coordinates and world coordinates
        vx, vy, vz = torch.meshgrid(torch.arange(self.res[0]), torch.arange(self.res[1]), torch.arange(self.res[2]),
                                    indexing='ij')
        self.vox_coords = torch.stack([vx, vy, vz], dim=-1).to(device)
        self.world_coords = self.vox_coords * self.vox_len + self.origin.view(1, 1, 1, 3)
        self.world_coords = torch.cat([self.world_coords,
                                       torch.ones(*self.res, 1).to(self.dt).to(self.dev)],
                                      dim=-1)
        self.sdf_info()

    def sdf_integrate(self, depth, intrinsic, camera_pose, weight=1.0, rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: (ndarray.float) [H, W] a depth map whose unit is meter.
        :param intrinsic: (ndarray.float) [3, 3] the camera intrinsic matrix
        :param camera_pose: (ndarray.float) [4, 4] the transformation from world to camera frame
        :param weight: (float) the fusing weight for current observation
        :param rgb: (ndarray.uint8) [H, W, 3] a color image
        :return: None
        """
        depth = self.to_tensor(depth)
        cam_intr = self.to_tensor(intrinsic)
        fx, fy, cx, cy = cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
        cam_pose = self.to_tensor(camera_pose)  # w2c
        im_h, im_w = depth.shape
        c2w = torch.inverse(cam_pose)
        cam_coords = self.world_coords @ c2w.T  # world coordinates represented in camera frame
        pix_z = cam_coords[..., 2]
        # project all the voxels back to image plane
        pix_x = torch.round((cam_coords[..., 0] * fx / cam_coords[..., 2]) + cx).long()
        pix_y = torch.round((cam_coords[..., 1] * fy / cam_coords[..., 2]) + cy).long()
        # eliminate pixels outside view frustum
        valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
        valid_vox_coords = self.vox_coords[valid_pix]
        depth_val = depth[pix_y[valid_pix], pix_x[valid_pix]]
        # integrate sdf
        depth_diff = depth_val - pix_z[valid_pix]
        # all points 1. inside frustum 2. with valid depth 3. outside -truncate_dist
        dist = torch.clamp(depth_diff / self.sdf_trunc, max=1)
        valid_pts = (depth_val > 0.) & (depth_diff >= -self.sdf_trunc)
        valid_vox_coords = valid_vox_coords[valid_pts]
        valid_dist = dist[valid_pts]
        w_old = self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        sdf_old = self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_dist) / w_new
        self.sdf_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = sdf_new
        self.w_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]] = w_new
        if self.fuse_color and rgb is not None:
            rgb = self.to_tensor(rgb)
            rgb_old = self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2]]
            rgb = rgb[pix_y[valid_pix], pix_x[valid_pix]]
            valid_rgb = rgb[valid_pts]
            rgb_new = (w_old[:, None] * rgb_old + weight * valid_rgb) / w_new[:, None]
            self.rgb_vol[valid_vox_coords.T[0], valid_vox_coords.T[1], valid_vox_coords.T[2], :] = rgb_new

    def reset(self):
        self.sdf_vol = torch.ones(*self.res).to(self.dt).to(self.dev)
        self.w_vol = torch.zeros(*self.res).to(self.dt).to(self.dev)
        if self.fuse_color:
            self.rgb_vol = torch.zeros(*self.res, 3).to(self.dt).to(self.dev)

    def to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.dt).to(self.dev)

    def marching_cubes(self, step_size=1):
        b = time.time()
        verts, faces, norms, vals = measure.marching_cubes(self.sdf_vol.cpu().numpy(), step_size=step_size)
        norms = -norms
        verts_ids = np.round(verts).astype(int)
        verts = verts * self.vox_len.cpu().numpy() + self.origin.cpu().numpy().reshape(1, 3)
        if self.fuse_color:
            rgbs = self.rgb_vol[verts_ids[:, 0], verts_ids[:, 1], verts_ids[:, 2]].cpu().numpy().astype(np.uint8)
        else:
            rgbs = np.ones_like(verts).astype(np.uint8) * 255
        e = time.time()
        print('elapse time on marching cubes: {:04f}ms'.format((e - b)*1000))
        return verts, faces, norms, rgbs

    def compute_pcl(self, threshold=0.2):
        b = time.time()
        valid_vox = torch.abs(self.sdf_vol) < threshold
        xyz = self.vox_coords[valid_vox] * self.vox_len + self.origin.reshape(1, 3)
        if self.fuse_color:
            rgb = self.rgb_vol[valid_vox]
        else:
            rgb = torch.ones_like(xyz) * 255
        e = time.time()
        return xyz, rgb.to(torch.uint8)

    def sdf_info(self):
        print('origin: ')
        print(self.origin)
        print('resolution')
        print(self.res)
        print('volume bounds: ')
        print('x_min: {:04f}m'.format(self.origin[0]))
        print('x_max: {:04f}m'.format(self.origin[0]+self.res[0]*self.vox_len))
        print('y_min: {:04f}m'.format(self.origin[1]))
        print('y_max: {:04f}m'.format(self.origin[1]+self.res[1]*self.vox_len))
        print('z_min: {:04f}m'.format(self.origin[2]))
        print('z_max: {:04f}m'.format(self.origin[2]+self.res[2]*self.vox_len))
        print('voxel length: {:06f}m'.format(self.vox_len))

    @staticmethod
    def write_mesh(filename, vertices, faces, normals, rgbs):
        """
        Save a 3D mesh to a polygon .ply file.
        """
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (vertices.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("element face %d\n" % (faces.shape[0]))
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")
        # Write vertex list
        for i in range(vertices.shape[0]):
            ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
                vertices[i, 0], vertices[i, 1], vertices[i, 2],
                normals[i, 0], normals[i, 1], normals[i, 2],
                rgbs[i, 0], rgbs[i, 1], rgbs[i, 2],))
        # Write face list
        for i in range(faces.shape[0]):
            ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))
        ply_file.close()

    @staticmethod
    def write_pcl(filename, xyz, rgb):
        """
        Save a point cloud to a polygon .ply file.
        :param filename: The file name of the.ply file.
        :param xyz: (torch.Tensor.float) [N, 3] the point cloud
        :param rgb: (torch.Tensor.uint8) [N, 3] the corresponding rgb values of the point cloud
        :return: None
        """
        xyz = xyz.cpu().numpy()
        rgb = rgb.cpu().numpy()
        # Write header
        ply_file = open(filename, 'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n" % (xyz.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        # Write vertex list
        for i in range(xyz.shape[0]):
            ply_file.write("%f %f %f %d %d %d\n" % (
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                rgb[i, 0], rgb[i, 1], rgb[i, 2],))
