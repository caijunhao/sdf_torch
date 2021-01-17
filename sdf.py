from skimage import measure
import numpy as np
import torch
import time


class SDF(object):
    def __init__(self, volume_bounds, resolution, rgb=False, device='cuda', dtype=torch.float32):
        """
        Initialize SDF module
        :param volume_bounds: A 3x2 numpy array specifying the min and max bound values of the volume for each axis
        :param resolution: The resolution for each voxel, the unit is meter.
        :param rgb: A flag specifying if fusing color information
        :param device: An object representing the device on which a torch.Tensor is or will be allocated.
        :param dtype: Data type for torch tensor.
        """
        self.volume_bounds = volume_bounds
        self.resolution = resolution
        self.rgb = rgb
        self.device = device
        self.dtype = dtype

        # Adjust volume_bounds
        self.volume_dims = np.ceil((self.volume_bounds[:, 1]-self.volume_bounds[:, 0])/self.resolution).astype(np.int)
        self.volume_bounds[:, 1] = self.volume_bounds[:, 0] + self.volume_dims * self.resolution
        self.origin = torch.from_numpy(self.volume_bounds[:, 0]).float().view(1, 3).to(device)

        # Initialize signed distance and weight volumes
        self.sdf_volume = torch.ones(self.volume_dims.tolist(), dtype=self.dtype).to(device)
        self.post_processed_volume = self.sdf_volume.clone()
        self.w_volume = torch.zeros_like(self.sdf_volume, dtype=self.dtype).to(device)
        # record if the voxel has been scan at least once
        self.valid = torch.zeros_like(self.sdf_volume, dtype=torch.bool).to(device)
        if self.rgb:
            self.rgb_volume = torch.zeros_like(self.sdf_volume, dtype=self.dtype).to(device)

        # get voxel coordinates and world positions
        vx, vy, vz = torch.meshgrid(torch.arange(self.volume_dims[0]),
                                    torch.arange(self.volume_dims[1]),
                                    torch.arange(self.volume_dims[2]))
        self.voxel_coordinates = torch.stack([vx, vy, vz], dim=-1).to(device)
        self.world_positions = self.origin + self.voxel_coordinates * self.resolution
        self.sdf_info()

    def sdf_integrate(self, depth, intrinsic, camera_pose, weight=1.0, rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: An HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
        :param camera_pose: A 4x4 numpy array representing the transformation from camera to reference frame (world)
        :param weight: A scalar representing the fusion weight for current observation
        :param rgb: (Optional) An HxWx3 numpy array representing a color image
        :return: None
        """
        depth = torch.from_numpy(depth).to(self.device)
        weight = torch.tensor(weight, device=self.device)
        height, width = depth.shape
        c2w = torch.from_numpy(camera_pose).to(self.device)
        w2c = torch.inverse(c2w)
        intrinsic = torch.from_numpy(intrinsic).to(self.device)
        # inv_intrinsic = torch.inverse(intrinsic)
        world_positions = torch.cat([self.world_positions,
                                     torch.ones(self.volume_dims.tolist() + [1], dtype=self.dtype).to(self.device)],
                                    dim=-1)
        camera_positions = world_positions @ w2c.T  # position represented in camera frame
        pixel_coors = camera_positions[..., :-1] @ intrinsic.T
        # ideally the camera is outside the volume space and directed toward the volume,
        # so all the z value should larger than 0
        flag_z = pixel_coors[..., 2] > 0
        pixel_coors[flag_z, :] = pixel_coors[flag_z, :] / pixel_coors[flag_z, :][:, -1:]
        pixel_coors = torch.round(pixel_coors).long()
        flag_w = (pixel_coors[..., 0] >= 0) * (pixel_coors[..., 0] < width)
        flag_h = (pixel_coors[..., 1] >= 0) * (pixel_coors[..., 1] < height)
        valid_voxels = flag_w * flag_h * flag_z  # dim0 x dim1 x dim2
        depth_volume = depth[pixel_coors[valid_voxels][:, 1], pixel_coors[valid_voxels][:, 0]]
        valid_depth = depth_volume > 0
        diff_depth = torch.zeros_like(depth_volume, dtype=self.dtype, device=self.device)
        diff_depth[valid_depth] = depth_volume[valid_depth] - camera_positions[valid_voxels][valid_depth][:, 2]
        distances = diff_depth / (self.resolution * 5.0)  # rescale the distance
        self.valid[valid_voxels] = True
        valid_x = self.voxel_coordinates[valid_voxels][valid_depth][:, 0]
        valid_y = self.voxel_coordinates[valid_voxels][valid_depth][:, 1]
        valid_z = self.voxel_coordinates[valid_voxels][valid_depth][:, 2]
        w_old = self.w_volume[valid_x, valid_y, valid_z]
        sdf_old = self.sdf_volume[valid_x, valid_y, valid_z]
        valid_distances = distances[valid_depth]
        # valid_distances = torch.clamp_max(valid_distances, 1.0)
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_distances) / w_new
        self.w_volume[valid_x, valid_y, valid_z] = w_new
        self.sdf_volume[valid_x, valid_y, valid_z] = sdf_new
        if self.rgb and rgb is not None:
            rgb_old = self.decode_rgb(self.rgb_volume[valid_x, valid_y, valid_z])
            rgb = torch.from_numpy(rgb).to(self.device)
            valid_color = rgb[
                pixel_coors[valid_voxels][valid_depth][:, 1], pixel_coors[valid_voxels][valid_depth][:, 0]]
            rgb_new = (torch.unsqueeze(w_old, dim=1) * rgb_old + weight * valid_color) / torch.unsqueeze(w_new, dim=1)
            rgb_new = torch.minimum(rgb_new, torch.tensor(255.0, device=self.device))
            self.rgb_volume[valid_x, valid_y, valid_z] = self.encode_rgb(rgb_new)

    def reset(self):
        self.sdf_volume = torch.ones(self.volume_dims.tolist(), dtype=self.dtype).to(self.device)
        self.w_volume = torch.ones_like(self.sdf_volume, dtype=self.dtype).to(self.device)
        if self.rgb:
            self.rgb_volume = torch.zeros_like(self.sdf_volume, dtype=self.dtype).to(self.device)

    def post_process_volume(self):
        """
        Process SDF volume before computing mesh or point cloud.
        Should be overwrote in subclass.
        :return: None
        """
        self.post_processed_volume = self.sdf_volume

    def compute_pcl(self, threshold=0.2):
        b = time.time()
        valid_voxels = torch.abs(self.post_processed_volume) < threshold
        # valid_voxels = curr_sdf_volume < threshold
        xyz = self.world_positions[valid_voxels]
        if self.rgb:
            rgb = self.decode_rgb(self.rgb_volume[valid_voxels])
        else:
            rgb = (torch.ones_like(xyz, dtype=self.dtype) * 255).to(self.device)
        pcl = torch.cat([xyz, rgb], dim=1)
        e = time.time()
        print('elapse time for computing point cloud: {:06f}s'.format(e - b))
        return pcl

    def compute_pcl_using_gradients(self, threshold=0.6):
        b = time.time()
        valid_voxels = torch.abs(self.post_processed_volume) < threshold
        gradients = self.compute_gradients(self.post_processed_volume)
        gradients = gradients[valid_voxels]
        norm = torch.norm(gradients, dim=1, keepdim=True)
        valid_gradients = norm[:, 0] > 0
        gradients = gradients[valid_gradients]
        norm = norm[valid_gradients]
        gradients = gradients / norm
        curr_sdf_value = self.post_processed_volume[valid_voxels][valid_gradients]
        delta = -torch.unsqueeze(curr_sdf_value, dim=1) * gradients * (self.resolution * 5.0)
        xyz = self.world_positions[valid_voxels][valid_gradients] + delta
        # import matplotlib.pyplot as plt
        # # gradients_np = self.gradients.cpu().numpy().reshape(-1, 3)
        # # gradients_np = gradients_np / np.linalg.norm(gradients_np, axis=1, keepdims=True)
        # # xyz_np = self.voxel_coordinates.cpu().numpy().reshape(-1, 3)
        # gradients_np = gradients.cpu().numpy()
        # # gradients_np = gradients_np / np.linalg.norm(gradients_np, axis=1, keepdims=True)
        # xyz_np = self.voxel_coordinates[valid_voxels][valid_gradients].cpu().numpy().reshape(-1, 3)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.quiver(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], gradients_np[:, 0], gradients_np[:, 1], gradients_np[:, 2],
        #           length=0.4)
        # plt.show()

        if self.rgb:
            rgb = self.decode_rgb(self.rgb_volume[valid_voxels][valid_gradients])
        else:
            rgb = (torch.ones_like(xyz, dtype=self.dtype) * 255).to(self.device)
        pcl = torch.cat([xyz, rgb], dim=1)
        e = time.time()
        print('elapse time for computing point cloud: {:06f}s'.format(e - b))
        return pcl

    def normal_propagation(self):
        # if the initial view frustum includes the entire volume space,
        # we can use weight volume to find in-object space
        curr_sdf_volume = self.post_processed_volume.clone()
        in_obj_ids = (self.w_volume == 0) * self.valid
        curr_sdf_volume[in_obj_ids] = -1.0
        # curr_sdf_volume = torch.squeeze(torch.nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(curr_sdf_volume, dim=0), dim=0), scale_factor=2, mode='trilinear'))
        sdf_volume = curr_sdf_volume.cpu().numpy()
        b = time.time()
        v, f, n, _ = measure.marching_cubes(sdf_volume, level=0)
        v = torch.from_numpy(v.copy()).to(self.device)
        unassigned = torch.ones_like(self.post_processed_volume, dtype=torch.bool).to(self.device)
        # p_sdf_volume = torch.zeros_like(self.sdf_volume).to(self.device)
        p_sdf_volume = self.post_processed_volume.clone()
        curr_sdf = 0.1
        v_c = torch.ceil(v).long()
        v_f = torch.floor(v).long()
        v = torch.cat([v_c, v_f], dim=0)
        i = 0
        while torch.any(unassigned) and i < 5:
            # boundary checking
            b_f = (v[:, 0] >= 0) * (v[:, 0] < self.volume_dims[0]) * \
                  (v[:, 1] >= 0) * (v[:, 1] < self.volume_dims[1]) * \
                  (v[:, 2] >= 0) * (v[:, 2] < self.volume_dims[2])
            v = v[b_f]
            # assigned checking
            v_u = v[unassigned[v[:, 0], v[:, 1], v[:, 2]]]
            torch.cuda.empty_cache()
            unassigned[v[:, 0], v[:, 1], v[:, 2]] = False
            # sign checking
            v_sdf = curr_sdf_volume[v_u[:, 0], v_u[:, 1], v_u[:, 2]]
            v_sdf = v_sdf + 1e-9
            sign = v_sdf / torch.abs(v_sdf)
            p_sdf_volume[v_u[:, 0], v_u[:, 1], v_u[:, 2]] = curr_sdf * sign
            curr_sdf += 0.2
            v0 = v_u.clone()
            v0[:, 0] += 1
            v1 = v_u.clone()
            v1[:, 0] -= 1
            v2 = v_u.clone()
            v2[:, 1] += 1
            v3 = v_u.clone()
            v3[:, 1] -= 1
            v4 = v_u.clone()
            v4[:, 2] += 1
            v5 = v_u.clone()
            v5[:, 0] += 1
            v = torch.cat([v0, v1, v2, v3, v4, v5], dim=0)
            i += 1
        v, f, n, _ = measure.marching_cubes(sdf_volume, level=0)
        n = -n
        e = time.time()
        print('elapse time for sdf propagation: {:06f}s'.format(e - b))
        ids = np.round(v).astype(np.int)
        v = v * self.resolution + self.volume_bounds[:, 0:1].T
        if self.rgb:
            rgb = self.decode_rgb(self.rgb_volume[ids[:, 0], ids[:, 1], ids[:, 2]])
            rgb = rgb.cpu().numpy()
            rgb = np.floor(rgb)
            rgb = rgb.astype(np.uint8)
        else:
            rgb = np.ones_like(v, dtype=np.uint8) * 255
        return v, f, n, rgb

    def compute_mesh(self):
        """
        Using marching cubes to compute a mesh from SDF volume.
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html
        :return: Vertices, faces, normals, and rgb values of the constructed mesh.
        """
        sdf_volume = self.post_processed_volume.cpu().numpy()
        b = time.time()
        v, f, n, _ = measure.marching_cubes(sdf_volume, level=0)
        n = -n
        e = time.time()
        print('elapse time for marching cubes: {:06f}s'.format(e-b))
        ids = np.round(v).astype(np.int)
        v = v * self.resolution + self.volume_bounds[:, 0:1].T
        if self.rgb:
            rgb = self.decode_rgb(self.rgb_volume[ids[:, 0], ids[:, 1], ids[:, 2]])
            rgb = rgb.cpu().numpy()
            rgb = np.floor(rgb)
            rgb = rgb.astype(np.uint8)
        else:
            rgb = np.ones_like(v, dtype=np.uint8) * 255
        return v, f, n, rgb

    def sdf_info(self):
        print('volume bounds: ')
        print('x_min: {:04f}m'.format(self.volume_bounds[0, 0]))
        print('x_max: {:04f}m'.format(self.volume_bounds[0, 1]))
        print('y_min: {:04f}m'.format(self.volume_bounds[1, 0]))
        print('y_max: {:04f}m'.format(self.volume_bounds[1, 1]))
        print('z_min: {:04f}m'.format(self.volume_bounds[2, 0]))
        print('z_max: {:04f}m'.format(self.volume_bounds[2, 1]))
        print('volume size: {}'.format(self.volume_dims))
        print('resolution: {:06f}m'.format(self.resolution))

    @staticmethod
    def compute_gradients(sdf_volume, device='cuda'):
        """
        Return the gradient of SDF volume.
        The gradient is computed using second order accurate central differences in the interior points and
        first order accurate one-sides differences at the boundaries.
        For more details, you can refer to this link.
        https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        :param sdf_volume: An SDF volume
        :param device: The desired device of returned tensor.
        :return: A torch tensor with the same shape as self.world_positions.
                 Each channel represents one derivative of the SDF volume
        """
        h, w, d = sdf_volume.size()
        gradients = torch.zeros([h, w, d, 3], dtype=sdf_volume.dtype).to(device)
        # interior points
        gradients[1:h - 1, ..., 0] = sdf_volume[2:h, ...] - sdf_volume[0:h - 2, ...]
        gradients[:, 1:w - 1, :, 1] = sdf_volume[:, 2:w, ...] - sdf_volume[:, 0:w - 2, ...]
        gradients[..., 1:d - 1, 2] = sdf_volume[..., 2:d] - sdf_volume[..., 0:d - 2]
        gradients = gradients / 2.0
        # boundaries
        gradients[0, ..., 0] = sdf_volume[1, ...] - sdf_volume[0, ...]
        gradients[-1, ..., 0] = sdf_volume[-1, ...] - sdf_volume[-2, ...]
        gradients[:, 0, :, 1] = sdf_volume[:, 1, ...] - sdf_volume[:, 0, ...]
        gradients[:, -1, :, 1] = sdf_volume[:, -1, ...] - sdf_volume[:, -2, ...]
        gradients[..., 0, 2] = sdf_volume[..., 1] - sdf_volume[..., 0]
        gradients[..., -1, 2] = sdf_volume[..., -1] - sdf_volume[..., -2]
        return gradients

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
    def write_pcl(filename, pcl):
        """
        Save a point cloud to a polygon .ply file.
        :param filename: The file name of the.ply file.
        :param pcl: An N x 6 torch tensor representing the point cloud with rgb information.
        :return: None
        """
        pcl = pcl.cpu().numpy()
        xyz = pcl[:, :3]
        rgb = pcl[:, 3:].astype(np.uint8)
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

    @staticmethod
    def decode_rgb(rgb_volume):
        """
        Decode an RGB volume to R, G, and B volumes separately
        :param rgb_volume: An N-d torch tensor in which each voxel encodes the RGB information as B * 256 * 256 + G * 256 + R
        :return: An N x 3 torch tensors representing the R, G, B values of selected pixels of the color image
        """
        b = torch.floor(rgb_volume / (256 * 256))
        g = torch.floor((rgb_volume - b * (256 * 256)) / 256)
        r = torch.floor(rgb_volume - b * (256 * 256) - g * 256)
        return torch.stack([r, g, b], dim=1)

    @staticmethod
    def encode_rgb(rgb):
        """
        Encode each pixel of color image as B * 256 * 256 + G * 256 + R
        :param rgb: An N x 3 torch tensor representing the R, G, B values of selected pixels of the color image
        :return: An N-d torch tensor
        """
        return rgb[:, 2] * 256 * 256 + rgb[:, 1] * 256 + rgb[:, 0]


class TSDF(SDF):
    def __init__(self, volume_bounds, resolution, rgb=False, device='cuda', dtype=torch.float32):
        """
        Initialize SDF module
        :param volume_bounds: A 3x2 numpy array specifying the min and max bound values of the volume for each axis
        :param resolution: The resolution for each voxel, the unit is meter.
        :param rgb: A flag specifying if fusing color information.
        :param device: An object representing the device on which a torch.Tensor is or will be allocated.
        :param dtype: Data type for torch tensor.
        """
        super(TSDF, self).__init__(volume_bounds, resolution, rgb, device, dtype)
        self.tsdf_volume = self.sdf_volume.clone()
        self.tsdf_w_volume = self.w_volume.clone()
        self.truncate_margin = resolution * 5

    def tsdf_integrate(self, depth, intrinsic, camera_pose, weight=1.0, rgb=None):
        """
        Integrate RGB-D frame into SDF volume (naive SDF)
        :param depth: An HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing the camera intrinsic matrix
        :param camera_pose: A 4x4 numpy array representing the transformation from camera to reference frame (world)
        :param weight: A scalar representing the fusion weight for current observation
        :param rgb: (Optional) An HxWx3 numpy array representing a color image
        :return: None
        """
        depth = torch.from_numpy(depth).to(self.device)
        weight = torch.tensor(weight, device=self.device)
        height, width = depth.shape
        c2w = torch.from_numpy(camera_pose).to(self.device)
        w2c = torch.inverse(c2w)
        intrinsic = torch.from_numpy(intrinsic).to(self.device)
        # inv_intrinsic = torch.inverse(intrinsic)
        world_positions = torch.cat([self.world_positions,
                                     torch.ones(self.volume_dims.tolist() + [1], dtype=self.dtype).to(self.device)],
                                    dim=-1)
        camera_positions = world_positions @ w2c.T  # position represented in camera frame
        pixel_coors = camera_positions[..., :-1] @ intrinsic.T
        # ideally the camera is outside the volume space and directed toward the volume,
        # so all the z value should larger than 0
        flag_z = pixel_coors[..., 2] > 0
        pixel_coors[flag_z, :] = pixel_coors[flag_z, :] / pixel_coors[flag_z, :][:, -1:]
        pixel_coors = torch.round(pixel_coors).long()
        flag_w = (pixel_coors[..., 0] >= 0) * (pixel_coors[..., 0] < width)
        flag_h = (pixel_coors[..., 1] >= 0) * (pixel_coors[..., 1] < height)
        valid_voxels = flag_w * flag_h * flag_z  # dim0 x dim1 x dim2
        depth_volume = depth[pixel_coors[valid_voxels][:, 1], pixel_coors[valid_voxels][:, 0]]
        valid_depth = depth_volume > 0
        valid_depth_tsdf = valid_depth.clone()
        diff_depth = torch.zeros_like(depth_volume, dtype=self.dtype, device=self.device)
        diff_depth[valid_depth] = depth_volume[valid_depth] - camera_positions[valid_voxels][valid_depth][:, 2]
        distances = diff_depth / self.truncate_margin  # rescale the distance
        # integrate sdf
        self.valid[valid_voxels] = True
        valid_x = self.voxel_coordinates[valid_voxels][valid_depth][:, 0]
        valid_y = self.voxel_coordinates[valid_voxels][valid_depth][:, 1]
        valid_z = self.voxel_coordinates[valid_voxels][valid_depth][:, 2]
        w_old = self.w_volume[valid_x, valid_y, valid_z]
        sdf_old = self.sdf_volume[valid_x, valid_y, valid_z]
        valid_distances = distances[valid_depth]
        # valid_distances = torch.clamp_max(valid_distances, 1.0)
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_distances) / w_new
        self.w_volume[valid_x, valid_y, valid_z] = w_new
        self.sdf_volume[valid_x, valid_y, valid_z] = sdf_new
        # integrate tsdf
        # truncate distance
        valid_depth_tsdf *= diff_depth >= -self.truncate_margin
        valid_x = self.voxel_coordinates[valid_voxels][valid_depth_tsdf][:, 0]
        valid_y = self.voxel_coordinates[valid_voxels][valid_depth_tsdf][:, 1]
        valid_z = self.voxel_coordinates[valid_voxels][valid_depth_tsdf][:, 2]
        w_old = self.tsdf_w_volume[valid_x, valid_y, valid_z]
        sdf_old = self.tsdf_volume[valid_x, valid_y, valid_z]
        valid_distances = distances[valid_depth_tsdf]
        # valid_distances = torch.clamp_max(valid_distances, 1.0)
        w_new = w_old + weight
        sdf_new = (w_old * sdf_old + weight * valid_distances) / w_new
        self.tsdf_w_volume[valid_x, valid_y, valid_z] = w_new
        self.tsdf_volume[valid_x, valid_y, valid_z] = sdf_new
        if self.rgb and rgb is not None:
            rgb_old = self.decode_rgb(self.rgb_volume[valid_x, valid_y, valid_z])
            rgb = torch.from_numpy(rgb).to(self.device)
            valid_color = rgb[pixel_coors[valid_voxels][valid_depth_tsdf][:, 1], pixel_coors[valid_voxels][valid_depth_tsdf][:, 0]]
            rgb_new = (torch.unsqueeze(w_old, dim=1) * rgb_old + weight * valid_color) / torch.unsqueeze(w_new, dim=1)
            rgb_new = torch.minimum(rgb_new, torch.tensor(255.0, device=self.device))
            self.rgb_volume[valid_x, valid_y, valid_z] = self.encode_rgb(rgb_new)

    def post_process_volume(self):
        # if the initial view frustum includes the entire volume space,
        # we can use weight volume to find in-object space
        sdf_volume = self.tsdf_volume.clone()
        in_obj_ids = (self.tsdf_w_volume == 0) * self.valid
        sdf_volume[in_obj_ids] = self.sdf_volume[in_obj_ids]
        self.post_processed_volume = sdf_volume








