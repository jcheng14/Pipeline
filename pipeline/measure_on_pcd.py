import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def load_ply(ply_path):

    # Load the point cloud from a PLY file
    # ply_path = '/home/jamesxyye/Documents/Test_0321_front_cropped.ply'  # Replace with the path to your PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    # pcd = remove_noise(pcd, nb_neighbors=1000, std_ratio=0.25)  # Adjust parameters as needed

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # Extract colors

    # Perform PCA on the point cloud
    pca = PCA(n_components=3)
    pca.fit(points)
    points_pca = pca.transform(points)

    # Project the points along the first principal component
    # This means we drop the first component and keep the other two
    projected_points = points_pca[:, 0:2]  # Change here for different components

    # First, normalize the points to be within the range [0, 1]
    min_val = np.min(projected_points, axis=0)
    max_val = np.max(projected_points, axis=0)
    normalized_points = (projected_points - min_val) / (max_val - min_val)

    # Then, scale the normalized points to the desired image size
    image_size = [600, 800]  # For example, 600x800 pixels
    scaled_points = (normalized_points * [image_size[1], image_size[0]]).astype(np.int32)

    # Create an empty image in which each pixel is initially black (intensity zero)
    cv_mat = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Now, plot each point onto the image. Here, I'm making each point white for visibility.
    # If you have color information, you can use it here.
    for point in scaled_points:
        cv2.circle(cv_mat, (point[0], point[1]), radius=1, color=(255, 255, 255), thickness=-1)

    return projected_points

# # Visualization
# plt.figure(figsize=(10, 8))
# plt.scatter(projected_points[:, 0], projected_points[:, 1], c=colors, s=0.1)  # s controls point size
# # plt.title('2D Projection of 3D Point Cloud along the First Principal Component')
# # plt.xlabel('Second Principal Component')
# # plt.ylabel('Third Principal Component')
# plt.axis('off')
# plt.savefig('/home/jamesxyye/Documents/2D_Projection_of_3D_Point_Cloud.png', dpi=300)  # Replace the path and filename as needed
# # plt.show()

# Assuming the third component represents depth information
depth_values = points_pca[:, 2]  # This will be your depth map values

# Normalize depth_values for better visualization
normalized_depth = (depth_values - np.min(depth_values)) / (np.max(depth_values) - np.min(depth_values))

# # Create a depth map
# plt.figure(figsize=(10, 8))
# plt.scatter(projected_points[:, 0], projected_points[:, 1], c=normalized_depth, cmap='gray', s=0.1)  # s controls point size
# # plt.colorbar(label='Normalized Depth')
# # plt.title('Depth Map of Projected 3D Point Cloud')
# # plt.xlabel('Second Principal Component')
# # plt.ylabel('Third Principal Component')
# plt.axis('off')
# plt.savefig('/home/jamesxyye/Documents/2D_Projection_of_3D_Point_Cloud_Depth.png', dpi=300)  # Replace the path and filename as needed
# # plt.show()


print("Loading model...")
sam = sam_model_registry["vit_h"](checkpoint="../weights/sam_vit_h_4b8939.pth")
_ = sam.to(device="cuda")
output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
amg_kwargs = get_amg_kwargs(args)
generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

masks = generator.generate(image)