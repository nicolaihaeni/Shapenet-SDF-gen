import os

import sys
import h5py
import argparse
import numpy as np
import point_cloud_utils as pcu

import constant


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh_dir",
    type=str,
    default="/home/isleri/haeni001/data/ShapeNetCore.v2",
    help="Orginal mesh directory",
)
parser.add_argument(
    "--out_dir", type=str, default="./outputs", help="Directory to save sdf"
)
parser.add_argument("--rank", type=int, default=0, help="rank of the process")
parser.add_argument("--num_procs", type=int, default=1, help="number of processes")
parser.add_argument(
    "--categories",
    type=str,
    default="shapenet_13",
    help="Short-handed categories to generate ground-truth",
)
parser.add_argument(
    "--num_surf_pts", type=int, default=125000, help="Number of surface sampled"
)
parser.add_argument("--expand_rate", type=float, default=1.1, help="Max value of x,y,z")
parser.add_argument(
    "--skip_all_exist",
    type=bool,
    default=True,
    help="Whether to skip existing ground-truth",
)
args = parser.parse_args()


def create_dataset(args):
    categories = args.categories
    if categories == "shapenet_13":
        cats = constant.shapenet_13
    elif categories == "shapenet_42":
        cats = constant.shapenet_42
    elif categories == "shapenet_55":
        cats = constant.shapenet_55
    elif categories == "bench":
        cats = constant.bench
    else:
        print("Category does not fit input string.")
        sys.exit()

    # Now we get all the filenames
    for cat_id in cats:
        cat_sdf_dir = os.path.join(args.out_dir, cat_id)
        if not os.path.exists(cat_sdf_dir):
            os.makedirs(cat_sdf_dir)

        cat_mesh_dir = os.path.join(args.mesh_dir, cat_id)
        list_obj = sorted(os.listdir(cat_mesh_dir))
        list_obj = np.array_split(list_obj, args.num_procs)[args.rank]

        for file_name in list_obj:
            mesh_path = os.path.join(
                cat_mesh_dir, file_name, "models", "model_normalized.obj"
            )
            if not os.path.exists(mesh_path):
                print(f"{file_name} does not exists... Skipping")
                continue

            out_sdf_dir = os.path.join(cat_sdf_dir, file_name)
            if not os.path.exists(out_sdf_dir):
                os.makedirs(out_sdf_dir)
            h5_file = os.path.join(out_sdf_dir, f"{file_name}.h5")

            if os.path.exists(h5_file):
                print(f"{file_name} exists... Skipping")
                continue

            # Resolution used to convert shapes to watertight manifolds
            # Higher value means better quality and slower
            manifold_resolution = 20_000

            # Load the mesh
            v, f = pcu.load_mesh_vf(mesh_path)

            # Normalize the vertices to unit sphere
            v = v - np.mean(v, axis=0)
            distances = np.linalg.norm(v, axis=1)
            v /= np.max(distances)

            # Convert the mesh to watertight manifold
            vm, fm = pcu.make_mesh_watertight(v, f, manifold_resolution)
            nm = pcu.estimate_mesh_vertex_normals(
                vm, fm
            )  # Compute vertex normals for watertight mesh

            # Sample points on the surface as face ids and barycentric coordinates
            fid_surf, bc_surf = pcu.sample_mesh_random(vm, fm, args.num_surf_pts)

            # Compute 3D coordinates and normals of surface samples
            p_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, vm)
            n_surf = pcu.interpolate_barycentric_coords(fm, fid_surf, bc_surf, nm)

            p_free = []
            p_free.append(
                p_surf + np.random.normal(scale=0.005, size=(args.num_surf_pts, 3))
            )
            p_free.append(
                p_surf + np.random.normal(scale=0.0005, size=(args.num_surf_pts, 3))
            )
            p_free.append(
                np.random.uniform(
                    -args.expand_rate, args.expand_rate, (args.num_surf_pts, 3)
                )
            )
            p_free = np.concatenate(p_free)

            # Comput the SDF of the random points
            sdf, _, _ = pcu.signed_distance_to_mesh(p_free, vm, fm)

            surface_data = np.concatenate([p_surf, n_surf], axis=-1)
            free_data = np.concatenate([p_free, sdf[:, None]], axis=-1)

            f1 = h5py.File(h5_file, "w")
            f1.create_dataset(
                "free_pts",
                data=free_data,
                compression="gzip",
                compression_opts=9,
                dtype="f",
            )
            f1.create_dataset(
                "surface_pts",
                data=surface_data,
                compression="gzip",
                compression_opts=9,
                dtype="f",
            )
            f1.close()


if __name__ == "__main__":
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    create_dataset(args)
