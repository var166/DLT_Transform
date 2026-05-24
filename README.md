# DLT Camera Calibration

Direct Linear Transformation for camera resection recovers the 3×4 projection matrix `P = K[R|t]` from N ≥ 6 known 3D-to-2D correspondences with non-coplanar 3D structure.

## What it does

Given an image with detected 2D corners and matching 3D world points, the script:

1. Normalizes both point sets (translate to centroid, scale to mean distance √2 in 2D, √3 in 3, cf. Hartley & Zisserman, 2003).
2. Builds the `(3N, 12+N)` DLT system from equation (18) of the lecture notes.
3. Solves it via SVD — the smallest right-singular vector of M.
4. Denormalizes to recover P in the original pixel / world units:
                           `P = inverse(N_img) · P_norm · N_world`.
5. Decomposes P into intrinsics K and extrinsics (R, t) via RQ factorization.
6. Reports reprojection error against the ground-truth P and visualizes both projections side by side.

## Requirements

- Python 3.9+
- `numpy`, `opencv-python`

```bash
pip install numpy opencv-python
```

## Data

Tested against the Oxford VGG Multi-View datasets (Model House and Corridor), downloadable from <https://www.robots.ox.ac.uk/~vgg/data/mview/>. Lay them out as:

```
$IP_PROJECT_DATA/
├── images_house/
│   ├── house.000.pgm     … house.009.pgm
│   ├── house.000.corners … house.009.corners
│   ├── house.000.P       … house.009.P
│   ├── house.p3d
│   └── house.nview-corners
└── images_corridor/
    └── (same layout, `bt.*` prefix)
```

## Usage

```bash
export IP_PROJECT_DATA="/path/to/data"
python dlt1.py
```

The script iterates the 10 views from each dataset, prints per-view reprojection error, and opens OpenCV windows comparing the DLT result to the provided ground-truth projection. Press any key to advance to the next view.

If `IP_PROJECT_DATA` is unset, the script falls back to `./data/`.

## Project layout

`dlt1.py` is a single file. Main functions:

| Function | Purpose |
| --- | --- |
| `normalize_2d(pts)` | Hartley normalization, 3×3 matrix for image points |
| `normalize_3d(pts)` | Hartley normalization, 4×4 matrix for world points |
| `build_M_matrix(x, X)` | Stack 3N rows of the DLT system from equation 18 |
| `get_dlt_solution(M)` | SVD; returns the smallest right-singular vector |
| `solve_resection(x, X)` | Full pipeline: normalize → build → SVD → denormalize |
| `reprojection_error(P, x, X)` | Project Xᵢ through P, measure pixel distance to xᵢ |
| `get_inner_params_of_camera(P)` | RQ-factorize the 3×3 left block of P → K, R, t |
| `show_projections(...)` | OpenCV side-by-side visualization of DLT vs ground-truth |

Two driver loops, `house_loop()` and `corridor_loop()`, iterate over the views of each dataset.

## Constraints

The DLT formulation here assumes non-coplanar 3D world points. If all world points lie on a plane, the system matrix becomes rank-deficient by more than one and the recovered P is not valid.

## References

- Lecture 3: *Camera Calibration, DLT, SVD* (forelas3.pdf))
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2003
- Oxford VGG Multi-View dataset: <https://www.robots.ox.ac.uk/~vgg/data/mview/>
