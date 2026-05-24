import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(os.getenv('IP_PROJECT_DATA', 'data'))
HOUSE = DATA_DIR / 'images_house'
CORRIDOR = DATA_DIR / 'images_corridor'


def detect_checkerboard(image_path, board_size):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not found:
        print(f"Checkerboard not found in {image_path}")
        return None, img
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    for pt in corners.reshape(-1, 2):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    return corners.reshape(-1, 2), img


def make_world_points(board_size, square_size):
    cols, rows = board_size
    pts = np.zeros((rows * cols, 3), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            pts[i * cols + j] = [j * square_size, i * square_size, 0.0]
    return pts

#aici se termina sectiunea cu creatu de puncte cunoscute

#am puncte cunoscute in viata reala si pe ecran, trebe sa scot matricea de rotatie, daca am K matricea de camera
#deci trebe sa scot P = K[R t], care e calibrata pentru ca K este cunoscut.
#[R t] este nnormalizata daca se cunoaste K, deci scot camera normalizata
#[R t] numeste camera normalizata
#gamma - aspect ratio il am din cum arata poza, pixelii is patrati deci aspect ratio ii 1
#focal length il am din matricea K - camera_matrix din yml, camera_matrix[1][1] ii focal length 
#principal pointu ii tot din camera matrix, ultima coloana, adica camera_matrix[:][2]
#lambda din pdf se refera la adancimi, nu la valori proprii
#daca nu se cunoaste K calibrarea, atunci P = K[R t] poate fi orice matrice
# de 3 X 4 si solutia se numeste 'projective reconstruction' si nu are solutie unica 
# ca sa scoatem ambiguitatea, ne trebe K calibrarea camerei
# daca cunoastem K atunci avem o camera calibrata
#problema structurii din miscare devine gasirea camerelor calibrate(sau normalizate, inseamna 
#acelasi lucru) astfel incat ecuatia §8§ din pdf, pagina 3
#cand se cunoaste K, solutia la problema se numeste 'Euclidean reconstruction' si are o solutie unica pana la o transformare similara (scalare, rotatie, translatia)
#ca sa aflii parametri intrinseci: aspect ratio, skew si focal length:
# calculezi o camera P, te asiguri ca nu exista ambiguitati de proiectie, si presupui ca punctele din scena sunt cunoscute
# asta se poate face utilizand o poza unde am masurat adancimi 
# dupa ce stii camera P, putem sa o descompunem in K[R t], unde K este o matrice
# triangulara si R este o rotatie, folosind RQ factorization
#  The resection problem --- aci ii DLT de care am io nevoie
#  am nevoie de macar 6 puncte ca problema sa fie bine definita
#  DLT formeaza un sistem omogen liniar de ecuatii si rezolva sistemul gasind null space aproximativ al matricii sistem
#  approximate null space ---- gaseste vectorii care inmultiti cu matricea sistem dau aproape 0 


#normalizarea punctelor cunoscute
def normalize_2d(pts):
    mean = pts.mean(axis=0)
    print(mean)
    shifted = pts - mean
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    s = np.sqrt(2) / (mean_dist + 1e-12)

    N = np.array([
        [s, 0, -s * mean[0]],
        [0, s, -s * mean[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    pts_h = np.column_stack([pts, np.ones(len(pts))])
    print(f"PTs_h sjape {pts_h.shape}")
    pts_n = (N @ pts_h.T).T
    return N, pts_n[:, :2]


def normalize_3d(pts):
    mean = pts.mean(axis=0)
    shifted = pts - mean
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    s = np.sqrt(3) / (mean_dist + 1e-12)
    N = np.array([
        [s, 0, 0, -s*mean[0]],
        [0, s, 0, -s*mean[1]],
        [0, 0, s, -s*mean[2]],
        [0, 0, 0, 1.0]
    ])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_n = (N @ pts_h.T).T
    return N, pts_n[:, :3]


def build_M_matrix(pts_2d, pts_world_3d):
    if len(pts_2d) != len(pts_world_3d) or len(pts_2d) < 6:
        print("Must have at least 6 point correspondences for DLT to be well defined")
        return None

    N = len(pts_world_3d)
    z4 = np.zeros(4)
    rows = []

    for i, X in enumerate(pts_world_3d):
        Xi = np.append(X, 1.0)
        xi, yi = pts_2d[i]

        lam_x = np.zeros(N); lam_x[i] = -xi
        lam_y = np.zeros(N); lam_y[i] = -yi
        lam_1 = np.zeros(N); lam_1[i] = -1.0

        rows.append(np.concatenate([Xi, z4, z4, lam_x]))
        rows.append(np.concatenate([z4, Xi, z4, lam_y]))
        rows.append(np.concatenate([z4, z4, Xi, lam_1]))

    return np.vstack(rows)


def get_dlt_solution(M):
    U, S, V = np.linalg.svd(M)
    if np.linalg.norm(V[:, -1]) < 1e-10:
        print("The matrix M is rank defficient, singular value decomposition returns a vector with norm ~=0 ")

    return V[-1]


def solve_resection(pts_2d, pts_3d):

    N_img,   pts_2d_n = normalize_2d(pts_2d)
    N_world, pts_3d_n = normalize_3d(pts_3d)

    M = build_M_matrix(pts_2d_n, pts_3d_n)
    print(f'M shape {M.shape}')

    v = get_dlt_solution(M)
    P_norm = v[:12].reshape(3, 4)
    print(f'v shape {v.shape}')

    P = np.linalg.inv(N_img) @ P_norm @ N_world
    return P


def reprojection_error(P, pts_2d, pts_3d):
    N = len(pts_2d)
    X_h = np.column_stack([pts_3d, np.ones(N)])
    projected = (P @ X_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - pts_2d, axis=1)
    return errors.mean(), errors.max(), np.median(errors), errors


def get_inner_params_of_camera(P):
    Q, R = np.linalg.qr(P[:, :3])
    K = R / R[2, 2]
    if(np.linalg.det(K) < 0):
        K = -K
    t = np.linalg.inv(K) @ P[:, 3]
    return K, t / np.linalg.norm(t)


def show_projections(P_dlt, P_gt, pts_3d, img):
    def project(P, pts):
        X_h = np.column_stack([pts, np.ones(len(pts))])
        projected = (P @ X_h.T).T
        return projected[:, :2] / projected[:, 2:3]

    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    img_dlt = base.copy()
    img_gt  = base.copy()
    img_all = base.copy()

    proj_dlt = project(P_dlt, pts_3d)
    proj_gt  = project(P_gt,  pts_3d)

    for pt in proj_dlt:
        cv2.circle(img_dlt, (int(round(pt[0])), int(round(pt[1]))),
                   4, (0, 0, 255), 1)
        cv2.circle(img_all, (int(round(pt[0])), int(round(pt[1]))),
                   4, (0, 0, 255), 1)

    for pt in proj_gt:
        cv2.circle(img_gt, (int(round(pt[0])), int(round(pt[1]))),
                   4, (255, 0, 0), 1)
        cv2.circle(img_all, (int(round(pt[0])), int(round(pt[1]))),
                   4, (255, 0, 0), 1)

    cv2.putText(img_dlt, 'DLT P', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_gt, 'Provided P', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('DLT projections', img_dlt)
    cv2.imshow('Provided P projections', img_gt)
    cv2.imshow("Original", img)
    cv2.imshow("Both original P and DLT P", img_all)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#todo: itereaza prin toate pozele cu house si apoi ia si dinozauru si fa ceva statistica pe erorile de reproiectie 
def house_loop():
    pts_3d_all = np.loadtxt(HOUSE / 'house.p3d')
    for i in range(10):
        pts_2d_all = np.loadtxt(HOUSE / f'house.00{i}.corners')
        nview = np.loadtxt(HOUSE / 'house.nview-corners', dtype=str)

        view0 = nview[:, i]
        valid = view0 != '*'

        pts_3d = pts_3d_all[valid]
        pts_2d = pts_2d_all[view0[valid].astype(int)]

        print(f'2dpoints shape {pts_2d.shape}')
        print(f'3dpoints shape {pts_3d.shape}')

        P = solve_resection(pts_2d=pts_2d, pts_3d=pts_3d)
        NormalP = np.loadtxt(HOUSE / f'house.00{i}.P')
        print(f"DLT P {P}")
        print(f'Provided P{NormalP}')
        print(f"Norm of difference between the 2 matrices{np.linalg.norm(P - NormalP)}")
        reprerror_mean, reprerror_max, reprerror_median, reprerrors = reprojection_error(P, pts_2d, pts_3d)
        print(f"Max ERROR from reprojection: {reprerror_max},\n MedianERROR {reprerror_median}")
        K, t = get_inner_params_of_camera(P)
        k_given, t_given = get_inner_params_of_camera(NormalP)
        print(f'obtained K = {K} \nvs \nnormal K = {k_given}\n obrained t = \n{t} vs given t = \n{t_given}')
        src = cv2.imread(str(HOUSE / f'house.00{i}.pgm'))
        show_projections(P, NormalP, pts_3d, src)


def corridor_loop():
    pts_3d_all = np.loadtxt(CORRIDOR / 'bt.p3d')
    nview = np.loadtxt(CORRIDOR / 'bt.nview-corners', dtype=str)
    for i in range(10):
        pts_2d_all = np.loadtxt(CORRIDOR / f'bt.00{i}.corners')

        view0 = nview[:, i]
        valid = view0 != '*'

        pts_3d = pts_3d_all[valid]
        pts_2d = pts_2d_all[view0[valid].astype(int)]

        print(f'2dpoints shape {pts_2d.shape}')
        print(f'3dpoints shape {pts_3d.shape}')

        P = solve_resection(pts_2d=pts_2d, pts_3d=pts_3d)
        NormalP = np.loadtxt(CORRIDOR / f'bt.00{i}.P')
        print(f"DLT P {P}")
        print(f'Provided P{NormalP}')
        print(f"Norm of difference between the 2 matrices{np.linalg.norm(P - NormalP)}")
        reprerror_mean, reprerror_max, reprerror_median, reprerrors = reprojection_error(P, pts_2d, pts_3d)
        print(f"Max ERROR from reprojection: {reprerror_max},\n MedianERROR {reprerror_median}")
        K, t = get_inner_params_of_camera(P)
        k_given, t_given = get_inner_params_of_camera(NormalP)
        print(f'obtained K = {K} \nvs \nnormal K = {k_given}\n obrained t = \n{t} vs given t = \n{t_given}')
        src = cv2.imread(str(CORRIDOR / f'bt.00{i}.pgm'))
        show_projections(P, NormalP, pts_3d, src)


def main():
    house_loop()
    corridor_loop()


if __name__ == "__main__":
    main()