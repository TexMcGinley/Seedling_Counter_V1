import os
import cv2
import numpy as np
import csv

data_folder_path = 'rough_data/run_1' # Path to the folder containing images

def load_pair(i, skip = 1, debug = False):
    # Funciton for loading 2 sequential image pairs and greyscaling them
    if debug:
        print(f"Loading images {i} and {i + skip}")
    exts = ['.jpg', '.jpeg', '.png']
    file_names = [f for f in os.listdir(data_folder_path) if f.lower().endswith(tuple(exts))]
    file_names = sorted(file_names)
    if i + skip >= len(file_names):
        raise IndexError("Index out of range for image loading.")



    img1_path = os.path.join(data_folder_path, file_names[i])
    img2_path = os.path.join(data_folder_path, file_names[i + skip])

    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    return image1, image2, img1_path, img2_path


def ORB_keypoints_and_descriptors(image, mask=None, debug=False):
    # Function to compute ORB keypoints and descriptors
    if debug:
        print("Computing ORB keypoints and descriptors.")
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(image, mask)
    if debug:
        print(f"Computed {len(keypoints)} keypoints.")
        print(f"Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}")
    return keypoints, descriptors

def match_descriptors(desc1, desc2, ratio=0.8, max_keep=200, debug=False):
    if desc1 is None or desc2 is None:
        return []

    if debug:
        print("Matching descriptors using BFMatcher with ratio test.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)
    good_total = len(good)
    good = good[:max_keep]
    good_used = len(good)
    if debug:
        print(f"Number of good matches after ratio test: {len(good)}")
    return good, good_total, good_used

def convert_to_point_arrays(kp1, kp2, matches):
    # Convert matched keypoints to point arrays
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return points1, points2

def affine_estimation_ransac(pts1, pts2, debug=False):
    if debug:
        print("Estimating affine transformation using RANSAC.")

    if len(pts1) < 3:
        if debug:
            print("Not enough points to estimate affine transformation.")
        return None, None

    max_iters = 10000
    if debug:
        print(f"Using maxIters = {max_iters} for RANSAC.")
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=max_iters
    )

    if matrix is None or inliers is None:
        if debug:
            print("Affine estimation failed.")
        return None, None

    n_inliers = int(np.sum(inliers))
    inlier_ratio = n_inliers / len(pts1)
    if debug:
        print(f"Number of inliers: {n_inliers} out of {len(pts1)} (inlier_ratio: {inlier_ratio:.2f})")
    if (n_inliers < 10) or (inlier_ratio < 0.08):
        if debug:
            print("Affine estimation failed.")
        return None, None

    if debug:
        print(f"Estimated affine matrix:\n{matrix}")
        print(f"Number of inliers: {n_inliers} out of {len(pts1)}")
    return matrix, inliers


def warp_image(image1, image2, matrix):
    warped2 = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))
    overlay = cv2.addWeighted(image1, 0.5, warped2, 0.5, 0)
    return warped2, overlay

def draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers):
    # Flatten inlier mask to (N,) boolean array
    inlier_mask = inliers.ravel().astype(bool)

    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep] 

    # Only display the top 50 inlier matches for clarity
    inlier_matches = inlier_matches[:50]

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=2)
    return img_matches

def build_feature_mask(image, mode = "fixed", debug = False):
    image_shape = image.shape
    h, w = image_shape[:2]
    border = int(0.12 * min(h, w))        # ~8% of smaller dimension
    inner_margin = int(0.20 * min(h, w))
    if mode == "fixed":
        if debug:
            print(f"Building fixed mask for image of size ({w}, {h})")
        # Create a mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Allow features in a border region
        mask[:border, :] = 255
        mask[h-border:, :] = 255
        mask[:, :border] = 255
        mask[:, w-border:] = 255

        # # Suppress features in the central region
        # y1, y2 = inner_margin, h - inner_margin
        # x1, x2 = inner_margin, w - inner_margin
        # if y2 > y1 and x2 > x1:
        #     mask[y1:y2, x1:x2] = 0

    elif mode == "tray_detected":
        return None
    
    return mask
    

def main(i, skip, debug = False):
    result = {
        "image_index": i,
        "img1": None, "img2": None,
        "kp1": 0, "kp2": 0,
        "matches": 0,
        "matches_total": 0,
        "matches_used": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "success": False,
    }

    img1, img2, p1, p2 = load_pair(i, skip, debug=debug) 
    result["img1"] = p1
    result["img2"] = p2
    
    if debug:
        print(p1, p2)
        print(img1.shape, img2.shape)

    mask1 = build_feature_mask(img1, mode="fixed", debug=debug)
    mask2 = build_feature_mask(img2, mode="fixed", debug=debug)

    if debug:
        cv2.imshow("Mask1", mask1)
        cv2.imshow("Mask2", mask2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kp1, des1 = ORB_keypoints_and_descriptors(img1, mask=mask1, debug=debug)
    kp2, des2 = ORB_keypoints_and_descriptors(img2, mask=mask2, debug=debug)
    result["kp1"] = len(kp1)
    result["kp2"] = len(kp2)


    if debug:
        print(f"Number of keypoints in image 1: {len(kp1)}")
        print(f"Number of keypoints in image 2: {len(kp2)}")


    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        print("Too few keypoints/descriptors after masking.")
        return result

    matches, matches_total, matches_used = match_descriptors(des1, des2, debug=debug)
    result["matches"] = len(matches)
    result["matches_total"] = matches_total
    result["matches_used"] = matches_used

    if len(matches) < 10:
        print("Not enough matches found.")
        return result
    if debug:
        print(f"Number of matches after filtering: {len(matches)}")

    pts1, pts2 = convert_to_point_arrays(kp1, kp2, matches)
    
    if debug:
        print(f"Number of matched points: {len(pts1)}")
    
    matrix, inliers = affine_estimation_ransac(pts1, pts2)
    if debug:
        print(f"Affine transformation matrix:\n{matrix}")
        print(f"Number of inliers: {np.sum(inliers) if inliers is not None else 0} out of {len(pts1)}")
    
    if matrix is None or inliers is None:
        print("Could not compute a valid affine transformation.")
        return result
    
    if debug:
        print("Affine transformation matrix:")
        print(matrix)

    warped2, overlay = warp_image(img1, img2, matrix)
    if debug:
        cv2.imshow('Warped Image 2', warped2)
        cv2.imshow('Overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    img_matches = draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers)
    
    if debug:
        cv2.imshow('Inlier Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    n_inliers = int(np.sum(inliers))  
    inlier_ratio = n_inliers / len(pts1)
    result["inliers"] = n_inliers
    result["inlier_ratio"] = inlier_ratio
    result["success"] = (n_inliers >= 10 and inlier_ratio >= 0.08)

    return result



if __name__ == "__main__":
    rows = []
    fieldnames = ['image_index', 'img1', 'img2', 'kp1', 'kp2', 'matches', 'matches_total', 'matches_used', 'inliers', 'inlier_ratio', 'success']
    for i in range(0,5):
        success = False
        row = main(i, 1, debug=False)
        
        rows.append(row)

        if row['success']:
            print(f"[BATCH] Success pair {i}->{i+1}, inlier_ratio={row['inlier_ratio']:.2f}")
    
    with open('batch_results.csv', mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        