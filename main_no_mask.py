import os
import cv2
import numpy as np


data_folder_path = 'rough_data/run_1' # Path to the folder containing images


def load_pair(i, skip = 1):
    # Funciton for loading 2 sequential image pairs and greyscaling them
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



def ORB_keypoints_and_descriptors(image):
    # Function to compute ORB keypoints and descriptors
    print("Computing ORB keypoints and descriptors.")
    orb = cv2.ORB_create(nfeatures=5000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    print(f"Computed {len(keypoints)} keypoints.")
    print(f"Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}")
    return keypoints, descriptors


def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        print("One of the descriptor sets is None, cannot match.")
        return []


    # Function to match descriptors using BFMatcher
    print("Matching descriptors.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    print(f"Found {len(matches)} matches before filtering.")


    # Apply Lowe's ratio test
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)


    good_matches = sorted(good_matches, key=lambda x: x.distance)    
    print(f"{len(good_matches)} matches passed Lowe's ratio test.")


    return good_matches


def convert_to_point_arrays(kp1, kp2, matches):
    # Convert matched keypoints to point arrays
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return points1, points2


def affine_estimation_ransac(pts1, pts2,
                             ransac_thresh=5.0,
                             max_iters=10000,
                             min_inliers=30,
                             min_inlier_ratio=0.08):
    print("Estimating affine transformation using RANSAC (estimateAffinePartial2D).")

    if pts1 is None or pts2 is None or len(pts1) < 3:
        print(f"Not enough matched points for affine: {0 if pts1 is None else len(pts1)}")
        return None, None

    matrix, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iters
    )

    if matrix is None or inliers is None:
        print("Affine estimation failed (OpenCV returned None).")
        return None, None

    n_inliers = int(np.sum(inliers))
    inlier_ratio = n_inliers / len(pts1)

    print(f"Inliers: {n_inliers}/{len(pts1)} (ratio: {inlier_ratio:.2f})")
    print(f"Gate: min_inliers={min_inliers}, min_inlier_ratio={min_inlier_ratio:.2f}")

    if (n_inliers < min_inliers) or (inlier_ratio < min_inlier_ratio):
        print("Affine estimation rejected by quality gate.")
        return None, None

    print(f"Estimated affine matrix:\n{matrix}")
    return matrix, inliers


def warp_image(image1, image2, matrix):
    warped2 = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))
    overlay = cv2.addWeighted(image1, 0.5, warped2, 0.5, 0)
    return warped2, overlay


def draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers):
    # Flatten inlier mask to (N,) boolean array
    inlier_mask = inliers.ravel().astype(bool)


    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]


    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=2)
    return img_matches
   


def main(i, skip):
    img1, img2, p1, p2 = load_pair(i, skip)
    print(p1, p2)
    print(img1.shape, img2.shape)


    kp1, des1 = ORB_keypoints_and_descriptors(img1)
    kp2, des2 = ORB_keypoints_and_descriptors(img2)
    # cv2.imshow('Image 1', img1)
    # cv2.imshow('Image 2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    matches = match_descriptors(des1, des2)
    pts1, pts2 = convert_to_point_arrays(kp1, kp2, matches)
    print(f"Number of matched points: {len(pts1)}")
    matrix, inliers = affine_estimation_ransac(pts1, pts2)
    print(f"Affine transformation matrix:\n{matrix}")
    print(f"Number of inliers: {np.sum(inliers) if inliers is not None else 0} out of {len(pts1)}")
    if matrix is None:
        print("Could not compute a valid affine transformation.")
        return
   
    print("Affine transformation matrix:")
    print(matrix)


    warped2, overlay = warp_image(img1, img2, matrix)
    print("Displaying warped images...")
    cv2.imshow('Warped Image 2', warped2)
    cv2.imshow('Overlay', overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
    img_matches = draw_inlier_matches(img1, kp1, img2, kp2, matches, inliers)
    cv2.imshow('Inlier Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main(2, 1)

