import os
from collections import Counter

import cv2
import numpy as np
import torch

from normalization import Normalization
from embedd import Embedd

def batch_normalization():
    count_successfull_img = 0
    normalization = Normalization()
    for i in range(100,200):
        source_path_1 = "../img-test/" + f"{i:03}"
        for j in range(1,11):
            for suffix in ['L', 'R']:
                source_path_2 = source_path_1 + f"{j:02}_" + f"{suffix}" + ".bmp"
                image = cv2.imread(source_path_2, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Không thể đọc ảnh từ: {source_path_2}")
                    continue
                normalization_img, segmentation_img = normalization.normalize(image, return_segmentation=True)
                rectangle_img_np = np.array(normalization_img)
                # cv2.imshow("normalization", rectangle_img_np)
                # cv2.waitKey(-1)
                # cv2.destroyAllWindows()
                output_path = "./normalization_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".png"
                output_segmentation_path = "./segmentation_img/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + "_segmentation.png"
                cv2.imwrite(output_path, rectangle_img_np)
                cv2.imwrite(output_segmentation_path, segmentation_img)
                print(f"Phân đoạn thành công: {source_path_2}")
                count_successfull_img+=1
    print(f"Đã phân đoạn {count_successfull_img} ảnh.")

def normalization(i, j, suffix):
    path = "../img-test/" + f"{i:03}" + f"{j:02}_" + f"{suffix}" + ".bmp"
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Không thể đọc ảnh từ: {path}")
        return
    normalization_img, seg_img = Normalization().normalize(image, return_segmentation=True)
    rectangle_img_np = np.array(normalization_img)

    return rectangle_img_np, seg_img

def embedd(i, j, suffix):
    normalization_img, seg_img = normalization(i, j, suffix)
    vector_embedd = Embedd("vector_embedding_pca.pth", "pca_pretrained_model.pth").embedding(normalization_img)
    print(vector_embedd.shape)
    return vector_embedd

def euclidean_distance(v1, v2):
    v1 = v1.detach().cpu().numpy() if isinstance(v1, torch.Tensor) else v1
    v2 = v2.detach().cpu().numpy() if isinstance(v2, torch.Tensor) else v2
    return np.sqrt(np.sum((v1 - v2) ** 2))

def classify_by_knn(query_vector, embedded_all_file,  k=10):
    vector_label_pairs = np.load(embedded_all_file)
    vectors = vector_label_pairs[:, :-1].astype(np.float32)
    labels = vector_label_pairs[:, -1]
    distances = [euclidean_distance(query_vector, db_vector) for db_vector in vectors]

    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = [labels[i][0:3]+labels[i][-1] for i in k_nearest_indices]

    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    frequency = Counter(k_nearest_labels).most_common(1)[0][1]

    # return Counter(k_nearest_labels).most_common(k), most_common_label, frequency
    return most_common_label

if __name__ == "__main__":
    # batch_normalization()

    # rectangle_img_np, seg_img = normalization(1, 1, "L")
    # cv2.imshow("segmentation_img", seg_img)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    # cv2.imshow("normalization_img", rectangle_img_np)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    true_predict = 0
    for i in range(1,101):
        for j in range(1,11):
            for suffix in ['L', 'R']:
                img_path = f"../img-test/{i:03}{j:02}_{suffix}.bmp"
                if not os.path.exists(img_path):
                    print(f"Warning: The file {img_path} does not exist.")
                    continue
                true_label = f"{i:03}{suffix}"
                vector = embedd(i, j, suffix)
                predict_label = classify_by_knn(vector, "db.npy")
                if predict_label == true_label:
                        true_predict += 1
                print(f"{img_path} predicted")
    print(f"accuracy: {true_predict}%")



