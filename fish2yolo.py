import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import cv2
from sklearn import preprocessing

names_class = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp',
'Striped Red Mullet', 'Trout']

def change_label(path_img, label):
    for i, filename in enumerate(os.listdir(path_img)):
        old = "{}/{}".format(path_img, filename)
        new = "{}/{}_{}.png".format(path_img, label, str(i))
        first_filename = filename.split("_")[0]
        if first_filename != label:
            os.rename(old, new)

def split(src_img):
    img_array = []
    label_array = []

    for i, name in enumerate(names_class):
        class_path = os.path.join(src_img, name, name)
        label_path = os.path.join(src_img, name, f"{name} Label")
        for file in os.listdir(class_path):
            file_title = file.split(".")[0]
            file_txt = file_title+".txt" # ex: "halo.jpg" --> "halo"

            img_array.append((file_title, name, os.path.join(class_path,file), os.path.join(label_path, file_txt)))
            label_array.append(i)

    X_main, X_test, y_main, y_test = train_test_split(
                                        img_array,
                                        label_array,
                                        test_size=0.2,
                                        random_state=42
                                        )
    X_train, X_val, y_train, y_val = train_test_split(
                                        X_main,
                                        y_main,
                                        test_size=0.125,
                                        random_state=42
                                    )


    print(f'Total data : {len(img_array)}')
    print(f'Train size : {len(X_train)} | {len(X_train)/len(img_array)}%')
    print(f'Test size  : {len(X_test)} | {len(X_test)/len(img_array)}%')
    print(f'Val size   : {len(X_val)} | {len(X_val)/len(img_array)}%')
    print("Split data success!")
    return X_train, X_test, X_val, y_train, y_test, y_val

def adjust_box(shape, box_array):
    x_center = (box_array[0]+box_array[1])/2
    y_center = (box_array[2]+box_array[3])/2
    width_box = box_array[1]-box_array[0]
    height_box = box_array[3]-box_array[2]
    x,y,w,h = x_center/shape[1], y_center/shape[0], width_box/shape[1], height_box/shape[0]
    # Return the value after normalize 0-1
    return x,y,w,h

def write_label(label, label_path, id_img, bb):
    with open(os.path.join(label_path, f'{id_img}.txt'),'w') as f:
        f.write("{} {} {} {} {}".format(names_class.index(label), *bb))

def bounding_box(label, label_path, gt_path):
    
    for filename in tqdm(os.listdir(gt_path)):
        try: 
            path_img = os.path.join(gt_path, filename)
            image = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
            id_image = filename.split(".")[0]

            height, width = image.shape
            # y_coord, x_coord = np.nonzero(image)
            y_coord, x_coord = image.nonzero()

            x_min = x_coord.min()
            x_max = x_coord.max()
            y_min = y_coord.min()
            y_max = y_coord.max()
  
            bb = adjust_box([height, width], [x_min, x_max, y_min, y_max])
  

            write_label(label, label_path, id_image, bb)

        except Exception as e:
            print(f'[Warning]: {e} Convert image with name {filename} is failed!')

def copy_file(X, split_folder, split_sub_folder):
    images_sub_folder = os.path.join(split_folder,'images', split_sub_folder)
    label_sub_folder = os.path.join(split_folder, 'labels', split_sub_folder)

    os.makedirs(images_sub_folder, exist_ok=True)
    os.makedirs(label_sub_folder, exist_ok=True)

    print("Copying file that has splitted...")
    for elem in tqdm(X):
        shutil.copy(elem[2], os.path.join(images_sub_folder, f"{elem[1]}_{elem[0]}.png"))
        shutil.copy(elem[3],os.path.join(label_sub_folder, f"{elem[1]}_{elem[0]}.txt"))
    print("Copying succeed")

        





if __name__=="__main__":

    src = "../datasets/Fish/Fish_Dataset/Fish_Dataset"
    for label in names_class:
        dataset_path = os.path.join(src, label, label)
        label_path = os.path.join(src, label, f"{label} Label")
        # change_label(dataset_path, label)
        gt_path = '{} GT'.format(dataset_path)
        label_path = '{} Label'.format(dataset_path)

        os.makedirs(label_path, exist_ok=True)

        bounding_box(label, label_path, gt_path)
        

    X_train, X_test, X_val, y_train, y_test, y_val = split(src)
    split_folder = "../datasets/Fish/Fish_split"
    os.makedirs(split_folder, exist_ok=True)

    print(X_train)
    # for name in names_class:
    #     path_image = os.path.join(src, name, name)
    #     path_label = os.path.join(src, name, f"{name} Label")
    copy_file(X_train, split_folder, "train")
    copy_file(X_test, split_folder, "test")
    copy_file(X_val, split_folder, "val")







