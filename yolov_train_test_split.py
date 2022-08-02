#%% Kütüphanelerin Yüklenmesi

from IPython.display import Image  
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

#%% Xml Dosyası Hakkında Bilgi Alınması

# def extract_info_from_xml(xml_file):
#     root = ET.parse(xml_file).getroot()
    
#     info_dict = {}
#     info_dict['bboxes'] = []

#     for elem in root:
#         if elem.tag == "filename":
#             info_dict['filename'] = elem.text
            
#         elif elem.tag == "size":
#             image_size = []
#             for subelem in elem:
#                 image_size.append(int(subelem.text))
            
#             info_dict['image_size'] = tuple(image_size)
        
#         elif elem.tag == "object":
#             bbox = {}
#             for subelem in elem:
#                 if subelem.tag == "name":
#                     bbox["class"] = subelem.text
                    
#                 elif subelem.tag == "bndbox":
#                     for subsubelem in subelem:
#                         bbox[subsubelem.tag] = int(subsubelem.text)            
#             info_dict['bboxes'].append(bbox)
    
#     return info_dict


# print(extract_info_from_xml('annotations/road4.xml'))
# class_name_to_id_mapping = {"trafficlight": 0,
#                            "stop": 1,
#                            "speedlimit": 2,
#                            "crosswalk": 3}

# #%% Xml Dosyasını Txt Dosyasına Çevirme

# def convert_to_yolov5(info_dict):
#     print_buffer = []
    
#     for b in info_dict["bboxes"]:
#         try:
#             class_id = class_name_to_id_mapping[b["class"]]
#         except KeyError:
#             print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
#         b_center_x = (b["xmin"] + b["xmax"]) / 2 
#         b_center_y = (b["ymin"] + b["ymax"]) / 2
#         b_width    = (b["xmax"] - b["xmin"])
#         b_height   = (b["ymax"] - b["ymin"])
        
#         image_w, image_h, image_c = info_dict["image_size"]  
#         b_center_x /= image_w 
#         b_center_y /= image_h 
#         b_width    /= image_w 
#         b_height   /= image_h 
        
#         print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
#     save_file_name = os.path.join("annotations", info_dict["filename"].replace("png", "txt"))
    
#     print("\n".join(print_buffer), file= open(save_file_name, "w"))

# annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "xml"]
# annotations.sort()

# for ann in tqdm(annotations):
#     info_dict = extract_info_from_xml(ann)
#     convert_to_yolov5(info_dict)
# annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]

# random.seed(0)

# class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

# #%% Xml Dosyalarını Silme

# def remove_xml():

#     cwd = os.chdir("annotations/")
#     files = os.listdir(cwd)
#     txt_files = glob.glob('*.xml')
#     for file in txt_files:
#         os.remove(file)
#     files = os.listdir(cwd)
#     txt_files = glob.glob('*.xml')

# remove_xml()

# # Bu işlemi yaptıktan annotations dosyasında olacaksınız. Tekrardan ana dizine gitmeniz gerekecektir. Yoksa hata alacaksınız.

# #%% Veri Setini %80-%10-%10 Train-Test-Val Olarak Bölme

images = [os.path.join('images', x) for x in os.listdir('images')]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
images.sort()
annotations.sort()
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

#%% Yolov5 Formatında Veri Setini Ayalarma

root_path = 'images/'
folders = ['train','test/','val']
for folder in folders:
    os.makedirs(os.path.join(root_path,folder))
    
root_path = 'annotations/'
folders = ['train','test/','val']
for folder in folders:
    os.makedirs(os.path.join(root_path,folder))    

#%% Resim ve Labelleri Dosyalara Taşıma

def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')
