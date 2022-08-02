#### clone repo
 1.``` git clone https://github.com/WongKinYiu/yolov7.git```

2. ```cd yolov7/```
3. ```!pip install requirements.txt```


 4. script for dividing data into train,test & validation
```python yolov_train_test_split.py```


create customdata folder inside yolov7
data set download link :[customdata](https://drive.google.com/drive/folders/1u4IL2sGy2Hh84Xp3YyLENsd1SH437lea?usp=sharing)

<img width="203" alt="image" src="https://user-images.githubusercontent.com/62583018/178202558-d7cd75e5-f906-4473-a5c8-a5d148faa277.png">



-inside images paste all train & valid images

-inside labels all labels of train & valid image


. go insiode data folder & create custom_data.yaml
& add path of image & labels (i.e .txt)

```
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./customdata/train.txt  
val: ./customdata/valid.txt  




# number of classes
nc: 3

# class names
names: ['pistol','rifle','revolver']
```

for creating train.txt & val.txt use 
(it contains the path of images )

!python [genrate_test.py]()

<img width="309" alt="image" src="https://user-images.githubusercontent.com/62583018/178203880-1036cfcf-e4e1-416e-bc6c-3db2946cf4f9.png">



5. go inside cfg/training & select any .yaml file 
do changes such as no classes
<img width="214" alt="image" src="https://user-images.githubusercontent.com/62583018/178204356-ed50ae03-c096-4a85-9e1b-c034a83fc6b7.png">


6. start training using

```!python train.py --workers 8 --device 0 --batch-size 16 --data data/custom_data.yaml --img 640 640 --cfg cfg/training/yolov7_custom_3.yaml --weights '' --name yolov7xcustom3 --hyp data/hyp.scratch.p5.yaml --epochs 200```



7. inference on video/image 
```python yolov7_inference.py ```  # add alll file path inside this file

```python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg```
