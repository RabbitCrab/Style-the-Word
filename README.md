# **Style the Word**
:link: https://github.com/RabbitCrab/Style-the-Word


## **How to run the code**
This project works module by module. You are required to git clone a project with the instructions and also install some of the requirements. Some of the module you might choose not to use while giving the input as required.


## **What is given / hand-in**
* The synthetic dataset: `./cg_project/ShapeMatchingGAN/imgs/` & `./cg_project/gan_style/data/`
* The selected dataset from ICDAR2013: `./cg_project/gan_style/data/`
* The sakura style pretrained weight: `./cg_project/ShapeMatchingGAN/save/sakura-GT.ckpt` & `./cg_project/ShapeMatchingGAN/save/sakura-GS.ckpt`
* The CycleGan module pretrained weight: `./cg_project/gan_style/weight/sakura/`
* `./cg_project/randomly_split.py`
* `./cg_project/ShapeMatchingGAN/src/test2.py`
* `./cg_project/gan_style/neural_style.py`
* `./cg_project/gan_style/cyclegan.py`


## **Requirements**
**Prerequisites:**
* python 3.6
* pytorch 1.1.0
* matplotlib
* scipy
* Pillow
* opencv
* easyocr (optional)


## **Folder Organization**
```
-cg_project
|---ShapeMatchingGAN
|   |---imgs
|   |   |---1
|   |---save
|   |   |---sakura-GT.ckpt
|   |   |---sakura-GS.ckpt
|   |---output
|   |   |---output.png
|   |---src
|   |   |---test2.py
|---gan_style
|   |---data
|   |   |---train_normal
|   |---weight
|   |   |---sakura
|   |   |   |----G_A.pth
|   |---neural_style.py
|   |---cyclegan.py
|---randomly_split.py

```


## **Step by step**


### **Step 1 (Optional)**
Given an input image, localise the word using text detector. It is optional to have this step. If not, directly given a cropped text image will work also.


#### **Text detector.**
Any scene text detector will do. For example:

```
pip install easyocr
```

Usage:

```
import easyocr
reader = easyocr.Reader(['en'])
result = reader.readtext('word_1.jpg')
```

For character detector, can refer to this [MultiLingual-OCR](https://github.com/RabbitCrab/MultiLingual-OCR).


### **Step 2**
For artistry render text. Also refer to [this](https://github.com/VITA-Group/ShapeMatchingGAN).

```
git clone https://github.com/TAMU-VITA/ShapeMatchingGAN.git
cd ShapeMatchingGAN/src
```

* For sakura pretrained model, please obtain from the hand-in file. Else, refer to the official page.
* Place the pretrained model to `../save/` (as shown in folder organization).


#### **Testing example**
**Single image**
* Run
```
python test.py --scale 0.0 --structure_model ../save/sakura-GS.ckpt --texture_model ../save/sakura-GT.ckpt --gpu
```

* For black-and-white text image, use option `--text_type 1`.

**Image folder**
* Modify `options.py`, add the following after `line 8`.
```
self.parser.add_argument('--text_dir', type=str, default='../imgs/', help='path of the text image directory')
```

* Run
```
python test.py --scale 0.0 --structure_model ../save/sakura-GS.ckpt --texture_model ../save/sakura-GT.ckpt --gpu --text_dir ../imgs/1/
```

#### **Training example**
* Please refer to the official page.


### **Step 3**
Randomly split output images from step 2 for train and validation.
* Modify `line 5` and `line 6` to the input source directory and output destination directory.
* Modify `line 27` and `line 34` if needed.

* Run
```
cd ./CG_Project/
python randomly_split.py
```

### **Step 4**
Neural style transfer.
* Modify `line 52` and `line 53` for original and style images.
* Modify `line 104` for output images.

* Run
```
cd ./CG_Project/gan_style/
python neural_style.py
```


### **Step 5**
CycleGAN.
* Modify `line 35` if needed (base directory for data).

#### **Test**
* Modify `line 60` for output images.
* Modify `line 484` and `line 485` for model path.
* Modify `line 488` and `line 490` for test subdirectory.

* Run
```
python cyclegan.py --mode 0
```


#### **Train**
* Modify `line 84` for output images.
* Modify `line 310`, `line 313`, `line 317` and `line 319` for train and validation subdirectory.
* Modify `line 325` and `line 326` for different output image.
* modify `line 471 to line 477` for model name.

* Run
```
python cyclegan.py --mode 1
```


## Contact
Feel free to drop an email if any question faced.
:email: jiaying.ee10@nycu.edu.tw


## Citation / References
1. [ShapeMatchingGAN](https://github.com/VITA-Group/ShapeMatchingGAN)
2. [EasyOCR](https://github.com/JaidedAI/EasyOCR)
3. [Neural Style Transfer](https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa)
4. [CycleGAN](https://ltquesada.medium.com/your-first-cyclegan-using-pytorch-85546dfe6317)
