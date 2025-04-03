import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from sklearn.model_selection import train_test_split

def augment(n_images_per_class: int, 
            resolution: int, 
            seed: int,
            input_path = "merged-db",
            output_path = "augmented-db"):
    
    rng = np.random.default_rng(seed=seed)
    
    classes = os.listdir(input_path)
    if not os.path.exists(output_path):
            os.mkdir(output_path)
    for Class in classes:
        if not os.path.exists(os.path.join(output_path,Class)):
            os.mkdir(os.path.join(output_path,Class))
    
    n_total_images = len(classes)*n_images_per_class
    X = torch.zeros(n_total_images,3,resolution,resolution)
    Y = torch.zeros(n_total_images)
    transf = transforms.ToTensor()
    
    with open(os.path.join(output_path,"data.txt"),"w") as file:
        i = 0
        for label, Class in enumerate(classes):
            images = os.listdir(os.path.join(input_path,Class))
            n = len(images)
            m = n_images_per_class // n
            n_transf_per_image = [m for i in range(n - n_images_per_class % n)] + [m+1 for i in range(n_images_per_class % n)]
            rng.shuffle(n_transf_per_image)
            
            for image_name,n_transf in zip(images,n_transf_per_image):
                img = Image.open(os.path.join(input_path,Class,image_name)).convert("RGB")
                for j in range(n_transf):
                    new_img, data = produce_image(img, resolution, rng)
                    new_image_name = f"{image_name[:-5]}_{j}.jpeg" 
                    new_img.save(os.path.join(output_path,Class,new_image_name))
                    X[i] = transf(new_img)
                    Y[i] = label
                    i += 1 
                    
                    file.write(new_image_name+"\n")
                    for dat in data:
                        file.write(dat+"\n")
                    file.write("\n")
                    
                    if i % 100 == 0:
                        print(f"class {label+1}/{len(classes)}, total {i}/{n_total_images}")
        
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8)
    torch.save(X_train,os.path.join(output_path,"X_train.pt"))
    torch.save(X_test,os.path.join(output_path,"X_test.pt"))
    torch.save(Y_train,os.path.join(output_path,"Y_train.pt"))
    torch.save(Y_test,os.path.join(output_path,"Y_test.pt"))
            
    
def produce_image(img, resolution, rng):
    transformation_list = [(rotate,True),
                           (crop,True),
                           (hor_flip,False),
                           (ver_flip,False),
                           (blur,True),
                           (brightness,False),
                           (contrast,False),
                           (color,False)]
    
    n_transformations = rng.choice([1,2,3,4],p=[0.2,0.4,0.3,0.1])
    chosen_transformations = rng.choice(list(range(len(transformation_list))),
                                              size=n_transformations,
                                              replace=False)
    
    data = []
    for i in chosen_transformations:
        if transformation_list[i][1]:
            img, dat = transformation_list[i][0](img, rng)
            data += [dat]
    
    img = transforms.functional.center_crop(img,min(img.size))    
    img = img.resize((resolution,resolution))
    
    for i in chosen_transformations:
        if not transformation_list[i][1]:
            img, dat = transformation_list[i][0](img, rng)
            data += [dat]
        
    return [img, data]


def hor_flip(img, rng):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return [img, "Horizontal Flip"]

def ver_flip(img, rng):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return [img, "Vertical Flip"]

def crop(img, rng):
    width, height = img.size
    scale = rng.uniform(0.5,0.75)
    maxOffset = (1-scale)*min(width,height)
    offsetX = rng.uniform(-1,1)
    offsetY = rng.uniform(-1,1)
    size = min(width,height)*scale
    img = img.crop((width/2 + offsetX*maxOffset/2 - size/2, height/2 + offsetY*maxOffset/2 - size/2, width/2 + offsetX*maxOffset/2 + size/2, height/2 + offsetY*maxOffset/2 + size/2))
    return [img, f"Crop: scale = {scale}, offsetX = {offsetX}, offsetY = {offsetY}"]

def rotate(img, rng):
    angle = rng.uniform(-45,45)
    img = img.rotate(angle,expand=False)
    return [img, f"Rotate: angle = {angle}"]

def blur(img, rng):
    value = rng.integers(1,7)
    img = img.filter(ImageFilter.GaussianBlur(radius = value))
    return [img, f"Gaussian Blur: value = {value}"]

def brightness(img, rng):
    value = rng.uniform(0.3,1.7)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(value)
    return [img, f"Brightness: value = {value}"]

def contrast(img, rng):
    value = rng.uniform(0.3,1.7)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(value)
    return [img, f"Constrast: value = {value}"]

def color(img, rng):
    value = rng.uniform(0.3,1.7)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(value)
    return [img, f"Color: value = {value}"]