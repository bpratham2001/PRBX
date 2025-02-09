import pandas as pd
#from pathlib import Path
#import os
import tensorflow as tf
import tensorflow_addons as tfa
from random import randrange as rr
from random import uniform as ru
#tf.compat.v1.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
#abs_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\github\\hmb1\\" #local
#new_path = "C:\\Users\\Pratham B\\Desktop\\UNI_Y3\\PRBX\\github\\hmb1_L1\\" #local
abs_path = "/shared/storage/cs/studentscratch/pb1028/new_venv/hmb1/"  #gpu
new_path = "/shared/storage/cs/studentscratch/pb1028/new_venv/hmb1_L1/"  #gpu


""" other ops
tf.image - adjust_contrast, adjust_gamma, adjust_saturation
"""

# probabilities 1 - 40% chance to be augmented - equal probability for each type of augmentation - p = (0.4/6)
p = (0.4/6)
# probabilities 2 - 50% chance to be augmented - equal probability for each type of augmentation - p = (0.5/6)
#p = (0.5/6)
# probabilities 3 - 75% chance to be augmented - equal probability for each type of augmentation - p = (0.75/6)
#p = (0.75/6)

def rand_L1(input_images, probs): # augments a list of images with the same transformation (only 1 augmentation per list of images)
    output_images = []
    op = rr(6)
    if ru(0.0, 1.0) < probs:
        rads = ru(-0.78, 0.78) # 0.78 rads => approx 45 degrees
        factor = ru(-1.0, 1.0)
        box = rr(128)
        while box % 2 != 0:
            box = rr(128)
        zoom_range = (ru(0.0, 2.5), ru(0.0, 2.5))
        intensity = ru(0.0, 200.0)
        delta = ru(-0.5, 0.5)
        for i in input_images:
            if op == 0:
                # rotate - correlates between cams
                output_images.append(tfa.image.rotate(i, rads))
            elif op == 1:
                #shear - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, intensity), tf.float64))
            elif op == 2:
                #zoom - does not correlate
                output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, zoom_range), tf.float64))
            elif op == 3:
                #hue - correlates
                output_images.append(tf.image.adjust_hue(i, factor))
            elif op == 4:
                #occlude - does not correlate
                output_images.append(tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (box, box))))
            elif op == 5:
                #brightness - correlates
                output_images.append(tf.clip_by_value(tf.image.adjust_brightness(i, delta), 0.0, 1.0))
            else:
                output_images.append(i)
        return output_images
    else:
        return input_images

#df = pd.read_csv("interpolated.csv")
df_train = pd.read_csv("interpolated_train.csv", header=None)
#df_test = pd.read_csv("interpolated_test.csv", header=None)

lst = df_train[5].to_list()

idx = 0
counter = 1
inputs = []
names = []
outputs = []
while idx < len(lst):
    name = str(lst[idx]) #str(lst[idx]).replace("/", "\\")
    names.append(name)
    img_raw = tf.io.read_file(abs_path + str(lst[idx])) #tf.io.read_file(abs_path + str(lst[idx]).replace("/", "\\"))
    img = tf.io.decode_image(img_raw)
    img = tf.image.convert_image_dtype(img, tf.float64)
    inputs.append(img)
    if idx > 0 and counter == 3:
        outputs += rand_L1(inputs, p)
        inputs = []
        counter = 0
    idx += 1
    counter += 1

# save tensors as images to folder
if len(names) != len(outputs):
    print("lst LEN ERROR")
else:
    print("saving images to " + new_path + " ...")

for i in range(len(names)):
    #img = Image.fromarray(outputs[i].numpy())
    #img.save(new_path+names[i])
    #filename = Path("C:/Users/Pratham B/Desktop/UNI_Y3/PRBX/github/hmb1_L1")
    print(new_path + names[i])
    tf.keras.utils.save_img((new_path + names[i]), outputs[i].numpy())
    #tf.keras.utils.save_img(os.path.join(filename, names[i]), outputs[i].numpy())
print("OK")

"""
def rand_L1_test(input_images, p):
    output_images = []
    rads = ru(-0.79, 0.79) # 0.79 rads => 45-46 degrees
    factor = ru(-1.0, 1.0)
    sharp = ru(0.0 ,0.01)
    box = rr(128)
    while box % 2 != 0:
        box = rr(128)
    zoom_range = (ru(0.0, 2.5), ru(0.0, 2.5))
    intensity = ru(0.0, 200.0)
    delta = ru(-0.5, 0.5)
    for i in input_images:
        if p == 0:
            # rotate - correlates between cams
            output_images.append(tfa.image.rotate(i, rads))
        elif p == 1:
            #shear - does not correlate
            output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_shear(i, intensity), tf.float64))
        elif p == 2:
            #zoom - does not correlate
            output_images.append(tf.convert_to_tensor(tf.keras.preprocessing.image.random_zoom(i, zoom_range), tf.float64))
        elif p == 3:
            #hue - correlates
            output_images.append(tf.image.adjust_hue(i, factor))
        elif p == 4:
            #occlude - does not correlate
            output_images.append(tf.squeeze(tfa.image.random_cutout(tf.expand_dims(i, axis=0), (box, box))))
        elif p == 5:
            #brightness - correlates
            output_images.append(tf.clip_by_value(tf.image.adjust_brightness(i, delta), 0.0, 1.0))
        else:
            #sharpness - is excessive always - correlates
            #print(i)
            #print(tfa.image.sharpness(i, 0.001))
            output_images.append(tfa.image.sharpness(i, 1.0))
    return output_images
"""
