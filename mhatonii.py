#!/usr/bin/env python3

import os
import SimpleITK  as sitk
import shutil 
datasetMHA = './Dataset_BRATS2015'
datasetNII = './Dataset_BRATS2015NII'

for X in os.listdir(datasetMHA):
    xx = os.path.join(datasetMHA, X)
    out = os.path.join(datasetNII, X)
    for Y in os.listdir(xx):
        yy = os.path.join(xx, Y)
        out1 = os.path.join(out, Y)
        for image_folder in os.listdir(yy):
            mri_path = os.path.join(yy, image_folder)
            o1 = os.path.join(out1, image_folder)
            
            for modular_folder in os.listdir(mri_path):
            
                mod_path = os.path.join(mri_path, modular_folder)
                o2 = os.path.join(o1, modular_folder)
                os.makedirs(o2, exist_ok=True)    
                
                for mod_images in os.listdir(mod_path):
                    
                    input_image_path = os.path.join(mod_path, mod_images)
                    
                    if mod_images.split(".")[-1] == 'mha':
                        
                        mod_images = mod_images[0:-3] + 'nii'
                        o3 = os.path.join(o2, mod_images)
                
                        image = sitk.ReadImage(input_image_path) 
                        print(o3)
                        sitk.WriteImage(image, o3)
                    
                    else:
                        
                        o3 = os.path.join(o2, mod_images)
                        shutil.copyfile(input_image_path, o3)
                    
            
            