
import os, sys, time, multiprocessing
import ants, shutil
from intensity_normalization.normalize import nyul

PROCESSED_DIR = '../pp/Train'
HGG_DIR = '../Dataset_BRATS2015NII/Train/HGG'
LGG_DIR = '../Dataset_BRATS2015NII/Train/LGG'
# HGG_DIR = '../DataNII/Train/HGG'
# LGG_DIR = '../DataNII/Train/LGG'
TEMP_DIR = '../pp/TEMP'

def createDir(name):
    if not os.path.exists(name):
        try:
            os.makedirs(name)
            os.chmod(name, 0o777)
        except Exception() as e:
            print(e)
            print("Could not create new directory")
            sys.exit(1)

def getPath(newDir, name, module):
    if 'T1c' in module:
        return os.path.join(newDir, name, name+'_t1c.nii')
    if 'T1' in module:
        return os.path.join(newDir, name, name+'_t1.nii')
    if 'T2' in module:
        return os.path.join(newDir, name, name+'_t2.nii')
    if 'Flair' in module:
        return os.path.join(newDir, name, name+'_flair.nii')
    return os.path.join(newDir, name, name+'_seg.nii')

def n4_bfc(param): # even for seg???????????????????
    INP_DIR = param[0]
    pat = param[1]
    out_pat = pat + ('_HGG' if 'HGG' in INP_DIR else '_LGG' )
    OUT_DIR = param[2]
    pat_path = os.path.join(INP_DIR, pat)
    createDir(os.path.join(OUT_DIR, out_pat))
    for mod in os.listdir(pat_path):
        temp = os.listdir(os.path.join(pat_path, mod))
        path = os.path.join(pat_path, mod, temp[0] if temp[0][-3:]=='nii' else temp[1])
        img = ants.image_read(path)
        final = getPath(OUT_DIR, out_pat, mod)
        if not 'seg' in final:
            img = ants.n4_bias_field_correction(img)
        ants.image_write(img, final, ri=False)

def norm(param):
    INP_DIR = param[0]
    pat = param[1]
    org = os.path.join(INP_DIR, pat)
    temp = os.path.join(param[2], pat)
    createDir(temp)
    for x in os.listdir(org):
        if not 'seg' in x:
            shutil.move(os.path.join(org, x), temp)
    nyul.nyul_normalize(temp, output_dir=org)

if os.path.exists(PROCESSED_DIR):
    print('ProcessedData directory already exists. Continuing will overwrite that directory. Are you sure(y/n): ', end='')
    response = input().strip()
    if not response.lower() in ['y', 'yes']:
        print("Preprocessing is halted")
        sys.exit()
    shutil.rmtree(PROCESSED_DIR)

createDir(PROCESSED_DIR)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# N4 Bias feild correction
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print('Starting N4 Bias Field Correction')
start = time.time()
with multiprocessing.Pool() as p:
    p.map(n4_bfc, [(HGG_DIR, x, PROCESSED_DIR) for x in os.listdir(HGG_DIR)])
print('HGG done')
mid = time.time()
print('n4 HGG time', mid-start)
start = mid
with multiprocessing.Pool() as p:
    p.map(n4_bfc, [(LGG_DIR, x, PROCESSED_DIR) for x in os.listdir(LGG_DIR)])
print('LGG Done')
mid = time.time()
print('n4 LGG time', mid-start)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Intensity Normalization
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

print("-------------------------------------------------------------")
print("-------------------------------------------------------------")
print('starting Internsity Normalization')
createDir(TEMP_DIR)
start = mid
with multiprocessing.Pool() as p:
    p.map(norm, [(PROCESSED_DIR, pat, TEMP_DIR) for pat in os.listdir(PROCESSED_DIR)])
mid = time.time()
shutil.rmtree(TEMP_DIR)
print('time', mid-start)

print('Normalization Done')
print('Preprocessing compleated successfuly')