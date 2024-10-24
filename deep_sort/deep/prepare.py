import os
from shutil import copyfile

download_path = 'Market1501'
download_path2 = 'Market-1501-v15.09.15'

if not os.path.isdir(download_path):
    if os.path.isdir(download_path2):
        os.system('mv %s %s'%(download_path2, download_path))
    else:
        print('please change the download_path')

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

# query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for _, _, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        ID = name.split('_')
        src_path = os.path.join(query_path, name)
        dst_path = os.path.join(query_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))

# multi-query
query_path = os.path.join(download_path, 'gt_bbox')
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)
    
    for _, _, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name.endswith('.jpg'):
                continue
            ID = name.split('_')
            src_path = os.path.join(query_path, name)
            dst_path = os.path.join(query_save_path, ID[0])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
        
# gallery
gallery_path = os.path.join(download_path, 'bounding_box_test')
gallery_save_path = os.path.join(download_path, 'pytorch/gallery')
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for _, _, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        ID = name.split('_')
        src_path = os.path.join(gallery_path, name)
        dst_path = os.path.join(gallery_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# test
test_path = os.path.join(download_path, 'bounding_box_test')
test_save_path = os.path.join(download_path, 'pytorch/test')
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)

for _, _, files in os.walk(test_path, topdown=True):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        ID = name.split('_')
        src_path = os.path.join(test_path, name)
        dst_path = os.path.join(test_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# train_all
train_path = os.path.join(download_path, 'bounding_box_train')
train_save_path = os.path.join(download_path, 'pytorch/train_all')
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for _, _, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        ID = name.split('_')
        src_path = os.path.join(train_path, name)
        dst_path = os.path.join(train_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# train_val
train_path = os.path.join(download_path, 'bounding_box_train')
train_save_path = os.path.join(download_path, 'pytorch/train')
val_save_path = os.path.join(download_path, 'pytorch/val')
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for _, _, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name.endswith('.jpg'):
            continue
        ID = name.split('_')
        src_path = os.path.join(train_path, name)
        dst_path = os.path.join(train_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = os.path.join(val_save_path, ID[0])
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)