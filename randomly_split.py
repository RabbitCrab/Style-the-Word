import os
import shutil
import random

img_dir = './ShapeMatchingGAN/output/sakura/'
output_dir = './gan_style/data/'

done = 0
final_count = 0

for i in range(5):
    img_source_dir = img_dir + str(i + 1) + '/'
    img_list = []
    count = 0
    for files in os.listdir(img_source_dir):
        img_name = img_source_dir + files
        count += 1
        img_list.append(files)

    print(count)
    val_num = int(count * 0.2)
    print(val_num)
    for j in range(val_num):
        now_len = len(img_list)
        random_num = random.randint(0, now_len - 1)
        source_file = img_source_dir + img_list[random_num]
        dest_file = output_dir + 'val_sakura/' + img_list[random_num]
        shutil.copy(source_file, dest_file)
        img_list.pop(random_num)
        final_count += 1
    
    for rest in img_list:
        source_file = img_source_dir + rest
        dest_file = output_dir + 'train_sakura/' + rest
        shutil.copy(source_file, dest_file)
        final_count += 1
    
    done += 1
    

print(done)
print(final_count)