import numpy as np
from matplotlib import pyplot as plt


def concat_jpg(fdir, png_dir, num):
    if num == 64:
        for i in range(64):
            im_path = fdir + png_dir+ '/layer_vis_' + png_dir + \
                            '_f' + str(i) + '_iter' + str(i) + '.jpg'
#             im_path = fdir + png_dir+ '/layer_vis_' + png_dir + \
#                             '_f' + str(i) + '_iter' + str(200) + '.jpg'
            img = plt.imread(im_path)
            plt.subplot(8, 8, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.imshow(img)
        im_path = fdir + png_dir+ '/layer_vis_' + png_dir + '.jpg'
        plt.savefig(im_path, dpi=500)
    else :
        if num == 512:
            for j in range(8):
                for i in range(64):
                    n = j*64+ i
                    im_path = fdir + png_dir+ '/layer_vis_' + png_dir + \
                                    '_f' + str(n) + '_iter' + str(n) + '.jpg'
#                     im_path = fdir + png_dir+ '/layer_vis_' + png_dir + \
#                                     '_f' + str(n) + '_iter' + str(200) + '.jpg'
                    img = plt.imread(im_path)
                    plt.subplot(8, 8, i+1)
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                    plt.imshow(img)
                im_path = fdir + png_dir+ '/layer_vis_' + png_dir + '_'+ str(j) +'.jpg'
                plt.savefig(im_path, dpi=500)
        else:
            return
    
if __name__ == '__main__':
    
    concat_jpg('./output/', 'conv1', 64)
    concat_jpg('./filter/', 'conv1', 64)
    concat_jpg('./output/', 'resblock4_2', 512)
    concat_jpg('./generated/', 'conv1', 64)
    concat_jpg('./generated/', 'resblock4_2', 512)
    
    