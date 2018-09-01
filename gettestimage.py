import struct
import numpy as np
from PIL import Image
filename='t10k-images.idx3-ubyte' #change
binfile=open(filename,'rb')
buf=binfile.read()

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
#save image
for image in range(0,numImages):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
   #convert type
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
   # fig=plt.figure()
   # plotwindow=fig.add_subplot(111)
   # plt.imshow(im,cmap='gray')
   # plt.show()
    im=Image.fromarray(im)
    im.save('testimage/image_%s.bmp'%image,'bmp') #change path