# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import cv2
import numpy as np

# %%
im =  cv2.imread('elephant.jpg') #Reads an image into BGR Format

im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
original_shape = im.shape
print(im.shape)

# %%
plt.imshow(im) # as RGB Format
plt.show()

# %%
# Flatten Each channel of the Image
all_pixels  = im.reshape((-1,3))
print(all_pixels.shape)

# %%
from sklearn.cluster import KMeans

# %%
dominant_colors = 4

km = KMeans(n_clusters=dominant_colors)
km.fit(all_pixels)

# %%
centers = km.cluster_centers_

# %%
centers = np.array(centers,dtype='uint8')

# %%
print(centers)

# %%
i = 1

plt.figure(0,figsize=(8,2))


colors = []

for each_col in centers:
    plt.subplot(1,4,i)
    plt.axis("off")
    i+=1
    
    colors.append(each_col)
    
    #Color Swatch
    a = np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    plt.imshow(a)
    
plt.show()

# %%
new_img = np.zeros((330*500,3),dtype='uint8')

print(new_img.shape)

# %%
colors

# %%
km.labels_

# %%
for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]
    
new_img = new_img.reshape((original_shape))
plt.imshow(new_img)
plt.show()

# %%
