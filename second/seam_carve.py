
# coding: utf-8

# In[25]:


import numpy as np
from skimage import io

def build_ymap(img):
    return np.dot(img, [0.299, 0.587, 0.114]) 

def grad_norm(ymap):
    grad = np.gradient(ymap)
    #grad[0][1:-1, 1:-1] *= 2
    #grad[1][1:-1, 1:-1] *= 2
    return sum(a * a for a in grad) ** 0.5

def build_seam_map(grad):
    ret = grad.copy()
    for i in range(1, ret.shape[0]):
        ret[i, 0] += ret[i - 1, 0:2].min()
        for j in range(1, ret.shape[1]):
            ret[i, j] += ret[i - 1, j - 1: j + 2].min()
    return ret

def rem_seam(img, idxs):
    n, m = img.shape[:2]
    rem = np.hstack((idxs * 3, idxs * 3 + 1, idxs * 3 + 2))
    return np.delete(img, rem).reshape(n, m - 1, 3)

def get_seam(seam_map):
    idxs = np.array([], dtype=np.int32)
    n = seam_map.shape[0] - 1
    m = seam_map.shape[1]
    j = np.argmin(seam_map[n])
    while n >= 0:
        if j == 0:
            if seam_map[n, 0] < seam_map[n, 1]:
                j = 1
        else:
            j += np.argmin(seam_map[n, j - 1: j + 2]) - 1
        idxs = np.append(idxs, n * m + j)
        n -= 1
    return idxs

def add_seam(img, idxs):
    n, m = img.shape[:2]
    idxs1 = idxs + 1 - (idxs == n * m - 1)
    return np.dstack((np.insert(it, idxs, (it[idxs] + it[idxs1]) // 2).reshape(n, m + 1)
         for it in map(np.ravel, np.split(img, 3, 2))))

def seam_carve(image, mode='horizontal shrink', mask=None):
    img = image.copy()
    if mask is None:
        msk = np.zeros(img.shape[:2])
    else:
        msk = mask.copy()
    if mode.split()[0] == 'vertical':
        img = img.transpose(1, 0, 2)
        msk = msk.T
    n, m = msk.shape
    seam_mask = np.zeros(msk.shape)
    ymap = build_ymap(img)
    grad = grad_norm(ymap)
    #print(grad.dtype)
    grad += 2 ** 0.5 * 255 * msk * n * m
    seam_map = build_seam_map(grad)
    idxs = get_seam(seam_map)
    if mode.split()[1] == 'shrink':
        img = rem_seam(img, idxs)
        seam_mask.ravel()[idxs] = 1
        msk = np.delete(msk, idxs).reshape(n, m - 1)
    else:
        img = add_seam(img, idxs)
        msk.ravel()[idxs] = 1
        seam_mask.ravel()[idxs] = 1
        #seam_mask.ravel()[idxs + 1] = 1
        msk = np.insert(msk.ravel(), idxs, np.ones(n)).reshape(n, m + 1)
    if mode.split()[0] == 'vertical':
        img = img.transpose(1, 0, 2)
        msk = msk.T
        seam_mask = seam_mask.T
    return img, msk, seam_mask

