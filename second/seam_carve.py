
# coding: utf-8

# In[51]:


import numpy as np
from skimage import io

def build_ymap(img):
     return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

def grad_norm(ymap):
    grad = np.gradient(ymap)
    grad[0][1:-1, :] *= 2
    grad[1][:, 1:-1] *= 2
    return sum(a ** 2 for a in grad) ** 0.5

def build_seam_map(grad):
    ret = grad.copy()
    for cur, prev in zip(ret[1:], ret[:-1]):
        cur[1:-1] += np.vstack((prev[:-2], prev[1:-1], prev[2:])).min(axis=0)
        cur[0] += min(prev[0], prev[1])
        cur[-1] += min(prev[-1], prev[-2])
    return ret

def rem_seam(img, idxs):
    n, m = img.shape[:2]
    rem = np.hstack((idxs * 3, idxs * 3 + 1, idxs * 3 + 2))
    return np.delete(img, rem).reshape(n, m - 1, 3)

def get_seam(seam_map):
    n = seam_map.shape[0] - 1
    m = seam_map.shape[1]
    j = np.argmin(seam_map[n])
    idxs = [n * m + j]
    n -= 1
    for cur in seam_map[-2::-1]:
        if j == 0:
            if cur[0] > cur[1]:
                j = 1
        else:
            j += np.argmin(cur[j - 1: j + 2]) - 1
        idxs += [n * m + j]
        n -= 1
    return np.array(idxs)

def add_seam(img, idxs):
    n, m = img.shape[:2]
    idxs1 = idxs + 1 - (idxs % m == m - 1)
    return np.dstack((np.insert(it, idxs, (it[idxs] + it[idxs1]) // 2).reshape(n, m + 1)
         for it in map(np.ravel, np.split(img, 3, 2))))

def seam_carve(img, mode='horizontal shrink', mask=None):
    if mask is None:
        mask = np.zeros(img.shape[:2])
    else:
        mask = mask.copy()
    img = img.copy()
    if mode.split()[0] == 'vertical':
        img = img.transpose(1, 0, 2)
        mask = mask.T
    n, m = img.shape[:2]
    seam_mask = np.zeros((n, m))
    ymap = build_ymap(img)
    grad = grad_norm(ymap)
    grad += 2 ** 0.5 * 255 * n * m * mask
    seam_map = build_seam_map(grad)
    idxs = get_seam(seam_map)
    seam_mask.ravel()[idxs] = 1
    if mode.split()[1] == 'shrink':
        mask = np.delete(mask, idxs).reshape(n, m - 1)
        img = rem_seam(img, idxs)
    else:
        mask.ravel()[idxs] = 1
        mask = np.insert(mask.ravel(), idxs, np.ones(n)).reshape(n, m + 1)
        img = add_seam(img, idxs)
    if mode.split()[0] == 'vertical':
        img = img.transpose(1, 0, 2)
        mask = mask.T
        seam_mask = seam_mask.T
    return img, mask, seam_mask


# In[52]:


a = np.arange(5)


# In[58]:


a[-1::-1]

