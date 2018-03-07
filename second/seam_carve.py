
# coding: utf-8

# In[106]:


import numpy as np
from skimage import io
from itertools import accumulate, chain

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
        cur[1:-1] += np.minimum(np.minimum(prev[:-2], prev[1:-1]), prev[2:])
        cur[0] += min(prev[0], prev[1])
        cur[-1] += min(prev[-1], prev[-2])
    return ret

def get_seam(seam_map):
    return np.fromiter(accumulate(chain([seam_map[-1].argmin()], seam_map[-2::-1]),
                                  lambda x, y: x + y[x - 1: x + 2].argmin() - 1 if x > 0 else y[:2].argmin()), 
                       dtype=np.int32) + np.arange(seam_map.shape[0] - 1, -1, -1) * seam_map.shape[1]

def add_seam(img, idxs):
    n, m = img.shape[:2]
    idxs1 = idxs + 1 - (idxs % m == m - 1)
    return np.dstack((np.insert(it, idxs, (it[idxs] + it[idxs1]) // 2).reshape(n, m + 1)
         for it in map(np.ravel, np.dsplit(img, 3))))

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
        mask = np.delete(mask, idxs)
        img = np.delete(img, idxs, axis=2)
    else:
        mask.ravel()[idxs] = 1
        mask = np.insert(mask, idxs, np.ones(n)).reshape(n, m + 1)
        img = add_seam(img, idxs)
    if mode.split()[0] == 'vertical':
        img = img.transpose(1, 0, 2)
        mask = mask.T
        seam_mask = seam_mask.T
    return img, mask, seam_mask


# In[109]:


x = np.arange(3).reshape(1, 1, 3)
a, b, c = np.dsplit(x, 3)


# In[110]:


x


# In[111]:


a


# In[112]:


a[:, :, 0] = 5


# In[113]:


a


# In[114]:


x


# In[116]:


g = np.vstack((a, b))


# In[117]:


g.shape


# In[118]:


g[0, 0, 0] = 4


# In[119]:


x


# In[124]:


g = np.array([1,0, 0, 1])


# In[125]:


g


# In[122]:


np.delete(g.ravel(), 0)


# In[127]:


g == 1

