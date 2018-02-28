import numpy as np
from skimage.transform import pyramid_gaussian

def trim(img, ratio):
    dh, dw = map(lambda x: int(x * ratio), img.shape[:2])
    return img[dh:-dh, dw:-dw]

def split(img, trim_ratio):
    h = img.shape[0] // 3
    return list(map(lambda x: trim(x, trim_ratio),
                    (img[i * h:(i + 1) * h] for i in range(3))))

def mse(a, b):
    return ((a - b) ** 2).mean()

def cross(a, b):
    return -(a * b).sum() / ((a * a).sum() * (b * b).sum()) ** 0.5

def shift2(a, b, x, y, n, scorer=mse):
    best_score, best_dx, best_dy = 2, 0, 0
    for dx in range(-n + x, n + x + 1):
        for dy in range(-n + y, n + y + 1):
            p = -abs(dx) if dx != 0 else a.shape[0]
            q = -abs(dy) if dy != 0 else a.shape[1]
            score = scorer(np.roll(a, (dx, dy), (0, 1))[:p, :q],
                           b[:p, :q])
            if score < best_score:
                best_score, best_dx, best_dy = score, dx, dy
    return best_dx - x, best_dy - y

def align(img, gc):
    h = img.shape[0] // 3
    img = img.astype(np.float16)
    
    gc = np.array(gc)
    rc = gc + (h, 0)
    bc = gc - (h, 0)
    
    bd = np.zeros(bc.shape, dtype=np.int32)
    rd = np.zeros(rc.shape, dtype=np.int32)
    
    if h > 1000:
        k, cnt = 4, 2
        
        pb, pg ,pr = map(lambda x: list(pyramid_gaussian(x, cnt, k)), split(img, 0.33))
        
        for i in range(cnt + 1):
            n = k ** (cnt - i) + 2
            rd *= k
            bd *= k
            
            bd += shift2(pg[cnt - i], pb[cnt - i], bd[0], bd[1], n)        
            rd += shift2(pg[cnt - i], pr[cnt - i], rd[0], rd[1], n)
        
        bc += bd
        rc += rd
        
        b, g, r = split(img, 0.05)
        
        ret = np.dstack((np.roll(r, (-rd), (0, 1)), 
                         g,
                         np.roll(b, (-bd), (0, 1))))
                        
    else:
        n = int(h * 0.05)
        
        b, g ,r = split(img, 0.33)
     
        bd += shift2(g, b, 0, 0, n)
        rd += shift2(g, r, 0, 0, n)
     
        bc += bd
        rc += rd
        
        b, g, r = split(img, 0.05)
        ret = np.dstack((np.roll(r, (rd[0], rd[1]), (0, 1)), 
                         g,
                         np.roll(b, (bd[0], bd[1]), (0, 1))))
                        
    return ret, (bc[0], bc[1]), (rc[0], rc[1])