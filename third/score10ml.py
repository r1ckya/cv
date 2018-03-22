def extract_hog(img):
    
    import numpy as np
    from skimage.transform import resize
    from skimage.util import crop
    from skimage.util import view_as_windows, view_as_blocks
    
    imgh = 30
    cellh = 3
    blockh = 2
    nbins = 9
    max_theta = 180
    
    def normalize(v, eps=1e-5):
        
        ret = v / np.sqrt(np.sum(v ** 2, axis=1) + eps ** 2)[:, np.newaxis]
        #ret = np.minimum(ret, 0.2)
        #ret = ret / np.sqrt(np.sum(ret ** 2, axis=1) + eps ** 2)[:, np.newaxis]
    
        return ret
    
    def build_cells(g_magn, theta):
        
        phi = max_theta // nbins
        
        cells_g_magn = view_as_blocks(g_magn, (cellh, cellh)).reshape(((imgh // cellh) ** 2, cellh ** 2))
        cells_theta = view_as_blocks(theta, (cellh, cellh)).reshape(((imgh // cellh) ** 2, cellh ** 2))
        cells_bin = np.zeros(((imgh // cellh) ** 2, nbins))
        
        for i, (g, t) in enumerate(zip(cells_g_magn, cells_theta)):
            idx = t // phi
            
            ratio = (t - idx * phi) / phi
            cells_bin[i, idx] += g * (1 - ratio)
            idx += 1
            idx[idx == nbins] = 0
            cells_bin[i, idx] += g * ratio
        
        return cells_bin.reshape(((imgh // cellh), (imgh // cellh), nbins))
    
    def build_blocks(cells):
        
        blocks = view_as_windows(cells, (blockh, blockh, nbins), (blockh // 2, blockh // 2, nbins))
        blocks = blocks.reshape((-1, blockh ** 2 * nbins))
        blocks = normalize(blocks)
        #print(len(blocks.ravel()))
        return blocks.ravel()
    
    def calc_gradient(img):
        
        gx = np.empty(img.shape, dtype=np.float32)
        gx[0, :, :] = 0
        gx[-1, :, :] = 0
        gx[1:-1, :] = img[2:, :, :] - img[:-2, :, :]
        
        gy = np.empty(img.shape, dtype=np.float32)
        gy[:, 0, :] = 0
        gy[:, -1, :] = 0
        gy[:, 1:-1, :] = img[:, 2:, :] - img[:, :-2, :]
        
        return gx, gy
    
    if img.dtype.kind == 'u':
        img = img.astype(np.float64) / 255.0

    h, w = img.shape[:2]
    h = int(h * 0.2)
    w = int(w * 0.2)
    img = crop(img, ((h, h), (w, w), (0, 0)))
    img = resize(img, (imgh, imgh))
    #img = np.sqrt(img)
    #img = np.log(img + 1)
    
    gx, gy = calc_gradient(img)
    g_magn = np.hypot(gx, gy)
    g_chmax = g_magn.argmax(axis=2)
    rr, cc = np.meshgrid(np.arange(imgh),
                         np.arange(imgh),
                         indexing='ij',
                         sparse=True)
    
    gx = gx[rr, cc, g_chmax]
    gy = gy[rr, cc, g_chmax]
    g_magn = g_magn[rr, cc, g_chmax]
    
    theta = np.rad2deg(np.arctan2(gy, gx)).astype(np.int32) % max_theta
    
    return build_blocks(build_cells(g_magn, theta))
    
    
def fit_and_classify(train_x, train_y, test_x):
    
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np

    params = dict(
        C=0.013)

    model = LinearSVC(**params)
    model.fit(train_x, train_y)

    preds = model.predict(test_x)
    return preds
