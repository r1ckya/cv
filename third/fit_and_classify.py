def extract_hog(img):
    
    from cv2 import HOGDescriptor, resize
    
    img_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = HOGDescriptor(img_size, block_size, block_stride, cell_size, nbins)

    return hog.compute(resize(img, img_size)).ravel()

def fit_and_classify(train_x, train_y, test_x):
    from sklearn.svm import SVC
    params = dict(
        C=100,
        random_state=213)
    model = SVC(**params)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    return preds
