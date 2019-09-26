import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Input image')
    plt.imshow(img, cmap='gray')

    a2 = fig.add_subplot(2, 2, 2)
    a2.set_title('Output mask')
    plt.imshow(mask[...,0], cmap='gray')

    a3 = fig.add_subplot(2, 2, 3)
    a3.set_title('Output mask')
    plt.imshow(mask[...,1], cmap='gray')

    b = fig.add_subplot(2, 2, 4)
    b.set_title('Output mask')
    plt.imshow(mask[...,2], cmap='gray')
    plt.show()