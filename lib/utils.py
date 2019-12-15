
import matplotlib.pyplot as plt


def show_image_compare(orig, result, title=''):
    figure = plt.figure()
    figure.canvas.set_window_title(title)
    figure.add_subplot(1, 2, 1)

    plt.imshow(orig)
    figure.add_subplot(1, 2, 2)
    plt.imshow(result)
    plt.show(block=True)
