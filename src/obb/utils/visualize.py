import matplotlib.pyplot as plt


def plot_obb(img, obb):
    fig, axes = plt.subplots()
    axes.imshow(img)

    for box in obb:
        box_corners = [(float(box[2 * i]), float(box[2 * i + 1])) for i in range(4)]
        axes.add_patch(plt.Polygon(box_corners, ec="b", alpha=0.3))

    axes.axis('off')
    plt.show()
