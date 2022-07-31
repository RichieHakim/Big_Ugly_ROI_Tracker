import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

def display_toggle_image_stack(images, clim=None):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imshow_FOV = ax.imshow(
        images[0],
#         vmax=clim[1]
    )

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        imshow_FOV.set_data(images[i_frame])
        imshow_FOV.set_clim(clim)


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def display_toggle_2channel_image_stack(images, clim=None):

    fig, axs = plt.subplots(1,2 , figsize=(14,8))
    ax_1 = axs[0].imshow(images[0][...,0], clim=clim)
    ax_2 = axs[1].imshow(images[0][...,1], clim=clim)

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        ax_1.set_data(images[i_frame][...,0])
        ax_2.set_data(images[i_frame][...,1])


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));

