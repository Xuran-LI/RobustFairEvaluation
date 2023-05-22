from matplotlib import pyplot

def draw_lines(x, y, names, x_label, y_label, P_title, output_file):
    """
    绘制折线图
    :return:
    """
    fig, axs = pyplot.subplots(1, 1, figsize=(5, 3.5))
    for i in range(len(names)):
        axs.plot(x, y[i], label=names[i])
        # axs[0].set_xlim(0, 2)
    axs.set_xlabel(x_label, fontsize=15)
    axs.set_ylabel(y_label, fontsize=15)
    # axs.xaxis.get_major_formatter().set_powerlimits((0, 1))
    axs.yaxis.get_major_formatter().set_powerlimits((0, 1))
    axs.set_title(P_title, fontsize=15)
    fig.tight_layout()
    pyplot.legend(loc="upper left")
    pyplot.savefig(output_file, bbox_inches='tight')
    pyplot.show()
