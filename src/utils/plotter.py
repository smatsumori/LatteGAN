# import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_multilabel_confusion_matrix(
    cmat, labels, save_path, fname='cmat.png',
    n_rows=6, n_columns=10, show=False,
):
    """Plots confusion matrix.

    Parameters
    ----------
    cmat :
        A multilabel confusion matrix instance whose shape is
        (n_class, 2, 2).
    labels :
        A list of labels. A length should be n_classes.
    save_path :
        A path to save confusion matrix.
    fname :
        A filename.
    n_rows :
        The number of rows to plot.
    n_columns :
        The number of colums to plot.
    show :
        If true then show the plot.

    Returns
    ----------
    f :
        matplotlib.figure.Figure.
    file_path :
        A path to a saved img.
    """
    f, axes = plt.subplots(n_rows, n_columns, figsize=(10*3, 6*3))
    axes = axes.ravel()
    for i, cm in enumerate(cmat):
        disp = ConfusionMatrixDisplay(cm, display_labels=[-1, i])
        disp.plot(ax=axes[i], values_format='.4g', cmap=plt.cm.Blues)
        disp.ax_.set_title(f'{labels[i]}')
        if i % 10 != 0:
            disp.ax_.set_ylabel('')
        if i < n_columns * (n_rows - 1):
            disp.ax_.set_xlabel('')
        disp.im_.colorbar.remove()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f.colorbar(disp.im_, ax=axes)

    # remove empty plots
    for j in range(i+1, n_rows*n_columns):
        axes[j].set_axis_off()

    if show:
        plt.show()

    # save confusion matrix
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, fname)
    f.savefig(file_path)
    return f, file_path
