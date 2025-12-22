import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def legend_shadow_old(fig, ax, legend, d=0.75):

    frame = legend.get_window_extent()
    fig_frame = fig.get_window_extent()

    xmin, ymin = fig.transFigure.inverted().transform((frame.xmin, frame.ymin))
    xmax, ymax = fig.transFigure.inverted().transform((frame.xmax, frame.ymax))

    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                             facecolor='white', edgecolor='white',
                             transform=fig.transFigure, clip_on=False)

    shadow = patches.Shadow(rect,
                            (d/fig.dpi)*fig_frame.ymax/fig_frame.xmax,
                            -(d/fig.dpi),
                            color='k', alpha=1.0, zorder=99)

    ax.add_patch(shadow)

    return None


def legend_shadow_old2(fig, ax, legend, d=1.0, color='k'):

    frame = legend.get_window_extent()
    fig_frame = fig.get_window_extent()

    xmin, ymin = fig.transFigure.inverted().transform((frame.xmin, frame.ymin))
    xmax, ymax = fig.transFigure.inverted().transform((frame.xmax, frame.ymax))

    linewidth = plt.rcParams['axes.linewidth']*0.75

    shift_x = (xmax-xmin)*d*0.015
    shift_y = shift_x/(fig_frame.ymax/fig_frame.xmax)
    rect = patches.Rectangle((xmin+shift_x, ymin-shift_y),
                             xmax-xmin, ymax-ymin,
                             facecolor=color, edgecolor=color,
                             linewidth=linewidth,
                             transform=fig.transFigure,
                             clip_on=False, zorder=99)

    ax.add_patch(rect)

    return None


def legend_shadow(fig, ax, legend, d=0.0, color='k'):
    # Axesの枠線の80%に設定
    # leg.get_frame().set_linewidth(ax_linewidth*2)

    # 凡例の背景パッチを取得
    patch = legend.legendPatch

    # パスエフェクトを設定（影の色、オフセット、アルファなどを指定）
    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(3.5, -3.5),
                             shadow_rgbFace=color,
                             alpha=1.0),
        pe.Normal(),
    ])

    return None
