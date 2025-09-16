import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.colors as mplC
from matplotlib.ticker import LogLocator

import numpy as np
from datetime import datetime


class ShareXaxis():
    def __init__(self):
        self.fontsize = 20
        # self.fontname = 'Liberation Sans Narrow'
        self.fontname = 'DejaVu Sans'   # default
        self.universalcolor = False
        self.hspace = 0   # default 0.2
        self.wspace = 0   # default 0.3

        return None

    def __enter__(self):
        print("前処理")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("後処理")

    def close(self):
        plt.clf()
        plt.close()

    # Globaly used
    def set_default(self):
        """
        引数なし。Matplotlibのデフォルト書式で、1枚絵をさくっと作りたいときに使う。 \\
        使い方: \\
        `F = ShareXaxis()` \\
        `F.fontsize = 15`  \\
        `F.fontname = 'Liberation Sans Narrow'` \\
        `F.set_default()` \\
        この先でイニシャライズしなくてOK
        """
        self.nrows = 1  # required
        self.ncols = 1  # required
        self.figsize = (6, 4)  # required
        self.dpi = 100  # required
        self.height_ratios = None  # required
        self.rcParamsUpdate(1.0)

        return None

    # Globaly used
    def set_figparams(
            self,
            nrows: int = 1,
            ncols: int = 1,
            figsize=(6, 4),
            dpi='M',
            height_ratios=None,
            ticksize=1.0,
            thickness=1.0,):

        self.nrows = nrows  # required
        self.ncols = ncols  # required
        self.figsize = figsize  # required

        if dpi == 'M':
            self.dpi = 150  # required
        elif dpi == 'S':
            self.dpi = 100
        elif dpi == 'L':
            self.dpi = 227
        elif dpi == 'XL':
            self.dpi = 326
        else:
            self.dpi = dpi

        self.height_ratios = None  # required
        if height_ratios is not None:
            self.height_ratios = height_ratios

        self.rcParamsUpdate(ticksize, thickness)

        return None

    # Locally used
    def rcParamsUpdate(self, ticksize=1.0, thickness=1.0):
        plt.rcParams.update({'font.sans-serif': self.fontname,
                            'font.family': 'sans-serif',
                             'mathtext.fontset': 'custom',
                             'mathtext.rm': self.fontname,
                             'mathtext.it': self.fontname+':italic',
                             'mathtext.bf': self.fontname+':bold',
                             })
        params = {
            # 'lines.markersize': 1,
            # 'lines.linewidth': 1,
            'axes.titlesize': self.fontsize,
            'axes.titlepad': 9.0,
            'axes.labelsize': self.fontsize,
            'axes.linewidth': 1.4*thickness,
            'xtick.major.size': 7.5*ticksize,
            'xtick.minor.size': 4.5*ticksize,
            'xtick.major.width': 1.4*thickness,
            'xtick.minor.width': 0.9*thickness,
            'ytick.major.size': 7.5*ticksize,
            'ytick.minor.size': 4.5*ticksize,
            'ytick.major.width': 1.4*thickness,
            'ytick.minor.width': 0.9*thickness,
        }
        plt.rcParams.update(params)
        plt.rcParams['font.size'] = self.fontsize

        return None

    # Globaly used
    def set_xaxis(
            self,
            label='$x$',
            labelcolor='k',
            min=0,
            max=1,
            ticks=[0, 1],
            ticklabels=[0, 1],
            minor_num=2,
            format='%H:%M',
            xscale='linear',):

        if type(min) is datetime:
            print('time format 0')
            if type(label) is str:
                if self.ncols == 1:
                    self.plot_xaxis_timeformat(
                        self.fig,
                        self.ax,
                        label=label,
                        labelcolor=labelcolor,
                        min=min,
                        max=max,
                        ticks=ticks,
                        ticklabels=ticklabels,
                        minor_num=minor_num,
                        format=format,)
                    print('time format 1')
                else:
                    print('Sorry, not compatible so far.')

            elif type(label) is list:
                print('time format 2')
                if self.ncols == 1:
                    self.plot_xaxis_timeformat(
                        self.fig,
                        self.ax,
                        label=None,
                        labelcolor=labelcolor,
                        min=min,
                        max=max,
                        ticks=ticks,
                        ticklabels=ticklabels,
                        minor_num=minor_num,)
                    print('time format 3')
                else:
                    print('Sorry, not compatible so far.')

                self.multiple_xlabels(label)
                ticklabels = self.ax[self.nrows-1].get_xticklabels()
                ticklabels[0].set_ha("left")

        else:
            if type(label) is str:
                if self.ncols == 1:
                    self.plot_xaxis(
                        self.fig,
                        self.ax,
                        label=label,
                        labelcolor=labelcolor,
                        min=min,
                        max=max,
                        ticks=ticks,
                        ticklabels=ticklabels,
                        minor_num=minor_num,
                        xscale=xscale,)
                else:
                    for i in range(self.ncols):
                        self.plot_xaxis(
                            self.fig,
                            self.ax[:, i],
                            label=label,
                            labelcolor=labelcolor,
                            min=min,
                            max=max,
                            ticks=ticks,
                            ticklabels=ticklabels,
                            minor_num=minor_num,
                            xscale=xscale,)

            elif type(label) is list:
                if self.ncols == 1:
                    self.plot_xaxis(
                        self.fig,
                        self.ax,
                        label=None,
                        labelcolor=labelcolor,
                        min=min,
                        max=max,
                        ticks=ticks,
                        ticklabels=ticklabels,
                        minor_num=minor_num,
                        xscale=xscale,)
                else:
                    for i in range(self.ncols):
                        self.plot_xaxis(
                            self.fig,
                            self.ax[:, i],
                            label=None,
                            labelcolor=labelcolor,
                            min=min,
                            max=max,
                            ticks=ticks,
                            ticklabels=ticklabels,
                            minor_num=minor_num,
                            xscale=xscale,)

                # set the alignment for outer ticklabels
                self.multiple_xlabels(label)
                ticklabels = self.ax[self.nrows-1].get_xticklabels()
                # if len(label) > 1:
                ticklabels[0].set_ha('left')

        return None

    # Globaly used
    def set_yaxis(
            self,
            ax_idx,
            label=r'$y$',
            labelcolor='k',
            min=0.0,
            max=1.0,
            ticks=None,
            ticklabels=None,
            minor_num=2,
            yscale='linear',
            adjust=False,
            panelname_bkg=True,):

        def _yaxis(ax):
            ax.set_yscale(yscale)
            ax.set_ylim(min, max)
            if ticks is not None:
                ax.set_yticks(ticks)
                ax.set_yticklabels(ticklabels, fontsize=self.fontsize)
                if yscale == 'linear':
                    ax.yaxis.set_minor_locator(
                        ptick.AutoMinorLocator(minor_num))
            if yscale == 'log':
                _logticklocate(ax=ax)
            return None

        def _logticklocate(ax):
            ax.yaxis.set_major_locator(ptick.LogLocator(numticks=999))
            ax.yaxis.set_minor_locator(
                ptick.LogLocator(numticks=999, subs='auto'))
            return None

        if self.ncols == 1:
            if self.nrows == 1:
                ax = self.ax
            else:
                ax = self.ax[ax_idx]
                self.set_panelname(ax_idx, panelname_bkg)

            _yaxis(ax)

            ax.set_ylabel(label, fontsize=self.fontsize, color=labelcolor)

            if yscale == 'log':
                if adjust:
                    print('Note: yaxis adjust is not compatible with log scale.')
                    adjust = False

            if adjust:
                if ticks is not None:
                    self.yadjust(ax, np.min(ticklabels),
                                 np.max(ticklabels), yscale)
                else:
                    self.yadjust(ax, min, max, yscale)

            if self.nrows != 1:
                self.fig.align_ylabels(self.ax[:])

        else:
            for i in range(self.ncols):
                ax = self.ax[ax_idx, i]
                ax.set_yscale(yscale)

                _yaxis(ax)

                if i == 0:
                    ax.set_ylabel(label, fontsize=self.fontsize)
                    ax2 = ax.secondary_yaxis(location='right')
                    _yaxis(ax2)
                    if self.wspace > 0:
                        ax2.tick_params(axis='y', which='major',
                                        direction='out')
                        ax2.tick_params(axis='y', which='minor',
                                        direction='out')
                    plt.setp(ax2.get_yticklabels(),
                             visible=False)  # ラベルを消す
                elif i == self.ncols-1:
                    # ax2 = ax.secondary_yaxis(location='right')
                    # ax2.set_ylabel(label, fontsize=self.fontsize)
                    # _yaxis(ax2)
                    if self.wspace == 0:
                        ax.tick_params(axis='y', which='major',
                                       direction='inout')
                        ax.tick_params(axis='y', which='minor',
                                       direction='inout')
                    plt.setp(ax.get_yticklabels(),
                             visible=False)  # ラベルを消す
                else:
                    if self.wspace == 0:
                        ax.tick_params(axis='y', which='major',
                                       direction='inout')
                        ax.tick_params(axis='y', which='minor',
                                       direction='inout')
                    plt.setp(ax.get_yticklabels(),
                             visible=False)  # ラベルを消す

                if yscale == 'log':
                    if adjust:
                        print('Note: yaxis adjust is not compatible with log scale.')
                        adjust = False

                if adjust:
                    self.yadjust(ax, np.min(ticklabels),
                                 np.max(ticklabels), yscale)
                    self.yadjust(ax2, np.min(ticklabels),
                                 np.max(ticklabels), yscale)

                self.fig.tight_layout()
                self.fig.subplots_adjust(hspace=self.hspace)
                self.fig.subplots_adjust(wspace=self.wspace)

            self.fig.align_ylabels(self.ax[:, 0])

            self.set_panelname(ax_idx, panelname_bkg)

        return None

    # Globaly used
    def initialize(self,
                   panel_beginwith=None,
                   panelname_position='left',
                   panelname_size_scale=1.0):
        fig, ax = plt.subplots(
            self.nrows,
            self.ncols,
            figsize=self.figsize,
            dpi=self.dpi,
            height_ratios=self.height_ratios)

        self.fig, self.ax = fig, ax

        fig.tight_layout()

        self.panelname = [' a ', ' b ', ' c ', ' d ',
                          ' e ', ' f ', ' g ', ' h ', ' i ', ' j ', ' k ']
        if panel_beginwith == 'b':
            self.panelname = [' b ', ' c ', ' d ', ' e ',
                              ' f ', ' g ', ' h ', ' i ', ' j ', ' k ', ' l ']
        elif panel_beginwith == 'c':
            self.panelname = [' c ', ' d ', ' e ', ' f ',
                              ' g ', ' h ', ' i ', ' j ', ' k ', ' l ', ' m ']
        elif panel_beginwith == 'd':
            self.panelname = [' d ', ' e ', ' f ', ' g ',
                              ' h ', ' i ', ' j ', ' k ', ' l ', ' m ', ' o ']
        elif panel_beginwith == 'A':
            self.panelname = [' A ', ' B ', ' C ', ' D ',
                              ' E ', ' F ', ' G ', ' H ', ' I ', ' J ', ' K ']
        elif panel_beginwith == 'B':
            self.panelname = [' B ', ' C ', ' D ', ' E ',
                              ' F ', ' G ', ' H ', ' I ', ' J ', ' K ', ' L ']
        elif panel_beginwith == 'C':
            self.panelname = [' C ', ' D ', ' E ', ' F ',
                              ' G ', ' H ', ' I ', ' J ', ' K ', ' L ', ' M ']
        elif panel_beginwith == 'D':
            self.panelname = [' D ', ' E ', ' F ', ' G ',
                              ' H ', ' I ', ' J ', ' K ', ' L ', ' M ', ' O ']
        elif panel_beginwith == 'i':
            self.panelname = [' i ', ' ii ', ' iii ', ' iv ', ' v ',
                              ' vi ', ' vii ', ' viii ', ' ix ', ' x ', ' xi ']
        elif panel_beginwith == 'ii':
            self.panelname = [' ii ', ' iii ', ' iv ', ' v ',
                              ' vi ', ' vii ', ' viii ', ' ix ', ' x ', ' xi ', 'xii']
        elif panel_beginwith == '1':
            self.panelname = [' 1 ', ' 2 ', ' 3 ', ' 4 ',
                              ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ', ' 10 ', ' 11 ']

        self.panelname_xposition = panelname_position
        self.panelname_fontsize = self.fontsize*0.9*panelname_size_scale

    # Locally used
    def yadjust(
            self,
            ax,
            min,
            max,
            yscale='linear'):

        if yscale == 'linear':
            delta = max-min
            min = min - delta*0.15
            max = max + delta*0.15

            ax.set_ylim(min, max)

        elif yscale == 'log':
            min *= 0.75
            max *= 1.25
            ax.set_ylim(min, max)

        return None

    # Locally used
    def set_panelname_old(self, ax_idx=0, background=True, coef=None):
        x = 0
        if self.panelname_xposition == 'right':
            x = 1

        # Panel name at top right
        pad = 2.5
        if self.dpi <= 150:
            coef = 2.2
        elif self.dpi > 150 and self.dpi <= 300:
            coef = 3.5
        else:
            coef = 5.1

        if not background:
            coef *= 1.5

        offset = transforms.ScaledTranslation(
            (1-2*x)*(pad*coef)/self.dpi, -(pad*coef)/self.dpi, self.fig.dpi_scale_trans,)

        if self.ncols == 1:
            panelname = self.panelname[ax_idx]
            if self.nrows == 1:
                ax = self.ax
            else:
                ax = self.ax[ax_idx]
        else:
            panelname = self.panelname[self.nrows*ax_idx[1] + ax_idx[0]]
            ax = self.ax[ax_idx[0], ax_idx[1]]

        if background:
            ax.text(x, 1, panelname, color='w', weight='bold',
                    horizontalalignment=self.panelname_xposition,
                    verticalalignment='top',
                    bbox=dict(facecolor='k', pad=pad),
                    transform=ax.transAxes + offset,
                    fontsize=self.fontsize*0.9)
        else:
            ax.text(x, 1, panelname, color='w', weight='bold',
                    fontsize=self.fontsize*0.9,
                    verticalalignment='top',
                    transform=ax.transAxes + offset,
                    path_effects=[pe.withStroke(linewidth=3, foreground='k')])

        return None

    def draw_panel_name(self, ax, panelname, a=1, background=True):
        if background:
            path_effects = [pe.PathPatchEffect(offset=(0.75, -0.75), linewidth=0.75, facecolor='k'),
                            pe.withStroke(linewidth=1., foreground='k')]
            ax.annotate(panelname,
                        xy=(0.5*(1-a), 1), xycoords='axes fraction',
                        xytext=(a*0.16*20/self.panelname_fontsize,
                                -0.16*20/self.panelname_fontsize),
                        textcoords='offset fontsize',
                        fontweight='bold',
                        color='w',
                        fontsize=self.panelname_fontsize,
                        verticalalignment='top',
                        horizontalalignment=self.panelname_xposition,
                        path_effects=path_effects,
                        bbox=dict(fc='k', ec='k', alpha=0.8, pad=2.6, lw=0.), zorder=99)
        else:
            path_effects = [pe.PathPatchEffect(offset=(0.8, -0.8), linewidth=0.8, facecolor='k'),
                            pe.withStroke(linewidth=1.25, foreground='k')]
            ax.annotate(panelname,
                        xy=(0.5*(1-a), 1), xycoords='axes fraction',
                        xytext=(a*0.17*20/self.panelname_fontsize,
                                -0.24*20/self.panelname_fontsize),
                        textcoords='offset fontsize',
                        fontweight='bold',
                        color='w',
                        fontsize=self.panelname_fontsize,
                        verticalalignment='top',
                        horizontalalignment='left',
                        path_effects=path_effects,
                        bbox=dict(fc='none', ec='none', pad=2.6, lw=0.), zorder=99)

        return None

    # Locally used
    def set_panelname(self, ax_idx=0, background=True, coef=1.0):

        a = 1
        if self.panelname_xposition == 'right':
            a = -1

        if self.ncols == 1:
            panelname = self.panelname[ax_idx]
            if self.nrows == 1:
                ax = self.ax
            else:
                ax = self.ax[ax_idx]
            self.draw_panel_name(ax, panelname, a, background)
        else:
            for i in range(self.ncols):
                panelname = self.panelname[self.nrows*i + ax_idx]
                ax = self.ax[ax_idx, i]
                self.draw_panel_name(ax, panelname, a, background)

        return None

    # Locally used
    def plot_xaxis(self, fig, ax, label, labelcolor, min, max, ticks, ticklabels, minor_num, xscale):
        fig.subplots_adjust(hspace=self.hspace)
        fig.subplots_adjust(wspace=self.wspace)

        def _bottom_xaxis(ax0):
            ax0.set_xscale(xscale)
            ax0.set_xlim(min, max)
            ax0.tick_params(axis='both', labelsize=self.fontsize)
            if ticks is not None:
                ax0.set_xticks(ticks)
                ax0.set_xticklabels(
                    ticklabels, fontsize=self.fontsize, linespacing=1.1)
                if xscale == 'linear':
                    ax0.xaxis.set_minor_locator(
                        ptick.AutoMinorLocator(minor_num))  # minor ticks
            if xscale == 'log':
                _logticklocate(ax0)
            return None

        def _logticklocate(ax0):
            ax0.xaxis.set_major_locator(ptick.LogLocator(numticks=999))
            ax0.xaxis.set_minor_locator(
                ptick.LogLocator(numticks=999, subs='auto'))
            return None

        if self.nrows == 1:
            ax0 = ax
            _bottom_xaxis(ax0)

        else:
            for i in range(ax.size):
                ax0 = ax[i]
                _bottom_xaxis(ax0)

                if self.hspace == 0:
                    if i != 0:
                        ax2 = ax0.secondary_xaxis('top')
                        if ticks is not None:
                            _bottom_xaxis(ax2)
                            ax2.tick_params(axis='x', which='major',
                                            length=10, direction='inout')
                            ax2.tick_params(axis='x', which='minor',
                                            length=7, direction='inout')
                        plt.setp(ax2.get_xticklabels(),
                                 visible=False)  # ラベルを消す
                        ax2.set_zorder(ax0.get_zorder()+5)
                    if i != ax.size-1:
                        ax0.xaxis.set_visible(False)
                        plt.setp(ax0.get_xticklabels(),
                                 visible=False)  # ラベルを消す
                else:
                    fig.tight_layout()
                    if i != ax.size-1:
                        ax0.tick_params(labelbottom=False)

        ax0.set_xlabel(label, fontsize=self.fontsize, color=labelcolor)

        return None

    # Locally used
    def plot_xaxis_timeformat(self, fig, ax, label, labelcolor, min, max,
                              ticks=None, ticklabels=None, minor_num=2, format='%H:%M'):
        fig.subplots_adjust(hspace=self.hspace)
        fig.subplots_adjust(wspace=self.wspace)

        def _set_ticks(ax0):
            ax0.set_xlim(min, max)
            if ticks is None:
                ax0.xaxis.set_major_formatter(mdates.DateFormatter(format))
                ax0.xaxis.set_major_locator(mdates.MinuteLocator())
                if format == '%H:%M':
                    ax0.xaxis.set_minor_locator(mdates.SecondLocator(30))
            else:
                ax0.set_xticks(ticks)
                ax0.set_xticklabels(ticklabels, linespacing=1.1)
                ax0.xaxis.set_minor_locator(
                    ptick.AutoMinorLocator(minor_num))  # minor ticks
            return None

        if self.nrows > 1:
            for i in range(ax.size):
                ax1 = ax[i]
                _set_ticks(ax1)
                ax1.tick_params(axis='both', labelsize=self.fontsize)

                if self.hspace == 0:
                    if i != 0:
                        ax2 = ax1.secondary_xaxis('top')
                        _set_ticks(ax2)
                        ax2.tick_params(axis='x', which='major',
                                        length=8, direction='inout')
                        ax2.tick_params(axis='x', which='minor',
                                        length=6, direction='inout')

                        plt.setp(ax2.get_xticklabels(),
                                 visible=False)  # ラベルを消す
                        ax2.set_zorder(ax1.get_zorder()+5)
                    if i != ax.size-1:
                        ax1.xaxis.set_visible(False)
                        plt.setp(ax1.get_xticklabels(),
                                 visible=False)  # ラベルを消す
                else:
                    fig.tight_layout()
                    if i != ax.size-1:
                        ax1.tick_params(labelbottom=False)

        elif self.nrows == 1:
            ax1 = ax
            _set_ticks(ax1)
            ax1.tick_params(axis='both', labelsize=self.fontsize)

        ax1.set_xlabel(label, fontsize=self.fontsize, color=labelcolor)

        # set the alignment for outer ticklabels
        ticklabels = ax1.get_xticklabels()
        ticklabels[0].set_ha('left')

        return None

    # Locally used
    def multiple_xlabels(self, labels):
        axpos = self.ax[self.nrows-1].get_position()
        pp_ax = self.fig.add_axes(
            [-0.12, axpos.y0, 0.14, axpos.height])

        label = ''
        for i in range(len(labels)):
            if i > 0 and i <= len(labels):
                label += '\n'
            label += labels[i]

        # pp_ax.set_xlabel(label, fontsize=self.fontsize)

        ticklab = pp_ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        pp_ax.xaxis.set_label_coords(0.5, 0, transform=trans)
        pp_ax.tick_params(bottom=False, left=False, right=False, top=False)
        pp_ax.spines['right'].set_visible(False)
        pp_ax.spines['top'].set_visible(False)
        pp_ax.spines['bottom'].set_visible(False)
        pp_ax.spines['left'].set_visible(False)
        pp_ax.set_xticks([0, 0.5, 1])
        pp_ax.set_xticklabels([None, label, None],
                              fontsize=self.fontsize, linespacing=1.1)
        pp_ax.axes.get_yaxis().set_visible(False)
        # pp_ax.set_facecolor("yellow")
        pp_ax.patch.set_alpha(0.)

        return 0

    # Globally used
    def legend(
            self,
            ax_idx,
            handles=None,
            labels=None,
            loc='auto',
            ncol=1,
            bbox_to_anchor=None,
            fancybox=False,
            facecolor='white',
            framealpha=1,
            edgecolor='k',
            labelspacing=0.34,
            columnspacing=0.9,
            handletextpad=0.8,
            handlelength=1.0,
            fontfamily=None,
            fontsize_scale=0.9,
            textcolor=True,
            title=None,
            titleweight='bold',
            markerscale=1,
            zorder=100,
            **kwargs):

        fontsize = self.fontsize*fontsize_scale

        if loc == 'auto':
            if self.panelname_xposition == 'left':
                loc = 'upper right'
            elif self.panelname_xposition == 'right':
                loc = 'upper left'

        if self.ncols == 1:
            if self.nrows > 1:
                ax = self.ax[ax_idx]
            else:
                ax = self.ax
        else:
            ax = self.ax[ax_idx[0], ax_idx[1]]

        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc=loc,
            ncol=ncol,
            bbox_to_anchor=bbox_to_anchor,
            fancybox=fancybox,
            facecolor=facecolor,
            framealpha=framealpha,
            edgecolor=edgecolor,
            labelspacing=labelspacing,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            handlelength=handlelength,
            prop={'family': self.fontname, 'size': fontsize},
            fontsize=fontsize,
            markerscale=markerscale,)

        if title is not None:
            self.legend_title(legend, title, fontfamily,
                              fontsize, titleweight)

        # get plot colors
        if textcolor is True:
            for line, text in zip(legend.get_lines(), legend.get_texts()):
                text.set_color(line.get_color())
        elif textcolor is False:
            None
        else:
            for text in legend.get_texts():
                text.set_color(textcolor)

        linewidth = plt.rcParams['axes.linewidth']*0.7
        legend.get_frame().set_linewidth(linewidth)
        legend.set_zorder(zorder)

        return legend

    # Locally used
    def legend_title(self, legend, title, fontfamily, fontsize, weight):
        legend.set_title(
            title,
            prop={'family': self.fontname,
                  'size': fontsize,
                  'weight': weight})

        return None

    # Globally used
    def colormap(
            self, ax_idx,
            xdata, ydata, zdata,
            vmin=None, vmax=None, log=False,
            colorbar_label='[Unit]', cmap='viridis',
            adjust=False, zorder=0.5):
        if self.ncols == 1:
            if self.nrows > 1:
                ax = self.ax[ax_idx]
            else:
                ax = self.ax
        else:
            ax = self.ax[ax_idx[0], ax_idx[1]]
        norm = None
        if log:
            if vmin is None:
                norm = mplC.LogNorm(vmin=zdata.min(), vmax=zdata.max())
            else:
                norm = mplC.LogNorm(vmin=vmin, vmax=vmax)
                vmin = None
                vmax = None

        if cmap == 'parula':
            cm_data = self._parula_colors()
            parula = self._generate_cmap(cm_data)
            cmap = parula

        mappable = ax.pcolormesh(
            xdata, ydata, zdata, vmin=vmin, vmax=vmax, norm=norm, cmap=cmap, zorder=zorder)

        # Color bar
        axpos = ax.get_position()
        h = 1.0
        space = 0.0
        if adjust:
            h = 0.80
            space = 0.02

        pp_ax = self.fig.add_axes([axpos.x1+space, axpos.y0+axpos.height*(1.0-h)*0.5,
                                  0.03, axpos.height*h])  # カラーバーのaxesを追加
        pp = self.fig.colorbar(mappable, cax=pp_ax)
        pp.set_label(colorbar_label, fontsize=self.fontsize*0.9)
        pp.ax.tick_params(labelsize=self.fontsize*0.9)

        return mappable, pp

    # Globally used
    def generate_ticklabel_list(
            self,
            xlabel1: list,
            xlabel2: list,
            xlabel3: list = None,
            xlabel4: list = None,
            xlabel5: list = None):

        label_list = []
        for i in range(len(xlabel1)):
            label = xlabel1[i]+'\n'+xlabel2[i]
            if xlabel3 is not None:
                label += '\n'+xlabel3[i]
            if xlabel4 is not None:
                label += '\n'+xlabel4[i]
            if xlabel5 is not None:
                label += '\n'+xlabel5[i]
            label_list.append(label)

        return label_list

    # Globally used
    def set_strikethrough(self, x, ymin=0, ymax=1, linestyle='zebra1', color='k', gapcolor='w', linewidth=1.7):
        def _strikethrough_line(ax):
            if linestyle == 'zebra1':
                ax.axvline(x=x, linestyle=(0, (4.0, 4.0)),
                           linewidth=linewidth, color=color, gapcolor=gapcolor)
            elif linestyle == 'zebra2':
                ax.axvline(x=x, linestyle='-', linewidth=2.8, color=color)
                ax.axvline(x=x, linestyle='--', linewidth=1.6, color=gapcolor)
            else:
                ax.axvline(x=x, linestyle=linestyle, linewidth=linewidth,
                           color=color, gapcolor=gapcolor)
            return None

        if self.nrows == 1:
            _strikethrough_line(self.ax)
        else:
            for ax_idx in range(self.nrows):
                ax = self.ax[ax_idx]
                _strikethrough_line(ax)

        return None

    # Globally used
    def textbox(self,
                ax_idx,
                x,
                y,
                text,
                textcolor='w',
                textshadow=True,
                fontsize=18,
                fontweight='normal',
                facecolor='k',
                edgecolor='k',
                facealpha=0.7,
                verticalalignment='center_baseline',
                horizontalalignment='left',
                transform=None,
                rotation=0,
                zorder=2.0,):

        if self.nrows == 1:
            ax = self.ax
        else:
            ax = self.ax[ax_idx]

        if facealpha < 1.0:
            if type(facecolor) is str:
                fc_rgba0 = mplC.to_rgba(facecolor)
                fc_rgba0 = fc_rgba0[:-1] + (facealpha,)
            else:
                print('Set facecolor with a HEX code or a string.')
                fc_rgba0 = (0.0, 0.0, 0.0, 0.7)
        else:
            fc_rgba0 = facecolor

        bbox_props = dict(boxstyle='round,pad=0.15',
                          fc=fc_rgba0, ec=edgecolor, lw=1.)

        if textshadow:
            width = 0.66
            path_effects = [pe.PathPatchEffect(offset=(width, -width),
                                               linewidth=width,
                                               facecolor='k'),
                            pe.withStroke(linewidth=0., foreground='w')]
        else:
            path_effects = None

        if transform is None:
            transform = ax.transData   # the default

        text_patch = ax.text(x, y, text,
                             fontsize=fontsize,
                             weight=fontweight,
                             verticalalignment=verticalalignment,
                             horizontalalignment=horizontalalignment,
                             color=textcolor,
                             path_effects=path_effects,
                             transform=transform,
                             bbox=bbox_props,
                             clip_on=False,
                             rotation=rotation,
                             zorder=zorder)

        return text_patch

    # Globally used
    def upper_ax(self, height=0.10):
        if self.nrows == 1:
            ax0 = self.ax
        else:
            ax0 = self.ax[0]
        axpos = ax0.get_position()
        pp_ax = self.fig.add_axes(
            [axpos.x0, axpos.y1, axpos.width, axpos.height*height])
        pp_ax.tick_params(bottom=False, left=False, right=False, top=False)
        pp_ax.spines[['right', 'top', 'left']].set_visible(False)
        pp_ax.spines['bottom'].set_linewidth(2)
        pp_ax.axes.get_xaxis().set_visible(False)
        pp_ax.axes.get_yaxis().set_visible(False)
        pp_ax.patch.set_alpha(0.)
        return pp_ax

    # Locally used
    def _generate_cmap(self, colors):
        values = range(len(colors))

        vmax = np.ceil(np.max(values))
        color_list = []
        for v, c in zip(values, colors):
            color_list.append((v / vmax, c))

        return mplC.LinearSegmentedColormap.from_list('custom_cmap', color_list)

    # Locally used
    def _parula_colors(self):
        cm_data = [[0.2422, 0.1504, 0.6603],
                   [0.2444, 0.1534, 0.6728],
                   [0.2464, 0.1569, 0.6847],
                   [0.2484, 0.1607, 0.6961],
                   [0.2503, 0.1648, 0.7071],
                   [0.2522, 0.1689, 0.7179],
                   [0.254, 0.1732, 0.7286],
                   [0.2558, 0.1773, 0.7393],
                   [0.2576, 0.1814, 0.7501],
                   [0.2594, 0.1854, 0.761],
                   [0.2611, 0.1893, 0.7719],
                   [0.2628, 0.1932, 0.7828],
                   [0.2645, 0.1972, 0.7937],
                   [0.2661, 0.2011, 0.8043],
                   [0.2676, 0.2052, 0.8148],
                   [0.2691, 0.2094, 0.8249],
                   [0.2704, 0.2138, 0.8346],
                   [0.2717, 0.2184, 0.8439],
                   [0.2729, 0.2231, 0.8528],
                   [0.274, 0.228, 0.8612],
                   [0.2749, 0.233, 0.8692],
                   [0.2758, 0.2382, 0.8767],
                   [0.2766, 0.2435, 0.884],
                   [0.2774, 0.2489, 0.8908],
                   [0.2781, 0.2543, 0.8973],
                   [0.2788, 0.2598, 0.9035],
                   [0.2794, 0.2653, 0.9094],
                   [0.2798, 0.2708, 0.915],
                   [0.2802, 0.2764, 0.9204],
                   [0.2806, 0.2819, 0.9255],
                   [0.2809, 0.2875, 0.9305],
                   [0.2811, 0.293, 0.9352],
                   [0.2813, 0.2985, 0.9397],
                   [0.2814, 0.304, 0.9441],
                   [0.2814, 0.3095, 0.9483],
                   [0.2813, 0.315, 0.9524],
                   [0.2811, 0.3204, 0.9563],
                   [0.2809, 0.3259, 0.96],
                   [0.2807, 0.3313, 0.9636],
                   [0.2803, 0.3367, 0.967],
                   [0.2798, 0.3421, 0.9702],
                   [0.2791, 0.3475, 0.9733],
                   [0.2784, 0.3529, 0.9763],
                   [0.2776, 0.3583, 0.9791],
                   [0.2766, 0.3638, 0.9817],
                   [0.2754, 0.3693, 0.984],
                   [0.2741, 0.3748, 0.9862],
                   [0.2726, 0.3804, 0.9881],
                   [0.271, 0.386, 0.9898],
                   [0.2691, 0.3916, 0.9912],
                   [0.267, 0.3973, 0.9924],
                   [0.2647, 0.403, 0.9935],
                   [0.2621, 0.4088, 0.9946],
                   [0.2591, 0.4145, 0.9955],
                   [0.2556, 0.4203, 0.9965],
                   [0.2517, 0.4261, 0.9974],
                   [0.2473, 0.4319, 0.9983],
                   [0.2424, 0.4378, 0.9991],
                   [0.2369, 0.4437, 0.9996],
                   [0.2311, 0.4497, 0.9995],
                   [0.225, 0.4559, 0.9985],
                   [0.2189, 0.462, 0.9968],
                   [0.2128, 0.4682, 0.9948],
                   [0.2066, 0.4743, 0.9926],
                   [0.2006, 0.4803, 0.9906],
                   [0.195, 0.4861, 0.9887],
                   [0.1903, 0.4919, 0.9867],
                   [0.1869, 0.4975, 0.9844],
                   [0.1847, 0.503, 0.9819],
                   [0.1831, 0.5084, 0.9793],
                   [0.1818, 0.5138, 0.9766],
                   [0.1806, 0.5191, 0.9738],
                   [0.1795, 0.5244, 0.9709],
                   [0.1785, 0.5296, 0.9677],
                   [0.1778, 0.5349, 0.9641],
                   [0.1773, 0.5401, 0.9602],
                   [0.1768, 0.5452, 0.956],
                   [0.1764, 0.5504, 0.9516],
                   [0.1755, 0.5554, 0.9473],
                   [0.174, 0.5605, 0.9432],
                   [0.1716, 0.5655, 0.9393],
                   [0.1686, 0.5705, 0.9357],
                   [0.1649, 0.5755, 0.9323],
                   [0.161, 0.5805, 0.9289],
                   [0.1573, 0.5854, 0.9254],
                   [0.154, 0.5902, 0.9218],
                   [0.1513, 0.595, 0.9182],
                   [0.1492, 0.5997, 0.9147],
                   [0.1475, 0.6043, 0.9113],
                   [0.1461, 0.6089, 0.908],
                   [0.1446, 0.6135, 0.905],
                   [0.1429, 0.618, 0.9022],
                   [0.1408, 0.6226, 0.8998],
                   [0.1383, 0.6272, 0.8975],
                   [0.1354, 0.6317, 0.8953],
                   [0.1321, 0.6363, 0.8932],
                   [0.1288, 0.6408, 0.891],
                   [0.1253, 0.6453, 0.8887],
                   [0.1219, 0.6497, 0.8862],
                   [0.1185, 0.6541, 0.8834],
                   [0.1152, 0.6584, 0.8804],
                   [0.1119, 0.6627, 0.877],
                   [0.1085, 0.6669, 0.8734],
                   [0.1048, 0.671, 0.8695],
                   [0.1009, 0.675, 0.8653],
                   [0.0964, 0.6789, 0.8609],
                   [0.0914, 0.6828, 0.8562],
                   [0.0855, 0.6865, 0.8513],
                   [0.0789, 0.6902, 0.8462],
                   [0.0713, 0.6938, 0.8409],
                   [0.0628, 0.6972, 0.8355],
                   [0.0535, 0.7006, 0.8299],
                   [0.0433, 0.7039, 0.8242],
                   [0.0328, 0.7071, 0.8183],
                   [0.0234, 0.7103, 0.8124],
                   [0.0155, 0.7133, 0.8064],
                   [0.0091, 0.7163, 0.8003],
                   [0.0046, 0.7192, 0.7941],
                   [0.0019, 0.722, 0.7878],
                   [0.0009, 0.7248, 0.7815],
                   [0.0018, 0.7275, 0.7752],
                   [0.0046, 0.7301, 0.7688],
                   [0.0094, 0.7327, 0.7623],
                   [0.0162, 0.7352, 0.7558],
                   [0.0253, 0.7376, 0.7492],
                   [0.0369, 0.74, 0.7426],
                   [0.0504, 0.7423, 0.7359],
                   [0.0638, 0.7446, 0.7292],
                   [0.077, 0.7468, 0.7224],
                   [0.0899, 0.7489, 0.7156],
                   [0.1023, 0.751, 0.7088],
                   [0.1141, 0.7531, 0.7019],
                   [0.1252, 0.7552, 0.695],
                   [0.1354, 0.7572, 0.6881],
                   [0.1448, 0.7593, 0.6812],
                   [0.1532, 0.7614, 0.6741],
                   [0.1609, 0.7635, 0.6671],
                   [0.1678, 0.7656, 0.6599],
                   [0.1741, 0.7678, 0.6527],
                   [0.1799, 0.7699, 0.6454],
                   [0.1853, 0.7721, 0.6379],
                   [0.1905, 0.7743, 0.6303],
                   [0.1954, 0.7765, 0.6225],
                   [0.2003, 0.7787, 0.6146],
                   [0.2061, 0.7808, 0.6065],
                   [0.2118, 0.7828, 0.5983],
                   [0.2178, 0.7849, 0.5899],
                   [0.2244, 0.7869, 0.5813],
                   [0.2318, 0.7887, 0.5725],
                   [0.2401, 0.7905, 0.5636],
                   [0.2491, 0.7922, 0.5546],
                   [0.2589, 0.7937, 0.5454],
                   [0.2695, 0.7951, 0.536],
                   [0.2809, 0.7964, 0.5266],
                   [0.2929, 0.7975, 0.517],
                   [0.3052, 0.7985, 0.5074],
                   [0.3176, 0.7994, 0.4975],
                   [0.3301, 0.8002, 0.4876],
                   [0.3424, 0.8009, 0.4774],
                   [0.3548, 0.8016, 0.4669],
                   [0.3671, 0.8021, 0.4563],
                   [0.3795, 0.8026, 0.4454],
                   [0.3921, 0.8029, 0.4344],
                   [0.405, 0.8031, 0.4233],
                   [0.4184, 0.803, 0.4122],
                   [0.4322, 0.8028, 0.4013],
                   [0.4463, 0.8024, 0.3904],
                   [0.4608, 0.8018, 0.3797],
                   [0.4753, 0.8011, 0.3691],
                   [0.4899, 0.8002, 0.3586],
                   [0.5044, 0.7993, 0.348],
                   [0.5187, 0.7982, 0.3374],
                   [0.5329, 0.797, 0.3267],
                   [0.547, 0.7957, 0.3159],
                   [0.5609, 0.7943, 0.305],
                   [0.5748, 0.7929, 0.2941],
                   [0.5886, 0.7913, 0.2833],
                   [0.6024, 0.7896, 0.2726],
                   [0.6161, 0.7878, 0.2622],
                   [0.6297, 0.7859, 0.2521],
                   [0.6433, 0.7839, 0.2423],
                   [0.6567, 0.7818, 0.2329],
                   [0.6701, 0.7796, 0.2239],
                   [0.6833, 0.7773, 0.2155],
                   [0.6963, 0.775, 0.2075],
                   [0.7091, 0.7727, 0.1998],
                   [0.7218, 0.7703, 0.1924],
                   [0.7344, 0.7679, 0.1852],
                   [0.7468, 0.7654, 0.1782],
                   [0.759, 0.7629, 0.1717],
                   [0.771, 0.7604, 0.1658],
                   [0.7829, 0.7579, 0.1608],
                   [0.7945, 0.7554, 0.157],
                   [0.806, 0.7529, 0.1546],
                   [0.8172, 0.7505, 0.1535],
                   [0.8281, 0.7481, 0.1536],
                   [0.8389, 0.7457, 0.1546],
                   [0.8495, 0.7435, 0.1564],
                   [0.86, 0.7413, 0.1587],
                   [0.8703, 0.7392, 0.1615],
                   [0.8804, 0.7372, 0.165],
                   [0.8903, 0.7353, 0.1695],
                   [0.9, 0.7336, 0.1749],
                   [0.9093, 0.7321, 0.1815],
                   [0.9184, 0.7308, 0.189],
                   [0.9272, 0.7298, 0.1973],
                   [0.9357, 0.729, 0.2061],
                   [0.944, 0.7285, 0.2151],
                   [0.9523, 0.7284, 0.2237],
                   [0.9606, 0.7285, 0.2312],
                   [0.9689, 0.7292, 0.2373],
                   [0.977, 0.7304, 0.2418],
                   [0.9842, 0.733, 0.2446],
                   [0.99, 0.7365, 0.2429],
                   [0.9946, 0.7407, 0.2394],
                   [0.9966, 0.7458, 0.2351],
                   [0.9971, 0.7513, 0.2309],
                   [0.9972, 0.7569, 0.2267],
                   [0.9971, 0.7626, 0.2224],
                   [0.9969, 0.7683, 0.2181],
                   [0.9966, 0.774, 0.2138],
                   [0.9962, 0.7798, 0.2095],
                   [0.9957, 0.7856, 0.2053],
                   [0.9949, 0.7915, 0.2012],
                   [0.9938, 0.7974, 0.1974],
                   [0.9923, 0.8034, 0.1939],
                   [0.9906, 0.8095, 0.1906],
                   [0.9885, 0.8156, 0.1875],
                   [0.9861, 0.8218, 0.1846],
                   [0.9835, 0.828, 0.1817],
                   [0.9807, 0.8342, 0.1787],
                   [0.9778, 0.8404, 0.1757],
                   [0.9748, 0.8467, 0.1726],
                   [0.972, 0.8529, 0.1695],
                   [0.9694, 0.8591, 0.1665],
                   [0.9671, 0.8654, 0.1636],
                   [0.9651, 0.8716, 0.1608],
                   [0.9634, 0.8778, 0.1582],
                   [0.9619, 0.884, 0.1557],
                   [0.9608, 0.8902, 0.1532],
                   [0.9601, 0.8963, 0.1507],
                   [0.9596, 0.9023, 0.148],
                   [0.9595, 0.9084, 0.145],
                   [0.9597, 0.9143, 0.1418],
                   [0.9601, 0.9203, 0.1382],
                   [0.9608, 0.9262, 0.1344],
                   [0.9618, 0.932, 0.1304],
                   [0.9629, 0.9379, 0.1261],
                   [0.9642, 0.9437, 0.1216],
                   [0.9657, 0.9494, 0.1168],
                   [0.9674, 0.9552, 0.1116],
                   [0.9692, 0.9609, 0.1061],
                   [0.9711, 0.9667, 0.1001],
                   [0.973, 0.9724, 0.0938],
                   [0.9749, 0.9782, 0.0872],
                   [0.9769, 0.9839, 0.0805]]

        return cm_data
