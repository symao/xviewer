#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2016-2023 maoshuyuan123@gmail.com. All Rights Reserved.
import sys
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_parser import DataParser

# PyQt6
from interface import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# Matplotlib must later than qt, otherwise, it will crush
from matplotlib.widgets import MultiCursor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

project_dir = os.path.abspath(os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))
cfg_dir = os.path.abspath(os.path.join(project_dir, 'cfg'))
tmp_dir = os.path.join(project_dir, 'tmp')


def generate_curve_colors():
    # colors of curves
    colors = ['#000000', '#0000FF', '#007F00', '#007FFF', '#7F00FF',
              '#00FF00', '#00FF7F', '#00FFFF', '#7F0000', '#7F007F',
              '#7F7F00', '#7F7F7F', '#7F7FFF', '#7FFF00', '#7FFF7F',
              '#FF0000', '#FF007F', '#FF00FF', '#FF7F00', '#FF7F7F']
    # shuffle colors
    colors = colors[0::3] + colors[1::3] + colors[2::3]
    colors = colors[2::3] + colors[0::3] + colors[1::3]
    colors = colors[1::3] + colors[2::3] + colors[0::3]
    return colors


curve_color_list = generate_curve_colors()
curve_linestyle_list = ['-', '--', ':', ':.', '-.',
                        '*', '+', 'o', '.', 's', '^', 'p',
                        '-*', '-+', '-o', '-s', '-^', '-p',
                        '--*', '--+', '--o', '--s', '--^', '--p']

# one curve package


class CurvePack():
    def __init__(self, filename, colname, x=None, y=None):
        self.filename = filename
        self.colname = colname
        self.x = x
        self.y = y
        self.x_shift = 0
        self.color = 'blue'
        self.linestyle = curve_linestyle_list[0]
        self.linewidth = 1
        self.widget = None
        col_indicator = '->'
        self.name = self.filename+col_indicator+self.colname
        self.var = self.name.replace(col_indicator, '_').replace('.', '_')
        self.expression = self.name


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("X Viewer")
        self.setMinimumSize(0, 0)

        # figure
        plt.rcParams['axes.unicode_minus'] = False
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(111)

        # init canvas
        self.canvas = FigureCanvas(self.figure)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

        # init axis layout
        self.gridlayout = QGridLayout(self.groupBox_axis)
        self.gridlayout.addWidget(NavigationToolbar(self.canvas, self))  # add navigation bar
        self.gridlayout.addWidget(self.canvas)  # add canvas

        # init video player
        self.timer = VideoTimer()
        self.timer.set_fps(20)
        self.video_player_run = 0
        self.pushButton_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        self.comboBox_linestyle.addItems(curve_linestyle_list)

        # init signal and slot
        self.listWidget_variable.clicked.connect(self.on_click_varlist)
        self.listWidget_file.clicked.connect(self.on_click_filelist)
        self.pushButton_play.clicked.connect(self.on_click_video_play)
        self.pushButton_clear.clicked.connect(self.on_click_clear_axes)
        self.pushButton_reset.clicked.connect(self.on_click_reset)
        self.pushButton_select_folder.clicked.connect(self.on_click_select_dir)
        self.pushButton_select_file.clicked.connect(self.on_click_select_file)
        self.pushButton_calc.clicked.connect(self.on_click_expression_cal)
        self.pushButton_color.clicked.connect(self.on_click_color)
        self.pushButton_about.clicked.connect(self.on_click_about)
        self.spinBox_linewidth.valueChanged.connect(self.on_change_linewidth)
        self.lineEdit_videoidx.editingFinished.connect(self.on_finish_video_idx)
        self.lineEdit_time_shift.editingFinished.connect(self.on_finish_time_shift)
        self.lineEdit_search_variable.textChanged.connect(self.on_change_variable_search)
        self.checkBox_display_grid.stateChanged.connect(self.on_change_grid_show)

        # reset
        self.reset()

    def reset(self):
        self.xrange = (0, 1)
        self.data_loaded = False
        self.video_loaded = False
        self.curve_dict = {}
        self.prev_click_row = None
        self.axvline_widget = None
        self.cur_x_axis = None
        self.pushButton_play.setEnabled(False)
        self.pushButton_clear.setEnabled(False)
        self.pushButton_reset.setEnabled(False)
        self.pushButton_calc.setEnabled(False)
        self.horizontalSlider_video_slider.setEnabled(False)
        self.spinBox_linewidth.setEnabled(False)
        self.comboBox_linestyle.setEnabled(False)
        self.lineEdit_search_variable.setEnabled(False)
        self.lineEdit_videoidx.setEnabled(False)
        self.lineEdit_expression.setEnabled(False)
        self.lineEdit_time_shift.setEnabled(False)
        self.label_img_show.clear()
        self.listWidget_file.clear()
        self.listWidget_variable.clear()
        self.horizontalSlider_video_slider.setValue(0)
        self.axes.cla()
        self.canvas.draw()

    def enableVideoUi(self):
        self.pushButton_play.setEnabled(True)
        self.horizontalSlider_video_slider.setEnabled(True)
        self.lineEdit_videoidx.setEnabled(True)

    def enablePlotUi(self):
        self.pushButton_clear.setEnabled(True)
        self.pushButton_reset.setEnabled(True)
        self.pushButton_calc.setEnabled(True)
        self.spinBox_linewidth.setEnabled(True)
        self.comboBox_linestyle.setEnabled(True)
        self.lineEdit_search_variable.setEnabled(True)
        self.lineEdit_expression.setEnabled(True)
        self.lineEdit_time_shift.setEnabled(True)

    def closeEvent(self, event):  # quit interface
        # exit_flag = QMessageBox.question(self, 'Dialog', "Do you want to exit the program?",
        #                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #                                  QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes
        exit_flag = True
        if exit_flag:
            self.timer.stop()
            cv2.destroyAllWindows()
            event.accept()
        else:
            event.ignore()

    def find_cfg_file(self, input_dir):
        # find 'format.yaml' in input dir
        cfg_file = os.path.join(input_dir, 'format.yaml')
        if os.path.exists(cfg_file):
            return cfg_file
        # find 'format_*.yaml' in input dir
        for f in os.listdir(input_dir):
            if f.startswith('format_') and f.endswith('.yaml'):
                cfg_file = os.path.join(input_dir, f)
                if os.path.isfile(cfg_file):
                    return cfg_file
        return None

    def open_data_path(self, data_path):
        self.reset()

        # handle opened path
        self.save_history_path(data_path)
        self.lineEdit_datadir.setText(str(data_path))  # show data path

        # load data from data path
        self.data_parser = DataParser()
        if isinstance(data_path, str) and os.path.isdir(data_path):
            cfg_file = self.find_cfg_file(data_path)
            if cfg_file is None:
                ret = QMessageBox.question(
                    self, 'Warning', "No format file in input dir, select config file in 'cfg' dir?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
                if ret == QMessageBox.StandardButton.Yes:
                    cfg_file, _ = QFileDialog.getOpenFileName(self, "select config file", cfg_dir)
                if cfg_file is None or len(cfg_file) == 0:
                    return
            self.data_parser.load(data_path, cfg_file)
        else:
            self.data_parser.load(data_path)
        self.data_loaded = True if len(self.data_parser.file_dict) > 0 else False
        self.video_loaded = True if len(self.data_parser.img_dict) > 0 else False
        # print(self.data_parser.file_dict.keys(),self.data_parser.get_variables())

        if not self.data_loaded and not self.video_loaded:
            QMessageBox.warning(
                self, 'Warning', "Cannot open data path '%s', please select another data dir." % data_path)
            return

        if self.data_loaded:
            self.enablePlotUi()
            # create curve pack for each variables
            self.file_list = self.data_parser.get_files()
            for i, (filename, colname) in enumerate(self.data_parser.get_variables()):
                curve_pack = CurvePack(
                    filename, colname, self.data_parser.file_dict[filename].ts, self.data_parser.file_dict[filename].col_dict[colname])
                curve_pack.color = curve_color_list[i % len(curve_color_list)]
                self.curve_dict[curve_pack.name] = curve_pack
            # init listWidget for files and variables
            self.listWidget_file.addItems(self.file_list)
            self.listWidget_variable.addItems([val.name for _, val in self.curve_dict.items()])
            self.cur_cursor = FollowCursor(self.canvas, self.axes)
            self.xrange = (np.min([np.min(curve.x) for curve in self.curve_dict.values()]),
                           np.max([np.max(curve.x) for curve in self.curve_dict.values()]))

            self.list_widget_default_background = self.listWidget_variable.item(0).background()

        if self.video_loaded:
            self.enableVideoUi()
            self.video_ts_list = [ts for ts, img in self.data_parser.img_dict.get(next(iter(self.data_parser.img_dict)))]
            self.video_max_idx = len(self.video_ts_list)
            self.horizontalSlider_video_slider.setMinimum(0)
            self.horizontalSlider_video_slider.setMaximum(self.video_max_idx-1)
            # video timer play video
            self.timer.timeSignal.signal[str].connect(self.video_play)  # play video
            self.reset_play()
            # plot first image
            self.plot_video(0)

    def on_click_select_file(self):
        # set default open dir, set history opendir as default dir if stored.
        remember_path = self.load_history_path()
        data_path, _ = QFileDialog.getOpenFileNames(
            self, "Open file/files", remember_path if remember_path is not None else project_dir)
        if len(data_path) == 0:
            return
        # open data path
        self.open_data_path(data_path)

    def on_click_select_dir(self):
        self.reset()
        # set default open dir, set history opendir as default dir if stored.
        remember_path = self.load_history_path()
        if remember_path is None:
            remember_path = project_dir
        data_path = QFileDialog.getExistingDirectory(self, "Open data dir", remember_path)
        if len(data_path) == 0:
            return
        self.open_data_path(data_path)

    def on_click_filelist(self):
        cur_row = self.listWidget_file.currentRow()
        cur_file = self.listWidget_file.item(cur_row).text()
        cur_file_ids = [i for i, curve in enumerate(self.curve_dict.values()) if curve.filename == cur_file]
        if len(cur_file_ids) > 0:
            self.listWidget_variable.setCurrentRow(cur_file_ids[0])

    def on_change_variable_search(self):
        if self.data_loaded:
            target = self.lineEdit_search_variable.text()
            for i, curve_name in enumerate(self.curve_dict.keys()):
                if target in curve_name:
                    self.listWidget_variable.setCurrentRow(i)
                    break

    def save_history_path(self, data_path):
        history_file = os.path.join(tmp_dir, 'history_file')
        if isinstance(data_path, list):
            self.save_history_path(data_path[0])
        elif os.path.exists(data_path):
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            open(history_file, 'w').write(data_path)

    def load_history_path(self):
        history_file = os.path.join(tmp_dir, 'history_file')
        if os.path.exists(history_file):
            tdir = open(history_file, 'r').readline()
            if os.path.exists(tdir):
                return tdir
        return None

    def update_curve_property_bar(self, curve_pack):
        self.lineEdit_expression.setText(curve_pack.expression)
        self.lineEdit_time_shift.setText(str(curve_pack.x_shift))
        self.pushButton_color.setStyleSheet('QPushButton{background-color:%s}' % curve_pack.color)
        self.comboBox_linestyle.setCurrentText(curve_pack.linestyle)
        self.spinBox_linewidth.setValue(curve_pack.linewidth)

    def on_click_varlist(self):
        click_row = self.listWidget_variable.currentRow()
        curve_pack = self.curve_dict[self.listWidget_variable.item(click_row).text()]
        if self.prev_click_row == click_row and curve_pack.widget is not None:
            # remove var once click again
            self.axes.lines.remove(curve_pack.widget)
            self.listWidget_variable.item(click_row).setBackground(self.list_widget_default_background)
            self.listWidget_variable.clearSelection()
            self.lineEdit_expression.setText('')
            self.lineEdit_time_shift.setText('')
            curve_pack.widget = None
            curve_pack.expression = curve_pack.name
            curve_pack.x_shift = 0
            self.update_axes()
        elif curve_pack.widget is None:
            self.listWidget_variable.item(click_row).setBackground(QColor('green'))
            self.plot_curve_pack(curve_pack)
            self.update_axes()
            self.update_curve_property_bar(curve_pack)
        else:
            self.update_curve_property_bar(curve_pack)
        self.prev_click_row = click_row

    def plot_curve_pack(self, curve_pack):
        # calculate x,y for plotting
        x = curve_pack.x + curve_pack.x_shift
        if curve_pack.expression != curve_pack.name:
            y = self.calc_expression(curve_pack.expression)
        else:
            y = curve_pack.y

        # clear history plot cache
        if curve_pack.widget is not None:
            self.axes.lines.remove(curve_pack.widget)
            curve_pack.widget = None

        # plot data
        if x is None:
            curve_pack.widget = self.axes.plot(y, curve_pack.linestyle, label=curve_pack.name,
                                               color=curve_pack.color, linewidth=curve_pack.linewidth)[0]
        else:
            n = min(len(x), len(y))  # size between x and y may be not the same
            curve_pack.widget = self.axes.plot(x[: n],
                                               y[: n],
                                               curve_pack.linestyle, label=curve_pack.name, color=curve_pack.color,
                                               linewidth=curve_pack.linewidth)[0]

        # set xrange
        plt.xlim(*self.xrange)
        self.lineEdit_time_shift.setText(str(curve_pack.x_shift))

    def calc_expression(self, expression):
        try:
            proc_expression = expression
            for _, curve in self.curve_dict.items():
                # create variable for later expression calculation
                exec("%s = self.curve_dict['%s'].y" % (curve.var, curve.name))
                # replace var name in expression to valid var string
                # replace curve name to var name which does not contain '->' or '.', thus can be calculated
                proc_expression = proc_expression.replace(curve.name, curve.var)
            # calculating expression
            res = eval(proc_expression)
            return res
        except:
            QMessageBox.warning(self, 'Warning', "Wrong expression:%s" % expression)
            return None

    def clear_axes(self):
        for i in range(self.listWidget_variable.count()):
            self.listWidget_variable.item(i).setBackground(self.list_widget_default_background)
        for pack in self.curve_dict.values():
            pack.widget = None
        self.axvline_widget = None
        self.axes.cla()  # clear the last curve
        self.canvas.draw()
        self.timer.stop()

    def update_axes(self, update_legend=True, curs=False):
        if update_legend:
            a, b = self.axes.get_legend_handles_labels()
            if len(a) > 0 and len(b) > 0:
                self.axes.legend()
        self.axes.grid(self.checkBox_display_grid.isChecked())
        self.canvas.draw()
        if curs:
            self.cur_cursor.cursor

    # compute variable's result, Note: expression is corresponding to curve which is selected in var list
    def on_click_expression_cal(self):
        var_widget = self.listWidget_variable.item(self.listWidget_variable.currentRow())
        if var_widget is not None:
            equation = self.lineEdit_expression.text()
            var_name = var_widget.text()
            curve_pack = self.curve_dict[var_name]
            res = self.calc_expression(equation)
            if res is not None and abs(len(res) == len(curve_pack.y)) < 3:
                curve_pack.expression = equation
                self.plot_curve_pack(curve_pack)
                self.update_axes()
        else:
            QMessageBox.warning(self, 'Warning', "No variable selected, select one variable before calculating.")

    def on_click_color(self):
        var_widget = self.listWidget_variable.item(self.listWidget_variable.currentRow())
        if var_widget is not None:
            var_name = var_widget.text()
            curve_pack = self.curve_dict[var_name]
            color = QColorDialog.getColor()
            if color.isValid():
                curve_pack.color = color.name()
                self.plot_curve_pack(curve_pack)
                self.pushButton_color.setStyleSheet('QPushButton{background-color:%s}' % color.name())
                self.update_axes()

    def on_click_about(self):
        QMessageBox.about(self, 'XViewer 1.0', "Copyright 2018-2023 @shuyuanmao\nEmail: maoshuyuan123@gmail.com")

    def on_comboBox_linestyle_activated(self, text):
        if isinstance(text, str):
            line_style = text
        elif isinstance(text, int):
            line_style = curve_linestyle_list[text]
        else:
            return
        var_widget = self.listWidget_variable.item(self.listWidget_variable.currentRow())
        if var_widget is not None:
            curve_pack = self.curve_dict[var_widget.text()]
            curve_pack.linestyle = line_style
            self.plot_curve_pack(curve_pack)
            self.update_axes()

    def on_change_linewidth(self, value):
        var_widget = self.listWidget_variable.item(self.listWidget_variable.currentRow())
        if var_widget is not None:
            var_name = var_widget.text()
            curve_pack = self.curve_dict[var_name]
            curve_pack.linewidth = value
            self.plot_curve_pack(curve_pack)
            self.update_axes()

    def on_change_grid_show(self):
        self.update_axes()

    def reset_play(self):
        self.timer.stop()
        self.video_player_run = 0
        self.pushButton_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def plot_video(self, idx):
        if not self.video_loaded:
            return
        
        # plot ts
        plot_ts = self.video_ts_list[idx]

        # gather all images at plot ts
        plot_imgs = []
        for name, img_list in self.data_parser.img_dict.items():
            for ts,img in img_list:
                if ts >= plot_ts:
                    plot_imgs.append(img)
                    break

        # show image
        self.label_img_show.setScaledContents(True)
        self.label_img_show.setPixmap(QPixmap.fromImage(self.img_cv2qt(plot_imgs)))
        self.lineEdit_videoidx.setText(str(idx))

        # show curser
        idx = np.clip(idx, 0, self.video_max_idx)
        self.plot_ts_curser(plot_ts)

        # show play time
        start_ts = self.video_ts_list[0]
        stop_ts = self.video_ts_list[-1]
        t_all = int(stop_ts - start_ts)
        t_cur = int(plot_ts - start_ts)
        self.label_videots.setText(
            '%02d:%02d:%02d/%02d:%02d:%02d' %
            (int(t_cur / 3600),
             int(t_cur % 3600 / 60),
             int(t_cur % 60),
             int(t_all / 3600),
             int(t_all % 3600 / 60),
             int(t_all % 60)))

    def video_play(self):
        idx = self.horizontalSlider_video_slider.value() + 1
        self.horizontalSlider_video_slider.setValue(idx)
        if idx >= self.video_max_idx:
            self.reset_play()

    def on_click_video_play(self):
        if self.video_player_run == 0:
            self.timer.start()
            self.video_player_run = 1
            self.pushButton_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.timer.stop()
            self.video_player_run = 0
            self.pushButton_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def img_cv2qt(self, cvimg):
        def img_standardize(img):
            out = cv2.resize(img, (320, 240))
            if len(out.shape) < 3 or out.shape[2] == 1:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
            else:
                out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            return out

        if isinstance(cvimg, list):
            img_list = [img_standardize(img) for img in cvimg]
            img = np.hstack(img_list)
        else:
            img = img_standardize(cvimg)

        height, width = img.shape[:2]
        return QImage(img.flatten(), width, height, QImage.Format.Format_RGB888)  # convert to Qt tormat

    def plot_ts_curser(self, ts):
        if self.axvline_widget is not None:
            self.axes.lines.remove(self.axvline_widget)
        self.axvline_widget = self.axes.axvline(ts, ls='--', color='black', linewidth=1)
        self.update_axes(False, False)

    def on_click_reset(self):
        if self.data_loaded:
            # replot all curves
            self.axes.cla()
            for curve_pack in self.curve_dict.values():
                if curve_pack.widget is not None:  # need draw
                    curve_pack.widget = None  # old widget is clear by axes.cla(), so we need to reset it
                    self.plot_curve_pack(curve_pack)
            if self.video_loaded:
                self.axvline_widget = self.axes.axvline(self.video_ts_list[self.horizontalSlider_video_slider.value()],
                                                        ls='--', color='black', linewidth=1)
            self.update_axes()

    def on_click_clear_axes(self):
        if self.data_loaded:
            self.clear_axes()

    def on_horizontalSlider_video_slider_valueChanged(self, value):
        self.plot_video(value)

    def on_finish_video_idx(self):
        if not self.data_loaded:
            return
        idx = int(self.lineEdit_videoidx.text())
        slider_val = self.horizontalSlider_video_slider.value()
        if idx != slider_val:
            idx = np.clip(idx, 0, self.video_max_idx-1)
            self.horizontalSlider_video_slider.setValue(idx)

    def on_finish_time_shift(self):
        if not self.data_loaded:
            return
        shift = float(self.lineEdit_time_shift.text())
        var_widget = self.listWidget_variable.item(self.listWidget_variable.currentRow())
        if var_widget is not None:
            var_name = var_widget.text()
            curve_pack = self.curve_dict[var_name]
            curve_pack.x_shift = shift
            self.plot_curve_pack(curve_pack)
            self.update_axes()


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):
    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()
        self.image_index = 0

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)
            self.image_index += 1  # index of image

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


class FollowCursor(object):
    def __init__(self, canvas, ax):
        self.ax = ax
        self.fig = ax.figure
        self.ax.xaxis.set_label_position('top')
        self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self)
        axs = list()
        axs.append(ax)
        self.cursor = MultiCursor(canvas, axs,  color='r', lw=1, ls='--', horizOn=True)

    def __call__(self, event):
        if event.inaxes is None:
            return
        ax = self.ax
        if ax != event.inaxes:
            inv = ax.transData.inverted()  # x, y = event.mouseevent.xdata, event.mouseevent.ydata
            self.x, self.y = inv.transform(np.array((event.x, event.y)).reshape(1, 2)).ravel()
        elif ax == event.inaxes:
            self.x, self.y = event.xdata, event.ydata
        else:
            return
        event.canvas.draw()


if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
