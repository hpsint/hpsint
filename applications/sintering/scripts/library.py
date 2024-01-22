import colorsys
import glob
import os
import re
import pathlib
import matplotlib.pyplot as plt

def get_hex_colors(N):
    hsv_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in hsv_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def get_solutions(files, do_print = True):

    if type(files) is not list:
        raise Exception("Variable files={} is not list".format(files))

    files_list = []
    for fm in files:
        current_files_list = glob.glob(fm, recursive=True)
        current_files_list.sort()
        files_list += current_files_list

    files_list = sorted(list(set(files_list)))

    if do_print:
        print("The complete list of files to process:")
        for f in files_list:
            print(f)

    return files_list

def generate_short_labels(files_list):
    if len(files_list) == 0:
        return []

    if len(files_list) == 1:
        labels = [pathlib.PurePath(files_list[0]).parent.name]
        return labels

    min_len = min([len(f) for f in files_list])

    s_start = 0
    s_end = 0
    
    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[i] == files_list[0][i]):
                string_equal = False
                break

        if string_equal:
            s_start += 1
        else:
            break

    for i in range(min_len):
        string_equal = True
        for f in files_list:
            if not(f[-1-i] == files_list[0][-1-i]):
                string_equal = False
                break

        if string_equal:
            s_end += 1
        else:
            break

    labels = [f[s_start:-s_end] for f in files_list]

    # If there is a single empty label then append everything that is before and/or after os.sep
    if '' in labels:
        labels = []

        do_start = any(os.sep != f[s_start] for f in files_list)
        do_end = any(os.sep != f[-s_end] for f in files_list)

        for f in files_list:

            s_start_current = s_start
            s_end_current = s_end

            if do_start:
                s_start_current = s_start - 1
                while f[s_start_current] is not os.sep and s_start_current > 0:
                    s_start_current -= 1
                if f[s_start_current] is os.sep:
                    s_start_current += 1

            if do_end:
                s_end_current = s_end + 1
                while f[-s_end_current] is not os.sep and s_end_current < 0:
                    s_end_current += 1
                if f[-s_end_current] is os.sep:
                    s_end_current -= 1
        
            labels.append(f[s_start_current:-s_end_current])

    return labels

def get_markers(n_plot, n_points, n_total_markers, available_markers = None):

    if available_markers and n_total_markers > 0:
        n_every = round(n_points / n_total_markers)
        n_every = max(1, n_every)

        m_type = available_markers[n_plot % len(available_markers)]
    else:
        n_every = 1
        m_type = None

    return m_type, n_every

def plot_distribution_history(ax_mu_std, ax_total, time, means, stds, total):
    ax_mu_std.plot(time, means, linewidth=2, color='blue')
    ax_mu_std.fill_between(time, means-stds, means+stds, alpha=0.4, color='#888888')
    ax_mu_std.fill_between(time, means-2*stds, means+2*stds, alpha=0.4, color='#cccccc')

    ax_total.plot(time, total, linewidth=2, color='red')

def format_distribution_plot(ax_mu_std, ax_total, qty_name):
    ax_mu_std.grid(True)
    ax_mu_std.set_title("Average value")
    ax_mu_std.set_xlabel("time")
    ax_mu_std.set_ylabel(qty_name)
    #ax_mu_std.set_ylim([0.9*np.min(means-3*stds), 1.1*np.max(means+3*stds)])

    ax_total.grid(True)
    ax_total.set_title("Number of entities")
    ax_total.set_xlabel("time")
    ax_total.set_ylabel("#")

def animation_init_plot(background_color, x_size, y_size):
    fig, ax = plt.subplots(1, 1, dpi=100, facecolor=background_color)
    fig.set_figheight(y_size)
    fig.set_figwidth(x_size)

    return fig, ax

def animation_format_plot(ax, main_color, background_color, ticks_size, axes_label_size):
    ax.spines['bottom'].set_color(main_color)
    ax.spines['top'].set_color(main_color)
    ax.spines['right'].set_color(main_color)
    ax.spines['left'].set_color(main_color)
    ax.xaxis.label.set_color(main_color)
    ax.yaxis.label.set_color(main_color)
    ax.tick_params(axis='x', colors=main_color, labelsize=ticks_size)
    ax.tick_params(axis='y', colors=main_color, labelsize=ticks_size)
    ax.set_facecolor(background_color)
    ax.xaxis.label.set_size(axes_label_size)
    ax.yaxis.label.set_size(axes_label_size)

def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
