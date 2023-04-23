import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit


class Plot:
    @staticmethod
    def one_arr(y, x=None, title="", x_label="", y_label=""):
        """
        plot one arr

        :param y: y-axis values
        :param x: x-axis values. if None then generates 0-len(y)
        :param title: the graph title
        :param x_label: x-axis label
        :param y_label: y-axis label
        """
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()

    @staticmethod
    def many_arrays(arrays: list, descriptions: list = [], title="", x_label="", y_label="", err=[]):
        """
        plot many arrays on one graph

        :param arrays: array of tuples - (x, y)
        :param descriptions: array of the graphs labels
        :param title: the graph title
        :param x_label: x-axis label
        :param y_label: y-axis label
        """
        for i, (x, y) in enumerate(arrays):
            if x == np.array([]):
                x = np.arange(0, len(y))
            if not descriptions:
                plt.plot(x, y)
            else:
                plt.plot(x, y, label=descriptions[i], c='r')
                if err != []:
                    plt.errorbar(x, y, err, linestyle='None')

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_csv(path):
        # Load CSV file into a pandas DataFrame
        df = pd.read_csv(path)

        # Set the plot style
        sns.set_style('whitegrid')

        # Create a line plot of the time and value columns
        plt.plot(df['Time'], df['Channel B'], color='blue')

        # Add axis labels and a title
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Voltage (mV)', fontsize=12)
        # plt.title('Voltage Measured in Channel B', fontsize=14)
        plt.text(0.5, 1.10, '2nd Voltage Measurement in Channel B', fontsize=14, ha='center', va='center',
                 transform=plt.gca().transAxes)
        # Add a subtitle
        plt.text(0.5, 1.02, 'Vin = 600[mV] --- R = 1.5[Ohm]', fontsize=10, ha='center', va='center',
                 transform=plt.gca().transAxes)

        # Add a legend
        plt.legend(['Voltage'], loc='upper left', fontsize=10)

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.xticks(range(-6, 7, 1), fontsize=10)
        plt.yticks(range(-3, 4, 1), fontsize=10)
        # Add gridlines to the plot
        plt.grid(True, alpha=0.5)

        # Show the plot
        plt.show()

    @staticmethod
    def plot_fit(x_points, y_points, fit_func, opt_params, descriptions: list = [], title="", x_label="", y_label="",
                 err=[]):
        x_fit = np.linspace(min(x_points), max(x_points), 10000)
        y_fit = FittingTools.apply_fit_function(fit_func, opt_params, x_fit)

        plt.scatter(x_points, y_points, label=descriptions[0], c='r')
        if err != []:
            plt.errorbar(x_points, y_points, err, linestyle='None')

        plt.plot(x_fit, y_fit, label=descriptions[1])

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.show()


class FittingTools:
    @staticmethod
    def find_fit(curve, xdata, ydata):
        """
        :param curve: a python function that dependents on  the params we want to optimize
        :param xdata: xdata
        :param ydata: ydata
        :return: optinal_parameters
        """
        optinal_parameters, covariance = curve_fit(curve, xdata, ydata)

        return optinal_parameters

    @staticmethod
    def apply_fit_function(func, opt_params, x_arr: list):
        res = []
        for x in x_arr:
            res.append(func(x, opt_params))
        return res



def calc_r1():
    v_in_arr = np.array([1000, 1000, 1000, 1000, 1000])
    R_arr = np.array([47, 470, 680, 1000, 2200])
    v_out_arr = np.array([50.5, 377, 430, 531.8, 712.7])

    r1_arr = R_arr * (v_in_arr - v_out_arr) / v_out_arr

    # calc the avg and std
    avg = np.average(r1_arr)
    std = np.std(r1_arr)
    std_arr = np.array([std] * 5)
    xfit = np.linspace(min(R_arr), max(R_arr), 10000)
    true_graph = 1000 * xfit / (avg + xfit)

    # calc err
    v_in_plus_arr = np.array([1010, 1010, 1010, 1010, 1010])
    v_in_minus_arr = np.array([990, 990, 990, 990, 990])

    vout_plus_arr = v_in_plus_arr * R_arr / (avg + R_arr)
    vout_minus_arr = v_in_minus_arr * R_arr / (avg + R_arr)

    vout_err = vout_plus_arr - vout_minus_arr

    Plot.plot_fit(R_arr, v_out_arr, xfit, true_graph, descriptions=['Experiment', 'Formula'],
                  x_label="R [ohm]", y_label="Vout [mili - volt]", err=vout_err)

    return avg, std, vout_err


if __name__ == '__main__':
    r1, std, vout_err = calc_r1()
    print('r1, std, vout_err = ', r1, std, vout_err)
