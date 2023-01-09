# PURPOSE:  Plot an eye diagram of a signal
#
# INPUTS:
#   y_s:    vector of signal samples out of the matched filter
#   N:      the number of samples per symbol.  Assumes that time 0 is 
#           at sample y_s[0].  If not, you must send in an offset integer.
#   offset: the number of samples at the start of y_s to ignore
#
# OUTPUTS:  h, the handle to the lines plotted


def plot_eye_diagram(y_s, N, offset=0)

    start_indices = range( int(np.floor(N/2.0)) + offset - 1, len(y_s) - N, N)
    time_vals     = np.arange(-0.5, 0.5+1.0/N, 1.0/N)

    plt.figure()
    for i, start_i in enumerate(start_indices):
        plt.plot(time_vals, y_s[start_i:(start_i+N+1)], 'b-', linewidth=2)
        
    plt.xlabel('Time t/T_s_y', fontsize=16)
    plt.ylabel('M.F. Output', fontsize=16)
    plt.show()
