import numpy
import matplotlib.pyplot as plt


# used by project1
def pic_1(theta_list, cost_list, x, y, interval):
    pause_time = 0.05
    n_theta = len(theta_list)
    n_cost = len(cost_list)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xlim(1999, 2015)
    ax1.set_ylim(1000.0, 15000.0)
    ax2.set_xlim(0, len(cost_list))
    ax2.set_ylim(cost_list[n_cost - 1] - 1, cost_list[0] + 1)
    ax1.set_title("Fitted Curve")
    ax2.set_title("Cost")
    for i in range(len(x.tolist())):
        ax1.scatter(x.tolist()[i][1] + 2000, y.T.tolist()[0][i] * 1000, c='red', marker='o')
    x_p = numpy.linspace(1998.0, 2015.0, 100)
    cost_temp_x = [0]
    cost_temp_y = [numpy.mat(cost_list[0]).tolist()[0][0]]
    plt.ion()  # open interactive model
    for i in range(n_cost):  # n_cost always large than n_theta
        if i % interval == 0:
            cost_temp_x.append(i)
            cost_temp_y.append(numpy.mat(cost_list[i]).tolist()[0][0])
            plt.pause(pause_time)
            try:
                ax1.lines.remove(lines1[0])
                ax2.lines.remove(lines2[0])
            except Exception:
                pass
            lines1 = ax1.plot(x_p, theta_list[i].T.tolist()[0][0] * 1000 + theta_list[i].T.tolist()[0][1] * 1000 * (
                    x_p - 2000))
            lines2 = ax2.plot(cost_temp_x, cost_temp_y)
    plt.ioff()
    plt.show()


# used by project2 , project3
def pic_2(theta_list, cost_list, x, y, interval):
    pause_time = 0.05
    n_theta = len(theta_list)
    n_cost = len(cost_list)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xlim(15, 65)
    ax1.set_ylim(40, 90)
    ax2.set_xlim(0, len(cost_list))
    ax2.set_ylim(cost_list[n_cost - 1] - 1, cost_list[0] + 1)
    ax1.set_title("Fitted Curve")
    ax2.set_title("Cost")
    for i in range(len(x.tolist())):
        if y.T.tolist()[0][i] == 1.0:
            ax1.scatter(x.tolist()[i][1], x.tolist()[i][2], c='green', marker='o')
        else:
            ax1.scatter(x.tolist()[i][1], x.tolist()[i][2], c='red', marker='x')
    x_p = numpy.linspace(15, 65, 100)
    cost_temp_x = [0]
    cost_temp_y = [numpy.mat(cost_list[0]).tolist()[0][0]]
    plt.ion()  # open interactive model
    for i in range(n_cost):  # n_cost always large than n_theta
        if i % interval == 0:
            cost_temp_x.append(i)
            cost_temp_y.append(numpy.mat(cost_list[i]).tolist()[0][0])
            plt.pause(pause_time)
            try:
                ax1.lines.remove(lines1[0])
                ax2.lines.remove(lines2[0])
            except Exception:
                pass
            if theta_list[i].T.tolist()[0][2] != 0.0:
                lines1 = ax1.plot(x_p, (-theta_list[i].T.tolist()[0][1] / theta_list[i].T.tolist()[0][2]) * x_p -
                                  theta_list[i].T.tolist()[0][0] / theta_list[i].T.tolist()[0][2])
            lines2 = ax2.plot(cost_temp_x, cost_temp_y)
    plt.ioff()
    plt.show()


# used by project2 , project4
def pic_3(theta_list, cost_list, x, y, interval):
    pause_time = 0.05
    n_theta = len(theta_list)
    n_cost = len(cost_list)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 2)
    ax2 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(15, 65)
    ax1.set_ylim(40, 90)
    ax2.set_xlim(0, len(cost_list))
    ax2.set_ylim(20, 1000)
    ax1.set_title("Fitted Curve")
    ax2.set_title("Cost")
    for i in range(len(x.tolist())):
        if y.T.tolist()[0][i] == 1.0:
            ax1.scatter(x.tolist()[i][1], x.tolist()[i][2], c='green', marker='o')
        else:
            ax1.scatter(x.tolist()[i][1], x.tolist()[i][2], c='red', marker='x')
    x_p = numpy.linspace(15, 65, 100)
    cost_temp_x = [0]
    cost_temp_y = [numpy.mat(cost_list[0]).tolist()[0][0]]
    plt.ion()  # open interactive model
    for i in range(n_cost):  # picture Cost figure
        if i % interval == 0:
            cost_temp_x.append(i)
            cost_temp_y.append(numpy.mat(cost_list[i]).tolist()[0][0])
            plt.pause(pause_time)
            try:
                ax2.lines.remove(lines2[0])
            except Exception:
                pass
            lines2 = ax2.plot(cost_temp_x, cost_temp_y)
    for i in range(n_theta):  # picture Fitted Curve figure
        plt.pause(pause_time)
        try:
            ax1.lines.remove(lines1[0])
        except Exception:
            pass
        if theta_list[i].T.tolist()[0][2] != 0.0:
            lines1 = ax1.plot(x_p, (-theta_list[i].T.tolist()[0][1] / theta_list[i].T.tolist()[0][2]) * x_p -
                              theta_list[i].T.tolist()[0][0] / theta_list[i].T.tolist()[0][2])
    plt.ioff()
    plt.show()


# used by project5
def pic_4(cost_list, accuracy_list, interval):
    pause_time = 0.001
    n_cost = len(cost_list[0])
    n_acc = len(accuracy_list[0])
    max_cost = cost_list[0][0]
    min_cost = cost_list[0][len(cost_list[0]) - 1]
    max_acc = accuracy_list[0][len(cost_list[0]) - 1]
    min_acc = accuracy_list[0][0]
    for i in range(5):
        if len(cost_list[i]) < n_cost:
            n_cost = len(cost_list[i])
        if cost_list[i][0] > max_cost:
            max_cost = cost_list[i][0]
        if cost_list[i][len(cost_list[i]) - 1] < min_cost:
            min_cost = cost_list[i][len(cost_list[i]) - 1]
        if len(accuracy_list[i]) > n_acc:
            n_acc = len(accuracy_list[i])
        if accuracy_list[i][len(accuracy_list[i]) - 1] > max_acc:
            max_acc = accuracy_list[i][len(accuracy_list[i]) - 1]
        if accuracy_list[i][0] < min_acc:
            min_acc = accuracy_list[i][0]
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlim(0, n_cost)
    ax1.set_ylim(min_cost - 1, max_cost + 1)
    ax1.set_title("Group 1-5 Cost")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlim(0, n_cost)
    ax2.set_ylim(min_acc - 0.3, max_acc + 0.3)
    ax2.set_title("Group 1-5 Accuracy")
    cost_temp_x = [0]
    cost_temp_y1 = [numpy.mat(cost_list[0][0]).tolist()[0][0]]
    cost_temp_y2 = [numpy.mat(cost_list[1][0]).tolist()[0][0]]
    cost_temp_y3 = [numpy.mat(cost_list[2][0]).tolist()[0][0]]
    cost_temp_y4 = [numpy.mat(cost_list[3][0]).tolist()[0][0]]
    cost_temp_y5 = [numpy.mat(cost_list[4][0]).tolist()[0][0]]
    acc_temp_x = [0]
    acc_temp_y1 = [numpy.mat(accuracy_list[0][0]).tolist()[0][0]]
    acc_temp_y2 = [numpy.mat(accuracy_list[1][0]).tolist()[0][0]]
    acc_temp_y3 = [numpy.mat(accuracy_list[2][0]).tolist()[0][0]]
    acc_temp_y4 = [numpy.mat(accuracy_list[3][0]).tolist()[0][0]]
    acc_temp_y5 = [numpy.mat(accuracy_list[4][0]).tolist()[0][0]]
    legend_flag = 1
    plt.ion()  # open interactive model
    for i in range(min(n_cost, n_acc)):  # n_cost always large than n_theta
        if i % interval == 0:
            cost_temp_x.append(i)
            cost_temp_y1.append(numpy.mat(cost_list[0][i]).tolist()[0][0])
            cost_temp_y2.append(numpy.mat(cost_list[1][i]).tolist()[0][0])
            cost_temp_y3.append(numpy.mat(cost_list[2][i]).tolist()[0][0])
            cost_temp_y4.append(numpy.mat(cost_list[3][i]).tolist()[0][0])
            cost_temp_y5.append(numpy.mat(cost_list[4][i]).tolist()[0][0])
            acc_temp_x.append(i)
            acc_temp_y1.append(numpy.mat(accuracy_list[0][i]).tolist()[0][0])
            acc_temp_y2.append(numpy.mat(accuracy_list[1][i]).tolist()[0][0])
            acc_temp_y3.append(numpy.mat(accuracy_list[2][i]).tolist()[0][0])
            acc_temp_y4.append(numpy.mat(accuracy_list[3][i]).tolist()[0][0])
            acc_temp_y5.append(numpy.mat(accuracy_list[4][i]).tolist()[0][0])
            plt.pause(pause_time)
            try:
                ax1.lines.remove(lines1)
                ax2.lines.remove(lines2)
            except Exception:
                pass
            lines1 = ax1.plot(cost_temp_x, cost_temp_y1, color='b', label='group 1')
            lines1 = ax1.plot(cost_temp_x, cost_temp_y2, color='g', label='group 2')
            lines1 = ax1.plot(cost_temp_x, cost_temp_y3, color='r', label='group 3')
            lines1 = ax1.plot(cost_temp_x, cost_temp_y4, color='y', label='group 4')
            lines1 = ax1.plot(cost_temp_x, cost_temp_y5, color='k', label='group 5')
            lines2 = ax2.plot(acc_temp_x, acc_temp_y1, color='b', label='group 1')
            lines2 = ax2.plot(acc_temp_x, acc_temp_y2, color='g', label='group 2')
            lines2 = ax2.plot(acc_temp_x, acc_temp_y3, color='r', label='group 3')
            lines2 = ax2.plot(acc_temp_x, acc_temp_y4, color='y', label='group 4')
            lines2 = ax2.plot(acc_temp_x, acc_temp_y5, color='k', label='group 5')
            if legend_flag == 1:
                ax1.legend()
                ax2.legend()
                legend_flag = 0
    plt.ioff()
    plt.show()


# be droped
# used by project5
def pic_4_2(cost_list, interval):
    pause_time = 0.02
    n_cost = len(cost_list[0])
    max_cost = cost_list[0][0]
    min_cost = cost_list[0][len(cost_list[0]) - 1]
    for i in range(5):
        if len(cost_list[i]) < n_cost:
            n_cost = len(cost_list[i])
        if cost_list[i][0] > max_cost:
            max_cost = cost_list[i][0]
        if cost_list[i][len(cost_list[i]) - 1] < min_cost:
            min_cost = cost_list[i][len(cost_list[i]) - 1]
    fig = plt.figure(figsize=(15, 8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax1.set_xlim(0, n_cost)
    ax2.set_xlim(0, n_cost)
    ax3.set_xlim(0, n_cost)
    ax4.set_xlim(0, n_cost)
    ax5.set_xlim(0, n_cost)
    ax1.set_ylim(min_cost - 1, max_cost + 1)
    ax2.set_ylim(min_cost - 1, max_cost + 1)
    ax3.set_ylim(min_cost - 1, max_cost + 1)
    ax4.set_ylim(min_cost - 1, max_cost + 1)
    ax5.set_ylim(min_cost - 1, max_cost + 1)
    ax1.set_title("Group 1 Cost")
    ax2.set_title("Group 2 Cost")
    ax3.set_title("Group 3 Cost")
    ax4.set_title("Group 4 Cost")
    ax5.set_title("Group 5 Cost")
    cost_temp_x = [0]
    cost_temp_y1 = [numpy.mat(cost_list[0][0]).tolist()[0][0]]
    cost_temp_y2 = [numpy.mat(cost_list[1][0]).tolist()[0][0]]
    cost_temp_y3 = [numpy.mat(cost_list[2][0]).tolist()[0][0]]
    cost_temp_y4 = [numpy.mat(cost_list[3][0]).tolist()[0][0]]
    cost_temp_y5 = [numpy.mat(cost_list[4][0]).tolist()[0][0]]
    plt.ion()  # open interactive model
    for i in range(n_cost):  # n_cost always large than n_theta
        if i % interval == 0:
            cost_temp_x.append(i)
            cost_temp_y1.append(numpy.mat(cost_list[0][i]).tolist()[0][0])
            cost_temp_y2.append(numpy.mat(cost_list[1][i]).tolist()[0][0])
            cost_temp_y3.append(numpy.mat(cost_list[2][i]).tolist()[0][0])
            cost_temp_y4.append(numpy.mat(cost_list[3][i]).tolist()[0][0])
            cost_temp_y5.append(numpy.mat(cost_list[4][i]).tolist()[0][0])
            plt.pause(pause_time)
            try:
                ax1.lines.remove(lines1[0])
                ax2.lines.remove(lines2[0])
                ax3.lines.remove(lines3[0])
                ax4.lines.remove(lines4[0])
                ax5.lines.remove(lines5[0])
            except Exception:
                pass
            lines1 = ax1.plot(cost_temp_x, cost_temp_y1)
            lines2 = ax2.plot(cost_temp_x, cost_temp_y2)
            lines3 = ax3.plot(cost_temp_x, cost_temp_y3)
            lines4 = ax4.plot(cost_temp_x, cost_temp_y4)
            lines5 = ax5.plot(cost_temp_x, cost_temp_y5)
    plt.ioff()
    plt.show()


# used by project5
def pic_5(accuracy, cost, it_time):
    name_list = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']
    max_a = accuracy[0]
    max_c = cost[0]
    max_i = it_time[0]
    for i in accuracy:
        if i > max_a:
            max_a = i
    for i in cost:
        if i > max_c:
            max_c = i
    for i in it_time:
        if i > max_i:
            max_i = i
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.set_ylim(0, max_a + 1)
    ax2.set_ylim(0, max_c + 3)
    ax3.set_ylim(0, max_i + 3000)
    x = list(range(len(accuracy)))
    ax1.bar(x, accuracy, label='accuracy', tick_label=name_list, fc='y')
    ax2.bar(x, cost, label='min cost', tick_label=name_list, fc='r')
    ax3.bar(x, it_time, label='iteration time', tick_label=name_list, fc='b')
    for a, b in zip(x, accuracy):
        ax1.text(a, b + 0.05, '%.4f' % b, ha='center', va='bottom')
    for a, b in zip(x, cost):
        ax2.text(a, b + 0.05, '%.4f' % b, ha='center', va='bottom')
    for a, b in zip(x, it_time):
        ax3.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom')
    ax1.set_title('accuracy')
    ax2.set_title('min cost')
    ax3.set_title('iteration time')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


# used by project6
def pic_6(mu0, mu1, sigma, x, y, pro_func, line_func):
    x_ = x.tolist()
    y_ = y.T.tolist()[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    for i in range(len(x_)):  # picture the points
        if y_[i] == 1.0:
            ax1.scatter(x_[i][0], x_[i][1], c='green', marker='+')
        else:
            ax1.scatter(x_[i][0], x_[i][1], c='blue', marker='x')

    x_0 = numpy.arange(15., 65., 0.1)
    y_0 = numpy.arange(40., 90., 0.1)
    x_1 = numpy.arange(15., 65., 0.1)
    y_1 = numpy.arange(40., 90., 0.1)
    X_0, Y_0 = numpy.meshgrid(x_0, y_0)
    X_1, Y_1 = numpy.meshgrid(x_1, y_1)
    C1 = ax1.contour(X_0, Y_0, pro_func(mu0, sigma, X_0, Y_0), 6, colors='red')
    C2 = ax1.contour(X_1, Y_1, pro_func(mu1, sigma, X_1, Y_1), 6, colors='black')

    x_line = numpy.linspace(15, 65, 100)  # picture the boundary line
    ax1.plot(x_line, line_func(mu0, mu1, sigma, x_line))

    plt.show()


# used by project7
def pic_7(phis0, phis1, predict_verify):
    phis0 = numpy.mat(phis0).tolist()[0]
    phis1 = numpy.mat(phis1).tolist()[0]
    predict_verify = numpy.mat(predict_verify).T.tolist()
    n = len(phis0)  # the dimension of phis
    m = len(predict_verify)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_ylim(0, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim(0, 1.2)

    name_list_1 = []
    for i in range(n):
        name_list_1.append("phi_" + str(i + 1))
    x_1 = list(range(n))
    total_width, k = 0.8, 2
    width = total_width / k
    ax1.bar(x_1, phis0, width=width, label='P(x|y=0)', tick_label=name_list_1, fc='y')
    for a, b in zip(x_1, phis0):
        ax1.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    for i in range(len(x_1)):
        x_1[i] = x_1[i] + width
    ax1.bar(x_1, phis1, width=width, label='P(x|y=1)', fc='r')
    for a, b in zip(x_1, phis1):
        ax1.text(a, b + 0.05, '%.2f' % b, ha='center', va='bottom')
    ax1.set_title('P(x|y=0) and P(x|y=1)')
    ax1.legend()

    name_list_2 = []
    for i in range(m):
        name_list_2.append("sample_" + str(i + 1))
    x_2 = list(range(m))
    total_width, k = 0.8, 2
    width = total_width / k
    ax2.bar(x_2, predict_verify[0], width=width, label='P(y=0|x)', tick_label=name_list_2, fc='y')
    for a, b in zip(x_2, predict_verify[0]):
        ax2.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom')
    for i in range(len(x_2)):
        x_2[i] = x_2[i] + width
    ax2.bar(x_2, predict_verify[1], width=width, label='P(y=1|x)', fc='r')
    for a, b in zip(x_2, predict_verify[1]):
        ax2.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom')
    ax2.set_title('the prediction of verify samples')
    ax2.legend()

    plt.show()
