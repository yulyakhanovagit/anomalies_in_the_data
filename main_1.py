import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# import data
data = pd.read_csv("supermarket_sales.csv")
test = pd.read_csv("supermarket_sales_test.csv")
data = data.values
test = test.values
X_train = data[:, 6:10]
y_train = data[:, 5]
X_test = test[:, 6:10]
y_test = test[:, 5]
normal = test[:, 17]

model_OCSVM = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
model_OCSVM = model_OCSVM.fit(X_train)
y_pred_OCSVM = model_OCSVM.predict(X_test)

model_IF = IsolationForest(n_estimators = 60, max_samples=300, random_state=1, contamination=0.1)
model_IF = model_IF.fit(X_train)
y_pred_IF = model_IF.predict(X_test)

model_DBSCAN = DBSCAN(min_samples=6, eps=100)
y_pred_DBSCAN = model_DBSCAN.fit_predict(X_test)
y_pred_DBSCAN[y_pred_DBSCAN != -1] = 1

y_pred = y_pred_OCSVM.transpose() + y_pred_IF.transpose() + y_pred_DBSCAN.transpose()
y_pred[y_pred < 0] = -1
y_pred[y_pred > 0] = 1

normal = normal.astype(int)


def OCSVM(y_pred_OCSVM):
    print(y_pred_OCSVM)
    n_error_test_OCSVM = y_pred_OCSVM[y_pred_OCSVM == -1].size
    cm = confusion_matrix(normal, y_pred_OCSVM)
    print("number of detected anomalies(OCSVM): ", n_error_test_OCSVM)
    print('error type 1: ', cm[0][1])
    print('error type 2: ', cm[1][0])
    print('accuracy: ', '{:.2%}\n'.format(accuracy(normal, y_pred_OCSVM)))
    return

def IF(y_pred_IF):
    print(y_pred_IF)
    n_error_test_IF = y_pred_IF[y_pred_IF == -1].size
    cm = confusion_matrix(normal, y_pred_IF)
    print('error type 1: ', cm[0][1])
    print('error type 2: ', cm[1][0])
    print("number of detected anomalies(IF): ", n_error_test_IF)
    print('accuracy: ', '{:.2%}\n'.format(accuracy(normal, y_pred_IF)))
    return

def DBSCAN_(y_pred_DBSCAN):
    print(y_pred_DBSCAN)
    n_error_test_DBSCAN = y_pred_DBSCAN[y_pred_DBSCAN == -1].size
    cm = confusion_matrix(normal, y_pred_DBSCAN)
    print('error type 1: ', cm[0][1])
    print('error type 2: ', cm[1][0])
    print("number of detected anomalies(DBSCAN): ", n_error_test_DBSCAN)
    print('accuracy: ', '{:.2%}\n'.format(accuracy(normal, y_pred_DBSCAN)))
    return

def ensemble(y_pred):
    y_pred[y_pred < 0] = -1
    y_pred[y_pred > 0] = 1
    print(y_pred)
    n_error_test = y_pred[y_pred == -1].size
    cm = confusion_matrix(normal, y_pred)
    print('error type 1: ', cm[0][1])
    print('error type 2: ', cm[1][0])
    print("number of detected anomalies: ", n_error_test)
    print('accuracy: ', '{:.2%}\n'.format(accuracy(normal, y_pred)))
    return

def accuracy(normal, y_pred):
    cm = confusion_matrix(normal, y_pred)
    a = cm[0][0] + cm[1][1]
    return a/len(y_pred)

def errors(normal, y_pred):
    err_1 = 0
    err_2 = 0
    if normal!=y_pred and normal==1:
        err_1 += 1
    elif normal!=y_pred and normal==-1:
        err_2 += 1
    err = [err_1, err_2]
    return err

def plots(title, normal):
    d = {-1: 'red', 1: '#a799ff'}
    fig, ax = plt.subplots()
    ax.scatter(X_test[:, 0], X_test[:, 3], c=[d[y] for y in normal], s=5)
    pop_a = mpatches.Patch(color='#a799ff', label='нормальные значения')
    pop_b = mpatches.Patch(color='red', label='аномалии')
    plt.legend(handles=[pop_a, pop_b])
    ax.set_title(title)
    plt.xlabel('признак 1')
    plt.ylabel('признак 2')
    plt.show()
    return

def plots_param(normal):
    d = {-1: 'red', 1: '#a799ff'}
    fig, ax = plt.subplots(2,2)
    ax[0][0].plot(range(0, 51, 1), X_test[:, 0], c='#a799ff')
    ax[0][0].scatter(range(0, 51, 1), X_test[:, 0], c=[d[y] for y in normal], s=7, label='аномалии')
    ax[0][0].set_title('признак 1')
    #ax[0][0].legend()

    ax[0][1].plot(range(0, 51, 1), X_test[:, 1], c='#a799ff')
    ax[0][1].scatter(range(0, 51, 1), X_test[:, 1], c=[d[y] for y in normal], s=7, label='аномалии')
    ax[0][1].set_title('признак 2')
    #ax[0][1].legend()

    ax[1][0].plot(range(0, 51, 1), X_test[:, 2], c='#a799ff')
    ax[1][0].scatter(range(0, 51, 1), X_test[:, 2], c=[d[y] for y in normal], s=7, label='аномалии')
    ax[1][0].set_title('признак 3')
    #ax[1][0].legend()

    ax[1][1].plot(range(0, 51, 1), X_test[:, 3], c='#a799ff')
    ax[1][1].scatter(range(0, 51, 1), X_test[:, 3], c=[d[y] for y in normal], s=7, label='аномалии')
    ax[1][1].set_title('признак 4')
    #ax[1][1].legend()

    fig.set_figheight(100)
    fig.set_figwidth(70)
    plt.show()
    return

def plot_matrix(y_pred, labels):

        title = labels
        cmap = plt.cm.Purples
        cm = confusion_matrix(y_test, y_pred)
        target_names = ['аномальные', 'нормальные']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        width, height = cm.shape
        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        return


#normal = normal.astype(int)
#plot_matrix(normal, y_pred, 'Ensemble')

#OCSVM(y_pred_OCSVM)
#print(normal)
#print(y_pred_OCSVM)
#print(errors(normal, y_pred_OCSVM))
#IF(y_pred_IF)
#DBSCAN_()
#ensemble(y_pred)
#print(y_train)
#print(X_test)
#plots_param(y_pred_OCSVM)
#plots('OCSVM', y_pred_OCSVM)
#y_pred = np.argmax(y_pred,axis=1)


#print(confusion_matrix(normal, y_pred))
#print(confusion_matrix(normal, y_pred_OCSVM))
#print(confusion_matrix(normal, y_pred_IF))
#print(confusion_matrix(normal, y_pred_DBSCAN))


while True:
    print("1 - OneClassSVM - метод опорных векторов для одного класса")
    print("2 - IsolationForest - метод изолирующего леса")
    print("3 - DBSCAN - пространственная кластеризация, основанная на плотности")
    print("4 - Ансамбль моделей")
    print("5 - Тестовые данные")
    print("0 - Выйти из программы")
    cmd = input("Выберите алгоритм выявления аномалий: ")

    if cmd == "1":
        print('OneClassSVM: ')
        OCSVM(y_pred_OCSVM)
        while True:
            print("1 - Вывести график полученных аномалий")
            print("2 - Вывести графики по каждому признаку")
            print("3 - Вывести матрицу путаницы")
            print("0 - Выйти к предыдущему выбору")
            cmd_1 = input("Выберите: ")
            if cmd_1 == "1":
                plots('OneClassSVM', y_pred_OCSVM)
            elif cmd_1 == "2":
                plots_param(y_pred_OCSVM)
            elif cmd_1 == "3":
                plot_matrix(normal, y_pred_OCSVM, 'OneClassSVM')
            elif cmd_1 == "0":
                break
            else:
                print("Вы ввели не правильное значение")
    elif cmd == "2":
        print('IsolationForest: ')
        IF(y_pred_IF)
        while True:
            print("1 - Вывести график полученных аномалий")
            print("2 - Вывести графики по каждому признаку")
            print("3 - Вывести матрицу путаницы")
            print("0 - Выйти к предыдущему выбору")
            cmd_2 = input("Выберите: ")
            if cmd_2 == "1":
                plots('IsolationForest', y_pred_IF)
            elif cmd_2 == "2":
                plots_param(y_pred_IF)
            elif cmd_2 == "3":
                plot_matrix(normal, y_pred_IF, 'IsolationForest')
            elif cmd_2 == "0":
                break
            else:
                print("Вы ввели не правильное значение")
    elif cmd == "3":
        print('DBSCAN: ')
        DBSCAN_(y_pred_DBSCAN)
        while True:
            print("1 - Вывести график полученных аномалий")
            print("2 - Вывести графики по каждому признаку")
            print("3 - Вывести матрицу путаницы")
            print("0 - Выйти к предыдущему выбору")
            cmd_3 = input("Выберите: ")
            if cmd_3 == "1":
                plots('DBSCAN', y_pred_DBSCAN)
            elif cmd_3 == "2":
                plots_param(y_pred_DBSCAN)
            elif cmd_3 == "3":
                plot_matrix(normal, y_pred_DBSCAN, 'DBSCAN')
            elif cmd_3 == "0":
                break
            else:
                print("Вы ввели не правильное значение")
    elif cmd == "4":
        print('Ensamble: ')
        ensemble(y_pred)
        while True:
            print("1 - Вывести график полученных аномалий")
            print("2 - Вывести графики по каждому признаку")
            print("3 - Вывести матрицу путаницы")
            print("0 - Выйти к предыдущему выбору")
            cmd_4 = input("Выберите: ")
            if cmd_4 == "1":
                plots('Ensamble', y_pred)
            elif cmd_4 == "2":
                plots_param(y_pred)
            elif cmd_4 == "3":
                plot_matrix(normal, y_pred, 'Ensamble')
            elif cmd_4 == "0":
                break
            else:
                print("Вы ввели не правильное значение")
    elif cmd == "5":
        print('Тестовые данные (аномалии - "-1")')
        print(normal)
        while True:
            print("1 - Вывести график полученных аномалий")
            print("2 - Вывести графики по каждому признаку")
            print("3 - Вывести матрицу путаницы")
            print("0 - Выйти к предыдущему выбору")
            cmd_5 = input("Выберите: ")
            if cmd_5 == "1":
                plots('Test', normal)
            elif cmd_5 == "2":
                plots_param(normal)
            elif cmd_5 == "0":
                break
            else:
                print("Вы ввели не правильное значение")
    elif cmd == "0":
        break
    else:
        print("Вы ввели не правильное значение")






