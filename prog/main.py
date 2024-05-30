from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidget, QProgressBar, QTableWidgetItem, QSizePolicy
from lingspam_filter import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

  

class LingSpamFilterGUI(QtWidgets.QMainWindow):
    DEFAULT_TEST_DIR = "ling_public/lemm_stop/test-mails"
    DEFAULT_TRAIN_DIR = "ling_public/lemm_stop/train-mails"

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
    # Создайте виджеты для интерфейса
        self.resize(800, 600)
        self.label_train_dir = QtWidgets.QLabel("Обучающее множество:")
        self.lineEdit_train_dir = QtWidgets.QLineEdit()
        self.lineEdit_train_dir.setText(self.DEFAULT_TRAIN_DIR)
        self.button_browse_train_dir = QtWidgets.QPushButton("Обзор")
        self.label_test_dir = QtWidgets.QLabel("Тестирующее множество:")
        self.lineEdit_test_dir = QtWidgets.QLineEdit()
        self.lineEdit_test_dir.setText(self.DEFAULT_TEST_DIR)
        self.button_browse_test_dir = QtWidgets.QPushButton("Обзор")
        self.button_run = QtWidgets.QPushButton("Запуск")

        # Создайте таблицы для отображения результатов
        self.table_test_results = QTableWidget()
        self.table_test_results.setRowCount(2)
        self.table_test_results.setColumnCount(3)
        
        self.table_test_results.setHorizontalHeaderLabels(['', 'Не спам','Спам'])
        # self.table_test_results.resizeColumnsToContents()
        self.table_test_results.setColumnHidden(0, True)
        self.table_test_results.setVerticalHeaderLabels(['Не спам', 'Спам'])
        

        # Создайте полосы загрузки для обучения и тестирования
        self.progress_bar_train = QProgressBar()
        self.progress_bar_test = QProgressBar()

        # Создайте layout и добавьте виджеты
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label_train_dir)
        layout.addWidget(self.lineEdit_train_dir)
        layout.addWidget(self.button_browse_train_dir)
        layout.addWidget(self.label_test_dir)
        layout.addWidget(self.lineEdit_test_dir)
        layout.addWidget(self.button_browse_test_dir)
        layout.addWidget(self.button_run)
        layout.addWidget(self.progress_bar_train)
        layout.addWidget(self.table_test_results)
        layout.addWidget(self.progress_bar_test)

        # Создайте central widget и установите layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Свяжите сигналы и слоты
        self.button_browse_train_dir.clicked.connect(self.browse_train_dir)
        self.button_browse_test_dir.clicked.connect(self.browse_test_dir)
        self.button_run.clicked.connect(self.run_filter)

        self.setWindowTitle("Ling Spam Filter") 
        
    # def on_resize(self, event):
    #     self.table_test_results.resizeColumnsToContents()
        
    #     super().resizeEvent(event)
        
    def browse_train_dir(self):
        train_dir = QFileDialog.getExistingDirectory(self, "Выберите обучающее множество")
        if train_dir:
            self.lineEdit_train_dir.setText(train_dir)

    def browse_test_dir(self):
        test_dir = QFileDialog.getExistingDirectory(self, "Выберите тестирующее множество")
        if test_dir:
            self.lineEdit_test_dir.setText(test_dir)

    def run_filter(self):
        train_dir = self.lineEdit_train_dir.text()
        test_dir = self.lineEdit_test_dir.text()
        if not train_dir or not test_dir:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, выберите обучающее и тренирующие множества.")
            return
        self.run_lingspam_filter(train_dir, test_dir)


    def run_lingspam_filter(self, train_dir, test_dir):
        self.progress_bar_train.setValue(0)
        self.progress_bar_test.setValue(0)
        # Шаг 1: Разбор входящих почтовых сообщений на термы и создание частотного словаря термов с коэффициентами спамности
        time1 = time.time()
        spam_index = make_spam_index(train_dir)
        # print(spam_index)
        train_feature_vector = extract_weighted_features(train_dir, spam_index)
        test_feature_vector = extract_weighted_features(test_dir, spam_index)
        self.progress_bar_train.setValue(25)
        # Шаг 2: Определение принадлежности каждого сообщения к спаму или не-спаму
        train_labels = np.zeros(702)
        train_labels[351:702] = 1
        test_labels = np.zeros(260)
        test_labels[130:260] = 1
        print("SVM TRAINING ", time.time() - time1)
        self.progress_bar_train.setValue(50)

        
        
        # Шаг 3: Обучение нейронной сети
        nn_model = MLPClassifier( 
            hidden_layer_sizes=(100, 100),
            activation='relu',
            max_iter=100,
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10 
                        
        )
        time1 = time.time()
        nn_model.fit(train_feature_vector, train_labels)
        print("NN TRAINING ", time.time() - time1)
        self.progress_bar_train.setValue(100)
        # Шаг 4: Получение результатов на тестовых данных
        
        
        result = nn_model.predict(test_feature_vector)
        
        self.progress_bar_test.setValue(100)

        # Шаг 6: Отображение результатов в таблицах
        print(accuracy_score(test_labels, result))
        self.update_test_results_table(confusion_matrix(test_labels, result))



    def update_test_results_table(self, test_results):
        for row in range(len(test_results)):
            for col in range(len(test_results[0])):
                self.table_test_results.setItem(row, col+1, QTableWidgetItem(str(test_results[row][col])))
   
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = LingSpamFilterGUI()
    window.show()
    app.exec_()
