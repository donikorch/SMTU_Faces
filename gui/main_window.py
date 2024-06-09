from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QDesktopWidget, QLineEdit, QLabel, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import importlib.util
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_encoding = None
        self.initUI()

    def initUI(self):
        # Создаем основной вертикальный layout
        self.main_layout = QVBoxLayout()

        # Устанавливаем шрифт для кнопок
        button_font = QFont('Arial', 16)

        # Добавляем логотип
        self.logo_label = QLabel(self)
        pixmap = QPixmap('src/etc/SMTU_Logo.png')  # Замените 'path_to_logo.png' на путь к вашему логотипу
        self.logo_label.setPixmap(pixmap.scaled(145, 250))
        self.logo_label.setAlignment(Qt.AlignCenter)

        # Создаем и настраиваем кнопки главного экрана
        self.add_person_button = QPushButton('Добавить человека', self)
        self.add_person_button.setFont(button_font)
        self.add_person_button.setFixedSize(300, 50)
        self.add_person_button.clicked.connect(self.show_add_person_interface)

        self.recognize_person_button = QPushButton('Распознать человека', self)
        self.recognize_person_button.setFont(button_font)
        self.recognize_person_button.setFixedSize(300, 50)
        self.recognize_person_button.clicked.connect(self.run_face_recognition)

        self.detect_disease_button = QPushButton('Определить заболевания', self)
        self.detect_disease_button.setFont(button_font)
        self.detect_disease_button.setFixedSize(300, 50)
        self.detect_disease_button.clicked.connect(self.run_face_acne_recognition)

        # Добавляем кнопки в основной layout
        self.main_layout.addWidget(self.logo_label)
        self.main_layout.addWidget(self.add_person_button)
        self.main_layout.addWidget(self.recognize_person_button)
        self.main_layout.addWidget(self.detect_disease_button)

        # Выровняем кнопки по центру
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Создаем layout для интерфейса добавления человека
        self.add_person_layout = QVBoxLayout()

        # Кнопка "Сделать снимок"
        self.capture_button = QPushButton('Сделать снимок', self)
        self.capture_button.setFont(button_font)
        self.capture_button.setFixedSize(300, 50)
        self.capture_button.clicked.connect(self.run_capture)
        self.add_person_layout.addWidget(self.capture_button)

        # Создаем QLabel для отображения "или" между кнопками
        self.or_label = QLabel("или", self)
        self.or_label.setFont(button_font)
        self.or_label.setAlignment(Qt.AlignCenter)
        self.or_label.setFixedSize(300, 15)
        self.add_person_layout.addWidget(self.or_label)

        # Кнопка "Загрузить изображение"
        self.load_image_button = QPushButton('Загрузить изображение', self)
        self.load_image_button.setFont(button_font)
        self.load_image_button.setFixedSize(300, 50)
        self.load_image_button.clicked.connect(self.load_image)
        self.add_person_layout.addWidget(self.load_image_button)

        # Поля ввода для имени, фамилии и отчества
        self.first_name_input = QLineEdit(self)
        self.first_name_input.setPlaceholderText('Имя')
        self.first_name_input.setFont(button_font)
        self.first_name_input.setFixedSize(300, 40)
        self.first_name_input.setDisabled(True)
        self.add_person_layout.addWidget(self.first_name_input)

        self.last_name_input = QLineEdit(self)
        self.last_name_input.setPlaceholderText('Фамилия')
        self.last_name_input.setFont(button_font)
        self.last_name_input.setFixedSize(300, 40)
        self.last_name_input.setDisabled(True)
        self.add_person_layout.addWidget(self.last_name_input)

        self.patronymic_input = QLineEdit(self)
        self.patronymic_input.setPlaceholderText('Отчество')
        self.patronymic_input.setFont(button_font)
        self.patronymic_input.setFixedSize(300, 40)
        self.patronymic_input.setDisabled(True)
        self.add_person_layout.addWidget(self.patronymic_input)

        # Кнопка "ОК"
        self.ok_button = QPushButton('ОК', self)
        self.ok_button.setFont(button_font)
        self.ok_button.setFixedSize(300, 50)
        self.ok_button.setDisabled(True)
        self.ok_button.clicked.connect(self.save_person)
        self.add_person_layout.addWidget(self.ok_button)

        # Кнопка "Назад"
        self.back_button = QPushButton('Назад', self)
        self.back_button.setFont(button_font)
        self.back_button.setFixedSize(300, 50)
        self.back_button.clicked.connect(self.show_main_interface)
        self.add_person_layout.addWidget(self.back_button)

        # Устанавливаем основной layout
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.main_layout)
        self.layout.addLayout(self.add_person_layout)

        self.layout.setAlignment(Qt.AlignCenter)

        # Скрываем интерфейс добавления человека по умолчанию
        self.set_interface_visible(self.add_person_layout, False)

        # Настройки окна
        self.setWindowTitle('SMTU Faces')
        self.setFixedSize(400, 500)
        self.center()

    def center(self):
        # Получаем прямоугольник экрана с учетом всех доступных экранов
        qr = self.frameGeometry()
        # Определяем центр экрана
        cp = QDesktopWidget().availableGeometry().center()
        # Перемещаем прямоугольник окна в центр экрана
        qr.moveCenter(cp)
        # Перемещаем верхний левый угол окна в верхний левый угол прямоугольника
        self.move(qr.topLeft())

    def set_interface_visible(self, layout, visible):
        # Показываем или скрываем интерфейс в зависимости от параметра visible
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            widget.setVisible(visible)

    def show_interface(self, show):
        # Показываем или скрываем интерфейс добавления человека в зависимости от флага
        self.set_interface_visible(self.add_person_layout, show)

        # Показываем или скрываем основной интерфейс в зависимости от флага
        self.set_interface_visible(self.main_layout, not show)

    def show_add_person_interface(self):
        self.show_interface(True)

    def show_main_interface(self):
        self.show_interface(False)

    def load_image(self):
        # Открыть диалоговое окно выбора файла
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_path, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "",
                                                    "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if image_path:
            # Передать выбранное изображение в функцию capture_and_save_face
            self.run_capture(image_path)

    def import_module(self, module_name):
        spec = importlib.util.spec_from_file_location(module_name, f"src/{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_capture(self, image=None):
        # Изменения в функции run_capture для обработки переданного изображения
        create_embedding = self.import_module("create_embedding")

        if image:
            # Если передано изображение, передать его в функцию capture_and_save_face
            success, face_encoding = create_embedding.capture_and_save_face(image)
        else:
            success, face_encoding = create_embedding.capture_and_save_face()

        if not success:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Лицо не обнаружено.")
            msg.setWindowTitle("Ошибка")
            msg.exec_()
            return

        # Сохраняем face_encoding в атрибуте класса
        self.face_encoding = face_encoding

        # Блокируем кнопку "Сделать снимок"
        self.capture_button.setDisabled(True)
        self.load_image_button.setDisabled(True)

        # Разблокируем поля ввода
        self.first_name_input.setDisabled(False)
        self.last_name_input.setDisabled(False)
        self.patronymic_input.setDisabled(False)

        # Подключаем разблокировку кнопки "ОК" после ввода данных
        self.first_name_input.textChanged.connect(self.enable_ok_button)
        self.last_name_input.textChanged.connect(self.enable_ok_button)
        self.patronymic_input.textChanged.connect(self.enable_ok_button)

    def enable_ok_button(self):
        if self.first_name_input.text() and self.last_name_input.text() and self.patronymic_input.text():
            self.ok_button.setDisabled(False)
        else:
            self.ok_button.setDisabled(True)

    def save_person(self):
        # Получаем введенные данные
        first_name = self.first_name_input.text()
        last_name = self.last_name_input.text()
        middle_name = self.patronymic_input.text()

        # Подключение и вызов capture_face.py
        create_embedding = self.import_module("create_embedding")

        # Сохранение лица в базу данных
        success = create_embedding.save_person_to_db(first_name, last_name, middle_name, np.array(self.face_encoding).tolist())

        if success:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Лицо успешно сохранено в базе данных.")
            msg.setWindowTitle("Успех")
            msg.exec_()

            # Очистка полей ввода и блокировка кнопки "ОК"
            self.first_name_input.clear()
            self.last_name_input.clear()
            self.patronymic_input.clear()
            self.ok_button.setDisabled(True)

            # Возврат на основной интерфейс
            self.show_main_interface()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Ошибка при сохранении лица в базу данных.")
            msg.setWindowTitle("Ошибка")
            msg.exec_()

    def run_script(self, script_name):
        try:
            # Подключаем и вызываем скрипт по его названию
            spec = importlib.util.spec_from_file_location(script_name, f"src/{script_name}.py")
            script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(script)
        except Exception as e:
            # Обработка возможных ошибок при запуске скрипта
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f"Ошибка при выполнении скрипта {script_name}: {str(e)}")
            msg.setWindowTitle("Ошибка")
            msg.exec_()

    def run_face_recognition(self):
        self.run_script("face")

    def run_face_acne_recognition(self):
        self.run_script("face_acne")