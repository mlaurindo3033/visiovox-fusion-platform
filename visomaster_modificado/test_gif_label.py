from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QMovie
import sys

app = QApplication(sys.argv)

label = QLabel()
label.setFixedSize(64, 64)
label.setWindowTitle('Teste GIF QLabel')

movie = QMovie('C:/Users/miche/Music/VisoMaster/icons/reset.gif')
if not movie.isValid():
    label.setText('GIF inválido!')
    label.setStyleSheet('background: #f00; color: #fff;')
else:
    label.setMovie(movie)
    movie.start()

label.show()
sys.exit(app.exec()) 