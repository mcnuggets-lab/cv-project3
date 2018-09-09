from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.Qt import *
from PyQt5.QtCore import *


STYLE = """
    QWidget {
    color: #FFFFFF;
    background-color: #212121;
    border-width: 1px;}
"""


class BaseApplication(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLE)
        self.setGeometry(60, 60, 920, 800)
        self.viewer = None
        self.gl_viewer = None
        self.stacked = None
        self._imagePressEvent = None
        self.toolBar = None
        self.statusMsg = None
        self.img = None

    def init_ui(self):
        self.viewer = ImageViewer(self)
        self.viewer.mouseMove.connect(self.viewerMouseMove)
        self.viewer.mousePress.connect(self.viewerMousePress)
        self.stacked = QStackedWidget(self)
        self.stacked.addWidget(self.viewer)
        self.setCentralWidget(self.stacked)

        self.statusMsg = self.statusBar()
        self.create_toolbar()
        self.show()

    def create_central_widget(self):
        self.viewer = ImageViewer(self)
        self.viewer.mouseMove.connect(self.viewerMouseMove)
        self.viewer.mousePress.connect(self.viewerMousePress)

    def create_toolbar(self):
        self.toolBar = QToolBar()
        self.toolBar.addAction(QIcon('icons/folder-open.png'), 'Open', self.open_file)
        self.toolBar.addAction(QIcon('icons/save.png'), 'Save Image', self.save_file)
        self.toolBar.addAction(QIcon('icons/clear.png'), 'Clear', self.clear_scene)
        self.addToolBar(self.toolBar)

    def clear_scene(self):
        raise NotImplementedError

    def viewerMouseMove(self, x, y):
        raise NotImplementedError

    def viewerMousePress(self, x, y):
        raise NotImplementedError


class ImageViewer(QGraphicsView):
    mouseMove = pyqtSignal(int, int)
    mousePress = pyqtSignal(int, int)

    def __init__(self, parent):
        super().__init__(parent)
        self._zoom = 0
        self._empty = True
        self._parent = self.parentWidget()
        self.scene = QGraphicsScene(self)
        self.pixmap = None
        self.image = None
        self._h, self._w = None, None
        self.curr_img = None
        self.point_items = []
        self.line_items = []
        self.circle_items = []
        self.curr_line = None
        self.selected_item, self.prev_color = None, None
        self.setScene(self.scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        self.setFrameShape(QGraphicsView.NoFrame)

    def fitInView(self, scale=True):
        rect = QRectF(self.pixmap.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if not self._empty:
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                view_rect = self.viewport().rect()
                scene_rect = self.transform().mapRect(rect)
                factor = min(view_rect.width() / scene_rect.width(), view_rect.height() / scene_rect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def draw_image(self, img):
        self._h, self._w = img.shape[:2]
        self.image = QImage(img.flatten(), self._w, self._h,  3 * self._w, QImage.Format_RGB888)
        self.update_image()

    def update_image(self):
        self.pixmap = QPixmap(self.image)
        if self.curr_img is None:
            self.curr_img = self.scene.addPixmap(self.pixmap)
            self.curr_img.setZValue(-1)
        else:
            self.curr_img.setPixmap(self.pixmap)
        self.scene.update()
        self._empty = False
        self.fitInView()

    def scene_clear_items(self):
        for item in self.scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        self.point_items = []
        self.line_items = []
        self.circle_items = []
        self.curr_line = None
        self.selected_item, self.prev_color = None, None

    def reset(self):
        self.scene_clear_items()
        self.pixmap = None
        self.image = None
        self._h, self._w = None, None
        self.curr_img = None

    def draw_point(self, x, y, color):
        cross = draw_cross(x, y)
        point = self.scene.addPath(cross, QPen(color))
        point.setZValue(1)
        self.point_items.append(point)

    def delete_point(self):
        point_item = self.point_items.pop(-1)
        self.scene.removeItem(point_item)
        if self.curr_line is not None:
            self.scene.removeItem(self.curr_line)
            self.curr_line = None

    def move_line(self, x1, y1, x2, y2, color):
        if self.curr_line is None:
            self.curr_line = self.scene.addLine(x1, y1, x2, y2, QPen(color))
            self.curr_line.setZValue(0)
        else:
            self.curr_line.setLine(x1, y1, x2, y2)

    def draw_line(self, color):
        self.curr_line.setPen(color)
        line_ = self.curr_line
        self.line_items.append(line_)
        self.curr_line = None

    def delete_line(self):
        if self.line_items:
            line_item = self.line_items.pop()
            line_item.scene().removeItem(line_item)
        if self.curr_line is not None:
            self.curr_line.scene().removeItem(self.curr_line)

    def change_color(self, x, y, color, temp=True):
        item = self.scene.itemAt(QPoint(x, y), QGraphicsView.transform(self))
        if item in self.point_items:
            if temp and self.selected_item is None:
                self.prev_color = item.pen().color()
                self.selected_item = item
            item.setPen(color)
            self.scene.update()

    def reset_color(self, new_color=None):
        if self.selected_item is not None:
            color = new_color or self.prev_color
            self.selected_item.setPen(color)
            self.selected_item = None
            self.scene.update()

    def draw_circle(self, x, y, c):
        circle = self.scene.addEllipse(x-10, y-10, 20, 20, QPen(Qt.transparent), QBrush(c))
        circle.setZValue(0.5)
        self.circle_items.append(circle)

    def delete_circle(self):
        for circle in self.circle_items:
            if circle is not None:
                circle.scene().removeItem(circle)
        self.circle_items = []

    def wheelEvent(self, QWheelEvent):
        if not self._empty:
            if QWheelEvent.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.NoDrag:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)

    def mapToImage(self, pos):
        pos = self.mapToScene(pos)
        if (0 <= pos.x() < self._w) and (0 <= pos.y() < self._h):
            return int(pos.x()), int(pos.y())
        return None, None

    def mouseMoveEvent(self, QMouseEvent):
        x, y = self.mapToImage(QMouseEvent.pos())
        if x is not None and y is not None:
            self.mouseMove.emit(x, y)
        super().mouseMoveEvent(QMouseEvent)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            x, y = self.mapToImage(QMouseEvent.pos())
            if x is not None and y is not None:
                self.mousePress.emit(x, y)
        super().mousePressEvent(QMouseEvent)


def draw_cross(x, y):
    path = QPainterPath()
    path.moveTo(x, y - 5)
    path.lineTo(x, y + 5)
    path.moveTo(x - 5, y)
    path.lineTo(x + 5, y)
    return path
