import os
import sys

from x3d import X3DIndexedFaceSet, X3DSphere, output_x3d

from gui import *
from SvmInterface import *
import matplotlib.image as mpimg
import cv2
# Hold Ctrl to choose existing point (move to existing point until it changes to yellow)
# Alt + d to delete point


class SVM(BaseApplication, SvmInterface):

    def __init__(self):
        super().__init__()
        self.lines = []
        self.points = {}
        self.pt_keys = []
        self.prev_pt = None
        self.face = []
        self.map_coord = []
        self.shapes = []
        self.is_same_z = False
        self.start_point = None
        self.reuse_point = None
        self.pressed_keys = []
        self.ground_pairs = []
        self.z_pairs = []
        self.use_vp = False
        self.texcoord = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])  # top-left, top-right, bottom-right, bottom-left
        self.menu = None
        self.edit_item = None
        self.compute_button = None
        self.cursor_loc = None
        self.init_ui()
        if self.img is None:
            self.init_img("images/sony_clie.800x600.bmp")

    def create_toolbar(self):
        super().create_toolbar()
        draw_point = QPushButton(QIcon('icons/point.png'), 'Points', None)
        draw_line = QPushButton(QIcon('icons/line.png'), 'Lines', None)
        draw_polygon = QPushButton(QIcon('icons/polygon.png'), 'Polygons', None)
        compute = QPushButton('Compute')

        self.toolBar.addWidget(draw_point)
        self.toolBar.addWidget(draw_line)
        self.toolBar.addWidget(draw_polygon)
        self.toolBar.addWidget(compute)

        pt_menu = QMenu()
        pt_menu.addAction(QIcon(''), '-', self.disable_pt)
        pt_menu.addAction(QIcon(''), 'Ground Ref. Point', self.set_ground_pts)
        pt_menu.addAction(QIcon(''), 'Z Ref. Point', self.set_z_pts)
        pt_menu.addAction(QIcon(''), '3D Coordinate', self.set_3d_coord)
        draw_point.setMenu(pt_menu)

        line_menu = QMenu()
        line_menu.addAction(QIcon(''), 'X Line', self.edit_xline)
        line_menu.addAction(QIcon(''), 'Y Line', self.edit_yline)
        line_menu.addAction(QIcon(''), 'Z Line', self.edit_zline)
        draw_line.setMenu(line_menu)

        polygon_menu = QMenu()
        polygon_menu.addAction(QIcon(''), 'Texture Map', self.set_face)
        polygon_menu.addAction(QIcon(''), 'Same Z Plane', self.change_z_plane)
        polygon_menu.addAction(QIcon(''), 'Same XY Plane', self.change_xy_plane)
        draw_polygon.setMenu(polygon_menu)

        compute_menu = QMenu()
        compute_menu.addAction(QIcon(''), 'X VP', self.compute_xvp)
        compute_menu.addAction(QIcon(''), 'Y VP', self.compute_yvp)
        compute_menu.addAction(QIcon(''), 'Z VP', self.compute_zvp)
        compute_menu.addAction(QIcon(''), 'Homography', self.compute_h)
        compute_menu.addAction(QIcon(''), 'Alpha Z', self.compute_z)
        compute_menu.addAction(QIcon(''), 'Texture', self.compute_texture)
        compute.setMenu(compute_menu)

        self.cursor_loc = QLabel()
        self.cursor_loc.setFixedWidth(70)
        self.cursor_loc.setAlignment(Qt.AlignCenter)
        self.toolBar.addWidget(self.cursor_loc)
        self.toolBar.addWidget(self.statusMsg)

    def disable_pt(self):
        self.edit_item = None
        self.statusMsg.showMessage('Disable edit point')

    def set_ground_pts(self):
        self.edit_item = 'Ground Ref. Point'
        self.statusMsg.showMessage('set {0}'.format(self.edit_item))

    def set_z_pts(self):
        self.edit_item = 'Z Ref. Point'
        self.statusMsg.showMessage('set {0}'.format(self.edit_item))

    def set_3d_coord(self):
        self.edit_item = '3D Point'
        self.statusMsg.showMessage('Add 3D coordinates to points')

    def set_face(self):
        self.face = []
        self.map_coord = []
        self.viewer.delete_circle()
        self.edit_item = 'Texture Point'
        self.statusMsg.showMessage('Add points to texture map')

    def edit_xline(self):
        self.edit_line('X Line')

    def edit_yline(self):
        self.edit_line('Y Line')

    def edit_zline(self):
        self.edit_line('Z Line')

    def edit_line(self, item):
        self.edit_item = item
        self.statusMsg.showMessage('draw {0}'.format(item))

    def compute_xvp(self):
        self.vx = self.compute_vp('X Line')
        self.statusMsg.showMessage('vanishing point x: {}'.format(self.vx))

    def compute_yvp(self):
        self.vy = self.compute_vp('Y Line')
        self.statusMsg.showMessage('vanishing point y: {}'.format(self.vy))

    def compute_zvp(self):
        self.vz = self.compute_vp('Z Line')
        self.statusMsg.showMessage('vanishing point z: {}'.format(self.vz))

    def compute_vp(self, item):
        lines = np.array([l for l, k in self.lines if k == item])
        if len(lines) >= 2:
            return self.bob_collins(lines)
        else:
            self.statusMsg.showMessage('Please add {0} lines'.format(item[0]))

    def change_xy_plane(self):
        self.is_same_z = False
        self.statusMsg.showMessage('change to same xy plane')

    def change_z_plane(self):
        self.is_same_z = True
        self.statusMsg.showMessage('change to same z plane')

    def compute_h(self):
        dialog = QDialog()
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Use vanishing points?"))
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No, Qt.Horizontal, self)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        result = dialog.exec()
        self.use_vp = result == QDialog.Accepted
        self.statusMsg.showMessage("Use vanishing point: {}".format(self.use_vp))
        if len(self.ground_pairs) < 4:
            self.statusMsg.showMessage("Only have {} ground reference points".format(len(self.ground_pairs)))
        else:
            self.calculate_vanishing_line()
            H = self.set_ground_homography(np.array(self.ground_pairs, dtype="float32"), use_vanishing_pt=self.use_vp)
            self.statusMsg.showMessage("Computed H")
            print(H, 'use vanishing point:', self.use_vp)

    def compute_z(self):
        if len(self.z_pairs) < 2:
            self.statusMsg.showMessage("Only have {} z reference point".format(len(self.z_pairs)))
        elif np.sum(self.Hxy) == 0 or self.vz is None:
            self.statusMsg.showMessage("Please compute ground homography and z-vanishing point before setting alpha_z.")
        else:
            alpha_z = self.set_alpha_z(self.z_pairs)
            self.statusMsg.showMessage("alpha z: {}".format(alpha_z))

    def compute_texture(self):
        input_, result = CoordDialog.get_result(is_coo=False)
        if result and input_:
            if len(self.map_coord) == 4:
                w, h = input_
                img = self.texture_map(np.array(self.map_coord).astype("float32"), width=w, height=h)

                qimg = QImage(img.flatten(), w, h, 3 * w, QImage.Format_RGB888)
                dialog = QDialog()
                layout = QVBoxLayout()
                label = QLabel()
                label.setPixmap(QPixmap(qimg))
                layout.addWidget(label)
                buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.No, Qt.Horizontal, self)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                layout.addWidget(buttons)
                dialog.setLayout(layout)
                res = dialog.exec()
                if res == QDialog.Accepted:
                    file_name, _ = QFileDialog.getSaveFileName(self, 'Save File')
                    if file_name:
                        mpimg.imsave(file_name, img)
                        self.append_x3d(file_name)
                        self.statusMsg.showMessage('Saved image to {}'.format(file_name))
                        self.face = []
                        self.map_coord = []
            else:
                self.statusMsg.showMessage("Only have {} points".format(len(self.map_coord)))

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File')
        if file_name:
            self.init_img(file_name)
            self.statusMsg.showMessage('Opened {}'.format(file_name))

    def save_file(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save File')
        if file_name:
            self.save_x3d(file_name)
            self.statusMsg.showMessage('saved {}'.format(file_name))

    def init_img(self, file_name):
        img = plt.imread(file_name)
        self._set_img(img)

    def _set_img(self, img):
        self.set_image(img)
        self.viewer.draw_image(self.img)

    def set_image(self, img):
        if len(img.shape) == 2:
            # expand grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype == 'float32':
            self.img = (img * 255).astype('uint8')
        else:
            self.img = np.copy(img)
        if self.img.shape[-1] > 3:
            # discard the alpha from RGBA
            self.img = self.img[:, :, :-1]

    def closest_pt(self, pt):
        deltas = np.array(self.pt_keys) - pt
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return self.pt_keys[np.argmin(dist_2)]

    def reset_reuse(self):
        if self.reuse_point is not None:
            self.viewer.reset_color()
            self.reuse_point = None

    def keyPressEvent(self, QKeyEvent):
        if (QKeyEvent.key() == Qt.Key_Control) and (not QKeyEvent.isAutoRepeat()):
            self.pressed_keys.append(QKeyEvent.key())

        elif (QKeyEvent.key() == Qt.Key_Alt) and (not QKeyEvent.isAutoRepeat()):
            if Qt.Key_D in self.pressed_keys:
                self.delete_previous_point()
            else:
                self.pressed_keys.append(QKeyEvent.key())

        elif (QKeyEvent.key() == Qt.Key_D) and (not QKeyEvent.isAutoRepeat()):
            if Qt.Key_Alt in self.pressed_keys:
                self.delete_previous_point()
            else:
                self.pressed_keys.append(QKeyEvent.key())

    def keyReleaseEvent(self, QKeyEvent):
        self.pressed_keys = []
        self.reset_reuse()

    def viewerMouseMove(self, x, y):
        self.cursor_loc.setText('({0:=3d}, {1:=3d})'.format(x, y))
        if Qt.Key_Control in self.pressed_keys and self.reuse_point is None and self.points:
            self.reuse_point = self.closest_pt((x, y))
            self.viewer.change_color(*self.reuse_point, Qt.yellow)
        if self.start_point is not None:
            if self.reuse_point is not None:
                x, y = self.reuse_point
            self.viewer.move_line(*self.start_point, x, y, Qt.cyan)

    def viewerMousePress(self, x, y):
        if self.edit_item is not None:
            if 'Point' in self.edit_item:
                self.mouse_edit_point(x, y)
            elif 'Line' in self.edit_item:
                self.mouse_edit_line(x, y)

    def mouse_edit_point(self, x, y):
        if 'Ground' in self.edit_item:
            color = Qt.darkBlue
        elif 'Z' in self.edit_item:
            color = Qt.darkRed
        else:
            color = Qt.cyan

        if Qt.Key_Control in self.pressed_keys:
            if self.reuse_point is None:
                return
            (x, y), self.reuse_point = self.reuse_point, None
            self.viewer.change_color(x, y, color, temp=False)
        else:
            self.pt_keys.append((x, y))
            self.points[(x, y)] = self.points.get((x, y), None)
            self.viewer.draw_point(x, y, color)
        self.statusMsg.showMessage('Set point at {}, {}'.format(x, y))

        if 'Ref.' in self.edit_item:
            self.mouse_set_ref(x, y)

        elif 'Texture' in self.edit_item:
            self.viewer.draw_circle(x, y, QColor(128, 204, 255, 215))
            self.map_coord.append((x, y))
            self.face.append(self.points.get((x, y))[:3])
            print(self.face)

        else:
            self.add_3d_coord(x, y)

    def mouse_set_ref(self, x, y):
        is_2d = 'Ground' in self.edit_item
        input_, result = CoordDialog.get_result(is_2d=is_2d)
        if result:
            if is_2d:
                self.ground_pairs.append((input_, [x, y]))
                print(self.ground_pairs)
            else:
                self.z_pairs.append((input_, [x, y]))
                print(self.z_pairs)

    def add_3d_coord(self, x, y):
        self.viewer.delete_circle()
        if self.is_same_z:
            prev_3d = self.points.get(self.prev_pt)
            if prev_3d is not None:
                coord_3d = self.same_xy((prev_3d, self.prev_pt), [x, y])
                self.change_xy_plane()
                self.viewer.draw_circle(x, y, QColor(244, 244, 244, 215))
        else:
            if self.prev_pt is None:
                if self.origin is None:
                    self.statusMsg.showMessage('Please add origin')
                    return
                if self.confirm_z0() == QDialog.Accepted:
                    self.prev_pt = (self.origin[0], self.origin[1])
                    self.points[self.prev_pt] = [0, 0, 0, 1]
                else:
                    self.statusMsg.showMessage('Please start with a point co-z with origin')
                    return
            prev_3d = self.points.get(self.prev_pt)
            # print((self.points[self.prev_pt], self.prev_pt), [x, y])
            if prev_3d is not None:
                coord_3d = self.same_z_plane((self.points[self.prev_pt], self.prev_pt), [x, y])
                self.viewer.draw_circle(x, y, QColor(128, 204, 255, 215))
            else:
                return
        self.points[(x, y)] = coord_3d.tolist()
        self.prev_pt = (x, y)
        self.statusMsg.showMessage("Added 3D: {}, 2D: {}".format(coord_3d, [x, y]))

    def mouse_edit_line(self, x, y):
        if 'X' in self.edit_item:
            color = Qt.red
        elif 'Y' in self.edit_item:
            color = Qt.green
        else:
            color = Qt.blue

        if self.start_point is None:
            if Qt.Key_Control in self.pressed_keys:
                if self.reuse_point is None:
                    return
                self.start_point, self.reuse_point = self.reuse_point, None
                # self.viewer.reset_color()
                self.viewer.change_color(*self.start_point, color, temp=False)
            else:
                self.start_point = (x, y)
                self.points[(x, y)] = self.points.get((x, y), None)
                self.pt_keys.append((x, y))
                self.viewer.draw_point(*self.start_point, color)
        else:
            if Qt.Key_Control in self.pressed_keys:
                if self.reuse_point is None:
                    return
                (x, y), self.reuse_point = self.reuse_point, None
                # self.viewer.reset_color()
                self.viewer.change_color(x, y, color, temp=False)
            else:
                self.points[(x, y)] = self.points.get((x, y), None)
                self.pt_keys.append((x, y))
                self.viewer.draw_point(x, y, color)
            curr_line = [self.start_point, (x, y)]
            self._draw_line(curr_line, self.edit_item, color)

    def delete_previous_point(self):
        if self.pt_keys:
            pt = self.pt_keys.pop(-1)
            self.points.pop(pt)
            self.viewer.delete_point()
            if self.start_point is None:
                self.viewer.delete_line()
                if self.lines:
                    prev_line, edit_item = self.lines.pop(-1)
                    self.statusMsg.showMessage('Deleted {} at {}'.format(edit_item, prev_line))
                    self.start_point = prev_line[0]
            else:
                self.start_point, self.reuse_point = None, None
            self.statusMsg.showMessage('Deleted point at {}'.format(pt))

    def confirm_z0(self):
        dialog = QDialog()
        layout = QVBoxLayout()
        label = QLabel('Is it same z plane with origin?')
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No, Qt.Horizontal, self)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        return dialog.exec()

    def append_x3d(self, file_name):
        shape = X3DIndexedFaceSet(texture={"url": '../textures/{}'.format(os.path.basename(file_name))})
        shape.coordinate = np.array(self.face, dtype="float32")
        shape.texture_coordinate = self.texcoord
        self.viewer.delete_circle()
        self.shapes.append(shape)

    def save_x3d(self, file_name):
        ref_pts_3d = np.array([
            [0, 0, 0],
            [300, 0, 0],
            [0, 400, 0],
            [300, 400, 0],
            [0, 0, 183],
            [300, 0, 183],
            [0, 400, 183],
            [300, 400, 183]
        ])

        face4_3d = np.array([ref_pts_3d[3], ref_pts_3d[1], ref_pts_3d[0], ref_pts_3d[2]], dtype="float32")
        shape4 = X3DIndexedFaceSet(texture={"url": "../textures/texture_wl.png"})
        shape4.coordinate = face4_3d
        shape4.texture_coordinate = self.texcoord
        self.shapes.append(shape4)

        face5_3d = np.array([ref_pts_3d[6], ref_pts_3d[7], ref_pts_3d[3], ref_pts_3d[2]], dtype="float32")
        shape5 = X3DIndexedFaceSet(texture={"url": "../textures/texture_wh.png"})
        shape5.coordinate = face5_3d
        shape5.texture_coordinate = self.texcoord
        self.shapes.append(shape5)

        face6_3d = np.array([ref_pts_3d[7], ref_pts_3d[5], ref_pts_3d[1], ref_pts_3d[3]], dtype="float32")
        shape6 = X3DIndexedFaceSet(texture={"url": "../textures/texture_lh.png"})
        shape6.coordinate = face6_3d
        shape6.texture_coordinate = self.texcoord
        self.shapes.append(shape6)

        camera_center = self.compute_camera_center()
        self.statusMsg.showMessage("Camera center: {}".format(camera_center))
        camera_shape = X3DSphere(material={"diffuseColor": "0 0 0"}, center=camera_center, radius=20)
        self.shapes.append(camera_shape)
        output_x3d(file_name, self.shapes)
        self.viewer.delete_circle()
        self.shapes = []

    def clear_scene(self):
        self.viewer.scene_clear_items()
        self.lines = []
        self.points = {}
        self.pt_keys = []
        self.prev_pt = None
        self.face = []
        self.map_coord = []
        self.shapes = []
        self.is_same_z = False
        self.start_point = None
        self.reuse_point = None
        self.pressed_keys = []
        self.ground_pairs = []
        self.z_pairs = []
        self.use_vp = False

    def _draw_line(self, curr_line, edit_item, color):
        self.viewer.draw_line(color)
        self.lines.append((curr_line, edit_item))
        self.statusMsg.showMessage('Set {} at {}'.format(edit_item, curr_line))
        self.start_point, self.reuse_point = None, None


class CoordDialog(QDialog):
    def __init__(self, parent=None, is_2d=False, is_coo=True):
        super(CoordDialog, self).__init__(parent)
        self.setFixedWidth(250)
        form = QFormLayout()
        if is_coo:
            form.addWidget(QLabel('Please input the coordinates'))
            self.x_coo = QLineEdit()
            form.addRow(QLabel('X'), self.x_coo)
            self.y_coo = QLineEdit()
            form.addRow(QLabel('Y'), self.y_coo)
            self.z_coo = QLineEdit()
            self.z_coo.setText('0')
            self.is_2d = is_2d
            self.z_coo.setDisabled(self.is_2d)
            self.z_coo.setReadOnly(self.is_2d)
            form.addRow(QLabel('Z'), self.z_coo)
        else:
            form.addWidget(QLabel('Please input the Width and Height'))
            self.w = QLineEdit()
            form.addRow(QLabel('Width'), self.w)
            self.h = QLineEdit()
            form.addRow(QLabel('Height'), self.h)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addWidget(buttons)
        self.setLayout(form)

    def get_coo(self):
        try:
            if self.is_2d:
                return [int(self.x_coo.text()), int(self.y_coo.text())]
            return [int(self.x_coo.text()), int(self.y_coo.text()), int(self.z_coo.text())]
        except ValueError:
            return None

    def get_wh(self):
        try:
            return [int(self.w.text()), int(self.h.text())]
        except ValueError:
            return None

    @staticmethod
    def get_result(parent=None, is_2d=False, is_coo=True):
        dialog = CoordDialog(parent, is_2d=is_2d, is_coo=is_coo)
        result = dialog.exec()
        user_input = dialog.get_coo() if is_coo else dialog.get_wh()
        return user_input, result == QDialog.Accepted and user_input is not None


def start():
    app = QApplication(['Single View Modeling'])
    svm = SVM()
    sys.exit(app.exec())


if __name__ == '__main__':
    start()