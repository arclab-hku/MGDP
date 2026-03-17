# gviz_ui.py
import sys
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import threading
from g_basic import Server 
from g_message import *

class ScalableLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(1, 1)
        self.original_pixmap = None
        self.scale_mode = True  # True: auto scale, False: original size

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self._update_scaled_pixmap()

    def set_scale_mode(self, enabled):
        self.scale_mode = enabled
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self.original_pixmap:
            if self.scale_mode:
                scaled_pixmap = self.original_pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                super().setPixmap(scaled_pixmap)
            else:
                super().setPixmap(self.original_pixmap)

class ImageControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_item_selected)
        layout.addWidget(QLabel("Images:"))
        layout.addWidget(self.image_list)
        
        self.controls_group = QGroupBox("Image Controls")
        controls_layout = QVBoxLayout()
 
        # visible checkbox
        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.setChecked(True)
        self.visible_checkbox.stateChanged.connect(self.on_visible_changed)
        controls_layout.addWidget(self.visible_checkbox)
        
        # scale checkbox
        self.scale_checkbox = QCheckBox("Auto Scale")
        self.scale_checkbox.setChecked(True)
        self.scale_checkbox.stateChanged.connect(self.on_scale_changed)
        controls_layout.addWidget(self.scale_checkbox)
        
        # image info
        self.info_group = QGroupBox("Image Info")
        info_layout = QFormLayout()
        self.size_label = QLabel()
        info_layout.addRow("Size:", self.size_label)
        self.info_group.setLayout(info_layout)
        controls_layout.addWidget(self.info_group)
        
        self.controls_group.setLayout(controls_layout)
        layout.addWidget(self.controls_group)
        layout.addStretch()
        self.setLayout(layout)
        
    def on_item_selected(self, item):
        if item and self.parent_widget:
            img_id = item.data(Qt.UserRole)
            self.parent_widget.show_detail(img_id)

    def on_visible_changed(self, state):
        if self.parent_widget and self.parent_widget.current_id is not None:
            self.parent_widget.show_detail(self.parent_widget.current_id)
            
    def on_scale_changed(self, state):
        if self.parent_widget:
            self.parent_widget.detail_label.set_scale_mode(bool(state))

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # detail label
        self.detail_label = ScalableLabel()
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setMinimumSize(400, 300)
        
        # control panel
        self.control_panel = ImageControlPanel(self)
        self.control_panel.setFixedWidth(250)
        
        # image show scroll area
        detail_scroll = QScrollArea()
        detail_scroll.setWidget(self.detail_label)
        detail_scroll.setWidgetResizable(True)
        
        self.layout.addWidget(detail_scroll)
        self.layout.addWidget(self.control_panel)
        
        # data
        self.images = {}  # {id: QImage}
        self.current_id = None
        
    def add_image(self, image, id=0):
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            bytes_per_line = channels * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Update image info
            info = f"Size: {width}x{height}\n"
            info += f"Channels: {channels}\n"
            info += f"Type: {image.dtype}"
            self.control_panel.size_label.setText(info)
            
        if id in self.images:
            # update existing image
            self.images[id] = qimage
            if id == self.current_id:
                self.show_detail(id)
        else:
            # add new image
            self.images[id] = qimage
            item = QListWidgetItem(f"Image {id}")
            item.setData(Qt.UserRole, id)
            self.control_panel.image_list.addItem(item)
            self.control_panel.visible_checkbox.setChecked(True)
        
        if self.current_id is None:
            self.show_detail(id)
            
    def show_detail(self, id):
        self.current_id = id
        if self.control_panel.visible_checkbox.isChecked():
            self.detail_label.setPixmap(QPixmap.fromImage(self.images[id]))
        else:
            self.detail_label.clear()

        # highlight item
        for i in range(self.control_panel.image_list.count()):
            item = self.control_panel.image_list.item(i)
            if item.data(Qt.UserRole) == id:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def on_visible_changed(self, state):
        item = self.image_list.currentItem()
        if item:
            img_id = item.data(Qt.UserRole)
            if img_id == self.current_id:
                if state:
                    self.show_detail(img_id, self.image_labels[img_id].pixmap().toImage())
                else:
                    self.detail_label.clear()

class ObjectInfo:
    def __init__(self, name, obj_type):
        self.name = name
        self.type = obj_type  # 'point_cloud' or 'mesh'
        self.visible = True
        self.color = [1.0, 0.0, 0.0] if obj_type == 'point_cloud' else [0.0, 1.0, 0.0]
        self.size = 3.0 if obj_type == 'point_cloud' else 1.0
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]

class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.points = {}
        self.meshes = {}
        self.objects_info = {}  # Store information of all objects
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.xTrans = 0
        self.yTrans = 0
        self.scale = 1.0
        self.lastPos = QPoint()

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glTranslatef(self.xTrans, self.yTrans, -5.0)
        glRotatef(self.xRot, 1.0, 0.0, 0.0)
        glRotatef(self.yRot, 0.0, 1.0, 0.0)
        glRotatef(self.zRot, 0.0, 0.0, 1.0)
        glScalef(self.scale, self.scale, self.scale)

        self.drawGrid()
        self.drawAxes()
        
        # Draw point clouds
        for name, points in self.points.items():
            obj_info = self.objects_info.get(f'point_cloud_{name}')
            if obj_info and obj_info.visible:
                glPointSize(obj_info.size)
                glPushMatrix()
                glTranslatef(*obj_info.position)
                glRotatef(obj_info.rotation[0], 1, 0, 0)
                glRotatef(obj_info.rotation[1], 0, 1, 0)
                glRotatef(obj_info.rotation[2], 0, 0, 1)
                
                glBegin(GL_POINTS)
                glColor3f(*obj_info.color)
                for point in points:
                    glVertex3f(*point)
                glEnd()
                glPopMatrix()

        # Draw meshes
        for name, mesh in self.meshes.items():
            obj_info = self.objects_info.get(f'mesh_{name}')
            if obj_info and obj_info.visible:
                glPushMatrix()
                glTranslatef(*obj_info.position)
                glRotatef(obj_info.rotation[0], 1, 0, 0)
                glRotatef(obj_info.rotation[1], 0, 1, 0)
                glRotatef(obj_info.rotation[2], 0, 0, 1)
                
                glBegin(GL_TRIANGLES)
                glColor3f(*obj_info.color)
                for triangle in mesh:
                    for vertex in triangle:
                        glVertex3f(*vertex)
                glEnd()
                glPopMatrix()

    def drawGrid(self):
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0.5)
        
        for i in np.arange(-0.5, 0.51, 0.1):
            glVertex3f(i, -0.5, 0)
            glVertex3f(i, 0.5, 0)
            glVertex3f(-0.5, i, 0)
            glVertex3f(0.5, i, 0)
        
        glEnd()

    def drawAxes(self):
        glBegin(GL_LINES)
        
        # x-axis
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(10.0, 0.0, 0.0)
        
        # y-axis
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 10.0, 0.0)
        
        # z-axis
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 10.0)
        
        glEnd()

    def drawPoints(self):
        if self.points:
            glBegin(GL_POINTS)
            glColor3f(1.0, 0.0, 0.0)
            for point in self.points:
                glVertex3f(*point)
            glEnd()

    def drawMeshes(self):
        if self.meshes:
            glBegin(GL_TRIANGLES)
            glColor3f(0.0, 1.0, 0.0)
            for mesh in self.meshes:
                for vertex in mesh:
                    glVertex3f(*vertex)
            glEnd()

    def reset_view(self):
        self.xRot = 0.0
        self.yRot = 0.0
        self.zRot = 0.0
        self.scale = 1.0
        self.xTrans = 0.0
        self.yTrans = 0.0
            
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / height
        gluPerspective(45.0, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.yRot += dx
            self.xRot += dy
        elif event.buttons() & Qt.RightButton:
            self.zRot += dx
        elif event.buttons() & Qt.MiddleButton:
            self.xTrans += dx * 0.05
            self.yTrans -= dy * 0.05

        self.update()
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        self.update()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.reset_view()
        self.update()        

class ControlPanel(QWidget):
    def __init__(self, gl_widget):
        super().__init__()
        self.gl_widget = gl_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Object list
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.on_item_selected)
        layout.addWidget(QLabel("Objects:"))
        layout.addWidget(self.object_list)

        # Control group
        self.controls_group = QGroupBox("Object Controls")
        controls_layout = QVBoxLayout()

        # Visibility control
        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.stateChanged.connect(self.on_visible_changed)
        controls_layout.addWidget(self.visible_checkbox)

        # Color selection
        color_btn = QPushButton("Change Color")
        color_btn.clicked.connect(self.on_color_click)
        controls_layout.addWidget(color_btn)

        # Point size control
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.size_spinner = QDoubleSpinBox()
        self.size_spinner.setRange(0.1, 20.0)
        self.size_spinner.setSingleStep(0.1)
        self.size_spinner.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_spinner)
        controls_layout.addLayout(size_layout)

        # Position control
        pos_group = QGroupBox("Position")
        pos_layout = QGridLayout()
        self.pos_spinners = []
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            pos_layout.addWidget(QLabel(label), i, 0)
            spinner = QDoubleSpinBox()
            spinner.setRange(-10, 10)
            spinner.setSingleStep(0.1)
            spinner.valueChanged.connect(self.on_position_changed)
            self.pos_spinners.append(spinner)
            pos_layout.addWidget(spinner, i, 1)
        pos_group.setLayout(pos_layout)
        controls_layout.addWidget(pos_group)

        # Rotation control
        rot_group = QGroupBox("Rotation")
        rot_layout = QGridLayout()
        self.rot_spinners = []
        for i, label in enumerate(['X:', 'Y:', 'Z:']):
            rot_layout.addWidget(QLabel(label), i, 0)
            spinner = QDoubleSpinBox()
            spinner.setRange(-360, 360)
            spinner.setSingleStep(5)
            spinner.valueChanged.connect(self.on_rotation_changed)
            self.rot_spinners.append(spinner)
            rot_layout.addWidget(spinner, i, 1)
        rot_group.setLayout(rot_layout)
        controls_layout.addWidget(rot_group)

        self.controls_group.setLayout(controls_layout)
        layout.addWidget(self.controls_group)
        
        # Add stretch
        layout.addStretch()
        
        self.setLayout(layout)
        self.current_item = None

    def add_object(self, obj_id, obj_info):
        if obj_id in self.gl_widget.objects_info.keys(): 
            item = self.object_list.findItems(obj_info.name, Qt.MatchExactly)
            assert len(item) == 1
            item = item[0]
            item.setData(Qt.UserRole, obj_id)
        else: # Add object to list
            self.gl_widget.objects_info[obj_id] = obj_info
            item = QListWidgetItem(obj_info.name)
            item.setData(Qt.UserRole, obj_id)
            print(item.data(Qt.UserRole))
            self.object_list.addItem(item)

    def on_item_selected(self, item):
        self.current_item = item
        obj_id = item.data(Qt.UserRole)
        obj_info = self.gl_widget.objects_info[obj_id]
        
        # Update control panel
        self.visible_checkbox.setChecked(obj_info.visible)
        self.size_spinner.setValue(obj_info.size)
        
        for i, value in enumerate(obj_info.position):
            self.pos_spinners[i].setValue(value)
        
        for i, value in enumerate(obj_info.rotation):
            self.rot_spinners[i].setValue(value)

    def on_visible_changed(self, state):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            self.gl_widget.objects_info[obj_id].visible = bool(state)
            self.gl_widget.update()

    def on_color_click(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            current_color = self.gl_widget.objects_info[obj_id].color
            color = QColorDialog.getColor(QColor.fromRgbF(*current_color))
            
            if color.isValid():
                self.gl_widget.objects_info[obj_id].color = [color.redF(), color.greenF(), color.blueF()]
                self.gl_widget.update()

    def on_size_changed(self, value):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            self.gl_widget.objects_info[obj_id].size = value
            self.gl_widget.update()

    def on_position_changed(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            position = [spinner.value() for spinner in self.pos_spinners]
            self.gl_widget.objects_info[obj_id].position = position
            self.gl_widget.update()

    def on_rotation_changed(self):
        if self.current_item:
            obj_id = self.current_item.data(Qt.UserRole)
            rotation = [spinner.value() for spinner in self.rot_spinners]
            self.gl_widget.objects_info[obj_id].rotation = rotation
            self.gl_widget.update()

class PointCloudView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        
        # Create GL widget and control panel
        self.glWidget = GLWidget()
        self.control_panel = ControlPanel(self.glWidget)
        self.control_panel.setFixedWidth(250)
        
        layout.addWidget(self.control_panel)
        layout.addWidget(self.glWidget, stretch=1)
        self.setLayout(layout)

class Visualizer(QMainWindow):
    update_points_signal = pyqtSignal(object, int)
    update_mesh_signal = pyqtSignal(object, int) 
    update_image_signal = pyqtSignal(object, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualizer")
        self.resize(800, 600)

        # Create stacked widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Create views
        self.pcl_view = PointCloudView()
        self.img_view = ImageWidget()

        # Add views to stack
        self.stack.addWidget(self.pcl_view)
        self.stack.addWidget(self.img_view)

        # Create menu bar
        self.create_menu_bar()

        # Initialize server
        self.server = Server(address="tcp://*:5555")
        self.server.start()
        
        
        self.update_points_signal.connect(self._update_points)
        self.update_mesh_signal.connect(self._update_mesh)
        self.update_image_signal.connect(self.img_view.add_image)

        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._receive_messages)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

    def create_menu_bar(self):
        menubar = self.menuBar()
        view_menu = menubar.addMenu('View')

        # Point cloud view action
        pcl_action = QAction('PointCloud View', self)
        pcl_action.triggered.connect(lambda: self.stack.setCurrentWidget(self.pcl_view))
        view_menu.addAction(pcl_action)

        # Image view action
        img_action = QAction('Image View', self)
        img_action.triggered.connect(lambda: self.stack.setCurrentWidget(self.img_view))
        view_menu.addAction(img_action)

    def _receive_messages(self):
        while self.server.running:
            import time
            time.sleep(0.01)
            message = self.server.get_message()
            if message:
                self._process_message(message)

    def _process_message(self, message):
        msg = GMesssage.deserialize(message)
        
        for pcl in msg.pointcloud:
            points, idx = GMesPointCloud.to_numpy(pcl)
            self.update_points_signal.emit(points, idx)
            
        for mesh in msg.trimesh:
            vertices, faces, idx = GMesTrimesh.to_numpy(mesh)
            self.update_mesh_signal.emit((vertices, faces), idx)
            
        for img in msg.image:
            image, idx = GMesImage.to_numpy(img)
            self.update_image_signal.emit(image, idx)

    def _update_points(self, points, id:int):
        point_cloud_id = f'point_cloud_{id}'
        self.pcl_view.glWidget.points[id] = points
        self.pcl_view.control_panel.add_object(point_cloud_id, ObjectInfo(point_cloud_id, 'point_cloud'))
        self.pcl_view.glWidget.update()

    def _update_mesh(self, vertices, id:int):
        mesh_id = f'mesh_{id}'
        self.pcl_view.glWidget.meshes[id] = vertices
        self.pcl_view.control_panel.add_object(mesh_id, ObjectInfo(mesh_id, 'mesh'))
        self.pcl_view.glWidget.update()

    def add_points(self, points, id=0):
        if isinstance(points, np.ndarray):
            points = points.tolist()
        self._update_points(points, id)

    def add_mesh(self, vertices, id=0):
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        self._update_mesh(vertices, id)
    
    def add_trimesh(self, vertices, faces, id=0):
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        if isinstance(faces, np.ndarray):
            faces = faces.tolist()
        mesh = [[vertices[face[0]], vertices[face[1]], vertices[face[2]]] for face in faces]
        self.add_mesh(mesh, id)
        
    def add_image(self, image , id=0):
        """Add image to image view"""
        self.img_view.add_image(image)

class VisualizerWrapper:
    _instance = None
    _app = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            if QApplication.instance() is None:
                cls._app = QApplication(sys.argv)
            cls._instance = Visualizer()
            cls._instance.show()
        return cls._instance

    @classmethod
    def run(cls):
        if cls._app is not None:
            cls._app.exec_()

def create_visualizer():
    """Create and return a visualizer instance"""
    return VisualizerWrapper.get_instance()

if __name__ == '__main__':
    # Create visualizer
    vis = create_visualizer()

    # Add some random point clouds
    points1 = np.random.rand(1000, 3) * 0.5 + 0.25
    vis.add_points(points1, id=0)

    points2 = np.random.rand(1000, 3) * 0.5 - 1.5
    vis.add_points(points2, id=1)

    # Add a simple triangle mesh
    triangle = [
        [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]
    ]
    vis.add_mesh(triangle)

    VisualizerWrapper.run()