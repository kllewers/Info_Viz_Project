"""
FileManagerWidget - Floating panel for managing loaded and derived datasets.

Shows a tree view of all available datasets with context menus for actions.
"""

import os
from PyQt5 import QtCore, QtWidgets, QtGui
from typing import Optional, Dict, List
from datetime import datetime

from data_manager import DataManager


class FileManagerWidget(QtWidgets.QWidget):
    """Floating file manager widget for dataset management."""
    
    # Signals
    dataset_selected = QtCore.pyqtSignal(str)  # dataset name
    dataset_activated = QtCore.pyqtSignal(str)  # double-clicked dataset name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.data_manager = DataManager()
        self.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        
        self._setup_ui()
        self._setup_connections()
        self._setup_style()
        
        # Set initial size and position
        self.resize(300, 400)
        self.setMinimumSize(200, 300)
        
        # Variables for dragging and resizing
        self.dragging = False
        self.resizing = False
        self.drag_start_position = QtCore.QPoint()
        self.resize_start_position = QtCore.QPoint()
        self.resize_start_size = QtCore.QSize()
        
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Title bar with close button
        title_bar_widget = QtWidgets.QWidget()
        title_bar_widget.setFixedHeight(25)
        title_bar_widget.setStyleSheet("background-color: rgba(70, 70, 70, 200);")
        
        title_bar_layout = QtWidgets.QHBoxLayout(title_bar_widget)
        title_bar_layout.setContentsMargins(4, 2, 4, 2)
        
        # Title label
        title_label = QtWidgets.QLabel("File Manager")
        title_label.setStyleSheet("font-weight: bold; color: white; background: transparent;")
        title_bar_layout.addWidget(title_label)
        
        title_bar_layout.addStretch()
        
        # Close button
        close_btn = QtWidgets.QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 150);
                border-radius: 3px;
            }
            QPushButton:pressed {
                background-color: rgba(200, 0, 0, 200);
            }
        """)
        close_btn.clicked.connect(self.hide)
        title_bar_layout.addWidget(close_btn)
        
        layout.addWidget(title_bar_widget)
        self.title_bar = title_bar_widget  # Store for drag handling
        
        # Tree widget for datasets
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Dataset", "Type", "Shape"])
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setRootIsDecorated(True)
        self.tree_widget.setIndentation(20)
        
        # Set column widths
        self.tree_widget.setColumnWidth(0, 180)
        self.tree_widget.setColumnWidth(1, 80)
        self.tree_widget.setColumnWidth(2, 100)
        
        layout.addWidget(self.tree_widget)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setMaximumWidth(80)
        button_layout.addWidget(self.refresh_btn)
        
        button_layout.addStretch()
        
        self.properties_btn = QtWidgets.QPushButton("Properties")
        self.properties_btn.setMaximumWidth(80)
        self.properties_btn.setEnabled(False)
        button_layout.addWidget(self.properties_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def _setup_connections(self):
        """Setup signal connections."""
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        
        self.refresh_btn.clicked.connect(self.refresh_datasets)
        self.properties_btn.clicked.connect(self._show_properties)
        
    def _setup_style(self):
        """Setup widget styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(50, 50, 50, 230);
                border: 2px solid #666;
                border-radius: 8px;
                color: white;
            }
            QTreeWidget {
                background-color: rgba(40, 40, 40, 200);
                border: 1px solid #555;
                alternate-background-color: rgba(60, 60, 60, 100);
            }
            QTreeWidget::item {
                padding: 4px;
                border: none;
            }
            QTreeWidget::item:selected {
                background-color: rgba(100, 150, 200, 150);
            }
            QTreeWidget::item:hover {
                background-color: rgba(80, 80, 80, 100);
            }
            QPushButton {
                background-color: rgba(70, 70, 70, 200);
                border: 1px solid #888;
                border-radius: 4px;
                padding: 6px;
                color: white;
            }
            QPushButton:hover {
                background-color: rgba(90, 90, 90, 200);
            }
            QPushButton:pressed {
                background-color: rgba(50, 50, 50, 200);
            }
            QPushButton:disabled {
                background-color: rgba(40, 40, 40, 100);
                color: #666;
            }
        """)
        
    def refresh_datasets(self):
        """Refresh the dataset list."""
        self.tree_widget.clear()
        
        datasets = self.data_manager.list_datasets()
        if not datasets:
            # Add a "No datasets" item
            no_data_item = QtWidgets.QTreeWidgetItem(["No datasets loaded", "", ""])
            no_data_item.setForeground(0, QtGui.QBrush(QtGui.QColor(150, 150, 150)))
            self.tree_widget.addTopLevelItem(no_data_item)
            return
            
        # Group by type
        original_item = QtWidgets.QTreeWidgetItem(["Original Files", "", ""])
        original_item.setExpanded(True)
        original_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        
        derived_item = QtWidgets.QTreeWidgetItem(["Processed Files", "", ""])
        derived_item.setExpanded(True)
        derived_item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        
        self.tree_widget.addTopLevelItem(original_item)
        self.tree_widget.addTopLevelItem(derived_item)
        
        # Add datasets
        for dataset_info in datasets:
            name = dataset_info['name']
            dtype = dataset_info['type']
            shape_str = str(dataset_info['shape']) if dataset_info['shape'] else "Unknown"
            
            item = QtWidgets.QTreeWidgetItem([name, dtype.title(), shape_str])
            item.setData(0, QtCore.Qt.UserRole, name)  # Store dataset name
            
            # Set icon based on type
            if dtype == 'original':
                item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileIcon))
                original_item.addChild(item)
            else:
                item.setIcon(0, self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
                derived_item.addChild(item)
                
            # Highlight active dataset
            if dataset_info['is_active']:
                item.setBackground(0, QtGui.QBrush(QtGui.QColor(100, 150, 100, 100)))
                item.setBackground(1, QtGui.QBrush(QtGui.QColor(100, 150, 100, 100)))
                item.setBackground(2, QtGui.QBrush(QtGui.QColor(100, 150, 100, 100)))
                
        # Remove empty categories
        if original_item.childCount() == 0:
            self.tree_widget.removeItemWidget(original_item, 0)
            original_item = None
        if derived_item.childCount() == 0:
            self.tree_widget.removeItemWidget(derived_item, 0)
            derived_item = None
            
    def _on_selection_changed(self):
        """Handle selection changes."""
        items = self.tree_widget.selectedItems()
        if items:
            item = items[0]
            dataset_name = item.data(0, QtCore.Qt.UserRole)
            if dataset_name:
                self.properties_btn.setEnabled(True)
                self.dataset_selected.emit(dataset_name)
            else:
                self.properties_btn.setEnabled(False)
        else:
            self.properties_btn.setEnabled(False)
            
    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        """Handle double-click on items."""
        dataset_name = item.data(0, QtCore.Qt.UserRole)
        if dataset_name:
            self.dataset_activated.emit(dataset_name)
            
    def _show_context_menu(self, position: QtCore.QPoint):
        """Show context menu for tree items."""
        item = self.tree_widget.itemAt(position)
        if not item:
            return
            
        dataset_name = item.data(0, QtCore.Qt.UserRole)
        if not dataset_name:
            return
            
        menu = QtWidgets.QMenu(self)
        
        # View in new tab
        view_new_action = menu.addAction("View in New Tab")
        view_new_action.triggered.connect(lambda: self.dataset_activated.emit(dataset_name))
        
        # Set as active
        set_active_action = menu.addAction("Set as Active")
        set_active_action.triggered.connect(lambda: self._set_active_dataset(dataset_name))
        
        menu.addSeparator()
        
        # Properties
        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(self._show_properties)
        
        menu.addSeparator()
        
        # Remove from memory
        remove_action = menu.addAction("Remove from Memory")
        remove_action.triggered.connect(lambda: self._remove_dataset(dataset_name))
        
        menu.exec_(self.tree_widget.mapToGlobal(position))
        
    def _set_active_dataset(self, dataset_name: str):
        """Set a dataset as active."""
        self.data_manager.set_active_dataset(dataset_name)
        self.refresh_datasets()
        
    def _remove_dataset(self, dataset_name: str):
        """Remove a dataset from memory."""
        reply = QtWidgets.QMessageBox.question(
            self, "Remove Dataset",
            f"Remove '{dataset_name}' from memory?\\nThis action cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.data_manager.remove_dataset(dataset_name)
            self.refresh_datasets()
            
    def _show_properties(self):
        """Show properties dialog for selected dataset."""
        items = self.tree_widget.selectedItems()
        if not items:
            return
            
        dataset_name = items[0].data(0, QtCore.Qt.UserRole)
        if not dataset_name:
            return
            
        info = self.data_manager.get_dataset_info(dataset_name)
        if not info:
            return
            
        dialog = DatasetPropertiesDialog(info, self)
        dialog.exec_()
        
    # Mouse events for dragging and resizing
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Check if we're near the bottom-right corner for resizing
            corner_size = 15
            widget_rect = self.rect()
            corner_rect = QtCore.QRect(
                widget_rect.width() - corner_size,
                widget_rect.height() - corner_size,
                corner_size, corner_size
            )
            
            if corner_rect.contains(event.pos()):
                self.resizing = True
                self.resize_start_position = event.globalPos()
                self.resize_start_size = self.size()
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
            elif self.title_bar.geometry().contains(event.pos()):
                # Check if click is not on the close button
                title_bar_rect = self.title_bar.geometry()
                close_btn_area = QtCore.QRect(title_bar_rect.width() - 30, 0, 30, title_bar_rect.height())
                relative_pos = event.pos() - title_bar_rect.topLeft()
                
                if not close_btn_area.contains(relative_pos):
                    self.dragging = True
                    self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                
    def mouseMoveEvent(self, event):
        if self.resizing and event.buttons() == QtCore.Qt.LeftButton:
            diff = event.globalPos() - self.resize_start_position
            new_size = QtCore.QSize(
                max(200, self.resize_start_size.width() + diff.x()),
                max(300, self.resize_start_size.height() + diff.y())
            )
            self.resize(new_size)
        elif self.dragging and event.buttons() == QtCore.Qt.LeftButton:
            self.move(event.globalPos() - self.drag_start_position)
        else:
            # Update cursor based on position
            corner_size = 15
            widget_rect = self.rect()
            corner_rect = QtCore.QRect(
                widget_rect.width() - corner_size,
                widget_rect.height() - corner_size,
                corner_size, corner_size
            )
            
            if corner_rect.contains(event.pos()):
                self.setCursor(QtCore.Qt.SizeFDiagCursor)
            else:
                self.setCursor(QtCore.Qt.ArrowCursor)
                
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = False
            self.resizing = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            
    def paintEvent(self, event):
        super().paintEvent(event)
        # Draw resize handle in bottom-right corner
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw resize grip
        grip_size = 15
        grip_rect = QtCore.QRect(
            self.width() - grip_size,
            self.height() - grip_size,
            grip_size, grip_size
        )
        
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
        for i in range(3):
            y_offset = i * 4 + 5
            painter.drawLine(
                grip_rect.x() + 5, grip_rect.y() + y_offset,
                grip_rect.x() + grip_size - 5, grip_rect.y() + y_offset
            )


class DatasetPropertiesDialog(QtWidgets.QDialog):
    """Dialog showing detailed dataset properties."""
    
    def __init__(self, dataset_info: dict, parent=None):
        super().__init__(parent)
        self.dataset_info = dataset_info
        self.setWindowTitle(f"Properties - {dataset_info['name']}")
        self.setModal(True)
        self.resize(400, 300)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the properties dialog UI."""
        layout = QtWidgets.QVBoxLayout()
        
        # Create form layout for properties
        form_layout = QtWidgets.QFormLayout()
        
        # Basic properties
        form_layout.addRow("Name:", QtWidgets.QLabel(self.dataset_info['name']))
        form_layout.addRow("Type:", QtWidgets.QLabel(self.dataset_info['type'].title()))
        
        if self.dataset_info['shape']:
            shape_str = f"{self.dataset_info['shape'][0]}×{self.dataset_info['shape'][1]}×{self.dataset_info['shape'][2]}"
            form_layout.addRow("Dimensions:", QtWidgets.QLabel(shape_str))
            
        form_layout.addRow("Data Type:", QtWidgets.QLabel(self.dataset_info['dtype']))
        
        # Memory usage
        memory_mb = self.dataset_info['memory_size'] / (1024 * 1024)
        memory_str = f"{memory_mb:.1f} MB"
        form_layout.addRow("Memory Usage:", QtWidgets.QLabel(memory_str))
        
        # Wavelengths
        if self.dataset_info['wavelengths']:
            wl_str = f"{self.dataset_info['num_wavelengths']} bands"
            form_layout.addRow("Wavelengths:", QtWidgets.QLabel(wl_str))
            
        # File path
        if self.dataset_info['file_path']:
            path_label = QtWidgets.QLabel(self.dataset_info['file_path'])
            path_label.setWordWrap(True)
            form_layout.addRow("File Path:", path_label)
            
        # Parent datasets
        if self.dataset_info['parents']:
            parents_str = ", ".join(self.dataset_info['parents'])
            form_layout.addRow("Parent Datasets:", QtWidgets.QLabel(parents_str))
            
        # Processing history
        if self.dataset_info['processing_history']:
            history_widget = QtWidgets.QTextEdit()
            history_widget.setMaximumHeight(100)
            history_widget.setReadOnly(True)
            
            history_text = ""
            for i, proc in enumerate(self.dataset_info['processing_history']):
                history_text += f"{i+1}. {proc['operation']} ({proc['timestamp']})\\n"
                
            history_widget.setPlainText(history_text)
            form_layout.addRow("Processing:", history_widget)
            
        layout.addLayout(form_layout)
        
        # Close button
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)