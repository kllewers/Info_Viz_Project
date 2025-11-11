"""
ImageView class for RGB display with cursor tracking and ROI drawing functionality.

Uses PyQtGraph for efficient image display and interaction.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from typing import Optional, Tuple, Callable, List
from matplotlib import cm
import matplotlib.pyplot as plt


class ClickableLabel(QtWidgets.QLabel):
    """QLabel that emits clicked signal when clicked."""
    clicked = QtCore.pyqtSignal()
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            print(f"ClickableLabel clicked: {self.text()}")
            self.clicked.emit()
        super().mousePressEvent(event)


class ClickableSpinBox(QtWidgets.QSpinBox):
    """QSpinBox that emits clicked signal when clicked."""
    clicked = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            print(f"ClickableSpinBox clicked: {self.value()}")
            self.clicked.emit()
        super().mousePressEvent(event)


class ImageView(QtWidgets.QWidget):
    """Image display widget with cursor tracking and ROI drawing capabilities."""
    
    # Signals
    pixel_selected = QtCore.pyqtSignal(int, int)  # x, y coordinates
    cursor_moved = QtCore.pyqtSignal(int, int)  # x, y coordinates for cursor sync
    roi_selected = QtCore.pyqtSignal(object)  # ROI coordinates/mask
    zoom_changed = QtCore.pyqtSignal(object)  # View range
    rgb_bands_changed = QtCore.pyqtSignal()  # RGB bands changed signal
    default_rgb_requested = QtCore.pyqtSignal()  # Request for default RGB bands
    line_changed = QtCore.pyqtSignal(int)  # Line index changed in frame view mode
    band_changed = QtCore.pyqtSignal(int, str)  # Band index and channel ('R', 'G', 'B', 'mono') changed
    spectrum_collect_requested = QtCore.pyqtSignal(int, int)  # x, y coordinates for spectrum collection
    
    def __init__(self, parent=None):
        super().__init__(parent)
        print(f"Creating ImageView instance: {id(self)}")

        self.image_data = None
        self.current_roi = None
        self.roi_mode = False
        self.roi_type = None  # 'point', 'rectangle', 'polygon'
        self.roi_points = []  # Store collected points for current ROI
        self.roi_start = None
        self.zoom_box = None

        # Spectrum collection mode
        self.collect_mode = False

        # Frame view mode state
        self.frame_view_mode = False
        self.current_line = 0

        # Colormap for monochromatic mode
        self.current_colormap = 'Gray'
        self.manual_colormap_bounds = None  # (min, max) for manual colormap scaling
        self.total_lines = 0

        # Monochromatic mode state
        self.mono_mode = False
        self.current_band = 29
        self.selected_rgb_channel = None  # 'R', 'G', 'B' for slider interaction

        # Contrast stretch settings
        self.stretch_percent = 2.0  # Default stretch percentage
        self.histogram_levels = None  # Saved histogram levels (min, max)
        self.view_range = None  # Saved zoom and position (view range)
        self.levels_initialized = False  # Track if levels have been auto-set initially

        # Pinch gesture tracking
        self.pinch_scale_factor = 1.0
        self.last_pinch_scale = 1.0

        # Enable touch and gesture events
        self.setAttribute(QtCore.Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(QtCore.Qt.PinchGesture)

        # Create colormap icons
        self._create_colormap_icons()

        self._setup_ui()
        self._setup_connections()
        
    def _create_colormap_icons(self):
        """Create small colormap preview icons for the dropdown."""
        self.colormap_icons = {}
        colormap_names = [
            'Gray', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
            'Turbo', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Spring', 'Summer',
            'Autumn', 'Winter', 'Bone', 'Copper', 'HSV'
        ]
        
        # Mapping to matplotlib colormap names
        colormap_mapping = {
            'Gray': 'gray', 'Viridis': 'viridis', 'Plasma': 'plasma',
            'Inferno': 'inferno', 'Magma': 'magma', 'Cividis': 'cividis',
            'Turbo': 'turbo', 'Rainbow': 'rainbow', 'Jet': 'jet',
            'Hot': 'hot', 'Cool': 'cool', 'Spring': 'spring',
            'Summer': 'summer', 'Autumn': 'autumn', 'Winter': 'winter',
            'Bone': 'bone', 'Copper': 'copper', 'HSV': 'hsv'
        }
        
        # Create a small gradient for each colormap
        icon_width = 100
        icon_height = 16
        
        for name in colormap_names:
            # Create gradient data
            gradient = np.linspace(0, 1, icon_width)
            gradient = np.tile(gradient, (icon_height, 1))
            
            # Apply colormap
            cmap = cm.get_cmap(colormap_mapping[name])
            colored = cmap(gradient)[:, :, :3]  # Get RGB, ignore alpha
            colored = (colored * 255).astype(np.uint8)
            
            # Create QImage and QPixmap
            height, width = colored.shape[:2]
            bytes_per_line = 3 * width
            qimage = QtGui.QImage(colored.data, width, height, bytes_per_line, 
                                   QtGui.QImage.Format_RGB888)
            qpixmap = QtGui.QPixmap.fromImage(qimage)
            
            # Create QIcon
            self.colormap_icons[name] = QtGui.QIcon(qpixmap)
    
    def _setup_ui(self):
        """Initialize the user interface."""
        print(f"Starting _setup_ui for ImageView {id(self)}")
        try:
            layout = QtWidgets.QVBoxLayout()
            
            # Create toolbar container with two rows
            self.toolbar_container = self._create_toolbar()
            layout.addWidget(self.toolbar_container)
            
            # Create image widget
            self.image_widget = pg.ImageView()
            self.image_widget.ui.roiBtn.hide()
            self.image_widget.ui.menuBtn.hide()
            self.image_widget.getView().setAspectLocked(True)
            
            # Disable default context menu to prevent conflicts with our custom ROI menu
            self.image_widget.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
            
            # Mouse tracking will be set up in _setup_connections()
            
            layout.addWidget(self.image_widget)
            
            # Status label
            self.status_label = QtWidgets.QLabel("Ready")
            layout.addWidget(self.status_label)
            
            self.setLayout(layout)

            # Ensure ImageView and toolbar are visible
            self.show()
            if hasattr(self, 'toolbar_container'):
                self.toolbar_container.show()

            print(f"Completed _setup_ui for ImageView {id(self)}")
        except Exception as e:
            print(f"ERROR in _setup_ui for ImageView {id(self)}: {e}")
            import traceback
            traceback.print_exc()
        
    def _create_toolbar(self) -> QtWidgets.QWidget:
        """Create toolbar widget with two rows of controls."""
        print(f"Starting _create_toolbar for ImageView {id(self)}")
        try:
            # Create container widget for two toolbars
            toolbar_widget = QtWidgets.QWidget()
            toolbar_layout = QtWidgets.QVBoxLayout(toolbar_widget)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            toolbar_layout.setSpacing(0)
            
            # First toolbar row
            toolbar1 = QtWidgets.QToolBar()
            
            # Zoom controls (keep only zoom fit)
            zoom_fit_action = toolbar1.addAction("Zoom Fit")
            zoom_fit_action.triggered.connect(self._zoom_fit)
            
            toolbar1.addSeparator()
            
            # ROI controls - dropdown menu instead of simple toggle
            self.roi_button = QtWidgets.QPushButton("ROI Mode")
            self.roi_button.clicked.connect(self._show_roi_type_menu)
            toolbar1.addWidget(self.roi_button)

            clear_roi_action = toolbar1.addAction("Clear ROI")
            clear_roi_action.triggered.connect(self._clear_roi)

            toolbar1.addSeparator()

            # Spectrum collection controls
            self.collect_button = QtWidgets.QPushButton("Collect Spectra")
            self.collect_button.setCheckable(True)
            self.collect_button.setToolTip("Toggle spectrum collection mode - click pixels to collect their spectra")
            self.collect_button.clicked.connect(self._toggle_collect_mode)
            toolbar1.addWidget(self.collect_button)

            toolbar1.addSeparator()
        
            # RGB band selection
            self.red_label = QtWidgets.QLabel("R:")
            self.red_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            toolbar1.addWidget(self.red_label)
            print(f"Created Red label in ImageView {id(self)}")
        
            self.red_spinbox = ClickableSpinBox()
            self.red_spinbox.setMinimum(0)
            self.red_spinbox.setMaximum(999)
            self.red_spinbox.setValue(29)
            self.red_spinbox.clicked.connect(lambda: self._select_rgb_channel('R'))
            self.red_spinbox.setToolTip("Click to select Red channel for band slider control")
            toolbar1.addWidget(self.red_spinbox)
            print(f"Created Red spinbox in ImageView {id(self)}")
        
            self.green_label = QtWidgets.QLabel("G:")
            self.green_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            toolbar1.addWidget(self.green_label)
            print(f"Created Green label in ImageView {id(self)}")
        
            self.green_spinbox = ClickableSpinBox()
            self.green_spinbox.setMinimum(0)
            self.green_spinbox.setMaximum(999)
            self.green_spinbox.setValue(19)
            self.green_spinbox.clicked.connect(lambda: self._select_rgb_channel('G'))
            self.green_spinbox.setToolTip("Click to select Green channel for band slider control")
            toolbar1.addWidget(self.green_spinbox)
            print(f"Created Green spinbox in ImageView {id(self)}")
        
            self.blue_label = QtWidgets.QLabel("B:")
            self.blue_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
            toolbar1.addWidget(self.blue_label)
        
            self.blue_spinbox = ClickableSpinBox()
            self.blue_spinbox.setMinimum(0)
            self.blue_spinbox.setMaximum(999)
            self.blue_spinbox.setValue(9)
            self.blue_spinbox.clicked.connect(lambda: self._select_rgb_channel('B'))
            self.blue_spinbox.setToolTip("Click to select Blue channel for band slider control")
            toolbar1.addWidget(self.blue_spinbox)
        
            update_rgb_action = toolbar1.addAction("Update RGB")
            update_rgb_action.triggered.connect(self._update_rgb_bands)
            
            toolbar1.addSeparator()
            
            # True color RGB button
            default_rgb_action = toolbar1.addAction("True Color RGB")
            default_rgb_action.setToolTip("Set RGB bands to true color (visible spectrum) or well-spaced bands")
            default_rgb_action.triggered.connect(self._set_default_rgb_bands)
            
            toolbar1.addSeparator()
            
            # Contrast stretch controls
            stretch_label = QtWidgets.QLabel("Stretch %:")
            stretch_label.setToolTip("Contrast stretch percentile (lower values = more stretch)")
            toolbar1.addWidget(stretch_label)
            
            self.stretch_spinbox = QtWidgets.QDoubleSpinBox()
            self.stretch_spinbox.setMinimum(0.1)
            self.stretch_spinbox.setMaximum(50.0)
            self.stretch_spinbox.setValue(self.stretch_percent)
            self.stretch_spinbox.setSingleStep(0.5)
            self.stretch_spinbox.setDecimals(1)
            self.stretch_spinbox.setToolTip("Contrast stretch percentile (0.1-50.0%)")
            self.stretch_spinbox.valueChanged.connect(self._on_stretch_changed)
            toolbar1.addWidget(self.stretch_spinbox)
            
            # Add first toolbar to layout
            toolbar_layout.addWidget(toolbar1)
            
            # Second toolbar row
            toolbar2 = QtWidgets.QToolBar()
            
            # Monochromatic mode toggle button
            self.mono_action = toolbar2.addAction("Monochromatic")
            self.mono_action.setCheckable(True)
            self.mono_action.setToolTip("Toggle monochromatic mode - use band slider to select single band")
            self.mono_action.triggered.connect(self._toggle_mono_mode)
            
            # Colormap dropdown for monochromatic mode (initially hidden)
            self.colormap_label = QtWidgets.QLabel("Colormap:")
            toolbar2.addWidget(self.colormap_label)
            self.colormap_label.hide()
            
            self.colormap_combo = QtWidgets.QComboBox()
            # Add items with icons
            colormap_names = [
                'Gray', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                'Turbo', 'Rainbow', 'Jet', 'Hot', 'Cool', 'Spring', 'Summer',
                'Autumn', 'Winter', 'Bone', 'Copper', 'HSV'
            ]
            for name in colormap_names:
                self.colormap_combo.addItem(self.colormap_icons[name], name)
            
            self.colormap_combo.setCurrentText('Gray')
            self.colormap_combo.setToolTip("Select colormap for monochromatic display")
            self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
            # Set a reasonable width to show the colormap previews
            self.colormap_combo.setMinimumWidth(150)
            self.colormap_combo.setIconSize(QtCore.QSize(80, 12))  # Set icon size for better visibility
            toolbar2.addWidget(self.colormap_combo)
            self.colormap_combo.hide()
            
            # Colormap bounds controls (initially hidden)
            self.apply_bounds_button = QtWidgets.QPushButton("Apply Histogram Bounds")
            self.apply_bounds_button.setToolTip("Apply current histogram levels as colormap bounds for focused color differentiation")
            self.apply_bounds_button.clicked.connect(self._apply_histogram_bounds_to_colormap)
            toolbar2.addWidget(self.apply_bounds_button)
            self.apply_bounds_button.hide()
            
            self.reset_bounds_button = QtWidgets.QPushButton("Reset Bounds")
            self.reset_bounds_button.setToolTip("Reset colormap to automatic percentile stretching")
            self.reset_bounds_button.clicked.connect(self.reset_colormap_bounds)
            toolbar2.addWidget(self.reset_bounds_button)
            self.reset_bounds_button.hide()
        
            toolbar2.addSeparator()
        
            # Frame View Mode Toggle
            self.frame_view_checkbox = QtWidgets.QCheckBox("Frame View")
            self.frame_view_checkbox.setToolTip("Toggle between RGB composite and spatial-spectral cross-section view")
            self.frame_view_checkbox.toggled.connect(self._toggle_frame_view_mode)
            # Ensure Frame View checkbox is visible
            self.frame_view_checkbox.show()
            toolbar2.addWidget(self.frame_view_checkbox)
        
            # Frame view controls (initially hidden)
            self.frame_controls = []
        
            # Line/row selection for frame view
            self.line_label = QtWidgets.QLabel("Line:")
            toolbar2.addWidget(self.line_label)
            self.frame_controls.append(self.line_label)
        
            self.line_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.line_slider.setMinimum(0)
            self.line_slider.setMaximum(99)  # Will be updated when data loads
            self.line_slider.setValue(0)
            self.line_slider.setFixedWidth(150)
            self.line_slider.setToolTip("Select spatial line (row) for cross-section display")
            self.line_slider.valueChanged.connect(self._on_line_slider_changed)
            toolbar2.addWidget(self.line_slider)
            self.frame_controls.append(self.line_slider)
        
            # Line spinbox
            self.line_spinbox = QtWidgets.QSpinBox()
            self.line_spinbox.setMinimum(0)
            self.line_spinbox.setMaximum(99)  # Will be updated when data loads
            self.line_spinbox.setValue(0)
            self.line_spinbox.setToolTip("Select specific line number")
            self.line_spinbox.valueChanged.connect(self._on_line_spinbox_changed)
            toolbar2.addWidget(self.line_spinbox)
            self.frame_controls.append(self.line_spinbox)
        
            # Initially hide frame controls
            for control in self.frame_controls:
                control.hide()
                
            # Add second toolbar to layout
            toolbar_layout.addWidget(toolbar2)

            # Ensure the toolbar container and both toolbars are visible
            toolbar_widget.show()
            toolbar1.show()
            toolbar2.show()

            print(f"Completed _create_toolbar for ImageView {id(self)}")
            return toolbar_widget
        except Exception as e:
            print(f"ERROR in _create_toolbar for ImageView {id(self)}: {e}")
            import traceback
            traceback.print_exc()
            return QtWidgets.QWidget()  # Return empty widget on error
        
    def _setup_connections(self):
        """Setup signal connections."""
        # Mouse tracking for pixel selection using SignalProxy
        self.mouse_proxy = pg.SignalProxy(
            self.image_widget.getView().scene().sigMouseMoved,
            rateLimit=60,
            slot=self._mouse_moved
        )
        
        # Mouse events for spectrum updates and ROI drawing  
        scene = self.image_widget.getView().scene()
        scene.sigMouseClicked.connect(self._on_mouse_clicked)
        
        # Additional mouse events for ROI drag functionality
        # PyQtGraph uses different signal names - we'll handle drag in the existing mouse click handler
        
        # View range changes for zoom tracking
        self.image_widget.getView().sigRangeChanged.connect(self._on_range_changed)
        
        # Histogram level changes for automatic colormap refresh
        hist_widget = self.image_widget.getHistogramWidget()
        if hist_widget:
            hist_widget.sigLevelsChanged.connect(self._on_histogram_levels_changed)
            
    def _on_histogram_levels_changed(self):
        """Handle histogram level changes to refresh colormap automatically."""
        # Only refresh colormap for single-band datasets in mono mode
        if self.mono_mode and hasattr(self, 'current_dataset') and self.current_dataset:
            if hasattr(self.current_dataset, 'shape') and len(self.current_dataset.shape) >= 3 and self.current_dataset.shape[2] == 1:
                try:
                    # Get current histogram levels for debugging
                    current_levels = self.get_histogram_levels()
                    if current_levels:
                        print(f"Histogram levels changed to: {current_levels[0]:.3f} - {current_levels[1]:.3f}")
                    
                    # Refresh the colormap display using current levels
                    raw_band_data = self.current_dataset.get_band_data(0)
                    if raw_band_data is not None:
                        colored_image = self.apply_colormap(raw_band_data, self.current_colormap)
                        self.set_image(colored_image)
                except Exception as e:
                    print(f"Error refreshing colormap after histogram change: {e}")
        
    def apply_colormap(self, gray_image: np.ndarray, colormap_name: str) -> np.ndarray:
        """
        Apply a colormap to a grayscale image with proper histogram stretch handling.
        
        Args:
            gray_image: 2D grayscale image array
            colormap_name: Name of the colormap to apply
            
        Returns:
            RGB image array with colormap applied
        """
        # Normalize the image to 0-1 range
        if gray_image.size == 0:
            return np.zeros((*gray_image.shape, 3))
        
        # Priority order for determining colormap bounds:
        # 1. Current histogram levels (user's stretch via sliders) 
        # 2. Manual colormap bounds (set via "Apply Histogram Bounds" button)
        # 3. Default percentile stretching
        
        current_levels = self.get_histogram_levels()
        if current_levels is not None and len(current_levels) == 2:
            # Use current histogram levels - this gives full colormap range to stretched area
            vmin, vmax = current_levels
        elif hasattr(self, 'manual_colormap_bounds') and self.manual_colormap_bounds is not None:
            # Use manual colormap bounds
            vmin, vmax = self.manual_colormap_bounds
        else:
            # Fall back to standard percentile-based stretching
            valid_data = gray_image[~np.isnan(gray_image)]
            if len(valid_data) > 0:
                vmin, vmax = np.percentile(valid_data, [self.stretch_percent, 100-self.stretch_percent])
            else:
                vmin, vmax = 0, 1
        
        # Ensure we have a valid range
        if vmin == vmax:
            data_min, data_max = gray_image.min(), gray_image.max()
            if data_min == data_max:
                vmin, vmax = data_min, data_min + 1
            else:
                vmin, vmax = data_min, data_max
        
        # Normalize to 0-1 range using the determined stretch range
        # This ensures the full colormap range is used across the stretched area
        normalized = np.clip((gray_image - vmin) / (vmax - vmin), 0, 1)
        
        # Get the colormap
        colormap_mapping = {
            'Gray': 'gray', 'Viridis': 'viridis', 'Plasma': 'plasma',
            'Inferno': 'inferno', 'Magma': 'magma', 'Cividis': 'cividis',
            'Turbo': 'turbo', 'Rainbow': 'rainbow', 'Jet': 'jet',
            'Hot': 'hot', 'Cool': 'cool', 'Spring': 'spring',
            'Summer': 'summer', 'Autumn': 'autumn', 'Winter': 'winter',
            'Bone': 'bone', 'Copper': 'copper', 'HSV': 'hsv'
        }
        
        cmap_name = colormap_mapping.get(colormap_name, 'gray')
        cmap = cm.get_cmap(cmap_name)
        
        # Apply colormap (returns RGBA, we only need RGB)
        colored = cmap(normalized)[:, :, :3]
        
        # Convert to 0-255 range for display
        return (colored * 255).astype(np.uint8)
        
    def refresh_colormap_display(self):
        """Refresh the colormap display for single-band datasets when histogram levels change."""
        if self.mono_mode and hasattr(self, 'current_dataset') and self.current_dataset:
            if hasattr(self.current_dataset, 'shape') and len(self.current_dataset.shape) >= 3 and self.current_dataset.shape[2] == 1:
                try:
                    raw_band_data = self.current_dataset.get_band_data(0)
                    if raw_band_data is not None:
                        colored_image = self.apply_colormap(raw_band_data, self.current_colormap)
                        self.set_image(colored_image)
                except Exception as e:
                    print(f"Error refreshing colormap display: {e}")
                    
    def _apply_histogram_bounds_to_colormap(self):
        """Apply current histogram levels as colormap bounds."""
        if self.mono_mode and hasattr(self, 'current_dataset') and self.current_dataset:
            current_levels = self.get_histogram_levels()
            if current_levels is not None and len(current_levels) == 2:
                # Set manual colormap bounds
                self.manual_colormap_bounds = current_levels
                print(f"Applied histogram bounds to colormap: {current_levels}")
                
                # Refresh the colormap display with new bounds
                self.refresh_colormap_display()
            else:
                QtWidgets.QMessageBox.information(self, "No Histogram Bounds", 
                    "No histogram levels detected. Please adjust the histogram first.")
                    
    def reset_colormap_bounds(self):
        """Reset colormap bounds to automatic percentile stretching."""
        self.manual_colormap_bounds = None
        if self.mono_mode:
            self.refresh_colormap_display()

    def set_image(self, image_data: np.ndarray):
        """
        Set the image data to display.
        
        Args:
            image_data: RGB image array (height, width, 3) = (rows, cols, 3) OR
                       grayscale array (height, width) for monochromatic mode
        """
        if image_data is None:
            return
            
        # Check if we're in mono mode and have a grayscale image
        if self.mono_mode and len(image_data.shape) == 2:
            # Apply colormap to grayscale data
            image_data = self.apply_colormap(image_data, self.current_colormap)
            
        self.image_data = image_data  # Keep original (h, w, 3) format for coordinate validation
        
        # Image data loaded successfully
        
        # Apply the original working transformation sequence
        flipped = np.fliplr(image_data)
        rotated = np.rot90(flipped, k=-1)
        flipped_again = np.fliplr(rotated)
        display_image = np.rot90(flipped_again, k=-2)
        
        # Debug: Store the transformation steps for coordinate mapping verification
        self.transformation_steps = {
            'original_shape': image_data.shape,
            'display_shape': display_image.shape,
            'transformations': ['fliplr', 'rot90(k=-1)', 'fliplr', 'rot90(k=-2)']
        }
        # Image transformation debug output removed
        
        # Set image - auto-levels only on first load, then preserve user settings
        if not self.levels_initialized:
            # First time loading this image - use auto-levels and auto-range
            self.image_widget.setImage(display_image, autoRange=True, autoLevels=True)
            self.levels_initialized = True
        else:
            # Subsequent loads - preserve user's zoom, position, and histogram levels
            self.image_widget.setImage(display_image, autoRange=False, autoLevels=False)
            
            # Restore saved histogram levels if we have them
            if self.histogram_levels is not None:
                QtCore.QTimer.singleShot(10, lambda: self.set_histogram_levels(self.histogram_levels))
                
            # Restore saved view range (zoom and position) if we have it
            if self.view_range is not None:
                QtCore.QTimer.singleShot(20, lambda: self.set_view_range(self.view_range))
        
        # Image set in widget
        self._update_status()
        
    def _update_status(self):
        """Update status label with image information."""
        if self.image_data is not None:
            h, w = self.image_data.shape[:2]
            self.status_label.setText(f"Image: {w} x {h} pixels")
        else:
            self.status_label.setText("No image loaded")
            
    def _mouse_moved(self, evt):
        """Handle mouse movement for cursor tracking using SignalProxy."""
        if self.image_data is None:
            return
            
        # SignalProxy passes events as a list, get the position from first element
        pos = evt[0]
        vb = self.image_widget.getView()
        
        # Check if mouse is within the view bounds
        if vb.sceneBoundingRect().contains(pos):
            # Map scene position to view coordinates
            mouse_point = vb.mapSceneToView(pos)
            # Image is flipped left-right then rotated 90° clockwise
            x_display = int(mouse_point.x())
            y_display = int(mouse_point.y())
            
            # Use the same transformation as mouse click
            height, width = self.image_data.shape[:2]
            
            # Fix 90-degree transpose by swapping the coordinate assignment
            x = x_display
            y = height - 1 - y_display
            
            # Check bounds and update status
            if self._is_valid_pixel(x, y):
                self.status_label.setText(f"Pixel: ({x}, {y}) [Display: ({x_display}, {y_display})]")
                # Emit cursor position for overview synchronization
                self.cursor_moved.emit(x, y)
                
                # Emit pixel selection for spectrum update if shift is held (drag spectrum mode)
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier and not self.roi_mode:
                    self.pixel_selected.emit(x, y)
            
    def _on_mouse_clicked(self, event):
        """Handle mouse clicks for spectrum updates and ROI drawing.
        
        Args:
            event: PyQtGraph MouseClickEvent with pos(), button(), and double() methods
        """
        if self.image_data is None:
            return
            
        # Get pixel coordinates using the view's coordinate mapping
        vb = self.image_widget.getView()
        mouse_point = vb.mapSceneToView(event.pos())
        # Image is flipped left-right then rotated 90° clockwise
        x_display = int(mouse_point.x())
        y_display = int(mouse_point.y())
        
        # Use the original working coordinate transformation
        height, width = self.image_data.shape[:2]
        
        # For ROI drawing, we need to use display coordinates directly
        # The PyQtGraph coordinate system matches the display coordinates
        if self.roi_mode:
            # For ROI mode, use display coordinates directly for consistent visual placement
            x = x_display
            y = y_display
        else:
            # For spectrum selection, use the data coordinate transformation
            x = x_display
            y = height - 1 - y_display
        
        # Validate coordinates 
        if not self._is_valid_pixel(x, y):
            return
        
        # Handle spectrum collection on single left click (collect mode)
        if event.button() == QtCore.Qt.LeftButton and self.collect_mode and not self.roi_mode:
            # Single click spectrum collection (ignore double clicks)
            if not event.double():
                print(f"[COLLECT] Collecting spectrum at pixel ({x}, {y})")
                # Add a persistent marker at collected location
                self._add_collect_marker(x_display, y_display)

                # Emit spectrum collection signal
                self.spectrum_collect_requested.emit(x, y)
                print(f"[COLLECT] Signal emitted: spectrum_collect_requested({x}, {y})")
                return

        # Handle spectrum plotting on single left click (not ROI or collect mode)
        if event.button() == QtCore.Qt.LeftButton and not self.roi_mode and not self.collect_mode:
            # Single click spectrum plotting (ignore double clicks)
            if not event.double():
                # Pixel selected - debug output removed

                # Visual verification: add a temporary crosshair at clicked location
                self._add_debug_crosshair(x_display, y_display)

                self.pixel_selected.emit(x, y)
        
        # Handle custom ROI modes
        elif self.roi_mode and event.button() == QtCore.Qt.LeftButton:
            if not event.double():
                self._handle_roi_left_click(x, y)
                    
        elif self.roi_mode and event.button() == QtCore.Qt.RightButton:
            self._handle_roi_right_click(event)
            event.accept()  # Consume event to prevent default context menu
                
    def _show_roi_context_menu(self, event):
        """Show context menu for ROI operations."""        
        if self.current_roi is None:
            return
            
        menu = QtWidgets.QMenu(self)
        
        # Save to new ROI tab
        save_new_action = menu.addAction("Save to new ROI tab")
        save_new_action.triggered.connect(lambda: self._save_roi_to_tab(new_tab=True))
        
        # Save to existing tab (if any exist)
        existing_tabs_action = menu.addAction("Save to existing ROI tab")
        existing_tabs_action.triggered.connect(lambda: self._save_roi_to_tab(new_tab=False))
        # Only enable if there might be existing tabs
        existing_tabs_action.setEnabled(True)  # Always enabled, main app will handle empty case
        
        menu.addSeparator()
        
        # Cancel/clear ROI
        cancel_action = menu.addAction("Cancel ROI")
        cancel_action.triggered.connect(self._clear_roi)
        
        # Show menu at cursor position
        menu.exec_(QtGui.QCursor.pos())
        
        # Prevent event propagation to avoid double menus
        event.accept()
        
    def _save_roi_to_tab(self, new_tab: bool = True):
        """Save current ROI to spectrum tab."""        
        if self.current_roi is None:
            return
            
        # Convert display coordinates to data coordinates for spectral extraction
        roi_data_coords = self._convert_roi_to_data_coordinates(self.current_roi.copy())
        
        # Emit signal with ROI data and tab preference
        roi_data = {
            'roi_definition': roi_data_coords,
            'create_new_tab': new_tab
        }
        self.roi_selected.emit(roi_data)
        
        # Clear the ROI from image (it's now saved)
        self._clear_roi()
        self.roi_start = None
        self.status_label.setText("✓ ROI saved to spectrum tab! Define another ROI or exit ROI mode.")
        
    def _convert_roi_to_data_coordinates(self, roi_definition: dict) -> dict:
        """Convert ROI from display coordinates to data coordinates for spectral extraction."""
        if self.image_data is None:
            return roi_definition
            
        height, width = self.image_data.shape[:2]
        
        # Convert points from display coordinates to data coordinates
        if 'points' in roi_definition and roi_definition['points']:
            converted_points = []
            for x_display, y_display in roi_definition['points']:
                # Convert display coordinates to data coordinates
                x_data = x_display
                y_data = height - 1 - y_display  # Flip Y coordinate for data access
                
                # Validate coordinates
                if 0 <= x_data < width and 0 <= y_data < height:
                    converted_points.append((x_data, y_data))
                    
            roi_definition['points'] = converted_points
            
        return roi_definition
    
            
    def _is_valid_pixel(self, x: int, y: int) -> bool:
        """Check if pixel coordinates are valid."""
        if self.image_data is None:
            return False
        h, w = self.image_data.shape[:2]
        return 0 <= x < w and 0 <= y < h
    
    def _try_alternative_mapping(self, x_display: int, y_display: int, height: int, width: int) -> Tuple[int, int]:
        """Try alternative coordinate mappings in case the current one is wrong."""
        alternatives = [
            # Direct mapping (no transformation)
            (x_display, y_display),
            # Swap x and y
            (y_display, x_display),
            # Various flip and rotation combinations
            (width - 1 - x_display, y_display),
            (x_display, height - 1 - y_display),
            (width - 1 - x_display, height - 1 - y_display),
            (height - 1 - y_display, x_display),
            (y_display, width - 1 - x_display),
            (height - 1 - y_display, width - 1 - x_display)
        ]
        
        for alt_x, alt_y in alternatives:
            if self._is_valid_pixel(alt_x, alt_y):
                # Debug output removed to reduce terminal spam
                return alt_x, alt_y
        
        # Debug output removed to reduce terminal spam
        return x_display, y_display  # Fallback to direct mapping
    
    def _add_debug_crosshair(self, x_display: int, y_display: int):
        """Add temporary crosshair at clicked display coordinates for visual debugging."""
        try:
            import pyqtgraph as pg
            
            # Remove any existing debug crosshair
            if hasattr(self, 'debug_crosshair_v'):
                self.image_widget.getView().removeItem(self.debug_crosshair_v)
            if hasattr(self, 'debug_crosshair_h'):
                self.image_widget.getView().removeItem(self.debug_crosshair_h)
            
            # Create new crosshair lines at clicked position
            self.debug_crosshair_v = pg.InfiniteLine(pos=x_display, angle=90, pen=pg.mkPen('yellow', width=2))
            self.debug_crosshair_h = pg.InfiniteLine(pos=y_display, angle=0, pen=pg.mkPen('yellow', width=2))
            
            # Add to view
            view = self.image_widget.getView()
            view.addItem(self.debug_crosshair_v, ignoreBounds=True)
            view.addItem(self.debug_crosshair_h, ignoreBounds=True)
            
            # Debug crosshair added
            
            # Remove crosshair after 2 seconds
            QtCore.QTimer.singleShot(2000, self._remove_debug_crosshair)
            
        except Exception as e:
            pass  # Crosshair error suppressed
    
    def _remove_debug_crosshair(self):
        """Remove debug crosshair."""
        try:
            if hasattr(self, 'debug_crosshair_v'):
                self.image_widget.getView().removeItem(self.debug_crosshair_v)
                del self.debug_crosshair_v
            if hasattr(self, 'debug_crosshair_h'):
                self.image_widget.getView().removeItem(self.debug_crosshair_h)
                del self.debug_crosshair_h
        except Exception as e:
            pass  # Crosshair removal error suppressed
        
    def _create_roi(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Create rectangular ROI."""
        x1, y1 = start
        x2, y2 = end
        
        # Ensure proper order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Validate ROI size (must be at least 1x1)
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
        # Create ROI rectangle
        roi_rect = {
            'x': x_min,
            'y': y_min, 
            'width': x_max - x_min,
            'height': y_max - y_min
        }
        
        self.current_roi = roi_rect
        self._draw_roi_overlay()
        
        # Don't emit roi_selected here - wait for user to save it via context menu
        
        # Show size information
        pixel_count = roi_rect['width'] * roi_rect['height']
        self.status_label.setText(f"ROI: {roi_rect['width']}×{roi_rect['height']} pixels ({pixel_count} total)")
        
    def _draw_roi_overlay(self):
        """Draw ROI overlay on image using simple rectangle graphics."""
        if self.current_roi is None:
            return
            
        # Remove existing overlay
        view = self.image_widget.getView()
        for item in view.allChildren():
            if hasattr(item, 'roi_overlay') and item.roi_overlay:
                view.removeItem(item)
                
        # Create simple rectangle graphic (not interactive ROI)
        roi = self.current_roi
        
        # Create rectangle using PlotDataItem to draw the outline
        x_coords = [roi['x'], roi['x'] + roi['width'], roi['x'] + roi['width'], roi['x'], roi['x']]
        y_coords = [roi['y'], roi['y'], roi['y'] + roi['height'], roi['y'] + roi['height'], roi['y']]
        
        rect_item = pg.PlotDataItem(
            x_coords, y_coords,
            pen=pg.mkPen('r', width=3),
            connect='all'
        )
        rect_item.roi_overlay = True
        rect_item.setZValue(1000)  # Ensure it's on top
        
        view.addItem(rect_item)
        
    def _draw_roi_start_marker(self):
        """Draw marker at ROI start point."""
        if self.roi_start is None:
            return
            
        # Remove existing start marker
        view = self.image_widget.getView()
        for item in view.allChildren():
            if hasattr(item, 'roi_start_marker') and item.roi_start_marker:
                view.removeItem(item)
                
        # Create crosshair at start point
        x, y = self.roi_start
        
        # Horizontal line
        h_line = pg.PlotDataItem(
            [x-10, x+10], [y, y],
            pen=pg.mkPen('r', width=2)
        )
        h_line.roi_start_marker = True
        h_line.setZValue(999)
        
        # Vertical line  
        v_line = pg.PlotDataItem(
            [x, x], [y-10, y+10],
            pen=pg.mkPen('r', width=2)
        )
        v_line.roi_start_marker = True
        v_line.setZValue(999)
        
        view.addItem(h_line)
        view.addItem(v_line)
        
    def _draw_point_marker(self, x: int, y: int, number: int):
        """Draw a numbered point marker."""
        view = self.image_widget.getView()
        
        # Create circle marker using PlotDataItem (more consistent)
        # Draw a small circle by creating points around the circumference
        import math
        circle_points_x = []
        circle_points_y = []
        radius = 3
        for i in range(13):  # 12 points + 1 to close the circle
            angle = 2 * math.pi * i / 12
            circle_points_x.append(x + radius * math.cos(angle))
            circle_points_y.append(y + radius * math.sin(angle))
        
        circle_item = pg.PlotDataItem(
            circle_points_x, circle_points_y,
            pen=pg.mkPen('r', width=2),
            connect='all'
        )
        circle_item.roi_overlay = True
        circle_item.setZValue(1001)
        view.addItem(circle_item)
        
        # Add number label
        text_item = pg.TextItem(str(number), color='red', anchor=(0.5, 0.5))
        text_item.setPos(x, y-8)
        text_item.roi_overlay = True
        text_item.setZValue(1002)
        view.addItem(text_item)
        
    def _draw_polygon_lines(self, close: bool = False):
        """Draw lines connecting polygon points."""
        if len(self.roi_points) < 2:
            return
            
        view = self.image_widget.getView()
        
        # Remove existing polygon lines
        for item in view.allChildren():
            if hasattr(item, 'polygon_line') and item.polygon_line:
                view.removeItem(item)
        
        points = self.roi_points.copy()
        if close and len(points) >= 3:
            points.append(points[0])  # Close the polygon
            
        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            line = pg.PlotDataItem([x1, x2], [y1, y2], pen=pg.mkPen('r', width=2))
            line.polygon_line = True
            line.roi_overlay = True
            line.setZValue(1000)
            view.addItem(line)
            
    def _draw_rectangle_outline(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """Draw rectangle outline."""
        view = self.image_widget.getView()
        
        # Rectangle coordinates
        x_coords = [x_min, x_max, x_max, x_min, x_min]
        y_coords = [y_min, y_min, y_max, y_max, y_min]
        
        rect_item = pg.PlotDataItem(
            x_coords, y_coords,
            pen=pg.mkPen('r', width=3),
            connect='all'
        )
        rect_item.roi_overlay = True
        rect_item.setZValue(1000)
        view.addItem(rect_item)
        
    def _get_polygon_interior_points(self, polygon_points: list) -> list:
        """Get all pixel coordinates inside a polygon using ray casting algorithm."""
        if len(polygon_points) < 3:
            return []
            
        # Get bounding box
        x_coords = [p[0] for p in polygon_points]
        y_coords = [p[1] for p in polygon_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        interior_points = []
        
        # Check each pixel in bounding box
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if self._is_valid_pixel(x, y) and self._point_in_polygon(x, y, polygon_points):
                    interior_points.append((x, y))
                    
        return interior_points
        
    def _point_in_polygon(self, x: int, y: int, polygon_points: list) -> bool:
        """Ray casting algorithm to determine if point is inside polygon."""
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def _clear_roi(self):
        """Clear current ROI."""
        self.current_roi = None
        self.roi_start = None
        
        # Remove all ROI visual elements
        view = self.image_widget.getView()
        for item in view.allChildren():
            if ((hasattr(item, 'roi_overlay') and item.roi_overlay) or 
                (hasattr(item, 'roi_start_marker') and item.roi_start_marker) or
                (hasattr(item, 'polygon_line') and item.polygon_line)):
                view.removeItem(item)
                
        self.status_label.setText("ROI cleared")
        
    def _show_roi_type_menu(self):
        """Show menu to select ROI type."""
        if self.roi_mode:
            # If already in ROI mode, exit it
            self._exit_roi_mode()
            return
            
        menu = QtWidgets.QMenu(self)
        
        # Point ROI
        point_action = menu.addAction("📍 Point ROI")
        point_action.setToolTip("Select individual pixels")
        point_action.triggered.connect(lambda: self._start_roi_mode('point'))
        
        # Rectangle ROI  
        rect_action = menu.addAction("⬜ Rectangle ROI")
        rect_action.setToolTip("Define rectangular region")
        rect_action.triggered.connect(lambda: self._start_roi_mode('rectangle'))
        
        # Polygon ROI
        poly_action = menu.addAction("🔺 Polygon ROI")
        poly_action.setToolTip("Define free-form polygon region")
        poly_action.triggered.connect(lambda: self._start_roi_mode('polygon'))
        
        menu.addSeparator()
        
        # Exit ROI mode option
        exit_action = menu.addAction("❌ Exit ROI Mode")
        exit_action.triggered.connect(self._exit_roi_mode)
        exit_action.setEnabled(self.roi_mode)  # Only enabled if in ROI mode
        
        # Show menu at button position
        button_pos = self.roi_button.mapToGlobal(self.roi_button.rect().bottomLeft())
        menu.exec_(button_pos)
        
    def _start_roi_mode(self, roi_type: str):
        """Start ROI mode with specified type."""
        self.roi_mode = True
        self.roi_type = roi_type
        self.roi_points = []
        self.current_roi = None
        
        # Update button appearance
        self.roi_button.setText(f"ROI Mode: {roi_type.title()}")
        self.roi_button.setStyleSheet("QPushButton { background-color: #ffcccc; }")
        
        # Set cursor and status
        self.setCursor(QtCore.Qt.CrossCursor)
        
        if roi_type == 'point':
            self.status_label.setText("POINT ROI: Click pixels to select, right-click when done")
        elif roi_type == 'rectangle':
            self.status_label.setText("RECTANGLE ROI: Click two corners to define rectangle")  
        elif roi_type == 'polygon':
            self.status_label.setText("POLYGON ROI: Click points to define polygon, right-click when done")
            
    def _exit_roi_mode(self):
        """Exit ROI mode and clean up."""
        self.roi_mode = False
        self.roi_type = None
        self.roi_points = []
        self.roi_start = None
        
        # Reset button appearance
        self.roi_button.setText("ROI Mode")
        self.roi_button.setStyleSheet("")
        
        # Reset cursor and status
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.status_label.setText("ROI mode disabled")
        
        # Clear any visual elements
        self._clear_roi()

    def _toggle_collect_mode(self, checked: bool):
        """Toggle spectrum collection mode."""
        self.collect_mode = checked

        if checked:
            # Entering collect mode
            self.collect_button.setText("Collecting...")
            self.collect_button.setStyleSheet("QPushButton { background-color: #ccffcc; }")
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.status_label.setText("COLLECT MODE: Click pixels to collect their spectra")

            # Exit ROI mode if active
            if self.roi_mode:
                self._exit_roi_mode()
        else:
            # Exiting collect mode
            self.collect_button.setText("Collect Spectra")
            self.collect_button.setStyleSheet("")
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.status_label.setText("Collect mode disabled")

            # Clear all collect markers
            self.clear_collect_markers()

    def _add_collect_marker(self, x_display: int, y_display: int):
        """Add a persistent marker at collected spectrum location."""
        try:
            import pyqtgraph as pg

            # Create a simple scatter plot point as marker instead of CircleROI
            # CircleROI is meant for interactive regions, not static markers
            scatter = pg.ScatterPlotItem(
                pos=[[x_display, y_display]],
                size=8,
                pen=pg.mkPen('#00ff00', width=2),
                brush=pg.mkBrush(255, 255, 255, 128),
                symbol='o'
            )

            scatter.setZValue(1050)  # Above other markers
            scatter.collect_marker = True  # Tag for cleanup

            # Add to view
            view = self.image_widget.getView()
            view.addItem(scatter)

        except Exception as e:
            print(f"Error adding collect marker: {e}")

    def clear_collect_markers(self):
        """Clear all collect markers from the display."""
        try:
            view = self.image_widget.getView()
            for item in view.allChildren():
                if hasattr(item, 'collect_marker') and item.collect_marker:
                    view.removeItem(item)
        except Exception as e:
            print(f"Error clearing collect markers: {e}")

    def _handle_roi_left_click(self, x: int, y: int):
        """Handle left click in ROI mode based on ROI type."""
        if self.roi_type == 'point':
            self._handle_point_roi_click(x, y)
        elif self.roi_type == 'rectangle':
            self._handle_rectangle_roi_click(x, y)
        elif self.roi_type == 'polygon':
            self._handle_polygon_roi_click(x, y)
            
    def _handle_roi_right_click(self, event):
        """Handle right click in ROI mode."""        
        if self.roi_type in ['point', 'polygon'] and len(self.roi_points) > 0:
            # Finish multi-point ROI and show save menu
            self._finalize_roi()
            if self.current_roi is not None:
                self._show_roi_context_menu(event)
        elif self.roi_type == 'rectangle' and self.current_roi is not None:
            # Rectangle already complete, show save menu
            self._show_roi_context_menu(event)
        else:
            # Cancel current ROI
            self._clear_roi()
            
    def _handle_point_roi_click(self, x: int, y: int):
        """Handle point ROI selection."""        
        # Add point to collection
        self.roi_points.append((x, y))
        
        # Draw point marker
        self._draw_point_marker(x, y, len(self.roi_points))
        
        # Update status
        count = len(self.roi_points)
        self.status_label.setText(f"POINT ROI: {count} point{'s' if count != 1 else ''} selected - right-click when done")
        
    def _handle_rectangle_roi_click(self, x: int, y: int):
        """Handle rectangle ROI selection."""        
        if self.roi_start is None:
            # First corner
            self.roi_start = (x, y)
            self._draw_roi_start_marker()
            self.status_label.setText(f"RECTANGLE: Corner 1 at ({x}, {y}) - click second corner")
        else:
            # Second corner - create rectangle
            self._create_rectangle_roi(self.roi_start, (x, y))
            self.status_label.setText("✓ Rectangle defined! Right-click to save to spectrum tab")
            
    def _handle_polygon_roi_click(self, x: int, y: int):
        """Handle polygon ROI selection."""
        # Add point to polygon
        self.roi_points.append((x, y))
        
        # Draw point marker
        self._draw_point_marker(x, y, len(self.roi_points))
        
        # Draw connecting lines
        if len(self.roi_points) > 1:
            self._draw_polygon_lines()
            
        # Update status
        count = len(self.roi_points)
        self.status_label.setText(f"POLYGON ROI: {count} point{'s' if count != 1 else ''} - right-click when done")
        
    def _finalize_roi(self):
        """Finalize multi-point ROI and create ROI definition."""
        if self.roi_type == 'point':
            # Point ROI: just use the points directly
            self.current_roi = {
                'type': 'point',
                'points': self.roi_points.copy()
            }
        elif self.roi_type == 'polygon':
            # Polygon ROI: close the polygon and get all interior points
            if len(self.roi_points) >= 3:
                # Close the polygon visually
                self._draw_polygon_lines(close=True)
                
                # Get all points inside the polygon
                polygon_points = self._get_polygon_interior_points(self.roi_points)
                
                self.current_roi = {
                    'type': 'polygon', 
                    'boundary_points': self.roi_points.copy(),
                    'points': polygon_points
                }
            else:
                self.status_label.setText("Need at least 3 points for polygon")
                return
                
    def _create_rectangle_roi(self, start: tuple, end: tuple):
        """Create rectangle ROI definition."""
        x1, y1 = start
        x2, y2 = end
        
        # Ensure proper order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Validate ROI size
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
            
        # Create all points in rectangle
        points = []
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if self._is_valid_pixel(x, y):
                    points.append((x, y))
        
        self.current_roi = {
            'type': 'rectangle',
            'bounds': {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max},
            'points': points
        }
        
        # Draw rectangle outline
        self._draw_rectangle_outline(x_min, y_min, x_max, y_max)
            
    def _zoom_in(self):
        """Zoom in on the image."""
        view = self.image_widget.getView()
        view.scaleBy((0.8, 0.8))

    def _zoom_out(self):
        """Zoom out from the image."""
        view = self.image_widget.getView()
        view.scaleBy((1.2, 1.2))
        
    def _zoom_fit(self):
        """Fit image to view."""
        if self.image_data is not None:
            self.image_widget.autoRange()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if self.image_data is not None:
            # Get wheel delta (positive = scroll up, negative = scroll down)
            delta = event.angleDelta().y()

            view = self.image_widget.getView()
            if delta > 0:
                # Scroll up = zoom out
                view.scaleBy((0.8, 0.8))
            else:
                # Scroll down = zoom in
                view.scaleBy((1.2, 1.2))

        # Accept the event to prevent propagation
        event.accept()

    def event(self, event):
        """Handle events including gestures."""
        if event.type() == QtCore.QEvent.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def gestureEvent(self, event):
        """Handle gesture events for pinch-to-zoom."""
        gesture = event.gesture(QtCore.Qt.PinchGesture)
        if gesture is not None:
            return self.pinchTriggered(gesture)
        return True

    def pinchTriggered(self, gesture):
        """Handle pinch gesture for zooming."""
        if self.image_data is None:
            return True

        state = gesture.state()
        view = self.image_widget.getView()

        if state == QtCore.Qt.GestureStarted:
            # Reset scale tracking at the start of a new pinch
            self.last_pinch_scale = 1.0

        elif state == QtCore.Qt.GestureUpdated:
            # Get the scale factor change since last update
            current_scale = gesture.scaleFactor()
            scale_delta = current_scale / self.last_pinch_scale

            # Apply zoom based on pinch scale delta
            # Pinch out (spread fingers) = zoom in
            # Pinch in (bring fingers together) = zoom out
            view.scaleBy((scale_delta, scale_delta))

            # Update last scale for next delta calculation
            self.last_pinch_scale = current_scale

        elif state == QtCore.Qt.GestureFinished:
            # Reset for next gesture
            self.last_pinch_scale = 1.0

        return True
            
    def _on_range_changed(self, view, range_):
        """Handle view range changes."""
        self.zoom_changed.emit(range_)
        
    def _update_rgb_bands(self):
        """Signal to update RGB band selection."""
        # Emit signal to notify main application of RGB band changes
        self.rgb_bands_changed.emit()
        
    def _on_stretch_changed(self, value: float):
        """Handle contrast stretch percentage change."""
        self.stretch_percent = value
        # Update display with new stretch
        self._update_rgb_bands()
    
    def _on_colormap_changed(self, colormap_name: str):
        """Handle colormap change for monochromatic display."""
        self.current_colormap = colormap_name
        if self.mono_mode:
            # For single-band datasets, directly update the display with colormap
            if hasattr(self, 'current_dataset') and self.current_dataset:
                if hasattr(self.current_dataset, 'shape') and len(self.current_dataset.shape) >= 3 and self.current_dataset.shape[2] == 1:
                    # Get raw single-band data
                    try:
                        raw_band_data = self.current_dataset.get_band_data(0)
                        if raw_band_data is not None:
                            # Apply colormap and display
                            colored_image = self.apply_colormap(raw_band_data, colormap_name)
                            self.set_image(colored_image)
                            return
                    except Exception as e:
                        print(f"Error applying colormap to single-band data: {e}")
            
            # Fallback: Trigger a redraw with the new colormap
            self._update_rgb_bands()
        
    def get_rgb_bands(self) -> Tuple[int, int, int]:
        """Get current RGB band selection."""
        return (self.red_spinbox.value(), 
                self.green_spinbox.value(), 
                self.blue_spinbox.value())
                
    def get_stretch_percent(self) -> float:
        """Get current contrast stretch percentage."""
        return self.stretch_percent
        
    def set_stretch_percent(self, percent: float):
        """Set contrast stretch percentage."""
        self.stretch_percent = percent
        if hasattr(self, 'stretch_spinbox'):
            self.stretch_spinbox.setValue(percent)
            
    def get_histogram_levels(self):
        """Get current histogram levels (left/right limits)."""
        try:
            if hasattr(self, 'image_widget') and self.image_widget:
                hist_widget = self.image_widget.getHistogramWidget()
                if hist_widget:
                    levels = hist_widget.getLevels()
                    return levels
        except Exception as e:
            print(f"Error getting histogram levels: {e}")
        return None
        
    def set_histogram_levels(self, levels):
        """Set histogram levels (left/right limits)."""
        try:
            if levels is not None and hasattr(self, 'image_widget') and self.image_widget:
                hist_widget = self.image_widget.getHistogramWidget()
                if hist_widget:
                    hist_widget.setLevels(levels)
                    self.histogram_levels = levels
        except Exception as e:
            print(f"Error setting histogram levels: {e}")
            
    def get_view_range(self):
        """Get current view range (zoom and position)."""
        try:
            if hasattr(self, 'image_widget') and self.image_widget:
                view = self.image_widget.getView()
                if view:
                    return view.viewRange()
        except Exception as e:
            print(f"Error getting view range: {e}")
        return None
        
    def set_view_range(self, view_range):
        """Set view range (zoom and position)."""
        try:
            if view_range is not None and hasattr(self, 'image_widget') and self.image_widget:
                view = self.image_widget.getView()
                if view:
                    view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
                    self.view_range = view_range
        except Exception as e:
            print(f"Error setting view range: {e}")
                
    def set_rgb_bands(self, red: int, green: int, blue: int):
        """Set RGB band selection."""
        self.red_spinbox.setValue(red)
        self.green_spinbox.setValue(green)
        self.blue_spinbox.setValue(blue)
        
    def set_band_limits(self, max_bands: int):
        """Set maximum band numbers for spinboxes."""
        max_val = max_bands - 1
        self.red_spinbox.setMaximum(max_val)
        self.green_spinbox.setMaximum(max_val)
        self.blue_spinbox.setMaximum(max_val)
        
        # Also update slider ranges if in mono or RGB channel selection mode
        if self.mono_mode or self.selected_rgb_channel:
            self.line_slider.setMaximum(max_val)
            self.line_spinbox.setMaximum(max_val)
            # Ensure current values are within range
            if self.current_band > max_val:
                self.current_band = max_val
                self.line_slider.setValue(self.current_band)
                self.line_spinbox.setValue(self.current_band)
        
    def _set_default_rgb_bands(self):
        """Set RGB bands to true color or well-spaced default."""
        # Exit mono mode if active
        if self.mono_mode:
            self.mono_action.setChecked(False)
            self._toggle_mono_mode(False)
        
        # Signal to main app to set default RGB bands
        # The main app will handle getting the wavelengths and setting the bands
        self.default_rgb_requested.emit()
        
    def get_current_roi(self) -> Optional[dict]:
        """Get current ROI definition."""
        return self.current_roi
        
    def set_zoom_box(self, rect: dict):
        """Set zoom box for overview display."""
        # Remove existing zoom box
        view = self.image_widget.getView()
        for item in view.allChildren():
            if hasattr(item, 'zoom_box') and item.zoom_box:
                view.removeItem(item)
                
        # Create new zoom box
        zoom_rect = pg.RectROI([rect['x'], rect['y']], [rect['width'], rect['height']], 
                             pen=pg.mkPen('y', width=1, style=QtCore.Qt.DashLine))
        zoom_rect.zoom_box = True
        zoom_rect.setZValue(999)
        
        view.addItem(zoom_rect)
        self.zoom_box = zoom_rect
        
    def get_view_range(self) -> dict:
        """Get current view range."""
        view_range = self.image_widget.getView().viewRange()
        return {
            'x': view_range[0][0],
            'y': view_range[1][0],
            'width': view_range[0][1] - view_range[0][0],
            'height': view_range[1][1] - view_range[1][0]
        }
    
    def _toggle_frame_view_mode(self, enabled: bool):
        """Toggle between RGB and frame view modes."""
        print(f"Frame View toggle called: enabled={enabled}, current_mode={self.frame_view_mode}")
        self.frame_view_mode = enabled
        
        # Show/hide appropriate controls
        rgb_controls = [
            self.red_spinbox, self.green_spinbox, self.blue_spinbox
        ]
        
        # RGB labels are now class attributes
        rgb_labels = [self.red_label, self.green_label, self.blue_label]
        
        if enabled:
            # Hide RGB controls, show frame controls
            for control in rgb_controls + rgb_labels:
                control.hide()
            for control in self.frame_controls:
                control.show()

            # Provide immediate user feedback
            self.status_label.setText("🔥 Frame View enabled - loading spatial-spectral heatmap...")
            print(f"   UI updated: RGB controls hidden, Frame controls shown")
        else:
            # Show RGB controls, hide frame controls
            for control in rgb_controls + rgb_labels:
                control.show()
            for control in self.frame_controls:
                control.hide()

            # Provide immediate user feedback
            self.status_label.setText("RGB composite view restored")
            print(f"   UI updated: RGB controls shown, Frame controls hidden")

        # Emit signal to update display mode
        if enabled:
            print(f"   Emitting line_changed signal with line {self.current_line}")
            self.line_changed.emit(self.current_line)
        else:
            print(f"   Emitting rgb_bands_changed signal")
            self.rgb_bands_changed.emit()
    
    def _toggle_mono_mode(self, enabled: bool):
        """Toggle between RGB and monochromatic modes."""
        self.mono_mode = enabled
        
        # Clear any RGB channel selection when changing modes
        self._clear_rgb_channel_selection()
        
        if enabled:
            # In mono mode: set all RGB channels to current_band
            self.red_spinbox.setValue(self.current_band)
            self.green_spinbox.setValue(self.current_band) 
            self.blue_spinbox.setValue(self.current_band)
            
            # Change frame slider to band selector
            self.line_label.setText("Band:")
            self.line_slider.setToolTip("Select spectral band for monochromatic display")
            
            # Show the slider controls for band selection
            for control in self.frame_controls:
                control.show()
                
            # Show colormap controls
            self.colormap_label.show()
            self.colormap_combo.show()
            self.apply_bounds_button.show()
            self.reset_bounds_button.show()
            
            # Update slider range to match available bands (0-based indexing)
            max_bands = self.red_spinbox.maximum()  # Use actual band count, not fallback
            self.line_slider.setMaximum(max_bands)  # Already 0-based since spinbox max is already max_band_index
            self.line_spinbox.setMaximum(max_bands)  # Sync spinbox range too
            self.line_slider.setValue(self.current_band)
            self.line_spinbox.setValue(self.current_band)
            
        else:
            # Back to RGB mode: hide band selector, restore frame controls
            self.line_label.setText("Line:")
            self.line_slider.setToolTip("Select spatial line (row) for cross-section display")
            
            # Hide colormap controls
            self.colormap_label.hide()
            self.colormap_combo.hide()
            self.apply_bounds_button.hide()
            self.reset_bounds_button.hide()
            
            # Hide controls unless in frame view mode
            if not self.frame_view_mode:
                for control in self.frame_controls:
                    control.hide()
            
        # Update the display
        self._update_rgb_bands()
        
        # Emit band change signal to update spectrum plot indicators
        if enabled:
            # Mono mode - emit mono signal
            self.band_changed.emit(self.current_band, 'mono')
        else:
            # RGB mode - emit RGB update (will trigger RGB indicators refresh)
            # The _update_rgb_bands() call above will trigger rgb_bands_changed signal
            pass
        
    def _select_rgb_channel(self, channel: str):
        """Select RGB channel for band slider control."""
        print(f"RGB Channel selected: {channel}")
        # Toggle behavior - if clicking on already selected channel, clear selection
        if self.selected_rgb_channel == channel:
            print(f"Toggling off channel {channel}")
            self._clear_rgb_channel_selection()
            return
            
        # Clear previous selection
        self.red_spinbox.setStyleSheet("")
        self.green_spinbox.setStyleSheet("")
        self.blue_spinbox.setStyleSheet("")
        
        # Highlight selected channel spinbox
        if channel == 'R':
            self.red_spinbox.setStyleSheet("QSpinBox { background-color: rgba(255, 0, 0, 50); }")
            self.selected_rgb_channel = 'R'
            self.current_band = self.red_spinbox.value()
            print(f"Red channel selected, current_band: {self.current_band}")
        elif channel == 'G':
            self.green_spinbox.setStyleSheet("QSpinBox { background-color: rgba(0, 255, 0, 50); }")
            self.selected_rgb_channel = 'G'
            self.current_band = self.green_spinbox.value()
            print(f"Green channel selected, current_band: {self.current_band}")
        elif channel == 'B':
            self.blue_spinbox.setStyleSheet("QSpinBox { background-color: rgba(0, 0, 255, 50); }")
            self.selected_rgb_channel = 'B'
            self.current_band = self.blue_spinbox.value()
            print(f"Blue channel selected, current_band: {self.current_band}")
            
        # Show band selection controls
        self.line_label.setText(f"Band ({channel}):")
        self.line_slider.setToolTip(f"Select spectral band for {channel} channel")
        
        for control in self.frame_controls:
            control.show()
            
        # Update slider range to match available bands (0-based indexing)
        max_bands = self.red_spinbox.maximum()  # Use actual band count, not fallback
        print(f"Setting slider range: 0 to {max_bands}, current_band: {self.current_band}")
        self.line_slider.setMaximum(max_bands)  # Already 0-based since spinbox max is already max_band_index
        self.line_spinbox.setMaximum(max_bands)  # Sync spinbox range too
        
        # Update slider to current band
        self.line_slider.setValue(self.current_band)
        self.line_spinbox.setValue(self.current_band)
    
    def _clear_rgb_channel_selection(self):
        """Clear RGB channel selection and hide controls."""
        # Reset spinbox styles to default
        self.red_spinbox.setStyleSheet("")
        self.green_spinbox.setStyleSheet("")
        self.blue_spinbox.setStyleSheet("")
        
        # Clear selected channel
        self.selected_rgb_channel = None
        
        # Reset label text and hide controls if not in other modes
        if not self.mono_mode and not self.frame_view_mode:
            self.line_label.setText("Line:")
            for control in self.frame_controls:
                control.hide()
    
    def _on_line_slider_changed(self, line_index: int):
        """Handle line slider value changes."""
        if self.mono_mode:
            # In mono mode, slider controls band selection
            # Clamp line_index to valid range
            max_band = self.red_spinbox.maximum()
            line_index = min(max(line_index, 0), max_band)
            
            if self.current_band != line_index:
                self.current_band = line_index
                # Update all RGB channels to the same band
                self.red_spinbox.setValue(self.current_band)
                self.green_spinbox.setValue(self.current_band)
                self.blue_spinbox.setValue(self.current_band)
                # Update display
                self._update_rgb_bands()
                # Emit band change signal
                self.band_changed.emit(self.current_band, 'mono')
        elif self.selected_rgb_channel:
            # RGB channel selection mode
            print(f"Slider changed in RGB mode: {line_index}, selected_channel: {self.selected_rgb_channel}")
            # Clamp line_index to valid range
            max_band = self.red_spinbox.maximum()
            line_index = min(max(line_index, 0), max_band)
            
            if self.current_band != line_index:
                self.current_band = line_index
                print(f"Updating {self.selected_rgb_channel} channel to band {self.current_band}")
                # Update the selected RGB channel
                if self.selected_rgb_channel == 'R':
                    self.red_spinbox.setValue(self.current_band)
                elif self.selected_rgb_channel == 'G':
                    self.green_spinbox.setValue(self.current_band)
                elif self.selected_rgb_channel == 'B':
                    self.blue_spinbox.setValue(self.current_band)
                # Update display
                self._update_rgb_bands()
                # Emit band change signal
                self.band_changed.emit(self.current_band, self.selected_rgb_channel)
        else:
            # Normal frame view mode
            if self.current_line != line_index:
                self.current_line = line_index
                # Update spinbox (this will trigger _on_line_spinbox_changed)
                self.line_spinbox.setValue(line_index)
    
    def _on_line_spinbox_changed(self, line_index: int):
        """Handle line spinbox value changes."""
        if self.mono_mode:
            # In mono mode, spinbox controls band selection
            if self.current_band != line_index:
                self.current_band = line_index
                # Update all RGB channels to the same band
                self.red_spinbox.setValue(self.current_band)
                self.green_spinbox.setValue(self.current_band)
                self.blue_spinbox.setValue(self.current_band)
                # Update slider
                self.line_slider.setValue(line_index)
                # Update display
                self._update_rgb_bands()
                # Emit band change signal
                self.band_changed.emit(self.current_band, 'mono')
        elif self.selected_rgb_channel:
            # RGB channel selection mode
            if self.current_band != line_index:
                self.current_band = line_index
                # Update the selected RGB channel
                if self.selected_rgb_channel == 'R':
                    self.red_spinbox.setValue(self.current_band)
                elif self.selected_rgb_channel == 'G':
                    self.green_spinbox.setValue(self.current_band)
                elif self.selected_rgb_channel == 'B':
                    self.blue_spinbox.setValue(self.current_band)
                # Update slider
                self.line_slider.setValue(line_index)
                # Update display
                self._update_rgb_bands()
                # Emit band change signal
                self.band_changed.emit(self.current_band, self.selected_rgb_channel)
        else:
            # Normal frame view mode
            if self.current_line != line_index:
                self.current_line = line_index
                # Update slider
                self.line_slider.setValue(line_index)
                # Update line display
                self._update_line_display()
                # Emit signal for main app to update display
                if self.frame_view_mode:
                    self.line_changed.emit(line_index)
    
    def _update_line_display(self):
        """Update the line label."""
        if hasattr(self, 'total_lines') and self.total_lines > 0:
            self.line_label.setText(f"Line: {self.current_line}")
        else:
            self.line_label.setText("Line: N/A")
    
    def set_line_limits_for_frame_view(self, total_lines: int):
        """Set line limits for frame view controls."""
        self.total_lines = total_lines
        
        if total_lines > 0:
            max_line = total_lines - 1
            self.line_slider.setMaximum(max_line)
            self.line_spinbox.setMaximum(max_line)
            self._update_line_display()
    
    def set_spatial_spectral_heatmap(self, line_spectra: np.ndarray, line_index: int):
        """
        Set spatial-spectral cross-section heatmap to display.

        Args:
            line_spectra: 2D array of shape (spatial_positions, bands) containing
                         spectral data for all positions along a single spatial line
            line_index: The spatial line index being displayed
        """
        print(f"🔥 set_spatial_spectral_heatmap called: line_index={line_index}")
        if line_spectra is not None:
            print(f"   Input data shape: {line_spectra.shape}")
            print(f"   Input data range: {np.nanmin(line_spectra):.6f} - {np.nanmax(line_spectra):.6f}")
        else:
            print("   Input data is None!")

        if line_spectra is None or line_spectra.size == 0:
            print("   Returning early - no data")
            return
            
        # Transpose for display: bands (y-axis) vs spatial positions (x-axis)
        heatmap_data = line_spectra.T  # Shape: (bands, spatial_positions)
        
        # Handle NaN and infinite values
        valid_data = heatmap_data[np.isfinite(heatmap_data)]
        
        if valid_data.size == 0:
            # All invalid data, create zero heatmap
            normalized = np.zeros_like(heatmap_data, dtype=np.uint8)
        else:
            # Apply percentile stretching for better contrast
            p2 = np.percentile(valid_data, 2)
            p98 = np.percentile(valid_data, 98)
            
            if p98 > p2:
                # Apply linear stretch
                stretched = np.clip((heatmap_data - p2) / (p98 - p2), 0, 1)
                normalized = (stretched * 255).astype(np.uint8)
            else:
                # Constant values, use mid-gray
                normalized = np.full_like(heatmap_data, 128, dtype=np.uint8)
        
        # Try different image formats for PyQtGraph compatibility
        print(f"   🖼️ Preparing image data for display...")

        # Option 1: Try 2D grayscale (PyQtGraph often prefers this)
        grayscale_heatmap = normalized.astype(np.float32)  # Use float32 for better compatibility

        # Convert to RGB format for coordinate validation storage
        rgb_heatmap = np.stack([normalized, normalized, normalized], axis=2)
        self.image_data = rgb_heatmap  # Store RGB version for coordinate validation

        print(f"   Grayscale shape: {grayscale_heatmap.shape}, dtype: {grayscale_heatmap.dtype}")
        print(f"   Data range: {grayscale_heatmap.min()} - {grayscale_heatmap.max()}")

        try:
            # Try 2D grayscale first (most compatible)
            self.image_widget.setImage(grayscale_heatmap, autoRange=True, autoLevels=True,
                                     axes={'x': 1, 'y': 0})  # Specify axis order
            print("   ✅ setImage call completed successfully (2D grayscale)")
        except Exception as e1:
            print(f"   ⚠️ 2D grayscale setImage failed: {e1}")
            try:
                # Fallback: Try RGB format
                print(f"   Trying RGB format: {rgb_heatmap.shape}")
                self.image_widget.setImage(rgb_heatmap, autoRange=True, autoLevels=True)
                print("   ✅ setImage call completed successfully (RGB)")
            except Exception as e2:
                print(f"   ❌ Both setImage attempts failed:")
                print(f"     2D error: {e1}")
                print(f"     RGB error: {e2}")
                import traceback
                traceback.print_exc()

        # Update status to show current line info
        bands, positions = heatmap_data.shape
        status_message = f"Frame View: Line {line_index} | {positions} positions × {bands} bands"
        self.status_label.setText(status_message)
        print(f"   📊 Status updated: {status_message}")