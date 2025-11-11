"""
SpectrumPlot class for real-time spectral display with overlay capabilities.

Supports live pixel spectra and ROI statistics display using PyQtGraph.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from typing import Dict, List, Optional, Tuple, Union
import time

# Import the collected spectra tab
try:
    from .collected_spectra_tab import CollectedSpectraTab
except ImportError:
    from collected_spectra_tab import CollectedSpectraTab

# Conditional scipy import for Savitzky-Golay filtering
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    SCIPY_AVAILABLE = False
    print(f"Warning: scipy not available ({type(e).__name__}). Spectral smoothing will be disabled.")


class DraggableTabWidget(QtWidgets.QTabWidget):
    """Tab widget that supports drag-and-drop between instances."""
    
    # Signal emitted when tabs change (added/removed)
    tabs_changed = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.drag_start_pos = None
        self.dragging_tab_index = -1
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Check if click is on a tab by testing against tab bar directly
            tab_bar = self.tabBar()
            # Map position from widget coordinates to tab bar coordinates  
            tab_pos = tab_bar.mapFromParent(event.pos())
            tab_index = tab_bar.tabAt(tab_pos)
            
            # Alternative: also check with direct event position
            if tab_index < 0:
                tab_index = tab_bar.tabAt(event.pos())
                
            if tab_index >= 0:
                self.drag_start_pos = event.pos()
                self.dragging_tab_index = tab_index
                # Tab press detected
            else:
                self.drag_start_pos = None
                self.dragging_tab_index = -1
                pass  # No tab at this position
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        # Only start drag if we have a valid drag start position and tab
        if (event.buttons() == QtCore.Qt.LeftButton and 
            self.drag_start_pos is not None and
            self.dragging_tab_index >= 0 and
            (event.pos() - self.drag_start_pos).manhattanLength() > QtWidgets.QApplication.startDragDistance()):
            
            # Start drag operation
            self._start_drag(self.dragging_tab_index)
            # Reset drag state
            self.drag_start_pos = None
            self.dragging_tab_index = -1
        else:
            super().mouseMoveEvent(event)
                
    def _start_drag(self, tab_index):
        """Start dragging a tab."""
        if tab_index < 0 or tab_index >= self.count():
            # Invalid tab index
            return
            
        tab_widget = self.widget(tab_index)
        tab_text = self.tabText(tab_index)
        
        # Dragging tab
        
        # Create drag object
        drag = QtGui.QDrag(self)
        mime_data = QtCore.QMimeData()
        
        # Store drag data as JSON-like string for easier parsing
        drag_data = f"draggable_tab:{id(self)}:{tab_index}:{tab_text}"
        mime_data.setText(drag_data)
        drag.setMimeData(mime_data)
        
        # Set drag data
        
        # Create a more visible drag pixmap
        pixmap = QtGui.QPixmap(150, 25)
        pixmap.fill(QtCore.Qt.lightGray)
        painter = QtGui.QPainter(pixmap)
        painter.drawText(5, 15, f"Tab: {tab_text}")
        painter.end()
        drag.setPixmap(pixmap)
        
        # Execute drag with both Move and Copy actions allowed
        drop_action = drag.exec_(QtCore.Qt.MoveAction | QtCore.Qt.CopyAction, QtCore.Qt.MoveAction)
        # Drag operation completed
        
    def dragEnterEvent(self, event):
        mime_text = event.mimeData().text()
        # Check drag enter event
        if mime_text.startswith('draggable_tab:'):
            # Extract source widget ID
            parts = mime_text.split(':', 3)
            if len(parts) >= 3:
                source_id = parts[1]
                # Check source and target widgets
                # Only accept drops from other widgets (not same widget)
                if str(id(self)) != source_id:
                    # Accept drag from different widget
                    event.acceptProposedAction()
                    return
                else:
                    # Reject drag from same widget
                    pass
        # Ignore invalid drag
        event.ignore()
            
    def dragMoveEvent(self, event):
        # Same logic as dragEnterEvent
        mime_text = event.mimeData().text()
        if mime_text.startswith('draggable_tab:'):
            parts = mime_text.split(':', 3)
            if len(parts) >= 3:
                source_id = parts[1]
                if str(id(self)) != source_id:
                    event.acceptProposedAction()
                    return
        event.ignore()
            
    def dropEvent(self, event):
        mime_text = event.mimeData().text()
        # Handle drop event
        if mime_text.startswith('draggable_tab:'):
            parts = mime_text.split(':', 3)
            if len(parts) >= 4:
                source_id = parts[1]
                tab_index = int(parts[2])
                tab_text = parts[3]
                
                # Process drop data
                
                # Find source widget and move the tab
                source_widget = self._find_widget_by_id(source_id)
                # Source widget found
                if source_widget and source_widget != self:
                    # Get the tab widget before removing it
                    tab_widget = source_widget.widget(tab_index)
                    # Tab widget retrieved
                    if tab_widget:
                        # Remove from source
                        source_widget.removeTab(tab_index)
                        # Add to destination
                        new_index = self.addTab(tab_widget, tab_text)
                        # Switch to the moved tab
                        self.setCurrentIndex(new_index)
                        # Tab successfully moved
                        
                        # Emit signal for both source and destination
                        source_widget.tabs_changed.emit()
                        self.tabs_changed.emit()
                        
                        event.acceptProposedAction()
                        return
                    else:
                        pass  # Tab widget is None
                else:
                    pass  # Source widget not found or same as target
            else:
                pass  # Invalid parts count
        # Drop not accepted
        event.ignore()
                        
    def _find_widget_by_id(self, widget_id):
        """Find a DraggableTabWidget by its ID."""
        target_id = int(widget_id)
        for widget in QtWidgets.QApplication.allWidgets():
            if isinstance(widget, DraggableTabWidget) and id(widget) == target_id:
                return widget
        return None


class SpectrumPlot(QtWidgets.QWidget):
    """Real-time spectrum display widget with overlay capabilities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.wavelengths = None
        self.current_spectrum = None
        self.roi_spectra = {}
        self.plot_items = {}
        self.colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ff8000', '#8000ff', '#00ff80', '#ff0080']
        self.color_index = 0
        
        # Track multiple ROIs per tab: {tab_index: [roi_data1, roi_data2, ...]}
        self.tab_roi_data = {}
        self.tab_color_index = {}  # Track color index per tab
        
        # RGB band indicators
        self.rgb_band_lines = {'red': None, 'green': None, 'blue': None}
        self.current_rgb_bands = (0, 0, 0)
        
        # Current band indicator (for mono mode and RGB selection)
        self.current_band_line = None
        self.current_band = 0
        
        # Split view state
        self.split_mode = False
        self.split_splitter = None
        self.split_view_1 = None  # Top view
        self.split_view_2 = None  # Bottom view

        # Smoothing state
        self.smoothing_enabled = False
        self.smoothing_polynomial_order = 2
        self.smoothing_window_length = 11

        self._setup_ui()
        self._setup_plot()
        
    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        
        # Create toolbar
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)
        
        # Create container for tab widget(s) - will hold single or split views
        self.tab_container = QtWidgets.QWidget()
        self.tab_layout = QtWidgets.QVBoxLayout(self.tab_container)
        self.tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main tab widget for different spectrum views
        self.tab_widget = DraggableTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_roi_tab)
        
        # Add tab widget to container
        self.tab_layout.addWidget(self.tab_widget)
        
        # Create main spectrum tab (pixel spectra)
        main_tab = self._create_spectrum_tab("Main")
        self.main_plot_widget = main_tab['plot_widget']
        self.main_info_panel = main_tab['info_panel']
        self.tab_widget.addTab(main_tab['widget'], "Pixel Spectra")

        # Create collected spectra tab
        self.collected_spectra_tab = CollectedSpectraTab()
        self.tab_widget.addTab(self.collected_spectra_tab, "Collected Spectra")

        # Set the main plot widget as the default one (for backward compatibility)
        self.plot_widget = self.main_plot_widget
        self.info_panel = self.main_info_panel
        
        layout.addWidget(self.tab_container)
        
        self.setLayout(layout)
        
    def _create_toolbar(self) -> QtWidgets.QToolBar:
        """Create toolbar with plot controls."""
        toolbar = QtWidgets.QToolBar()
        
        # Clear all spectra
        clear_action = toolbar.addAction("Clear All")
        clear_action.triggered.connect(self.clear_all_spectra)
        
        # Export options
        export_action = toolbar.addAction("Export Data")
        export_action.triggered.connect(self._export_spectra)
        
        toolbar.addSeparator()
        
        # RGB band indicators toggle
        self.show_rgb_bands_checkbox = QtWidgets.QCheckBox("Show RGB Bands")
        self.show_rgb_bands_checkbox.setChecked(True)
        self.show_rgb_bands_checkbox.toggled.connect(self._toggle_rgb_bands)
        toolbar.addWidget(self.show_rgb_bands_checkbox)
        
        toolbar.addSeparator()
        
        # Y-axis controls
        toolbar.addWidget(QtWidgets.QLabel("Y-axis:"))
        self.y_auto_checkbox = QtWidgets.QCheckBox("Auto")
        self.y_auto_checkbox.setChecked(True)
        self.y_auto_checkbox.toggled.connect(self._toggle_y_auto)
        toolbar.addWidget(self.y_auto_checkbox)
        
        self.y_min_spinbox = QtWidgets.QDoubleSpinBox()
        self.y_min_spinbox.setDecimals(4)
        self.y_min_spinbox.setRange(-100000, 100000)  # Expanded range for uncalibrated data
        self.y_min_spinbox.setValue(0)
        self.y_min_spinbox.setEnabled(False)
        toolbar.addWidget(self.y_min_spinbox)
        
        self.y_max_spinbox = QtWidgets.QDoubleSpinBox()
        self.y_max_spinbox.setDecimals(4)
        self.y_max_spinbox.setRange(-100000, 100000)  # Expanded range for uncalibrated data
        self.y_max_spinbox.setValue(1)
        self.y_max_spinbox.setEnabled(False)
        toolbar.addWidget(self.y_max_spinbox)
        
        apply_y_action = toolbar.addAction("Apply Y")
        apply_y_action.triggered.connect(self._apply_y_range)

        toolbar.addSeparator()

        # Smoothing controls (only if scipy is available)
        if SCIPY_AVAILABLE:
            self.smoothing_checkbox = QtWidgets.QCheckBox("Smoothing")
            self.smoothing_checkbox.setToolTip("Enable Savitzky-Golay spectral smoothing")
            self.smoothing_checkbox.toggled.connect(self._toggle_smoothing)
            toolbar.addWidget(self.smoothing_checkbox)

            # Polynomial order dropdown
            toolbar.addWidget(QtWidgets.QLabel("Order:"))
            self.poly_order_combo = QtWidgets.QComboBox()
            self.poly_order_combo.addItems(["2", "3"])
            self.poly_order_combo.setCurrentText("2")
            self.poly_order_combo.setToolTip("Polynomial order for Savitzky-Golay filter")
            self.poly_order_combo.currentTextChanged.connect(self._on_poly_order_changed)
            self.poly_order_combo.setEnabled(False)
            toolbar.addWidget(self.poly_order_combo)

            # Window length dropdown
            toolbar.addWidget(QtWidgets.QLabel("Window:"))
            self.window_length_combo = QtWidgets.QComboBox()
            window_lengths = [str(w) for w in range(11, 51, 4)]  # 11, 15, 19, 23, 27, 31, 35, 39, 43, 47
            self.window_length_combo.addItems(window_lengths)
            self.window_length_combo.setCurrentText("11")
            self.window_length_combo.setToolTip("Window length for Savitzky-Golay filter")
            self.window_length_combo.currentTextChanged.connect(self._on_window_length_changed)
            self.window_length_combo.setEnabled(False)
            toolbar.addWidget(self.window_length_combo)
        else:
            # Add disabled smoothing notice if scipy is not available
            self.smoothing_checkbox = None
            self.poly_order_combo = None
            self.window_length_combo = None

        toolbar.addSeparator()

        # Split view controls
        self.split_action = QtWidgets.QAction("Split View", self)
        self.split_action.setCheckable(True)
        self.split_action.setToolTip("Split spectrum view vertically")
        self.split_action.triggered.connect(self._toggle_split_view)
        toolbar.addAction(self.split_action)
        
        toolbar.addSeparator()
        
        # Image source selector (for selecting which image view to track)
        toolbar.addWidget(QtWidgets.QLabel("Track Image:"))
        self.image_source_combo = QtWidgets.QComboBox()
        self.image_source_combo.setToolTip("Select which image view to track for spectra")
        self.image_source_combo.addItems(["Active View", "View 1", "View 2"])
        self.image_source_combo.currentTextChanged.connect(self._on_image_source_changed)
        toolbar.addWidget(self.image_source_combo)
        self.tracked_image_view = "Active View"  # Default to active view
        
        return toolbar
        
    def _create_spectrum_tab(self, tab_name: str) -> dict:
        """Create a new spectrum tab with plot and info panel."""
        tab_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Create toolbar for this tab
        tab_toolbar = QtWidgets.QToolBar()
        tab_toolbar.setMaximumHeight(30)
        
        # View selector for this tab
        tab_toolbar.addWidget(QtWidgets.QLabel("Track View:"))
        view_selector = QtWidgets.QComboBox()
        view_selector.setToolTip("Select which image view this spectrum tab should track")
        view_selector.addItem("None", "")
        # We'll populate this dynamically
        # Store the tab name as a property so we can identify which tab this selector belongs to
        view_selector.setProperty("tab_name", tab_name)
        view_selector.currentTextChanged.connect(
            lambda text, selector=view_selector: self._on_tab_view_selector_changed(selector)
        )
        tab_toolbar.addWidget(view_selector)
        
        # Store selector reference for later updates
        if not hasattr(self, 'tab_view_selectors'):
            self.tab_view_selectors = {}
        self.tab_view_selectors[tab_name] = view_selector
        
        layout.addWidget(tab_toolbar)
        
        # Create plot widget
        plot_widget = pg.PlotWidget()
        plot_widget.setLabel('left', 'Reflectance/Radiance')
        plot_widget.setLabel('bottom', 'Wavelength (nm)')
        plot_widget.showGrid(True, True, alpha=0.3)
        plot_widget.setMouseEnabled(x=True, y=True)
        plot_widget.setMinimumSize(300, 200)
        
        layout.addWidget(plot_widget)
        
        # Create info panel for this tab
        info_panel = self._create_info_panel()
        layout.addWidget(info_panel)
        
        tab_widget.setLayout(layout)
        
        # Setup plot appearance
        plot_widget.setBackground('w')
        
        # Add crosshair cursor
        crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        plot_widget.addItem(crosshair_v, ignoreBounds=True)
        plot_widget.addItem(crosshair_h, ignoreBounds=True)
        
        # Connect mouse events
        def on_mouse_moved(pos):
            self._on_mouse_moved_tab(pos, plot_widget, info_panel, crosshair_v, crosshair_h)
        
        plot_widget.scene().sigMouseMoved.connect(on_mouse_moved)
        
        return {
            'widget': tab_widget,
            'plot_widget': plot_widget,
            'info_panel': info_panel,
            'crosshair_v': crosshair_v,
            'crosshair_h': crosshair_h
        }
        
    def _create_info_panel(self) -> QtWidgets.QWidget:
        """Create information panel."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        
        # Current pixel info
        pixel_label = QtWidgets.QLabel("Pixel: None")
        pixel_label.setObjectName("pixel_label")
        layout.addWidget(pixel_label)
        
        # Statistics info
        stats_label = QtWidgets.QLabel("Stats: None")
        stats_label.setObjectName("stats_label")
        layout.addWidget(stats_label)
        
        # Wavelength info
        wavelength_label = QtWidgets.QLabel("Wavelength: None")
        wavelength_label.setObjectName("wavelength_label")
        layout.addWidget(wavelength_label)
        
        panel.setLayout(layout)
        panel.setMaximumHeight(30)
        
        # Set references for main tab
        if not hasattr(self, 'pixel_label'):
            self.pixel_label = pixel_label
            self.stats_label = stats_label
            self.wavelength_label = wavelength_label
        
        return panel
        
    def _close_roi_tab(self, index: int):
        """Close an ROI tab."""
        if index > 0:  # Don't allow closing the main tab
            # Clean up tab data tracking
            if index in self.tab_roi_data:
                del self.tab_roi_data[index]
            if index in self.tab_color_index:
                del self.tab_color_index[index]
            
            # Remove all ROIs associated with this tab
            roi_ids_to_remove = []
            for roi_id, roi_info in self.roi_spectra.items():
                if roi_info['tab_index'] == index:
                    roi_ids_to_remove.append(roi_id)
            
            for roi_id in roi_ids_to_remove:
                del self.roi_spectra[roi_id]
            
            # Remove tab
            self.tab_widget.removeTab(index)
            
            # Update tab indices for remaining tabs (since they shift down)
            for roi_id, roi_info in self.roi_spectra.items():
                if roi_info['tab_index'] > index:
                    roi_info['tab_index'] -= 1
                    roi_info['tab_data']['tab_index'] -= 1
                    
    def clear_tab_rois(self, tab_index: int):
        """Clear all ROI spectra from a specific tab."""
        if tab_index <= 0:  # Don't allow clearing main tab
            return
            
        # Remove ROI data for this tab
        roi_ids_to_remove = []
        for roi_id, roi_info in self.roi_spectra.items():
            if roi_info['tab_index'] == tab_index:
                roi_ids_to_remove.append(roi_id)
        
        for roi_id in roi_ids_to_remove:
            del self.roi_spectra[roi_id]
        
        # Clear tab data tracking
        if tab_index in self.tab_roi_data:
            del self.tab_roi_data[tab_index]
        if tab_index in self.tab_color_index:
            del self.tab_color_index[tab_index]
            
        # Clear the plot and re-add crosshairs
        tab_data = None
        for roi_info in self.roi_spectra.values():
            if roi_info['tab_index'] == tab_index:
                tab_data = roi_info['tab_data']
                break
                
        if tab_data:
            plot_widget = tab_data['plot_widget']
            plot_widget.clear()
            plot_widget.addItem(tab_data['crosshair_v'], ignoreBounds=True)
            plot_widget.addItem(tab_data['crosshair_h'], ignoreBounds=True)
            
            # Update info panel
            info_panel = tab_data['info_panel']
            pixel_label = info_panel.findChild(QtWidgets.QLabel, "pixel_label")
            stats_label = info_panel.findChild(QtWidgets.QLabel, "stats_label")
            if pixel_label:
                pixel_label.setText("No ROI data")
            if stats_label:
                stats_label.setText("")
            
    def add_roi_tab(self, roi_id: str, roi_definition: dict, roi_stats = None) -> int:
        """Add new ROI tab with spectrum data."""
        
        tab_name = f"ROI {len([t for t in range(self.tab_widget.count()) if 'ROI' in self.tab_widget.tabText(t)]) + 1}"
        
        # Create new tab
        roi_tab = self._create_spectrum_tab(tab_name)
        
        # Add tab to widget
        tab_index = self.tab_widget.addTab(roi_tab['widget'], tab_name)
        
        # Store tab_index in tab_data for tracking
        roi_tab['tab_index'] = tab_index
        
        # Store ROI data
        self.roi_spectra[roi_id] = {
            'tab_index': tab_index,
            'roi_definition': roi_definition,
            'roi_stats': roi_stats,
            'tab_data': roi_tab
        }
        
        # If ROI stats are available, plot them
        if roi_stats is not None:
            self._plot_roi_in_tab(roi_id, roi_stats, roi_tab)
        
        # Switch to new tab
        self.tab_widget.setCurrentIndex(tab_index)
        
        return tab_index
        
    def add_roi_to_existing_tab(self, roi_id: str, roi_definition: dict, roi_stats, tab_index: int) -> bool:
        """Add ROI spectrum to an existing tab for overlay comparison.
        
        Args:
            roi_id: ROI identifier
            roi_definition: ROI definition
            roi_stats: ROI statistics
            tab_index: Index of existing tab to add to
            
        Returns:
            True if successful, False otherwise
        """
        if tab_index <= 0 or tab_index >= self.tab_widget.count():
            return False
            
        # Find tab_data from existing ROI
        tab_data = None
        for existing_roi_id, roi_info in self.roi_spectra.items():
            if roi_info['tab_index'] == tab_index:
                tab_data = roi_info['tab_data']
                break
                
        if tab_data is None:
            return False
            
        # Store new ROI data
        self.roi_spectra[roi_id] = {
            'tab_index': tab_index,
            'roi_definition': roi_definition,
            'roi_stats': roi_stats,
            'tab_data': tab_data
        }
        
        # Plot ROI without clearing existing plots
        if roi_stats is not None:
            self._plot_roi_in_tab(roi_id, roi_stats, tab_data, clear_existing=False)
            
        # Switch to the tab
        self.tab_widget.setCurrentIndex(tab_index)
        
        return True
        
    def update_roi_in_existing_tab(self, roi_id: str, roi_stats):
        """Update ROI data in existing tab."""
        if roi_id in self.roi_spectra:
            tab_data = self.roi_spectra[roi_id]['tab_data']
            self._plot_roi_in_tab(roi_id, roi_stats, tab_data)
            self.roi_spectra[roi_id]['roi_stats'] = roi_stats
            
    def _plot_roi_in_tab(self, roi_id: str, roi_stats, tab_data, clear_existing=True):
        """Plot ROI statistics in the specified tab.
        
        Args:
            roi_id: ROI identifier
            roi_stats: ROI statistics object
            tab_data: Tab data dictionary
            clear_existing: If True, clear existing plots; if False, overlay new ROI
        """
        if self.wavelengths is None or roi_stats is None:
            return
            
        plot_widget = tab_data['plot_widget']
        info_panel = tab_data['info_panel']
        tab_index = tab_data.get('tab_index', -1)
        
        # Clear existing plots if requested
        if clear_existing:
            plot_widget.clear()
            # Reset tab data tracking
            if tab_index >= 0:
                self.tab_roi_data[tab_index] = []
                self.tab_color_index[tab_index] = 0
            
            # Re-add crosshair
            plot_widget.addItem(tab_data['crosshair_v'], ignoreBounds=True)
            plot_widget.addItem(tab_data['crosshair_h'], ignoreBounds=True)
        
        # Get color for this ROI
        if tab_index >= 0:
            if tab_index not in self.tab_color_index:
                self.tab_color_index[tab_index] = 0
            color_idx = self.tab_color_index[tab_index] % len(self.colors)
            roi_color = self.colors[color_idx]
            self.tab_color_index[tab_index] += 1
        else:
            roi_color = '#ff0000'  # Default red for backward compatibility
        
        # Store ROI data for this tab
        if tab_index >= 0:
            if tab_index not in self.tab_roi_data:
                self.tab_roi_data[tab_index] = []
            
            roi_data_entry = {
                'roi_id': roi_id,
                'roi_stats': roi_stats,
                'color': roi_color
            }
            self.tab_roi_data[tab_index].append(roi_data_entry)
        
        # Plot mean spectrum
        if hasattr(roi_stats, 'mean') and roi_stats.mean is not None:
            # Apply smoothing if enabled
            smoothed_mean = self._apply_smoothing(roi_stats.mean)

            plot_widget.plot(
                self.wavelengths, smoothed_mean,
                pen=pg.mkPen(roi_color, width=3),
                name=f'ROI {roi_id} Mean'
            )

            # Plot standard deviation envelope if available
            if hasattr(roi_stats, 'std') and roi_stats.std is not None:
                smoothed_std = self._apply_smoothing(roi_stats.std)
                upper = smoothed_mean + smoothed_std
                lower = smoothed_mean - smoothed_std
                
                fill_color = pg.mkColor(roi_color)
                fill_color.setAlphaF(0.15)  # More transparent for overlays
                
                fill_item = pg.FillBetweenItem(
                    pg.PlotCurveItem(self.wavelengths, upper),
                    pg.PlotCurveItem(self.wavelengths, lower),
                    brush=pg.mkBrush(fill_color)
                )
                plot_widget.addItem(fill_item)
        
        # Update info panel to show all ROIs in this tab
        self._update_tab_info_panel(tab_index, info_panel)
        
    def _update_tab_info_panel(self, tab_index: int, info_panel):
        """Update info panel to show all ROIs in the tab."""
        if tab_index < 0 or tab_index not in self.tab_roi_data:
            return
            
        pixel_label = info_panel.findChild(QtWidgets.QLabel, "pixel_label")
        stats_label = info_panel.findChild(QtWidgets.QLabel, "stats_label")
        
        roi_data_list = self.tab_roi_data[tab_index]
        
        if pixel_label:
            if len(roi_data_list) == 1:
                roi_data = roi_data_list[0]
                roi_stats = roi_data['roi_stats']
                pixel_count = roi_stats.count if hasattr(roi_stats, 'count') else 'N/A'
                pixel_label.setText(f"ROI: {roi_data['roi_id']} ({pixel_count} pixels)")
            else:
                total_pixels = sum(
                    rd['roi_stats'].count if hasattr(rd['roi_stats'], 'count') else 0 
                    for rd in roi_data_list
                )
                pixel_label.setText(f"ROIs: {len(roi_data_list)} overlaid ({total_pixels} total pixels)")
        
        if stats_label:
            if len(roi_data_list) == 1:
                roi_data = roi_data_list[0]
                roi_stats = roi_data['roi_stats']
                if hasattr(roi_stats, 'mean') and roi_stats.mean is not None:
                    mean_val = np.mean(roi_stats.mean)
                    std_val = np.std(roi_stats.mean)
                    min_val = np.min(roi_stats.mean)
                    max_val = np.max(roi_stats.mean)
                    
                    stats_label.setText(
                        f"ROI {roi_data['roi_id']} - Mean: {mean_val:.4f}, Std: {std_val:.4f}, "
                        f"Min: {min_val:.4f}, Max: {max_val:.4f}"
                    )
            else:
                # Show legend of ROIs with colors
                roi_names = []
                for roi_data in roi_data_list:
                    color = roi_data['color']
                    roi_id = roi_data['roi_id']
                    roi_names.append(f"● {roi_id}")
                
                stats_label.setText("Multiple ROIs: " + ", ".join(roi_names))
            
    def _on_mouse_moved_tab(self, pos, plot_widget, info_panel, crosshair_v, crosshair_h):
        """Handle mouse movement for crosshair and wavelength display in specific tab."""
        if plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Update crosshair
            crosshair_v.setPos(x)
            crosshair_h.setPos(y)
            
            # Find closest wavelength
            if self.wavelengths is not None:
                idx = np.argmin(np.abs(self.wavelengths - x))
                if 0 <= idx < len(self.wavelengths):
                    wl = self.wavelengths[idx]
                    
                    # Update wavelength label in info panel
                    wavelength_label = info_panel.findChild(QtWidgets.QLabel, "wavelength_label")
                    if wavelength_label:
                        wavelength_label.setText(f"Wavelength: {wl:.1f} nm")
        
    def get_existing_roi_tabs(self) -> list:
        """Get list of existing ROI tab names for selection."""
        roi_tabs = []
        for i in range(1, self.tab_widget.count()):  # Skip main tab (index 0)
            tab_text = self.tab_widget.tabText(i)
            if 'ROI' in tab_text:
                roi_tabs.append((i, tab_text))
        return roi_tabs
        
    def _setup_plot(self):
        """Setup plot appearance and behavior."""
        self.plot_widget.setBackground('w')
        
        # Set fixed axis formatting to prevent layout shifting
        bottom_axis = self.plot_widget.getAxis('bottom')
        left_axis = self.plot_widget.getAxis('left')
        
        # Create custom axis items with fixed formatting
        class FixedFormatAxis(pg.AxisItem):
            def __init__(self, orientation, decimals=1):
                super().__init__(orientation)
                self.decimals = decimals
                
            def tickStrings(self, values, scale, spacing):
                return [f"{v:.{self.decimals}f}" for v in values]
        
        # Replace axes with fixed format versions
        self.plot_widget.setAxisItems({'bottom': FixedFormatAxis('bottom', decimals=1), 
                                      'left': FixedFormatAxis('left', decimals=1)})
        
        # Add crosshair cursor
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Connect mouse events
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
    def set_wavelengths(self, wavelengths: np.ndarray):
        """Set wavelength values for x-axis."""
        self.wavelengths = wavelengths
        if wavelengths is not None:
            # Use fixed format for range display to prevent layout shifts
            self.plot_widget.setLabel('bottom', f'Wavelength (nm): {wavelengths[0]:.1f} - {wavelengths[-1]:.1f}')

            # Set fixed X-axis range to show full wavelength spectrum
            self.plot_widget.setXRange(wavelengths[0], wavelengths[-1], padding=0)
            # Disable X-axis auto-range to keep full spectrum view
            self.plot_widget.getViewBox().setLimits(xMin=wavelengths[0], xMax=wavelengths[-1])

            # Set wavelengths for collected spectra tab
            if hasattr(self, 'collected_spectra_tab'):
                self.collected_spectra_tab.set_wavelengths(wavelengths)
            self.plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.XAxis, enable=False)

    def collect_spectrum(self, spectrum: np.ndarray, x: int, y: int, name: str = None, source_file: str = None) -> str:
        """
        Collect a spectrum in the collected spectra tab.

        Args:
            spectrum: Spectral values array
            x: Pixel x coordinate
            y: Pixel y coordinate
            name: Optional custom name
            source_file: Optional source file path

        Returns:
            Spectrum ID
        """
        if hasattr(self, 'collected_spectra_tab'):
            return self.collected_spectra_tab.collect_spectrum(spectrum, x, y, name, source_file)
        else:
            print("Warning: No collected spectra tab available")
            return None
            
    def update_pixel_spectrum(self, spectrum: np.ndarray, x: int, y: int, source_view: str = None, 
                              active_bands: List[int] = None, dataset_name: str = None):
        """
        Update the current pixel spectrum.
        
        Args:
            spectrum: Spectral values
            x: Pixel x coordinate
            y: Pixel y coordinate
            source_view: Optional identifier for the source image view
            active_bands: List of active (good) band indices to display
            dataset_name: Name of the dataset for bad bands filtering
        """
        if spectrum is None or self.wavelengths is None:
            return
        
        # Apply bad bands filtering if active_bands is provided
        if active_bands is not None:
            spectrum, plot_wavelengths = self._filter_spectrum_for_bad_bands(spectrum, self.wavelengths, active_bands)
        else:
            plot_wavelengths = self.wavelengths
            
        # Determine which plot widget to use based on split mode and tracking preferences
        target_plot_widget = self.plot_widget
        target_info_panel = self.info_panel if hasattr(self, 'info_panel') else None
        
        if self.split_mode and self.split_view_1 and self.split_view_2:
            # In split mode, route to appropriate view based on tracking preferences
            # For now, update both views until tracking is fully implemented
            for view_widget in [self.split_view_1, self.split_view_2]:
                if view_widget and view_widget.count() > 0:
                    widget = view_widget.widget(0)
                    plot_widgets = widget.findChildren(pg.PlotWidget)
                    if plot_widgets:
                        plot_widget = plot_widgets[0]
                        # Remove existing pixel spectrum
                        existing_items = [item for item in plot_widget.listDataItems() 
                                        if hasattr(item, 'name') and item.name() and 'Pixel' in item.name()]
                        for item in existing_items:
                            plot_widget.removeItem(item)
                        # Plot new spectrum with filtered wavelengths
                        plot_widget.plot(
                            plot_wavelengths, spectrum,
                            pen=pg.mkPen('#000000', width=2),
                            name=f'Pixel ({x}, {y})'
                        )
        else:
            # Single view mode - use main plot widget
            self._original_spectrum = spectrum.copy()  # Store original for smoothing
            self.current_spectrum = spectrum

            # Apply smoothing if enabled
            display_spectrum = self._apply_smoothing(spectrum)

            # Remove existing pixel spectrum
            if 'current_pixel' in self.plot_items:
                self.plot_widget.removeItem(self.plot_items['current_pixel'])

            # Plot new spectrum with filtered wavelengths (apply smoothing to display_spectrum)
            if active_bands is not None:
                display_spectrum, plot_wavelengths = self._filter_spectrum_for_bad_bands(display_spectrum, plot_wavelengths, active_bands)

            plot_item = self.plot_widget.plot(
                plot_wavelengths, display_spectrum,
                pen=pg.mkPen('#000000', width=2),
                name=f'Pixel ({x}, {y})'
            )
            self.plot_items['current_pixel'] = plot_item

            # Update info (use original spectrum for stats)
            self.pixel_label.setText(f"Pixel: ({x}, {y})")
            self._update_stats_display(spectrum, f"Pixel ({x}, {y})")
    
    def update_pixel_spectrum_with_id(self, spectrum: np.ndarray, x: int, y: int, view_id: str,
                                       active_bands: List[int] = None, dataset_name: str = None):
        """
        Update pixel spectrum with source view ID for proper routing.
        
        Args:
            spectrum: Spectral values
            x: Pixel x coordinate
            y: Pixel y coordinate
            view_id: ID of the source image view
            active_bands: List of active (good) band indices to display
            dataset_name: Name of the dataset for bad bands filtering
        """
        if spectrum is None or self.wavelengths is None:
            return
        
        # Apply bad bands filtering if active_bands is provided
        if active_bands is not None:
            spectrum, plot_wavelengths = self._filter_spectrum_for_bad_bands(spectrum, self.wavelengths, active_bands)
        else:
            plot_wavelengths = self.wavelengths
        
        # Route to appropriate spectrum tab(s) based on view tracking configuration
        target_tabs = self._find_spectrum_tabs_tracking_view(view_id)
        
        print(f"[DEBUG] Pixel selected from view_id '{view_id}', found {len(target_tabs)} target spectrum tabs")
        
        if not target_tabs:
            # NO FALLBACK - only update spectrum tabs that are explicitly configured to track this view
            # This ensures spectrum windows only respond when their specific image view is clicked
            print(f"[DEBUG] No spectrum tabs configured to track view '{view_id}' - not updating any spectra")
            return
        
        # Update each target tab
        for plot_widget, info_panel, tab_name in target_tabs:
            # Apply smoothing if enabled
            display_spectrum = self._apply_smoothing(spectrum)

            # Remove existing pixel spectrum
            existing_items = [item for item in plot_widget.listDataItems()
                            if hasattr(item, 'name') and item.name() and 'Pixel' in item.name()]
            for item in existing_items:
                plot_widget.removeItem(item)

            # Apply bad bands filtering to display spectrum if needed
            if active_bands is not None:
                display_spectrum, plot_wavelengths_filtered = self._filter_spectrum_for_bad_bands(display_spectrum, plot_wavelengths, active_bands)
            else:
                plot_wavelengths_filtered = plot_wavelengths

            # Plot new spectrum with filtered wavelengths
            plot_widget.plot(
                plot_wavelengths_filtered, display_spectrum,
                pen=pg.mkPen('#000000', width=2),
                name=f'Pixel ({x}, {y}) from {view_id}'
            )
            
            # Update info panel if available
            if info_panel:
                # Find pixel label within this specific tab's info panel
                pixel_labels = info_panel.findChildren(QtWidgets.QLabel)
                for label in pixel_labels:
                    if hasattr(label, 'objectName') and label.objectName() == "pixel_label":
                        label.setText(f"Pixel: ({x}, {y}) from {view_id}")
                        break
        
        # Store current spectrum for reference
        self.current_spectrum = spectrum
    
    def _filter_spectrum_for_bad_bands(self, spectrum: np.ndarray, wavelengths: np.ndarray, active_bands: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Filter spectrum and wavelengths to exclude bad bands by inserting NaN values.
        
        This creates gaps in the plotted line where bad bands exist instead of 
        connecting good bands across bad band gaps.
        
        Args:
            spectrum: Original spectrum values
            wavelengths: Original wavelength values  
            active_bands: List of active (good) band indices
            
        Returns:
            Tuple of (filtered_spectrum, filtered_wavelengths) with NaN for bad bands
        """
        if not active_bands or len(active_bands) == len(spectrum):
            # No filtering needed
            return spectrum, wavelengths
        
        # Create boolean mask for active bands
        active_mask = np.zeros(len(spectrum), dtype=bool)
        for band_idx in active_bands:
            if 0 <= band_idx < len(spectrum):
                active_mask[band_idx] = True
        
        # Create filtered arrays with NaN for inactive bands
        filtered_spectrum = spectrum.copy()
        filtered_spectrum[~active_mask] = np.nan
        
        # Wavelengths remain the same (no gaps in x-axis)
        filtered_wavelengths = wavelengths
        
        return filtered_spectrum, filtered_wavelengths
        
    def add_roi_spectrum(self, roi_id: str, mean_spectrum: np.ndarray, 
                        std_spectrum: Optional[np.ndarray] = None,
                        min_spectrum: Optional[np.ndarray] = None,
                        max_spectrum: Optional[np.ndarray] = None,
                        roi_info: Optional[str] = None):
        """
        Add or update ROI spectrum with optional statistics.
        
        Args:
            roi_id: Unique identifier for ROI
            mean_spectrum: Mean spectrum values
            std_spectrum: Standard deviation (optional)
            min_spectrum: Minimum values (optional) 
            max_spectrum: Maximum values (optional)
            roi_info: ROI description
        """
        if mean_spectrum is None or self.wavelengths is None:
            return
            
        # Get color for this ROI
        color = self._get_roi_color(roi_id)
        
        # Remove existing ROI plots
        self._remove_roi_plots(roi_id)
        
        # Apply smoothing if enabled
        smoothed_mean = self._apply_smoothing(mean_spectrum)
        smoothed_std = self._apply_smoothing(std_spectrum) if std_spectrum is not None else None
        smoothed_min = self._apply_smoothing(min_spectrum) if min_spectrum is not None else None
        smoothed_max = self._apply_smoothing(max_spectrum) if max_spectrum is not None else None

        # Plot mean spectrum
        mean_plot = self.plot_widget.plot(
            self.wavelengths, smoothed_mean,
            pen=pg.mkPen(color, width=2),
            name=f'ROI {roi_id} Mean'
        )

        roi_plots = {'mean': mean_plot}

        # Plot standard deviation envelope if available
        if smoothed_std is not None:
            upper = smoothed_mean + smoothed_std
            lower = smoothed_mean - smoothed_std
            
            fill_color = pg.mkColor(color)
            fill_color.setAlphaF(0.3)
            
            fill_item = pg.FillBetweenItem(
                pg.PlotCurveItem(self.wavelengths, upper),
                pg.PlotCurveItem(self.wavelengths, lower),
                brush=pg.mkBrush(fill_color)
            )
            self.plot_widget.addItem(fill_item)
            roi_plots['std_fill'] = fill_item
            
        # Plot min/max envelope if available
        if smoothed_min is not None and smoothed_max is not None:
            min_plot = self.plot_widget.plot(
                self.wavelengths, smoothed_min,
                pen=pg.mkPen(color, width=1, style=QtCore.Qt.DashLine),
                name=f'ROI {roi_id} Min'
            )
            max_plot = self.plot_widget.plot(
                self.wavelengths, smoothed_max,
                pen=pg.mkPen(color, width=1, style=QtCore.Qt.DotLine),
                name=f'ROI {roi_id} Max'
            )
            roi_plots['min'] = min_plot
            roi_plots['max'] = max_plot
            
        self.plot_items[f'roi_{roi_id}'] = roi_plots
        
        # Store ROI spectrum data
        self.roi_spectra[roi_id] = {
            'mean': mean_spectrum,
            'std': std_spectrum,
            'min': min_spectrum,
            'max': max_spectrum,
            'info': roi_info or f'ROI {roi_id}'
        }
        
    def _get_roi_color(self, roi_id: str) -> str:
        """Get consistent color for ROI."""
        # Use hash of roi_id to get consistent color
        color_idx = hash(roi_id) % len(self.colors)
        return self.colors[color_idx]
        
    def _remove_roi_plots(self, roi_id: str):
        """Remove existing plots for ROI."""
        plot_key = f'roi_{roi_id}'
        if plot_key in self.plot_items:
            roi_plots = self.plot_items[plot_key]
            for plot_item in roi_plots.values():
                self.plot_widget.removeItem(plot_item)
            del self.plot_items[plot_key]
            
    def remove_roi_spectrum(self, roi_id: str):
        """Remove ROI spectrum from display."""
        self._remove_roi_plots(roi_id)
        if roi_id in self.roi_spectra:
            del self.roi_spectra[roi_id]
            
    def clear_all_spectra(self):
        """Clear all displayed spectra."""
        self.plot_widget.clear()
        self.plot_items.clear()
        self.roi_spectra.clear()
        self.current_spectrum = None

        # Clear smoothing-related data
        if hasattr(self, '_original_spectrum'):
            self._original_spectrum = None
        
        # Re-add crosshair
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Re-add RGB indicators if enabled
        if self.show_rgb_bands_checkbox.isChecked():
            self._update_rgb_indicators()
        
        # Clear info
        self.pixel_label.setText("Pixel: None")
        self.stats_label.setText("Stats: None")
        self.wavelength_label.setText("Wavelength: None")
        
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair and wavelength display."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Update crosshair
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
            
            # Find closest wavelength
            if self.wavelengths is not None:
                idx = np.argmin(np.abs(self.wavelengths - x))
                if 0 <= idx < len(self.wavelengths):
                    wl = self.wavelengths[idx]
                    
                    # Get spectrum value at this wavelength
                    spectrum_value = ""
                    if self.current_spectrum is not None and idx < len(self.current_spectrum):
                        spectrum_value = f", Value: {self.current_spectrum[idx]:.4f}"
                        
                    self.wavelength_label.setText(f"Wavelength: {wl:.1f} nm{spectrum_value}")
                    
    def _update_stats_display(self, spectrum: np.ndarray, label: str):
        """Update statistics display."""
        if spectrum is not None and len(spectrum) > 0:
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            min_val = np.min(spectrum)
            max_val = np.max(spectrum)
            
            self.stats_label.setText(
                f"{label} - Mean: {mean_val:.4f}, Std: {std_val:.4f}, "
                f"Min: {min_val:.4f}, Max: {max_val:.4f}"
            )
        else:
            self.stats_label.setText("Stats: None")
            
    def _toggle_y_auto(self, enabled: bool):
        """Toggle automatic y-axis scaling."""
        self.y_min_spinbox.setEnabled(not enabled)
        self.y_max_spinbox.setEnabled(not enabled)
        
        if enabled:
            self.plot_widget.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            self._apply_y_range()
            
    def _apply_y_range(self):
        """Apply manual y-axis range."""
        if not self.y_auto_checkbox.isChecked():
            y_min = self.y_min_spinbox.value()
            y_max = self.y_max_spinbox.value()
            self.plot_widget.setYRange(y_min, y_max)

    def _toggle_smoothing(self, enabled: bool):
        """Toggle spectral smoothing."""
        if not SCIPY_AVAILABLE:
            return

        self.smoothing_enabled = enabled
        if self.poly_order_combo:
            self.poly_order_combo.setEnabled(enabled)
        if self.window_length_combo:
            self.window_length_combo.setEnabled(enabled)

        # Refresh all displayed spectra with/without smoothing
        self._refresh_all_spectra()

    def _on_poly_order_changed(self, order_text: str):
        """Handle polynomial order change."""
        if not SCIPY_AVAILABLE:
            return

        self.smoothing_polynomial_order = int(order_text)
        if self.smoothing_enabled:
            self._refresh_all_spectra()

    def _on_window_length_changed(self, length_text: str):
        """Handle window length change."""
        if not SCIPY_AVAILABLE:
            return

        self.smoothing_window_length = int(length_text)
        if self.smoothing_enabled:
            self._refresh_all_spectra()

    def _apply_smoothing(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to spectrum if enabled."""
        if not self.smoothing_enabled or spectrum is None or not SCIPY_AVAILABLE:
            return spectrum

        if len(spectrum) < self.smoothing_window_length:
            # Window length too large for spectrum, return original
            return spectrum

        # Ensure window length is odd
        window_length = self.smoothing_window_length
        if window_length % 2 == 0:
            window_length += 1

        # Ensure polynomial order is less than window length
        poly_order = min(self.smoothing_polynomial_order, window_length - 1)

        try:
            smoothed = savgol_filter(spectrum, window_length, poly_order)
            return smoothed
        except (ValueError, np.linalg.LinAlgError):
            # If smoothing fails, return original spectrum
            return spectrum

    def _refresh_all_spectra(self):
        """Refresh all displayed spectra with current smoothing settings."""
        # Update current pixel spectrum
        if hasattr(self, 'current_spectrum') and self.current_spectrum is not None:
            # Store original spectrum without smoothing if not already stored
            if not hasattr(self, '_original_spectrum'):
                self._original_spectrum = self.current_spectrum.copy()

            # Apply current smoothing settings and update display
            spectrum_to_display = self._apply_smoothing(self._original_spectrum)

            # Remove existing pixel spectrum
            if 'current_pixel' in self.plot_items:
                self.plot_widget.removeItem(self.plot_items['current_pixel'])

            # Plot updated spectrum
            if self.wavelengths is not None:
                plot_item = self.plot_widget.plot(
                    self.wavelengths, spectrum_to_display,
                    pen=pg.mkPen('#000000', width=2),
                    name='Current Pixel'
                )
                self.plot_items['current_pixel'] = plot_item

        # Update all ROI spectra
        roi_ids_to_refresh = list(self.roi_spectra.keys())
        for roi_id in roi_ids_to_refresh:
            roi_data = self.roi_spectra[roi_id]
            self.add_roi_spectrum(
                roi_id,
                roi_data.get('mean'),
                roi_data.get('std'),
                roi_data.get('min'),
                roi_data.get('max'),
                roi_data.get('info')
            )
            
    def _export_spectra(self):
        """Export current spectra to file."""
        if not self.current_spectrum and not self.roi_spectra:
            QtWidgets.QMessageBox.information(self, "Export", "No spectra to export")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Spectra", "", "CSV Files (*.csv);;Text Files (*.txt)"
        )
        
        if filename:
            self._write_spectra_file(filename)
            
    def _write_spectra_file(self, filename: str):
        """Write spectra data to file."""
        try:
            with open(filename, 'w') as f:
                # Write header
                header = ["Wavelength"]
                data_arrays = [self.wavelengths]
                
                if self.current_spectrum is not None:
                    header.append("Current_Pixel")
                    data_arrays.append(self.current_spectrum)
                    
                for roi_id, roi_data in self.roi_spectra.items():
                    if roi_data['mean'] is not None:
                        header.append(f"ROI_{roi_id}_Mean")
                        data_arrays.append(roi_data['mean'])
                        
                    if roi_data['std'] is not None:
                        header.append(f"ROI_{roi_id}_Std")
                        data_arrays.append(roi_data['std'])
                        
                    if roi_data['min'] is not None:
                        header.append(f"ROI_{roi_id}_Min")
                        data_arrays.append(roi_data['min'])
                        
                    if roi_data['max'] is not None:
                        header.append(f"ROI_{roi_id}_Max")
                        data_arrays.append(roi_data['max'])
                        
                f.write(','.join(header) + '\n')
                
                # Write data
                for i in range(len(self.wavelengths)):
                    row = [str(arr[i] if i < len(arr) else '') for arr in data_arrays]
                    f.write(','.join(row) + '\n')
                    
            QtWidgets.QMessageBox.information(self, "Export", f"Spectra exported to {filename}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export spectra: {e}")
            
    def get_legend(self) -> QtWidgets.QWidget:
        """Get legend widget for external display."""
        legend_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Add current pixel
        if self.current_spectrum is not None:
            pixel_label = QtWidgets.QLabel("● Current Pixel")
            pixel_label.setStyleSheet("color: black; font-weight: bold;")
            layout.addWidget(pixel_label)
            
        # Add ROI entries
        for roi_id, roi_data in self.roi_spectra.items():
            color = self._get_roi_color(roi_id)
            roi_label = QtWidgets.QLabel(f"● {roi_data['info']}")
            roi_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            layout.addWidget(roi_label)
            
        legend_widget.setLayout(layout)
        return legend_widget
    
    def update_rgb_bands(self, red_band: int, green_band: int, blue_band: int):
        """
        Update RGB band indicators on the plot.
        
        Args:
            red_band: Red band index
            green_band: Green band index
            blue_band: Blue band index
        """
        self.current_rgb_bands = (red_band, green_band, blue_band)
        # Remove current band indicator when showing RGB indicators
        self._remove_current_band_indicator()
        self._update_rgb_indicators()
        
    def update_current_band(self, band_index: int, channel: str = None):
        """
        Update current band indicator on the plot.
        
        Args:
            band_index: Current band index
            channel: Optional channel identifier ('R', 'G', 'B', 'mono')
        """
        self.current_band = band_index
        
        # Clear indicators based on mode change
        if channel == 'mono':
            # Switching to mono mode - remove RGB indicators, show current band
            self._remove_rgb_indicators()
            self._update_current_band_indicator(channel)
        elif channel in ['R', 'G', 'B']:
            # In RGB mode - remove current band indicator, RGB will be handled by update_rgb_bands
            self._remove_current_band_indicator()
        else:
            # Default behavior
            self._update_current_band_indicator(channel)
        
    def _update_current_band_indicator(self, channel: str = None):
        """Update current band indicator line on the plot."""
        if self.wavelengths is None:
            return
            
        # Remove existing indicator
        if self.current_band_line:
            self.plot_widget.removeItem(self.current_band_line)
            self.current_band_line = None
            
        # Only show current band indicator in mono mode
        # In RGB mode, the RGB indicators will handle band display
        if channel == 'mono' and 0 <= self.current_band < len(self.wavelengths):
            wavelength = self.wavelengths[self.current_band]
            color = pg.mkPen('orange', width=3, style=QtCore.Qt.DashLine)
                
            self.current_band_line = pg.InfiniteLine(
                pos=wavelength,
                angle=90,
                pen=color,
                movable=False
            )
            self.plot_widget.addItem(self.current_band_line)
        
    def _update_rgb_indicators(self):
        """Update RGB band indicator lines on the plot."""
        if self.wavelengths is None or not self.show_rgb_bands_checkbox.isChecked():
            self._remove_rgb_indicators()
            return
            
        # Remove existing indicators
        self._remove_rgb_indicators()
        
        red_idx, green_idx, blue_idx = self.current_rgb_bands
        
        # Create RGB band indicator lines
        if 0 <= red_idx < len(self.wavelengths):
            red_wl = self.wavelengths[red_idx]
            self.rgb_band_lines['red'] = pg.InfiniteLine(
                pos=red_wl, angle=90, 
                pen=pg.mkPen('#ff0000', width=3, style=QtCore.Qt.DashLine),
                label=f'R:{red_idx}',
                labelOpts={'position': 0.95, 'color': '#ff0000', 'fill': '#ffcccc', 'movable': True}
            )
            self.plot_widget.addItem(self.rgb_band_lines['red'], ignoreBounds=True)
            
        if 0 <= green_idx < len(self.wavelengths):
            green_wl = self.wavelengths[green_idx]
            self.rgb_band_lines['green'] = pg.InfiniteLine(
                pos=green_wl, angle=90,
                pen=pg.mkPen('#00aa00', width=3, style=QtCore.Qt.DashLine),
                label=f'G:{green_idx}',
                labelOpts={'position': 0.85, 'color': '#00aa00', 'fill': '#ccffcc', 'movable': True}
            )
            self.plot_widget.addItem(self.rgb_band_lines['green'], ignoreBounds=True)
            
        if 0 <= blue_idx < len(self.wavelengths):
            blue_wl = self.wavelengths[blue_idx]
            self.rgb_band_lines['blue'] = pg.InfiniteLine(
                pos=blue_wl, angle=90,
                pen=pg.mkPen('#0000ff', width=3, style=QtCore.Qt.DashLine),
                label=f'B:{blue_idx}',
                labelOpts={'position': 0.75, 'color': '#0000ff', 'fill': '#ccccff', 'movable': True}
            )
            self.plot_widget.addItem(self.rgb_band_lines['blue'], ignoreBounds=True)
            
    def _remove_rgb_indicators(self):
        """Remove RGB band indicator lines from the plot."""
        for color, line in self.rgb_band_lines.items():
            if line is not None:
                self.plot_widget.removeItem(line)
                self.rgb_band_lines[color] = None
    
    def _remove_current_band_indicator(self):
        """Remove current band indicator line from the plot."""
        if self.current_band_line:
            self.plot_widget.removeItem(self.current_band_line)
            self.current_band_line = None
            
    def clear_all_band_indicators(self):
        """Clear all band indicators (RGB and current band)."""
        self._remove_rgb_indicators()
        self._remove_current_band_indicator()
                
    def _toggle_rgb_bands(self, show: bool):
        """Toggle RGB band indicator visibility."""
        if show:
            self._update_rgb_indicators()
        else:
            self._remove_rgb_indicators()
            
    def add_header_tab(self, header_content: str, tab_name: str = "Header"):
        """Add a tab displaying header file contents."""
        # Check if header tab already exists and remove it
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_name:
                self.tab_widget.removeTab(i)
                break
        
        # Create header viewer widget
        header_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        
        # Create text display
        text_display = QtWidgets.QTextEdit()
        text_display.setReadOnly(True)
        text_display.setPlainText(header_content)
        text_display.setFont(QtGui.QFont("Courier", 10))  # Monospace font for header data
        
        # Set background and text colors for better readability
        text_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                color: #333;
                border: 1px solid #ddd;
            }
        """)
        
        layout.addWidget(text_display)
        header_widget.setLayout(layout)
        
        # Add the tab
        tab_index = self.tab_widget.addTab(header_widget, tab_name)
        
        # Switch to the header tab
        self.tab_widget.setCurrentIndex(tab_index)
        
        return tab_index
    
    def _toggle_split_view(self, enabled: bool):
        """Toggle split view mode for spectrum display."""
        if enabled:
            self._enable_split_view()
        else:
            self._disable_split_view()
            
    def _enable_split_view(self):
        """Enable split view mode."""
        if self.split_mode:
            return
            
        self.split_mode = True
        
        # Remove single tab widget from layout
        self.tab_layout.removeWidget(self.tab_widget)
        self.tab_widget.hide()  # Hide but don't destroy
        
        # Create vertical splitter
        self.split_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split_splitter.setHandleWidth(5)
        self.split_splitter.setChildrenCollapsible(False)
        
        # Create container widgets for each split view
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        bottom_container = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create toolbars for each view
        self.top_toolbar = self._create_split_toolbar("Top")
        top_layout.addWidget(self.top_toolbar)
        
        self.bottom_toolbar = self._create_split_toolbar("Bottom")
        bottom_layout.addWidget(self.bottom_toolbar)
        
        # Move existing tab widget to top view
        self.split_view_1 = self.tab_widget
        self.split_view_1.show()
        top_layout.addWidget(self.split_view_1)
        
        # Create new draggable tab widget for bottom view
        self.split_view_2 = DraggableTabWidget()
        self.split_view_2.setTabsClosable(True)
        self.split_view_2.tabCloseRequested.connect(self._close_roi_tab)
        bottom_layout.addWidget(self.split_view_2)
        
        # Connect tabs_changed signals to check for auto-exit
        self.split_view_1.tabs_changed.connect(self._check_auto_exit_split)
        self.split_view_2.tabs_changed.connect(self._check_auto_exit_split)
        
        # Add containers to splitter
        self.split_splitter.addWidget(top_container)
        self.split_splitter.addWidget(bottom_container)
        
        # Set equal sizes
        self.split_splitter.setSizes([300, 300])
        
        # Add splitter to layout
        self.tab_layout.addWidget(self.split_splitter)
        
        # Create a default spectrum tab in the bottom view
        bottom_tab = self._create_spectrum_tab("Bottom")
        self.split_view_2.addTab(bottom_tab['widget'], "Pixel Spectra")
        
        # Update image source combo visibility
        if hasattr(self, 'image_source_combo'):
            self.image_source_combo.setVisible(True)
        
    def _disable_split_view(self):
        """Disable split view mode."""
        if not self.split_mode:
            return
            
        self.split_mode = False
        
        # Remove splitter from layout
        if self.split_splitter:
            # Move all tabs from bottom view to top view (if any remain)
            while self.split_view_2 and self.split_view_2.count() > 0:
                # Get the tab widget and text
                tab_widget = self.split_view_2.widget(0)
                tab_text = self.split_view_2.tabText(0)
                
                # Remove from bottom view
                self.split_view_2.removeTab(0)
                
                # Add to top view
                self.split_view_1.addTab(tab_widget, tab_text)
            
            # Disconnect signals
            if self.split_view_1:
                try:
                    self.split_view_1.tabs_changed.disconnect(self._check_auto_exit_split)
                except:
                    pass
            if self.split_view_2:
                try:
                    self.split_view_2.tabs_changed.disconnect(self._check_auto_exit_split)
                except:
                    pass
            
            # Remove and clean up splitter
            self.tab_layout.removeWidget(self.split_splitter)
            self.split_splitter.setParent(None)
            self.split_splitter.deleteLater()
            self.split_splitter = None
            
            # Clean up toolbars
            if hasattr(self, 'top_toolbar'):
                self.top_toolbar.setParent(None)
                self.top_toolbar.deleteLater()
                self.top_toolbar = None
            if hasattr(self, 'bottom_toolbar'):
                self.bottom_toolbar.setParent(None)
                self.bottom_toolbar.deleteLater()
                self.bottom_toolbar = None
            
            self.split_view_2 = None
            self.split_view_1 = None
        
        # Restore single tab widget to layout
        self.tab_widget.show()
        self.tab_layout.addWidget(self.tab_widget)
        
        # Update split action
        self.split_action.setChecked(False)
        
        # Update image source combo visibility
        if hasattr(self, 'image_source_combo'):
            # Keep visible but reset to "Active View" for single mode
            self.image_source_combo.setCurrentText("Active View")
    
    def _create_split_toolbar(self, view_name: str) -> QtWidgets.QToolBar:
        """Create a minimal toolbar for split view panes."""
        toolbar = QtWidgets.QToolBar()
        toolbar.setMaximumHeight(30)
        
        # View label
        label = QtWidgets.QLabel(f"{view_name} View - Track Image:")
        toolbar.addWidget(label)
        
        # Image source selector
        image_combo = QtWidgets.QComboBox()
        image_combo.setToolTip(f"Select which image view to track for {view_name} spectra")
        image_combo.addItems(["Active View", "View 1", "View 2"])
        image_combo.setProperty("view_name", view_name)
        image_combo.currentTextChanged.connect(lambda text: self._on_split_image_source_changed(view_name, text))
        toolbar.addWidget(image_combo)
        
        # Store reference
        if view_name == "Top":
            self.top_image_combo = image_combo
        else:
            self.bottom_image_combo = image_combo
        
        toolbar.addSeparator()
        
        # Clear button
        clear_action = toolbar.addAction("Clear")
        clear_action.triggered.connect(lambda: self._clear_split_view_spectra(view_name))
        
        return toolbar
    
    def _check_auto_exit_split(self):
        """Check if we should automatically exit split view when one pane is empty."""
        if not self.split_mode:
            return
            
        # Check if one of the views is empty
        if self.split_view_1 and self.split_view_2:
            view1_empty = self.split_view_1.count() == 0
            view2_empty = self.split_view_2.count() == 0
            
            # If one view is empty, exit split mode
            if view1_empty or view2_empty:
                # Auto-exit split view - one pane is empty
                self._disable_split_view()
    
    def _on_image_source_changed(self, source: str):
        """Handle image source selection change for main view."""
        self.tracked_image_view = source
        # Main spectrum view tracking updated
    
    def _on_split_image_source_changed(self, view_name: str, source: str):
        """Handle image source selection change for split views."""
        # Split view tracking updated
        # Store the tracking preference for each split view
        if not hasattr(self, 'split_view_tracking'):
            self.split_view_tracking = {}
        self.split_view_tracking[view_name] = source
    
    def _clear_split_view_spectra(self, view_name: str):
        """Clear all spectra in a specific split view."""
        if view_name == "Top" and self.split_view_1:
            # Clear spectra in top view tabs
            for i in range(self.split_view_1.count()):
                widget = self.split_view_1.widget(i)
                # Find plot widget in the tab and clear it
                plot_widgets = widget.findChildren(pg.PlotWidget)
                for plot_widget in plot_widgets:
                    plot_widget.clear()
        elif view_name == "Bottom" and self.split_view_2:
            # Clear spectra in bottom view tabs
            for i in range(self.split_view_2.count()):
                widget = self.split_view_2.widget(i)
                # Find plot widget in the tab and clear it
                plot_widgets = widget.findChildren(pg.PlotWidget)
                for plot_widget in plot_widgets:
                    plot_widget.clear()
    
    def _find_spectrum_tabs_tracking_view(self, view_id: str) -> List[tuple]:
        """Find spectrum tabs that are configured to track a specific view ID."""
        target_tabs = []
        
        # Check if we have tab tracking configuration
        if not hasattr(self, 'tab_view_tracking'):
            self.tab_view_tracking = {}
        
        print(f"[DEBUG] Looking for spectrum tabs tracking view '{view_id}'")
        print(f"[DEBUG] Current tab_view_tracking: {self.tab_view_tracking}")
        
        # Check all tabs in single mode and split mode
        tab_widgets_to_check = []
        
        if self.split_mode:
            if self.split_view_1:
                tab_widgets_to_check.append(("split_top", self.split_view_1))
            if self.split_view_2:
                tab_widgets_to_check.append(("split_bottom", self.split_view_2))
        else:
            tab_widgets_to_check.append(("main", self.tab_widget))
        
        for container_name, tab_widget in tab_widgets_to_check:
            for i in range(tab_widget.count()):
                tab_name = tab_widget.tabText(i)
                tab_key = f"{container_name}_{tab_name}"
                
                # Check if this tab is configured to track the specified view
                if tab_key in self.tab_view_tracking and self.tab_view_tracking[tab_key] == view_id:
                    widget = tab_widget.widget(i)
                    plot_widgets = widget.findChildren(pg.PlotWidget)
                    info_panels = widget.findChildren(QtWidgets.QWidget)
                    
                    if plot_widgets:
                        plot_widget = plot_widgets[0]
                        info_panel = info_panels[0] if info_panels else None
                        target_tabs.append((plot_widget, info_panel, tab_name))
        
        return target_tabs
    
    def _on_tab_view_selector_changed(self, selector):
        """Handle view selector change for a spectrum tab."""
        if not hasattr(self, 'tab_view_tracking'):
            self.tab_view_tracking = {}
        
        tab_name = selector.property("tab_name")
        view_id = selector.currentData()
        
        # Find which tab container this selector belongs to
        tab_key = self._find_tab_key_for_selector(selector, tab_name)
        
        if not tab_key:
            print(f"Warning: Could not determine tab container for {tab_name}")
            return
        
        # Update tracking configuration
        if view_id:
            self.tab_view_tracking[tab_key] = view_id
            print(f"Spectrum tab '{tab_key}' now tracking image view '{view_id}'")
        else:
            # Remove tracking if "None" selected
            if tab_key in self.tab_view_tracking:
                del self.tab_view_tracking[tab_key]
                print(f"Spectrum tab '{tab_key}' stopped tracking")
    
    def _find_tab_key_for_selector(self, selector, tab_name: str) -> str:
        """Find the correct tab key for a selector by checking which tab container it belongs to."""
        # Check all tab containers to find where this selector belongs
        tab_containers = []
        
        if self.split_mode:
            if self.split_view_1:
                tab_containers.append(("split_top", self.split_view_1))
            if self.split_view_2:
                tab_containers.append(("split_bottom", self.split_view_2))
        else:
            tab_containers.append(("main", self.tab_widget))
        
        # Search for the selector in each container
        for container_name, tab_widget in tab_containers:
            for i in range(tab_widget.count()):
                if tab_widget.tabText(i) == tab_name:
                    # Found the tab, check if this selector is in this tab's widget
                    widget = tab_widget.widget(i)
                    selectors = widget.findChildren(QtWidgets.QComboBox)
                    if selector in selectors:
                        return f"{container_name}_{tab_name}"
        
        # Fallback
        return f"main_{tab_name}"
    
    def update_view_selectors(self, available_views: List[tuple]):
        """Update all view selector dropdowns with available image views."""
        if not hasattr(self, 'tab_view_selectors'):
            return
            
        for tab_name, selector in self.tab_view_selectors.items():
            current_selection = selector.currentData()
            
            # Clear and repopulate
            selector.clear()
            selector.addItem("None", "")
            
            for view_id, view_label in available_views:
                selector.addItem(view_label, view_id)
            
            # Restore previous selection if still available
            if current_selection:
                index = selector.findData(current_selection)
                if index >= 0:
                    selector.setCurrentIndex(index)