"""
TabbedImageView class for managing multiple image views in tabs.

Extends the single ImageView to support multiple datasets simultaneously.
"""

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from typing import Optional, Dict, List
import weakref

from image_view import ImageView
from data_manager import DataManager


class TabbedImageView(QtWidgets.QWidget):
    """Tabbed container for multiple ImageView instances."""
    
    # Signals - forward from active tab
    pixel_selected = QtCore.pyqtSignal(int, int)  # Legacy signal
    pixel_selected_with_id = QtCore.pyqtSignal(int, int, str)  # x, y, view_id
    cursor_moved = QtCore.pyqtSignal(int, int)
    roi_selected = QtCore.pyqtSignal(object)
    zoom_changed = QtCore.pyqtSignal(object)
    rgb_bands_changed = QtCore.pyqtSignal()
    default_rgb_requested = QtCore.pyqtSignal()
    line_changed = QtCore.pyqtSignal(int)
    band_changed = QtCore.pyqtSignal(int, str)  # Forward from active tab
    spectrum_collect_requested = QtCore.pyqtSignal(int, int)  # x, y for spectrum collection
    dataset_changed = QtCore.pyqtSignal(str)  # Emitted when active dataset changes
    
    def __init__(self, parent=None, view_id: str = None):
        super().__init__(parent)
        
        self.data_manager = DataManager()
        self.image_views: Dict[str, ImageView] = {}  # dataset_name -> ImageView
        
        # Store each tab's individual state to prevent reset when switching
        self.tab_states: Dict[str, dict] = {}  # tab_key -> state dict
        
        # Unique identifier for this tabbed view instance
        self.view_id = view_id if view_id else f"view_{id(self)}"
        
        # Track tab view IDs for spectrum linking
        self.tab_view_ids: Dict[int, str] = {}  # tab_index -> view_id
        self._next_view_counter = 1
        
        # No data values per dataset
        self.no_data_values: Dict[str, float] = {}  # dataset_name -> no_data_value
        
        # Bad bands per dataset (True = active/good, False = inactive/bad)
        self.bad_bands: Dict[str, Dict[int, bool]] = {}  # dataset_name -> {band_index: active_status}
        
        self._setup_ui()
        self._setup_connections()
        
        # Track current tab for state saving
        self._current_tab_key = None
        
    def _setup_ui(self):
        """Initialize the tabbed interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Create tab widget FIRST (before toolbar which may access it)
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.North)
        
        # Add placeholder tab when no data is loaded
        self._add_placeholder_tab()
        
        # Create toolbar for this view (after tab_widget exists)
        self.toolbar = self._create_toolbar()
        layout.addWidget(self.toolbar)
        
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
    def _setup_connections(self):
        """Setup signal connections."""
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        
    def _add_placeholder_tab(self):
        """Add a placeholder tab when no datasets are loaded."""
        placeholder = QtWidgets.QLabel("No data loaded.\nUse File → Open to load hyperspectral data.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14pt;
                padding: 50px;
            }
        """)
        placeholder_index = self.tab_widget.addTab(placeholder, "No Data")
        
        # Make placeholder tab non-closable by removing close button
        self.tab_widget.tabBar().setTabButton(placeholder_index, QtWidgets.QTabBar.RightSide, None)
        self.tab_widget.tabBar().setTabButton(placeholder_index, QtWidgets.QTabBar.LeftSide, None)
        
    def _remove_placeholder_tab(self):
        """Remove placeholder tab when real data is added."""
        if self.tab_widget.count() > 0:
            widget = self.tab_widget.widget(0)
            if isinstance(widget, QtWidgets.QLabel):
                self.tab_widget.removeTab(0)
                
    def add_dataset_tab(self, dataset_name: str, display_name: Optional[str] = None) -> bool:
        """
        Add a new tab for a dataset.
        
        Args:
            dataset_name: Name of dataset in DataManager
            display_name: Display name for tab (defaults to dataset_name)
            
        Returns:
            True if tab was added successfully
        """
        # Get dataset from manager
        dataset = self.data_manager.get_dataset(dataset_name)
        if not dataset:
            return False
            
        # Check if this exact dataset already has a tab
        for existing_name in self.image_views.keys():
            # Check base name without counter suffix
            base_name = existing_name.split(' (')[0]
            if base_name == dataset_name:
                # Just activate the existing tab
                self.set_active_tab(existing_name)
                return True
            
        # If we get here, create a new tab
        tab_key = dataset_name
        counter = 1
        
        # Find a unique tab key if multiple tabs are needed
        while tab_key in self.image_views:
            tab_key = f"{dataset_name} ({counter})"
            counter += 1
            
        # Remove placeholder if this is the first real tab
        self._remove_placeholder_tab()
            
        # Create new ImageView
        image_view = ImageView()
        
        # Set the data
        if hasattr(dataset, 'data') and dataset.data is not None:
            # Check if this is a single-band dataset
            is_single_band = hasattr(dataset, 'shape') and len(dataset.shape) >= 3 and dataset.shape[2] == 1
            
            if is_single_band:
                # For single-band datasets (like SAM results), enable monochromatic mode
                try:
                    # Get no data value for this dataset
                    no_data_value = self.no_data_values.get(dataset_name, None)
                    
                    # Use single band for all RGB channels (grayscale)
                    rgb_data = dataset.get_rgb_composite(0, 0, 0, 2.0, no_data_value)
                    if rgb_data is not None:
                        image_view.set_image(rgb_data)
                        # Automatically enable monochromatic mode for single-band datasets
                        image_view._toggle_mono_mode(True)
                        image_view.mono_action.setChecked(True)
                except Exception as e:
                    print(f"Error setting single-band image data for {dataset_name}: {e}")
            else:
                # Multi-band datasets: Display as RGB composite using default bands
                try:
                    # Get no data value for this dataset
                    no_data_value = self.no_data_values.get(dataset_name, None)
                    
                    rgb_data = dataset.get_rgb_composite(29, 19, 9, 2.0, no_data_value)  # Default bands
                    if rgb_data is not None:
                        image_view.set_image(rgb_data)
                    else:
                        # Fallback to first three bands if available
                        if dataset.shape and len(dataset.shape) == 3 and dataset.shape[2] >= 3:
                            rgb_data = dataset.get_rgb_composite(2, 1, 0, 2.0, no_data_value)
                            if rgb_data is not None:
                                image_view.set_image(rgb_data)
                except Exception as e:
                    print(f"Error setting image data for {dataset_name}: {e}")
                    
        # Set band limits for this new image view
        if hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
            band_count = dataset.shape[2]
            image_view.set_band_limits(band_count)
            
            # For single-band datasets, ensure at least minimum ranges for controls
            if band_count == 1:
                # Set current band to 0 (the only band)
                image_view.current_band = 0
        
        # Set the dataset reference so the image view can access raw data
        image_view.current_dataset = dataset
        image_view.current_dataset_name = dataset_name
        
        # Load bad band list from ENVI header if available
        self._load_bad_bands_from_header(dataset_name, dataset)
        
        # Connect signals to forward from this image view
        self._connect_image_view_signals(image_view)
        
        # Store reference with the unique tab key
        self.image_views[tab_key] = image_view
        
        # Add tab with appropriate display name
        if tab_key != dataset_name:
            tab_name = display_name or tab_key  # Show the counter in tab name
        else:
            tab_name = display_name or dataset_name
        tab_index = self.tab_widget.addTab(image_view, tab_name)
        
        # Assign a unique view ID to this tab
        view_id = self._get_or_create_tab_view_id(tab_index)
        image_view.view_id = view_id  # Store ID in the image view itself
        
        # Save the current tab before potentially switching
        current_index = self.tab_widget.currentIndex()
        
        # Always switch to the new tab when loading a file
        # This ensures the correct tab is updated with the new data
        self.tab_widget.setCurrentIndex(tab_index)
        self._current_tab_key = tab_key
        
        # Don't save initial state - let it be saved when user actually modifies something
        # This prevents saving uninitialized values
        # self._save_tab_state(tab_key)
        
        # Update data manager active dataset to match the new tab
        self.data_manager.set_active_dataset(dataset_name)
        
        return True
        
    def _connect_image_view_signals(self, image_view: ImageView):
        """Connect signals from ImageView to forward them."""
        image_view.pixel_selected.connect(self._on_pixel_selected)
        image_view.cursor_moved.connect(self._on_cursor_moved)
        image_view.roi_selected.connect(self._on_roi_selected)
        image_view.zoom_changed.connect(self._on_zoom_changed)
        image_view.rgb_bands_changed.connect(self._on_rgb_bands_changed)
        image_view.default_rgb_requested.connect(self._on_default_rgb_requested)
        image_view.line_changed.connect(self._on_line_changed)
        image_view.band_changed.connect(self._on_band_changed)

        # Connect spectrum collection signal if it exists
        if hasattr(image_view, 'spectrum_collect_requested'):
            image_view.spectrum_collect_requested.connect(self._on_spectrum_collect_requested)
        
    def _on_pixel_selected(self, x: int, y: int):
        """Forward pixel selection from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            # Get the view ID for the current tab
            current_index = self.tab_widget.currentIndex()
            if current_index in self.tab_view_ids:
                view_id = self.tab_view_ids[current_index]
            else:
                # Generate view ID if not exists
                view_id = self._get_or_create_tab_view_id(current_index)
            
            # Emit both legacy and new signals
            self.pixel_selected.emit(x, y)
            self.pixel_selected_with_id.emit(x, y, view_id)
            
    def _on_cursor_moved(self, x: int, y: int):
        """Forward cursor movement from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.cursor_moved.emit(x, y)
            
    def _on_roi_selected(self, roi_data):
        """Forward ROI selection from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.roi_selected.emit(roi_data)
            
    def _on_zoom_changed(self, view_range):
        """Forward zoom changes from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.zoom_changed.emit(view_range)
            
    def _on_rgb_bands_changed(self):
        """Forward RGB band changes from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.rgb_bands_changed.emit()
            
    def _on_default_rgb_requested(self):
        """Forward default RGB request from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.default_rgb_requested.emit()
            
    def _on_line_changed(self, line_index: int):
        """Forward line changes from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.line_changed.emit(line_index)
            
    def _on_band_changed(self, band_index: int, channel: str):
        """Forward band changes from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            self.band_changed.emit(band_index, channel)

    def _on_spectrum_collect_requested(self, x: int, y: int):
        """Forward spectrum collection request from active tab."""
        sender = self.sender()
        if sender == self.get_current_image_view():
            print(f"[TABBED_VIEW] Forwarding spectrum_collect_requested({x}, {y})")
            self.spectrum_collect_requested.emit(x, y)

    def _on_tab_changed(self, index: int):
        """Handle tab changes."""
        if index >= 0:
            # Save current tab state before switching
            if self._current_tab_key is not None:
                self._save_tab_state(self._current_tab_key)
                
            # Get the dataset name for this tab
            widget = self.tab_widget.widget(index)
            if isinstance(widget, ImageView):
                # Find dataset name for this image view
                for tab_key, image_view in self.image_views.items():
                    if image_view == widget:
                        # Extract base dataset name (remove counter if present)
                        if ' (' in tab_key and ')' in tab_key:
                            base_dataset_name = tab_key.split(' (')[0]
                        else:
                            base_dataset_name = tab_key
                        
                        # Update current tab tracking
                        self._current_tab_key = tab_key
                        
                        # Restore this tab's saved state before setting as active
                        self._restore_tab_state(tab_key)
                        
                        # Update dataset combo to reflect current tab
                        if hasattr(self, 'dataset_combo'):
                            self._combo_updating = True
                            combo_index = self.dataset_combo.findText(base_dataset_name)
                            if combo_index >= 0:
                                self.dataset_combo.setCurrentIndex(combo_index)
                            self._combo_updating = False
                        
                        self.data_manager.set_active_dataset(base_dataset_name)
                        
                        # Emit signal that dataset has changed
                        self.dataset_changed.emit(base_dataset_name)
                        break
                        
    def _on_tab_close_requested(self, index: int):
        """Handle tab close requests."""
        if index >= 0:
            widget = self.tab_widget.widget(index)
            
            # Don't allow closing placeholder tabs
            if isinstance(widget, QtWidgets.QLabel):
                return
                
            if isinstance(widget, ImageView):
                # Find tab key and dataset name for this widget
                tab_key = None
                for key, image_view in self.image_views.items():
                    if image_view == widget:
                        tab_key = key
                        break
                        
                if tab_key:
                    # Extract base dataset name (remove counter if present)
                    if ' (' in tab_key and ')' in tab_key:
                        base_dataset_name = tab_key.split(' (')[0]
                    else:
                        base_dataset_name = tab_key
                    
                    # Ask for confirmation
                    reply = QtWidgets.QMessageBox.question(
                        self, "Close Tab",
                        f"Close tab for {tab_key}?\nDataset '{base_dataset_name}' will remain available in File Manager.",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No
                    )
                    
                    if reply == QtWidgets.QMessageBox.Yes:
                        # Clean up saved state for this tab
                        if tab_key in self.tab_states:
                            del self.tab_states[tab_key]
                        
                        # Remove tab but keep dataset in manager for reopening
                        del self.image_views[tab_key]
                        self.tab_widget.removeTab(index)
                        
                        # Update current tab tracking if this was the current tab
                        if self._current_tab_key == tab_key:
                            self._current_tab_key = None
                        
                        # Add placeholder if no tabs left
                        if self.tab_widget.count() == 0:
                            self._add_placeholder_tab()
                            
    def get_current_image_view(self) -> Optional[ImageView]:
        """Get the currently active ImageView."""
        current_widget = self.tab_widget.currentWidget()
        if isinstance(current_widget, ImageView):
            return current_widget
        return None
        
    def get_current_dataset_name(self) -> Optional[str]:
        """Get the name of the currently active dataset."""
        current_view = self.get_current_image_view()
        if current_view:
            for dataset_name, image_view in self.image_views.items():
                if image_view == current_view:
                    return dataset_name
        return None
        
    def set_active_tab(self, dataset_name: str) -> bool:
        """Set the active tab by dataset name."""
        if dataset_name in self.image_views:
            image_view = self.image_views[dataset_name]
            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == image_view:
                    self.tab_widget.setCurrentIndex(i)
                    return True
        return False
        
    def update_tab_name(self, dataset_name: str, new_name: str):
        """Update the display name of a tab."""
        if dataset_name in self.image_views:
            image_view = self.image_views[dataset_name]
            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == image_view:
                    self.tab_widget.setTabText(i, new_name)
                    break
                    
    # Forward common methods to current image view for backward compatibility
    def set_image(self, image_data: np.ndarray):
        """Set image data on the current tab."""
        current_view = self.get_current_image_view()
        if current_view:
            current_view.set_image(image_data)
            
    def get_rgb_bands(self):
        """Get RGB bands from current tab."""
        current_view = self.get_current_image_view()
        if current_view:
            return current_view.get_rgb_bands()
        return (0, 0, 0)
        
    def set_rgb_bands(self, r: int, g: int, b: int):
        """Set RGB bands on current tab."""
        current_view = self.get_current_image_view()
        if current_view:
            current_view.set_rgb_bands(r, g, b)
            
    def get_stretch_percent(self):
        """Get stretch percentage from current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'get_stretch_percent'):
            return current_view.get_stretch_percent()
        return 2.0
        
    def set_stretch_percent(self, percent: float):
        """Set stretch percentage on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'set_stretch_percent'):
            current_view.set_stretch_percent(percent)
            
    def get_histogram_levels(self):
        """Get histogram levels from current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'get_histogram_levels'):
            return current_view.get_histogram_levels()
        return None
        
    def set_histogram_levels(self, levels):
        """Set histogram levels on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'set_histogram_levels'):
            current_view.set_histogram_levels(levels)
            
    def get_view_range(self):
        """Get view range from current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'get_view_range'):
            return current_view.get_view_range()
        return None
        
    def set_view_range(self, view_range):
        """Set view range on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'set_view_range'):
            current_view.set_view_range(view_range)
        
    def set_band_limits(self, num_bands: int):
        """Set band limits on current tab only."""
        # Only set limits on the current tab (different datasets may have different band counts)
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'set_band_limits'):
            current_view.set_band_limits(num_bands)
            
    def set_line_limits_for_frame_view(self, num_lines: int):
        """Set line limits for frame view on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, 'set_line_limits_for_frame_view'):
            current_view.set_line_limits_for_frame_view(num_lines)
            
    def _zoom_fit(self):
        """Zoom fit on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, '_zoom_fit'):
            current_view._zoom_fit()
            
    def _zoom_in(self):
        """Zoom in on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, '_zoom_in'):
            current_view._zoom_in()
            
    def _zoom_out(self):
        """Zoom out on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, '_zoom_out'):
            current_view._zoom_out()
            
    def _show_roi_type_menu(self):
        """Show ROI type menu on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, '_show_roi_type_menu'):
            current_view._show_roi_type_menu()
            
    def _clear_roi(self):
        """Clear ROI on current tab."""
        current_view = self.get_current_image_view()
        if current_view and hasattr(current_view, '_clear_roi'):
            current_view._clear_roi()
            
    def _save_tab_state(self, tab_key: str):
        """Save the current state of a tab."""
        if tab_key not in self.image_views:
            return
            
        image_view = self.image_views[tab_key]
        
        # Save all relevant state
        state = {
            'rgb_bands': image_view.get_rgb_bands(),
            'mono_mode': image_view.mono_mode,
            'current_band': image_view.current_band,
            'selected_rgb_channel': image_view.selected_rgb_channel,
            'stretch_percent': image_view.get_stretch_percent() if hasattr(image_view, 'get_stretch_percent') else 2.0,
            'histogram_levels': image_view.get_histogram_levels() if hasattr(image_view, 'get_histogram_levels') else None,
            'view_range': image_view.get_view_range() if hasattr(image_view, 'get_view_range') else None,
            'levels_initialized': getattr(image_view, 'levels_initialized', False),
        }
        
        # Save any additional state if needed
        if hasattr(image_view, 'frame_view_mode'):
            state['frame_view_mode'] = image_view.frame_view_mode
            
        self.tab_states[tab_key] = state
        print(f"Saved state for tab {tab_key}: {state}")
        
    def _restore_tab_state(self, tab_key: str):
        """Restore the saved state of a tab."""
        if tab_key not in self.tab_states or tab_key not in self.image_views:
            # No saved state yet - this is normal for new tabs
            return
            
        state = self.tab_states[tab_key]
        image_view = self.image_views[tab_key]
        
        print(f"Restoring state for tab {tab_key}: {state}")
        
        # Restore RGB bands
        if 'rgb_bands' in state:
            r, g, b = state['rgb_bands']
            image_view.set_rgb_bands(r, g, b)
            
        # Restore mono mode
        if 'mono_mode' in state:
            image_view.mono_mode = state['mono_mode']
            
        # Restore current band
        if 'current_band' in state:
            image_view.current_band = state['current_band']
            
        # Restore RGB channel selection
        if 'selected_rgb_channel' in state:
            image_view.selected_rgb_channel = state['selected_rgb_channel']
            
        # Restore frame view mode if applicable
        if 'frame_view_mode' in state and hasattr(image_view, 'frame_view_mode'):
            image_view.frame_view_mode = state['frame_view_mode']
            
        # Restore stretch percentage
        if 'stretch_percent' in state and hasattr(image_view, 'set_stretch_percent'):
            image_view.set_stretch_percent(state['stretch_percent'])
            
        # Restore histogram levels (contrast left/right limits)
        if 'histogram_levels' in state and hasattr(image_view, 'set_histogram_levels'):
            levels = state['histogram_levels']
            if levels is not None:
                # Store and apply levels
                image_view.histogram_levels = levels
                # Apply immediately using QTimer to ensure proper order
                QtCore.QTimer.singleShot(10, lambda: image_view.set_histogram_levels(levels))
        
        # Restore view range (zoom and position) only if explicitly saved
        # Don't restore on first tab switch to avoid jumping
        if 'view_range' in state and hasattr(image_view, 'set_view_range'):
            view_range = state['view_range']
            # Only restore if view_range looks valid (not default/uninitialized values)
            if view_range is not None and 'width' in view_range and view_range['width'] > 0:
                # Store and apply view range
                image_view.view_range = view_range
                # Apply immediately using QTimer to ensure proper order
                QtCore.QTimer.singleShot(20, lambda: image_view.set_view_range(view_range))
                
        # Restore levels initialization state
        if 'levels_initialized' in state:
            image_view.levels_initialized = state['levels_initialized']
            
        # Trigger display update to reflect restored state
        # Emit RGB bands changed to update the main application display
        self.rgb_bands_changed.emit()
        
    def _create_toolbar(self):
        """Create toolbar for this image view instance."""
        toolbar = QtWidgets.QToolBar()
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setMaximumHeight(32)
        
        # View operations
        zoom_fit_action = toolbar.addAction('Fit')
        zoom_fit_action.setToolTip('Zoom to fit image')
        zoom_fit_action.triggered.connect(self._zoom_fit)
        
        zoom_in_action = toolbar.addAction('In')
        zoom_in_action.setToolTip('Zoom in')
        zoom_in_action.triggered.connect(self._zoom_in)
        
        zoom_out_action = toolbar.addAction('Out')
        zoom_out_action.setToolTip('Zoom out')
        zoom_out_action.triggered.connect(self._zoom_out)
        
        toolbar.addSeparator()
        
        # ROI operations
        roi_action = toolbar.addAction('ROI')
        roi_action.setToolTip('ROI Mode')
        roi_action.triggered.connect(self._show_roi_type_menu)
        
        clear_roi_action = toolbar.addAction('Clear')
        clear_roi_action.setToolTip('Clear ROIs')
        clear_roi_action.triggered.connect(self._clear_roi)
        
        toolbar.addSeparator()
        
        # Dataset selector dropdown
        toolbar.addWidget(QtWidgets.QLabel('Dataset:'))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setMinimumWidth(120)
        self.dataset_combo.setToolTip('Select dataset to display in this view')
        # Flag to prevent triggering during initialization
        self._combo_updating = False
        toolbar.addWidget(self.dataset_combo)
        
        # Refresh dataset list initially
        self._refresh_dataset_combo()
        
        # Connect after initial refresh to prevent triggering
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_combo_changed)
        # Also connect to activated signal which fires when user explicitly selects an item
        self.dataset_combo.activated.connect(lambda index: self._on_dataset_combo_changed(self.dataset_combo.itemText(index)))
        
        toolbar.addSeparator()
        
        # No Data value setting
        no_data_action = toolbar.addAction('No Data')
        no_data_action.setToolTip('Set no data value to exclude from contrast stretching')
        no_data_action.triggered.connect(self._show_no_data_dialog)
        
        # Bad Bands setting
        bad_bands_action = toolbar.addAction('Bad Bands')
        bad_bands_action.setToolTip('Select bad/inactive bands to exclude from spectral plots')
        bad_bands_action.triggered.connect(self._show_bad_bands_dialog)
        
        return toolbar
        
    def _refresh_dataset_combo(self):
        """Refresh the dataset combo box with available datasets."""
        self._combo_updating = True
        current_text = self.dataset_combo.currentText()
        self.dataset_combo.clear()
        
        # Get all available datasets from data manager
        datasets = self.data_manager.list_datasets()
        for dataset_info in datasets:
            self.dataset_combo.addItem(dataset_info['name'])
            
        # Restore selection if possible
        if current_text:
            index = self.dataset_combo.findText(current_text)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
        
        self._combo_updating = False
                
    def _on_dataset_combo_changed(self, dataset_name: str):
        """Handle dataset selection change in combo box."""
        # Ignore changes during combo update
        if self._combo_updating:
            return
            
        if not dataset_name:
            return
            
        current_dataset = self.get_current_dataset_name()
        
        # Allow loading if:
        # 1. Different dataset selected, OR
        # 2. No current dataset (placeholder tab only), OR
        # 3. Current dataset name is just the base name but we have countered tabs
        if (dataset_name != current_dataset or 
            current_dataset is None or
            len(self.image_views) == 0):
            
            # Add tab for this dataset if not already present
            success = self.add_dataset_tab(dataset_name)
            if success:
                # Update combo to reflect the current selection
                self._combo_updating = True
                current_index = self.dataset_combo.findText(dataset_name)
                if current_index >= 0:
                    self.dataset_combo.setCurrentIndex(current_index)
    
    def _get_or_create_tab_view_id(self, tab_index: int) -> str:
        """Get or create a unique view ID for a tab."""
        if tab_index not in self.tab_view_ids:
            # Create a unique ID combining the parent view ID and a counter
            view_id = f"{self.view_id}_tab{self._next_view_counter}"
            self.tab_view_ids[tab_index] = view_id
            self._next_view_counter += 1
        return self.tab_view_ids[tab_index]
    
    def get_all_tab_view_ids(self) -> List[tuple]:
        """Get all tab view IDs with their labels."""
        result = []
        for i in range(self.tab_widget.count()):
            tab_text = self.tab_widget.tabText(i)
            view_id = self._get_or_create_tab_view_id(i)
            result.append((view_id, tab_text))
        return result
    
    def _show_no_data_dialog(self):
        """Show dialog to set no data value for current dataset."""
        current_dataset = self.get_current_dataset_name()
        if not current_dataset:
            QtWidgets.QMessageBox.warning(self, "No Dataset", "No dataset is currently loaded.")
            return
        
        # Get current no data value if set
        current_value = self.no_data_values.get(current_dataset, None)
        
        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Set No Data Value - {current_dataset}")
        dialog.setModal(True)
        dialog.resize(400, 200)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Set the no data value to exclude from contrast stretching and histogram calculation.\n"
            "Common values include -9999, -9998, NaN, or 0."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Input section
        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(QtWidgets.QLabel("No Data Value:"))
        
        value_input = QtWidgets.QLineEdit()
        value_input.setPlaceholderText("e.g., -9999, -9998, nan")
        if current_value is not None:
            if np.isnan(current_value):
                value_input.setText("nan")
            else:
                value_input.setText(str(current_value))
        input_layout.addWidget(value_input)
        
        layout.addLayout(input_layout)
        
        # Preset buttons
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Common values:"))
        
        for preset_value in [-9999, -9998, -1, 0]:
            btn = QtWidgets.QPushButton(str(preset_value))
            btn.clicked.connect(lambda checked, val=preset_value: value_input.setText(str(val)))
            preset_layout.addWidget(btn)
        
        nan_btn = QtWidgets.QPushButton("NaN")
        nan_btn.clicked.connect(lambda: value_input.setText("nan"))
        preset_layout.addWidget(nan_btn)
        
        layout.addLayout(preset_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self._clear_no_data_value(current_dataset, dialog))
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(lambda: self._apply_no_data_value(current_dataset, value_input.text(), dialog))
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Show current status
        if current_value is not None:
            if np.isnan(current_value):
                status_text = f"Current no data value: NaN"
            else:
                status_text = f"Current no data value: {current_value}"
        else:
            status_text = "No data value: Not set"
        
        status_label = QtWidgets.QLabel(status_text)
        status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.insertWidget(1, status_label)
        
        dialog.exec_()
    
    def _apply_no_data_value(self, dataset_name: str, value_text: str, dialog):
        """Apply the no data value and refresh display."""
        try:
            if value_text.strip().lower() in ['nan', 'none', '']:
                if value_text.strip().lower() == 'nan':
                    no_data_value = np.nan
                else:
                    # Clear the no data value
                    if dataset_name in self.no_data_values:
                        del self.no_data_values[dataset_name]
                    dialog.accept()
                    self._refresh_display()
                    return
            else:
                no_data_value = float(value_text)
            
            # Store the no data value
            self.no_data_values[dataset_name] = no_data_value
            
            dialog.accept()
            
            # Refresh the display to apply the new no data value
            self._refresh_display()
            
        except ValueError:
            QtWidgets.QMessageBox.warning(dialog, "Invalid Value", 
                                        f"'{value_text}' is not a valid number.")
    
    def _clear_no_data_value(self, dataset_name: str, dialog):
        """Clear the no data value for the current dataset."""
        if dataset_name in self.no_data_values:
            del self.no_data_values[dataset_name]
        
        dialog.accept()
        
        # Refresh the display
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the current display to apply no data value changes."""
        current_view = self.get_current_image_view()
        if current_view:
            # Trigger a redraw with the current RGB bands
            current_view._update_rgb_display()
    
    def get_no_data_value(self, dataset_name: str) -> Optional[float]:
        """Get the no data value for a dataset."""
        return self.no_data_values.get(dataset_name, None)
    
    def _show_bad_bands_dialog(self):
        """Show dialog to select bad/inactive bands."""
        current_dataset = self.get_current_dataset_name()
        if not current_dataset:
            QtWidgets.QMessageBox.warning(self, "No Dataset", "No dataset is currently loaded.")
            return
        
        # Get dataset to access band information
        dataset = self.data_manager.get_dataset(current_dataset)
        if not dataset or not dataset.is_loaded:
            QtWidgets.QMessageBox.warning(self, "Dataset Error", "Dataset is not loaded.")
            return
        
        # Get wavelengths and band count
        wavelengths = dataset.wavelengths if hasattr(dataset, 'wavelengths') else None
        if not hasattr(dataset, 'shape') or len(dataset.shape) < 3:
            QtWidgets.QMessageBox.warning(self, "Dataset Error", "Dataset doesn't have band information.")
            return
        
        num_bands = dataset.shape[2]
        
        # Initialize bad bands for this dataset if not exists
        if current_dataset not in self.bad_bands:
            self.bad_bands[current_dataset] = {i: True for i in range(num_bands)}  # All bands active by default
        
        # Create dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Bad Bands Selection - {current_dataset}")
        dialog.setModal(True)
        dialog.resize(600, 500)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Info label
        info_label = QtWidgets.QLabel(
            f"Dataset: {current_dataset} ({num_bands} bands)\n"
            "Select active bands (checked = active/good, unchecked = inactive/bad).\n"
            "Bad bands will be excluded from spectral plots."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-weight: bold; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(info_label)
        
        # Create main content area with splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Left side: Band list
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        # Band list header
        list_header = QtWidgets.QLabel("Band List:")
        list_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        left_layout.addWidget(list_header)
        
        # Band list with checkboxes
        self.band_list = QtWidgets.QListWidget()
        self.band_list.setMinimumWidth(300)
        
        # Populate band list
        for i in range(num_bands):
            item = QtWidgets.QListWidgetItem()
            
            # Format band info
            if wavelengths is not None and i < len(wavelengths):
                wl = wavelengths[i]
                band_text = f"Band {i:3d}: {wl:7.2f} nm"
            else:
                band_text = f"Band {i:3d}"
            
            item.setText(band_text)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            
            # Set current status
            is_active = self.bad_bands[current_dataset].get(i, True)
            item.setCheckState(QtCore.Qt.Checked if is_active else QtCore.Qt.Unchecked)
            
            self.band_list.addItem(item)
        
        left_layout.addWidget(self.band_list)
        
        # Bulk selection buttons for band list
        bulk_buttons_layout = QtWidgets.QHBoxLayout()
        
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self._set_all_bands_status(True))
        bulk_buttons_layout.addWidget(select_all_btn)
        
        clear_all_btn = QtWidgets.QPushButton("Clear All")
        clear_all_btn.clicked.connect(lambda: self._set_all_bands_status(False))
        bulk_buttons_layout.addWidget(clear_all_btn)
        
        invert_btn = QtWidgets.QPushButton("Invert")
        invert_btn.clicked.connect(self._invert_band_selection)
        bulk_buttons_layout.addWidget(invert_btn)
        
        left_layout.addLayout(bulk_buttons_layout)
        
        # Right side: Range selection
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        
        # Range selection header
        range_header = QtWidgets.QLabel("Range Selection:")
        range_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        right_layout.addWidget(range_header)
        
        # Range input
        range_input_layout = QtWidgets.QFormLayout()
        
        self.start_band_input = QtWidgets.QSpinBox()
        self.start_band_input.setRange(0, num_bands - 1)
        self.start_band_input.setValue(0)
        range_input_layout.addRow("Start Band:", self.start_band_input)
        
        self.end_band_input = QtWidgets.QSpinBox()
        self.end_band_input.setRange(0, num_bands - 1)
        self.end_band_input.setValue(num_bands - 1)
        range_input_layout.addRow("End Band:", self.end_band_input)
        
        right_layout.addLayout(range_input_layout)
        
        # Range action buttons
        range_buttons_layout = QtWidgets.QVBoxLayout()
        
        select_range_btn = QtWidgets.QPushButton("Select Range (Active)")
        select_range_btn.clicked.connect(lambda: self._select_band_range(True))
        range_buttons_layout.addWidget(select_range_btn)
        
        deselect_range_btn = QtWidgets.QPushButton("Deselect Range (Inactive)")
        deselect_range_btn.clicked.connect(lambda: self._select_band_range(False))
        range_buttons_layout.addWidget(deselect_range_btn)
        
        right_layout.addLayout(range_buttons_layout)
        
        # Preset bad band ranges
        right_layout.addWidget(QtWidgets.QLabel("Common Bad Band Presets:"))
        
        preset_buttons_layout = QtWidgets.QVBoxLayout()
        
        # Water absorption bands around 1400nm
        water1_btn = QtWidgets.QPushButton("Water Bands ~1400nm")
        water1_btn.clicked.connect(lambda: self._apply_water_bands_preset(wavelengths, 1400, 50))
        preset_buttons_layout.addWidget(water1_btn)
        
        # Water absorption bands around 1900nm  
        water2_btn = QtWidgets.QPushButton("Water Bands ~1900nm")
        water2_btn.clicked.connect(lambda: self._apply_water_bands_preset(wavelengths, 1900, 50))
        preset_buttons_layout.addWidget(water2_btn)
        
        # Atmospheric bands
        atmos_btn = QtWidgets.QPushButton("Atmospheric Bands")
        atmos_btn.clicked.connect(lambda: self._apply_atmospheric_bands_preset(wavelengths))
        preset_buttons_layout.addWidget(atmos_btn)
        
        right_layout.addLayout(preset_buttons_layout)
        
        right_layout.addStretch()
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)
        
        # Status label
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
        self._update_bad_bands_status()
        
        # Dialog buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        reset_btn = QtWidgets.QPushButton("Reset to Default")
        reset_btn.clicked.connect(lambda: self._reset_bad_bands(current_dataset))
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(lambda: self._apply_bad_bands(current_dataset, dialog))
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        # Connect signals
        self.band_list.itemChanged.connect(self._update_bad_bands_status)
        
        dialog.exec_()
    
    def _set_all_bands_status(self, status: bool):
        """Set all bands to active (True) or inactive (False)."""
        for i in range(self.band_list.count()):
            item = self.band_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if status else QtCore.Qt.Unchecked)
    
    def _invert_band_selection(self):
        """Invert the current band selection."""
        for i in range(self.band_list.count()):
            item = self.band_list.item(i)
            current_state = item.checkState()
            new_state = QtCore.Qt.Unchecked if current_state == QtCore.Qt.Checked else QtCore.Qt.Checked
            item.setCheckState(new_state)
    
    def _select_band_range(self, status: bool):
        """Select or deselect a range of bands."""
        start = self.start_band_input.value()
        end = self.end_band_input.value()
        
        if start > end:
            start, end = end, start  # Swap if needed
        
        for i in range(start, end + 1):
            if i < self.band_list.count():
                item = self.band_list.item(i)
                item.setCheckState(QtCore.Qt.Checked if status else QtCore.Qt.Unchecked)
    
    def _apply_water_bands_preset(self, wavelengths, center_wavelength: float, width: float):
        """Apply preset for water absorption bands."""
        if wavelengths is None:
            QtWidgets.QMessageBox.information(self, "No Wavelengths", 
                                            "Cannot apply wavelength-based preset: no wavelength information available.")
            return
        
        # Find bands within the specified range around the center wavelength
        for i, wl in enumerate(wavelengths):
            if abs(wl - center_wavelength) <= width:
                if i < self.band_list.count():
                    item = self.band_list.item(i)
                    item.setCheckState(QtCore.Qt.Unchecked)  # Mark as bad/inactive
    
    def _apply_atmospheric_bands_preset(self, wavelengths):
        """Apply preset for common atmospheric absorption bands."""
        if wavelengths is None:
            QtWidgets.QMessageBox.information(self, "No Wavelengths", 
                                            "Cannot apply wavelength-based preset: no wavelength information available.")
            return
        
        # Common atmospheric absorption bands
        atmos_bands = [
            (1340, 1440),  # Water vapor
            (1820, 1950),  # Water vapor  
            (760, 770),    # Oxygen A-band
            (1268, 1278),  # Oxygen
        ]
        
        for i, wl in enumerate(wavelengths):
            for band_start, band_end in atmos_bands:
                if band_start <= wl <= band_end:
                    if i < self.band_list.count():
                        item = self.band_list.item(i)
                        item.setCheckState(QtCore.Qt.Unchecked)  # Mark as bad/inactive
                    break
    
    def _reset_bad_bands(self, dataset_name: str):
        """Reset all bands to active (good) status."""
        self._set_all_bands_status(True)
    
    def _update_bad_bands_status(self):
        """Update the status label with current bad band count."""
        if hasattr(self, 'band_list'):
            active_count = 0
            inactive_count = 0
            
            for i in range(self.band_list.count()):
                item = self.band_list.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    active_count += 1
                else:
                    inactive_count += 1
            
            total = active_count + inactive_count
            self.status_label.setText(f"Active bands: {active_count}, Inactive (bad) bands: {inactive_count}, Total: {total}")
    
    def _apply_bad_bands(self, dataset_name: str, dialog):
        """Apply the bad bands selection and close dialog."""
        # Update the bad bands dictionary
        bad_bands_dict = {}
        
        for i in range(self.band_list.count()):
            item = self.band_list.item(i)
            is_active = item.checkState() == QtCore.Qt.Checked
            bad_bands_dict[i] = is_active
        
        self.bad_bands[dataset_name] = bad_bands_dict
        
        # Save bad band list to ENVI header file
        try:
            dataset = self.data_manager.get_dataset(dataset_name)
            if dataset and hasattr(dataset, 'set_bad_band_list'):
                # Create BBL array (0=bad, 1=good) from the UI selections
                num_bands = self.band_list.count()
                bbl_array = np.zeros(num_bands, dtype=np.int32)
                
                for i in range(num_bands):
                    item = self.band_list.item(i)
                    is_active = item.checkState() == QtCore.Qt.Checked
                    bbl_array[i] = 1 if is_active else 0
                
                # Save to ENVI header
                success = dataset.set_bad_band_list(bbl_array)
                if success:
                    print(f"✅ Bad band list saved to ENVI header for dataset '{dataset_name}'")
                else:
                    print(f"⚠️ Failed to save bad band list to ENVI header for dataset '{dataset_name}'")
            else:
                print(f"⚠️ Dataset '{dataset_name}' does not support bad band list saving to header")
        except Exception as e:
            print(f"Error saving bad band list to header: {e}")
        
        dialog.accept()
        
        # Emit signal to update spectrum plots if needed
        self.rgb_bands_changed.emit()  # This will trigger spectrum plot updates
    
    def get_active_bands(self, dataset_name: str) -> List[int]:
        """Get list of active (good) band indices for a dataset."""
        if dataset_name not in self.bad_bands:
            # If no bad bands set, all bands are active
            dataset = self.data_manager.get_dataset(dataset_name)
            if dataset and hasattr(dataset, 'shape') and len(dataset.shape) >= 3:
                return list(range(dataset.shape[2]))
            return []
        
        # Return only active bands
        return [band_idx for band_idx, is_active in self.bad_bands[dataset_name].items() if is_active]
    
    def get_bad_bands(self, dataset_name: str) -> List[int]:
        """Get list of bad (inactive) band indices for a dataset."""
        if dataset_name not in self.bad_bands:
            return []  # No bad bands set
        
        # Return only inactive bands
        return [band_idx for band_idx, is_active in self.bad_bands[dataset_name].items() if not is_active]
    
    def _load_bad_bands_from_header(self, dataset_name: str, dataset):
        """Load bad band list from ENVI header file if available."""
        try:
            if hasattr(dataset, 'get_bad_band_list'):
                bbl_array = dataset.get_bad_band_list()
                if bbl_array is not None:
                    print(f"Loading bad band list from header for dataset '{dataset_name}': {np.sum(bbl_array == 0)} bad bands")
                    
                    # Convert BBL array to bad_bands dictionary format
                    bad_bands_dict = {}
                    for i, is_good in enumerate(bbl_array):
                        bad_bands_dict[i] = bool(is_good)  # Convert to boolean (True=good/active, False=bad/inactive)
                    
                    # Store in the UI's bad bands dictionary
                    self.bad_bands[dataset_name] = bad_bands_dict
                    print(f"✅ Loaded bad band list from header for dataset '{dataset_name}'")
                else:
                    print(f"No bad band list found in header for dataset '{dataset_name}'")
            else:
                print(f"Dataset '{dataset_name}' does not support bad band list loading from header")
        except Exception as e:
            print(f"Error loading bad band list from header for dataset '{dataset_name}': {e}")