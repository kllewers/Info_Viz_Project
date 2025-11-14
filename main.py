#!/usr/bin/env python3
"""
Hyperspectral Viewer - ENVI Classic-like viewer for hyperspectral data.

Main application entry point with integrated GUI components.
"""

import sys
import os
import argparse
from typing import Optional
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui, QtSvg
import pyqtgraph as pg

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from data_handler import DataHandler
from tabbed_image_view import TabbedImageView
from spectrum_plot import SpectrumPlot
from roi_manager import ROIManager
from utils import ConfigManager, estimate_optimal_rgb_bands, get_true_color_rgb_bands, validate_envi_file_pair
from data_manager import DataManager
from file_manager_widget import FileManagerWidget
from sam_dialog import SAMDialog
from whitened_similarity_dialog import WhitenedSimilarityDialog
from nc_viewer import NCViewerWindow
from indices_dialog import IndicesDialog


class HyperspectralViewer(QtWidgets.QMainWindow):
    """Main application window for hyperspectral data viewing."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.data_handler = DataHandler()
        self.roi_manager = ROIManager()
        self.config = ConfigManager()
        self.data_manager = DataManager()
        
        # GUI components
        self.image_view = None
        self.spectrum_plot = None
        self.overview_window = None
        self.file_manager = None
        
        # State
        self.current_file = None
        self.recent_files = []
        self.split_mode = None  # None, 'horizontal', or 'vertical'
        self.split_splitter = None  # QSplitter for split view
        self.split_views = []  # List of TabbedImageView instances in split mode
        self.focused_split_view = None  # Track which split view is focused
        
        self._setup_ui()
        self._setup_connections()
        self._load_settings()
        
        # Load file from command line if provided
        if len(sys.argv) > 1:
            self.load_data_file(sys.argv[1])
            
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Info_i - Hyperspectral Viewer")
        self.setWindowIcon(QtGui.QIcon())  # Add icon if available
        
        # Get window size from config
        window_width = self.config.get('ui.window_width', 1200)
        window_height = self.config.get('ui.window_height', 800)
        
        # Start in fullscreen by default
        self.is_starting_fullscreen = self.config.get('ui.start_fullscreen', True)
        if self.is_starting_fullscreen:
            self.showFullScreen()
        elif self.config.get('ui.start_maximized', True):
            self.showMaximized()
        else:
            self.resize(window_width, window_height)
        
        # Setup application font
        self._setup_font()
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # Create status bar
        if self.config.get('ui.show_status_bar', True):
            self.status_bar = self.statusBar()
            self.status_bar.showMessage("Ready")
        
        # Create central widget with splitter
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create main splitter (horizontal) - 2 sections: image, controls
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setHandleWidth(10)  # Increase handle width for better usability
        main_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing
        
        # Left side: Image view container (will hold either single view or split views)
        self.image_container = QtWidgets.QWidget()
        self.image_container.setMinimumSize(400, 300)
        
        # Create layout for image container
        self.image_layout = QtWidgets.QVBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Single image view (default)
        self.image_view = TabbedImageView(view_id="main_view")
        self.image_layout.addWidget(self.image_view)
        
        main_splitter.addWidget(self.image_container)
        
        # Create floating file manager
        if self.config.get('ui.show_file_manager', True):
            try:
                self.file_manager = FileManagerWidget(self)
                self.file_manager.hide()  # Initially hidden
            except Exception as e:
                print(f"Failed to create FileManagerWidget: {e}")
                self.file_manager = None
        
        # Create floating overview inside the image view
        if self.config.get('ui.show_overview', True):
            self.overview_window = self._create_floating_overview()
        
        # Right side: Spectrum plot and controls in vertical splitter
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.setHandleWidth(10)
        right_splitter.setChildrenCollapsible(False)

        self.spectrum_plot = SpectrumPlot()
        self.spectrum_plot.setMinimumSize(400, 300)  # Set minimum size
        right_splitter.addWidget(self.spectrum_plot)

        # Create tabbed panel for ROI management and base map
        tabbed_panel = self._create_tabbed_panel()
        tabbed_panel.setMinimumHeight(100)  # Minimum height for tabbed panel
        right_splitter.addWidget(tabbed_panel)
        
        # Set stretch factors: spectrum plot gets more space
        right_splitter.setStretchFactor(0, 3)  # Spectrum plot
        right_splitter.setStretchFactor(1, 1)  # ROI panel
        
        main_splitter.addWidget(right_splitter)
        
        # Set main splitter ratios for 2-section layout: image, controls
        ratios = self.config.get('ui.splitter_ratios', [0.6, 0.4])
        if len(ratios) == 3:
            # Convert old 3-section ratios to 2-section: combine overview + image
            ratios = [ratios[0] + ratios[1], ratios[2]]
        elif len(ratios) != 2:
            # Fallback to default if invalid
            ratios = [0.6, 0.4]
            
        total = sum(ratios)
        sizes = [int(ratio/total * window_width) for ratio in ratios]
        main_splitter.setSizes(sizes)
        main_splitter.setStretchFactor(0, int(ratios[0] * 100))  # Image view
        main_splitter.setStretchFactor(1, int(ratios[1] * 100))  # Right controls
        
        layout.addWidget(main_splitter)
        
    def _setup_font(self):
        """Setup application font based on configuration."""
        try:
            # Get font configuration
            font_families = self.config.get('ui.font_family', ["SF Pro Display", "SF Pro Text", "Monument Grotesk", "Instrument Sans", "Segoe UI", "Arial"])
            font_size = self.config.get('ui.font_size', 9)
            
            # Create font with fallback chain
            font = QtGui.QFont()
            if isinstance(font_families, list):
                # Try each font family in order until one is found
                for family in font_families:
                    font.setFamily(family)
                    if QtGui.QFontInfo(font).family() == family:
                        break
            else:
                # Single font family
                font.setFamily(font_families)
            
            font.setPointSize(font_size)
            font.setStyleHint(QtGui.QFont.SansSerif)
            
            # Apply to the entire application
            QtWidgets.QApplication.instance().setFont(font)
            
            # Also set a custom stylesheet to ensure consistency with sparkly pink and purple theme
            self.setStyleSheet(f"""
                QMainWindow {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #E91E63, stop:0.5 #9C27B0, stop:1 #673AB7);
                }}
                QWidget {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    font-size: {font_size}pt;
                    background-color: rgba(156, 39, 176, 0.1);
                    color: #FFFFFF;
                }}
                QLabel {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    color: #FFE4F5;
                    background-color: transparent;
                }}
                QMenuBar {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    font-size: {font_size}pt;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #E91E63, stop:1 #9C27B0);
                    color: #FFFFFF;
                    border-bottom: 2px solid #FFD1DC;
                }}
                QMenuBar::item:selected {{
                    background-color: rgba(255, 255, 255, 0.2);
                    border-radius: 4px;
                }}
                QMenu {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    font-size: {font_size}pt;
                    background-color: #9C27B0;
                    color: #FFFFFF;
                    border: 2px solid #E91E63;
                    border-radius: 5px;
                }}
                QMenu::item:selected {{
                    background-color: #E91E63;
                    color: #FFFFFF;
                }}
                QPushButton {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    font-size: {font_size}pt;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #E91E63, stop:1 #9C27B0);
                    color: #FFFFFF;
                    border: 2px solid #FFD1DC;
                    border-radius: 8px;
                    padding: 5px 15px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #FF4081, stop:1 #BA68C8);
                    border: 2px solid #FFFFFF;
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #C2185B, stop:1 #7B1FA2);
                }}
                QStatusBar {{
                    font-family: {', '.join([f'"{f}"' for f in font_families]) if isinstance(font_families, list) else f'"{font_families}"'};
                    font-size: {font_size}pt;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #9C27B0, stop:1 #673AB7);
                    color: #FFE4F5;
                    border-top: 2px solid #FFD1DC;
                }}
                QGroupBox {{
                    color: #FFE4F5;
                    border: 2px solid #E91E63;
                    border-radius: 8px;
                    margin-top: 10px;
                    font-weight: bold;
                    background-color: rgba(156, 39, 176, 0.3);
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 5px;
                    color: #FFD1DC;
                }}
                QTabWidget::pane {{
                    border: 2px solid #E91E63;
                    border-radius: 5px;
                    background-color: rgba(103, 58, 183, 0.2);
                }}
                QTabBar::tab {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #9C27B0, stop:1 #673AB7);
                    color: #FFFFFF;
                    border: 2px solid #E91E63;
                    border-bottom: none;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    padding: 8px 15px;
                    margin-right: 2px;
                }}
                QTabBar::tab:selected {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #E91E63, stop:1 #9C27B0);
                    border-bottom: 2px solid #E91E63;
                }}
                QTabBar::tab:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #BA68C8, stop:1 #9575CD);
                }}
                QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
                    background-color: rgba(255, 255, 255, 0.9);
                    color: #9C27B0;
                    border: 2px solid #E91E63;
                    border-radius: 5px;
                    padding: 5px;
                }}
                QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                    border: 2px solid #FFD1DC;
                    background-color: #FFFFFF;
                }}
                QSlider::groove:horizontal {{
                    border: 1px solid #E91E63;
                    height: 8px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #9C27B0, stop:1 #E91E63);
                    border-radius: 4px;
                }}
                QSlider::handle:horizontal {{
                    background: #FFD1DC;
                    border: 2px solid #E91E63;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }}
                QSlider::handle:horizontal:hover {{
                    background: #FFFFFF;
                    border: 3px solid #FF4081;
                }}
            """)
            
        except Exception as e:
            # Fallback to system default
            pass
    
    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        open_action = QtWidgets.QAction('&Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu('Recent Files')
        self._update_recent_files_menu()
        
        file_menu.addSeparator()
        
        save_project_action = QtWidgets.QAction('&Save Project...', self)
        save_project_action.setShortcut('Ctrl+S')
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)
        
        load_project_action = QtWidgets.QAction('&Load Project...', self)
        load_project_action.triggered.connect(self._load_project)
        file_menu.addAction(load_project_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('&View')
        
        # Split view submenu
        split_menu = view_menu.addMenu('&Split View')
        
        horizontal_split_action = QtWidgets.QAction('&Horizontal (Side by Side)', self)
        horizontal_split_action.setShortcut('Ctrl+Shift+H')
        horizontal_split_action.triggered.connect(lambda: self._toggle_split_view('horizontal'))
        split_menu.addAction(horizontal_split_action)
        
        vertical_split_action = QtWidgets.QAction('&Vertical (Top/Bottom)', self)
        vertical_split_action.setShortcut('Ctrl+Shift+V')
        vertical_split_action.triggered.connect(lambda: self._toggle_split_view('vertical'))
        split_menu.addAction(vertical_split_action)
        
        split_menu.addSeparator()
        
        close_split_action = QtWidgets.QAction('&Close Split View', self)
        close_split_action.setShortcut('Ctrl+Shift+C')
        close_split_action.triggered.connect(self._close_split_view)
        split_menu.addAction(close_split_action)
        
        view_menu.addSeparator()
        
        # Spectrum split view
        spectrum_split_action = QtWidgets.QAction('Split &Spectrum View', self)
        spectrum_split_action.setShortcut('Ctrl+Shift+S')
        spectrum_split_action.setCheckable(True)
        spectrum_split_action.triggered.connect(self._toggle_spectrum_split_view)
        view_menu.addAction(spectrum_split_action)
        
        view_menu.addSeparator()
        
        zoom_fit_action = QtWidgets.QAction('Zoom &Fit', self)
        zoom_fit_action.setShortcut('Ctrl+F')
        zoom_fit_action.triggered.connect(lambda: self.image_view._zoom_fit() if self.image_view else None)
        view_menu.addAction(zoom_fit_action)
        
        zoom_in_action = QtWidgets.QAction('Zoom &In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(lambda: self.image_view._zoom_in() if self.image_view else None)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QtWidgets.QAction('Zoom &Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(lambda: self.image_view._zoom_out() if self.image_view else None)
        view_menu.addAction(zoom_out_action)
        
        view_menu.addSeparator()
        
        toggle_overview_action = QtWidgets.QAction('Toggle &Overview', self)
        toggle_overview_action.setCheckable(True)
        toggle_overview_action.setChecked(self.config.get('ui.show_overview', True))
        toggle_overview_action.triggered.connect(self._toggle_overview)
        view_menu.addAction(toggle_overview_action)
        
        view_menu.addSeparator()
        
        # Fullscreen toggle
        toggle_fullscreen_action = QtWidgets.QAction('&Fullscreen', self)
        toggle_fullscreen_action.setShortcut('F11')
        toggle_fullscreen_action.setCheckable(True)
        toggle_fullscreen_action.triggered.connect(self._toggle_fullscreen)
        # Set initial checked state based on startup mode
        if hasattr(self, 'is_starting_fullscreen'):
            toggle_fullscreen_action.setChecked(self.is_starting_fullscreen)
        view_menu.addAction(toggle_fullscreen_action)
        self.fullscreen_action = toggle_fullscreen_action  # Store reference for updating check state
        
        view_menu.addSeparator()
        
        # Toolbar visibility
        toggle_toolbar_action = QtWidgets.QAction('Show &Toolbar', self)
        toggle_toolbar_action.setCheckable(True)
        toggle_toolbar_action.setChecked(True)
        toggle_toolbar_action.triggered.connect(self._toggle_toolbar)
        view_menu.addAction(toggle_toolbar_action)
        
        # File Manager
        toggle_file_manager_action = QtWidgets.QAction('Show &File Manager', self)
        toggle_file_manager_action.setCheckable(True)
        toggle_file_manager_action.setChecked(False)
        toggle_file_manager_action.triggered.connect(self._toggle_file_manager)
        view_menu.addAction(toggle_file_manager_action)
        
        # Header viewer
        header_viewer_action = QtWidgets.QAction('Show &Header', self)
        header_viewer_action.setShortcut('Ctrl+H')
        header_viewer_action.triggered.connect(self._show_header_viewer)
        view_menu.addAction(header_viewer_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('&Tools')

        roi_mode_action = QtWidgets.QAction('&ROI Mode', self)
        roi_mode_action.setShortcut('R')
        roi_mode_action.triggered.connect(lambda: self.image_view._show_roi_type_menu() if self.image_view else None)
        tools_menu.addAction(roi_mode_action)

        clear_rois_action = QtWidgets.QAction('&Clear All ROIs', self)
        clear_rois_action.setShortcut('Ctrl+R')
        clear_rois_action.triggered.connect(self._clear_all_rois)
        tools_menu.addAction(clear_rois_action)

        tools_menu.addSeparator()

        export_spectra_action = QtWidgets.QAction('&Export Spectra...', self)
        export_spectra_action.setShortcut('Ctrl+E')
        export_spectra_action.triggered.connect(lambda: self.spectrum_plot._export_spectra() if self.spectrum_plot else None)
        tools_menu.addAction(export_spectra_action)

        tools_menu.addSeparator()

        # NetCDF Structure Viewer
        nc_viewer_action = QtWidgets.QAction('&NetCDF Structure Viewer...', self)
        nc_viewer_action.setShortcut('Ctrl+N')
        nc_viewer_action.triggered.connect(self._open_nc_viewer)
        tools_menu.addAction(nc_viewer_action)
        
        # Spectral Analysis menu
        analysis_menu = menubar.addMenu('&Spectral Analysis')

        indices_action = QtWidgets.QAction('&Spectral Indices Calculator...', self)
        indices_action.setShortcut('Ctrl+I')
        indices_action.triggered.connect(self._show_indices_dialog)
        analysis_menu.addAction(indices_action)

        analysis_menu.addSeparator()

        sam_action = QtWidgets.QAction('&Spectral Angle Mapper...', self)
        sam_action.triggered.connect(self._show_sam_dialog)
        analysis_menu.addAction(sam_action)

        whitened_action = QtWidgets.QAction('&Whitened Similarity...', self)
        whitened_action.triggered.connect(self._show_whitened_similarity_dialog)
        analysis_menu.addAction(whitened_action)
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
        about_action = QtWidgets.QAction('&About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def _create_toolbar(self):
        """Create application toolbar."""
        toolbar = self.addToolBar('Main')
        icon_size = self.config.get('ui.toolbar_icon_size', 24)
        toolbar.setIconSize(QtCore.QSize(icon_size, icon_size))
        
        # File operations
        open_action = toolbar.addAction('Open')
        open_action.triggered.connect(self._open_file_dialog)
        
        toolbar.addSeparator()
        
        # Interleave override controls
        toolbar.addWidget(QtWidgets.QLabel('Interleave:'))
        self.interleave_combo = QtWidgets.QComboBox()
        self.interleave_combo.addItems(['Auto', 'BSQ', 'BIL', 'BIP'])
        self.interleave_combo.setToolTip('Override interleave format (requires reload)')
        toolbar.addWidget(self.interleave_combo)
        
        reload_action = toolbar.addAction('Reload')
        reload_action.setToolTip('Reload current file with selected interleave')
        reload_action.triggered.connect(self._reload_with_interleave)
        
        toolbar.addSeparator()
        
        # View operations
        zoom_fit_action = toolbar.addAction('Fit')
        zoom_fit_action.triggered.connect(lambda: self.image_view._zoom_fit() if self.image_view else None)
        
        zoom_in_action = toolbar.addAction('Zoom In')
        zoom_in_action.triggered.connect(lambda: self.image_view._zoom_in() if self.image_view else None)
        
        zoom_out_action = toolbar.addAction('Zoom Out')
        zoom_out_action.triggered.connect(lambda: self.image_view._zoom_out() if self.image_view else None)
        
        toolbar.addSeparator()
        
        # File Manager
        file_manager_action = toolbar.addAction('Files')
        file_manager_action.setCheckable(True)
        file_manager_action.setChecked(False)
        file_manager_action.setToolTip('Toggle File Manager')
        file_manager_action.triggered.connect(self._toggle_file_manager)
        
        # Header viewer
        header_action = toolbar.addAction('Header')
        header_action.setToolTip('View header file contents')
        header_action.triggered.connect(self._show_header_viewer)
        
        toolbar.addSeparator()
        
        # ROI operations
        roi_action = toolbar.addAction('ROI Mode')
        roi_action.triggered.connect(lambda: self.image_view._show_roi_type_menu() if self.image_view else None)
        
        clear_roi_action = toolbar.addAction('Clear ROIs')
        clear_roi_action.triggered.connect(self._clear_all_rois)
        
        # Add spacer to push logo to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Add app title to toolbar
        self._add_logo_to_toolbar(toolbar)

    def _add_logo_to_toolbar(self, toolbar: QtWidgets.QToolBar):
        """Add application title to toolbar."""
        try:
            # Text label for app branding
            logo_label = QtWidgets.QLabel("Hyperspectral Viewer")
            logo_label.setStyleSheet("""
                QLabel {
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    background-color: transparent;
                    margin: 2px;
                    padding: 2px;
                }
            """)
            logo_label.setToolTip("Hyperspectral Data Viewer")
            toolbar.addWidget(logo_label)

        except Exception as e:
            print(f"Could not add toolbar label: {e}")
        
    def _create_floating_overview(self):
        """Create floating overview widget that can be moved and resized."""
        # Create a custom floating widget class
        class FloatingOverview(QtWidgets.QWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowFlags(QtCore.Qt.Widget)  # Make it a regular widget, not a window
                self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
                
                # Make it semi-transparent with minimal styling
                self.setStyleSheet("""
                    FloatingOverview {
                        background-color: rgba(40, 40, 40, 200);
                        border: 1px solid #555;
                        border-radius: 5px;
                    }
                """)
                
                # Set initial size and position
                self.resize(250, 200)
                self.setMinimumSize(150, 120)
                
                # Variables for dragging
                self.dragging = False
                self.drag_start_position = QtCore.QPoint()
                
                # Variables for resizing
                self.resizing = False
                self.resize_start_position = QtCore.QPoint()
                self.resize_start_size = QtCore.QSize()
                
                self._setup_ui()
                
            def _setup_ui(self):
                layout = QtWidgets.QVBoxLayout(self)
                layout.setContentsMargins(5, 5, 5, 5)
                layout.setSpacing(0)
                
                # Title bar with just close button (no text)
                title_layout = QtWidgets.QHBoxLayout()
                title_layout.setContentsMargins(5, 0, 5, 0)
                
                # Spacer to push close button to the right
                title_layout.addStretch()
                
                # Close button
                close_button = QtWidgets.QPushButton("✕")
                close_button.setFixedSize(16, 16)
                close_button.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(200, 60, 60, 180);
                        border: none;
                        border-radius: 8px;
                        color: white;
                        font-weight: bold;
                        font-size: 10px;
                    }
                    QPushButton:hover {
                        background-color: rgba(220, 80, 80, 200);
                    }
                """)
                close_button.clicked.connect(self.hide)
                title_layout.addWidget(close_button)
                
                # Title container (draggable area)
                title_container = QtWidgets.QWidget()
                title_container.setLayout(title_layout)
                title_container.setStyleSheet("background-color: rgba(60, 60, 60, 180);")
                title_container.setFixedHeight(20)
                layout.addWidget(title_container)
                
                # Store reference for dragging
                self.title_bar = title_container
                
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
                        # Start resizing
                        self.resizing = True
                        self.resize_start_position = event.globalPos()
                        self.resize_start_size = self.size()
                        self.setCursor(QtCore.Qt.SizeFDiagCursor)
                    elif self.title_bar.geometry().contains(event.pos()):
                        # Start dragging from title bar
                        self.dragging = True
                        self.drag_start_position = event.globalPos() - self.frameGeometry().topLeft()
                        self.setCursor(QtCore.Qt.ClosedHandCursor)
                        
            def mouseMoveEvent(self, event):
                if self.resizing and event.buttons() == QtCore.Qt.LeftButton:
                    # Resize the widget
                    diff = event.globalPos() - self.resize_start_position
                    new_size = QtCore.QSize(
                        max(150, self.resize_start_size.width() + diff.x()),
                        max(120, self.resize_start_size.height() + diff.y())
                    )
                    self.resize(new_size)
                elif self.dragging and event.buttons() == QtCore.Qt.LeftButton:
                    # Move the widget
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
                
                painter.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150), 1))
                for i in range(3):
                    y_offset = i * 4 + 5
                    painter.drawLine(
                        grip_rect.x() + 5, grip_rect.y() + y_offset,
                        grip_rect.x() + grip_size - 5, grip_rect.y() + y_offset
                    )
        
        # Create the floating overview widget as child of image_view
        self.floating_overview = FloatingOverview(self.image_view)
        
        # Add the PyQtGraph ImageView to the floating widget
        layout = self.floating_overview.layout()
        
        # PyQtGraph ImageView for proper overview with zoom box
        self.overview_view = pg.ImageView()
        self.overview_view.ui.roiBtn.hide()
        self.overview_view.ui.menuBtn.hide()
        self.overview_view.ui.histogram.hide()  # Hide contrast scaling controls
        
        # Configure overview view
        overview_viewbox = self.overview_view.getView()
        overview_viewbox.setAspectLocked(True)
        # Disable direct pan/zoom but keep mouse events for zoom box
        overview_viewbox.setMouseEnabled(x=False, y=False)
        overview_viewbox.setMenuEnabled(False)  # Disable context menu
        # Disable wheel zoom
        overview_viewbox.wheelEvent = lambda ev: None
        
        # Add click-to-center functionality
        overview_viewbox.scene().sigMouseClicked.connect(self._on_overview_clicked)
        
        # Create zoom box (ROI rectangle) - more visible styling
        self.zoom_box = pg.RectROI([0, 0], [50, 50], 
                                   pen=pg.mkPen('cyan', width=2),
                                   hoverPen=pg.mkPen('yellow', width=2))
        
        # Make it movable but not resizable by removing resize handles
        # Clear all default handles first (removeHandle expects indices, so go backwards)
        while len(self.zoom_box.handles) > 0:
            self.zoom_box.removeHandle(0)
        
        # Make the whole box draggable without visible handles
        self.zoom_box.translatable = True
        
        # Connect signal for when zoom box is moved
        self.zoom_box.sigRegionChanged.connect(self._on_zoom_box_changed)
        
        layout.addWidget(self.overview_view)
        
        # Add zoom box to the overview view  
        overview_viewbox = self.overview_view.getView()
        # Remove any existing zoom boxes first to prevent duplicates
        for item in overview_viewbox.allChildren():
            if isinstance(item, pg.RectROI):
                overview_viewbox.removeItem(item)
        try:
            overview_viewbox.addItem(self.zoom_box)
        except Exception as e:
            pass
        
        # Position it in bottom-left corner initially  
        # We'll position it after the parent is shown, using a timer
        QtCore.QTimer.singleShot(100, self._position_overview_bottom_left)
        self.floating_overview.show()
        
        return self.floating_overview
    
    def _position_overview_bottom_left(self):
        """Position the overview in the bottom-left corner of the image view."""
        if hasattr(self, 'floating_overview') and hasattr(self, 'image_view') and self.floating_overview:
            try:
                # Get the image view widget geometry relative to the main window
                image_widget_pos = self.image_view.mapTo(self, QtCore.QPoint(0, 0))
                image_size = self.image_view.size()
                overview_size = self.floating_overview.size()
                
                # Position at bottom-left with margins (higher up from bottom)
                margin = 15
                bottom_margin = 50  # More space from bottom
                x = image_widget_pos.x() + margin
                y = image_widget_pos.y() + image_size.height() - overview_size.height() - bottom_margin
                
                # Ensure it's within the main window bounds
                main_window_size = self.size()
                x = max(margin, min(x, main_window_size.width() - overview_size.width() - margin))
                y = max(margin, min(y, main_window_size.height() - overview_size.height() - margin))
                
                self.floating_overview.move(x, y)
            except Exception as e:
                # Fallback to simple positioning
                self.floating_overview.move(15, self.height() - 250)
        
    def _create_tabbed_panel(self) -> QtWidgets.QWidget:
        """Create tabbed panel with ROI management and base map tabs."""
        tab_widget = QtWidgets.QTabWidget()

        # ROI Management tab
        roi_tab = self._create_roi_panel()
        tab_widget.addTab(roi_tab, "ROI Management")

        # Base Map tab
        base_map_tab = self._create_base_map_panel()
        tab_widget.addTab(base_map_tab, "Base Map")

        return tab_widget

    def _create_roi_panel(self) -> QtWidgets.QWidget:
        """Create ROI management panel."""
        roi_widget = QtWidgets.QWidget()
        # Remove maximum height to let splitter control it
        roi_widget.setMinimumHeight(100)

        layout = QtWidgets.QVBoxLayout(roi_widget)

        # ROI list
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(100)
        layout.addWidget(self.roi_list)

        # ROI controls
        controls_layout = QtWidgets.QHBoxLayout()

        export_roi_btn = QtWidgets.QPushButton("Export ROIs")
        export_roi_btn.clicked.connect(self._export_rois)
        controls_layout.addWidget(export_roi_btn)

        import_roi_btn = QtWidgets.QPushButton("Import ROIs")
        import_roi_btn.clicked.connect(self._import_rois)
        controls_layout.addWidget(import_roi_btn)

        layout.addLayout(controls_layout)

        return roi_widget

    def _create_base_map_panel(self) -> QtWidgets.QWidget:
        """Create base map panel for loading and managing base maps."""
        base_map_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(base_map_widget)

        # Base map source selection
        source_group = QtWidgets.QGroupBox("Base Map Source")
        source_layout = QtWidgets.QVBoxLayout()

        # Tile provider dropdown
        provider_layout = QtWidgets.QHBoxLayout()
        provider_layout.addWidget(QtWidgets.QLabel("Provider:"))
        self.base_map_provider_combo = QtWidgets.QComboBox()
        self.base_map_provider_combo.addItems([
            "OpenStreetMap",
            "OpenTopoMap",
            "Stamen Terrain",
            "Stamen Toner",
            "CartoDB Positron",
            "CartoDB Dark Matter",
            "ESRI World Imagery",
            "ESRI World Street Map",
            "Custom Image File..."
        ])
        self.base_map_provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self.base_map_provider_combo)
        source_layout.addLayout(provider_layout)

        # Load button
        self.load_base_map_btn = QtWidgets.QPushButton("Load Tile Base Map")
        self.load_base_map_btn.clicked.connect(self._load_tile_base_map)
        source_layout.addWidget(self.load_base_map_btn)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Status/Info
        self.base_map_label = QtWidgets.QLabel("No base map loaded")
        self.base_map_label.setWordWrap(True)
        layout.addWidget(self.base_map_label)

        # Coordinate input group
        coord_group = QtWidgets.QGroupBox("Map Coordinates (Optional)")
        coord_layout = QtWidgets.QVBoxLayout()

        coord_info = QtWidgets.QLabel("Enter coordinates if your hyperspectral data has geolocation:")
        coord_info.setWordWrap(True)
        coord_info.setStyleSheet("font-size: 10px; color: #FFE4F5;")
        coord_layout.addWidget(coord_info)

        # Center coordinates
        center_layout = QtWidgets.QHBoxLayout()
        center_layout.addWidget(QtWidgets.QLabel("Center Lat:"))
        self.center_lat_input = QtWidgets.QDoubleSpinBox()
        self.center_lat_input.setRange(-90, 90)
        self.center_lat_input.setDecimals(6)
        self.center_lat_input.setValue(0.0)
        center_layout.addWidget(self.center_lat_input)
        center_layout.addWidget(QtWidgets.QLabel("Lon:"))
        self.center_lon_input = QtWidgets.QDoubleSpinBox()
        self.center_lon_input.setRange(-180, 180)
        self.center_lon_input.setDecimals(6)
        self.center_lon_input.setValue(0.0)
        center_layout.addWidget(self.center_lon_input)
        coord_layout.addLayout(center_layout)

        # Zoom level
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(QtWidgets.QLabel("Zoom Level:"))
        self.zoom_level_input = QtWidgets.QSpinBox()
        self.zoom_level_input.setRange(1, 19)
        self.zoom_level_input.setValue(10)
        zoom_layout.addWidget(self.zoom_level_input)
        zoom_layout.addStretch()
        coord_layout.addLayout(zoom_layout)

        coord_group.setLayout(coord_layout)
        layout.addWidget(coord_group)

        # Overlay controls
        overlay_group = QtWidgets.QGroupBox("Overlay Settings")
        overlay_layout = QtWidgets.QVBoxLayout()

        # Opacity slider
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(QtWidgets.QLabel("Opacity:"))
        self.base_map_opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.base_map_opacity_slider.setMinimum(0)
        self.base_map_opacity_slider.setMaximum(100)
        self.base_map_opacity_slider.setValue(50)
        self.base_map_opacity_slider.setEnabled(False)
        self.base_map_opacity_slider.valueChanged.connect(self._update_base_map_opacity)
        opacity_layout.addWidget(self.base_map_opacity_slider)
        self.base_map_opacity_label = QtWidgets.QLabel("50%")
        opacity_layout.addWidget(self.base_map_opacity_label)
        overlay_layout.addLayout(opacity_layout)

        # Show/hide toggle
        self.base_map_visible_checkbox = QtWidgets.QCheckBox("Show Base Map")
        self.base_map_visible_checkbox.setChecked(True)
        self.base_map_visible_checkbox.setEnabled(False)
        self.base_map_visible_checkbox.stateChanged.connect(self._toggle_base_map_visibility)
        overlay_layout.addWidget(self.base_map_visible_checkbox)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # Clear base map button
        clear_base_map_btn = QtWidgets.QPushButton("Clear Base Map")
        clear_base_map_btn.clicked.connect(self._clear_base_map)
        layout.addWidget(clear_base_map_btn)

        # Add stretch to push everything to the top
        layout.addStretch()

        # Initialize base map data
        self.base_map_data = None
        self.base_map_item = None
        self.base_map_path = None
        self.tile_providers = self._get_tile_providers()

        return base_map_widget
        
    def _setup_connections(self):
        """Setup signal connections between components."""
        self._setup_image_view_connections(self.image_view)
        
    def _setup_image_view_connections(self, image_view):
        """Setup connections for an image view instance."""
        if image_view and self.spectrum_plot:
            # Pixel selection updates spectrum
            image_view.pixel_selected.connect(self._on_pixel_selected)
            # New signal with view ID
            if hasattr(image_view, 'pixel_selected_with_id'):
                image_view.pixel_selected_with_id.connect(self._on_pixel_selected_with_id)

            # Spectrum collection
            if hasattr(image_view, 'spectrum_collect_requested'):
                image_view.spectrum_collect_requested.connect(self._on_spectrum_collect_requested)

            # ROI selection
            image_view.roi_selected.connect(self._on_roi_selected)
            
            # Zoom changes update overview zoom box
            image_view.zoom_changed.connect(self._on_main_view_changed)
            
            # RGB band changes
            image_view.rgb_bands_changed.connect(self._update_rgb_display)
            
            # Default RGB request
            image_view.default_rgb_requested.connect(self._set_true_color_rgb_bands)
            
            # Frame view line changes
            image_view.line_changed.connect(self._update_frame_display)
            
            # Band selection changes
            image_view.band_changed.connect(self._on_band_changed)
            
            # Dataset changes (tab switching)
            if hasattr(image_view, 'dataset_changed'):
                image_view.dataset_changed.connect(self._on_dataset_changed)
            
        # Setup file manager connections
        self._setup_dataset_connections()
        
    def _disconnect_image_view_connections(self, image_view):
        """Disconnect connections for an image view instance."""
        if image_view and self.spectrum_plot:
            try:
                # Disconnect all the signals we connected
                image_view.pixel_selected.disconnect(self._on_pixel_selected)
                image_view.roi_selected.disconnect(self._on_roi_selected) 
                image_view.zoom_changed.disconnect(self._on_main_view_changed)
                image_view.rgb_bands_changed.disconnect(self._update_rgb_display)
                image_view.default_rgb_requested.disconnect(self._set_true_color_rgb_bands)
                image_view.line_changed.disconnect(self._update_frame_display)
                image_view.band_changed.disconnect(self._on_band_changed)
                if hasattr(image_view, 'dataset_changed'):
                    image_view.dataset_changed.disconnect(self._on_dataset_changed)
            except Exception as e:
                # Some signals might not be connected, ignore errors
                pass
        
    def _get_active_image_view(self):
        """Get the currently active image view (single or focused split view)."""
        if self.split_mode and self.split_views:
            # Return the focused split view
            return self.focused_split_view if self.focused_split_view else self.split_views[0]
        return self.image_view
    
    def eventFilter(self, obj, event):
        """Event filter to track focus in split views."""
        # Track focus when a split view is clicked or receives focus
        if self.split_mode and obj in self.split_views:
            if event.type() in [QtCore.QEvent.MouseButtonPress, QtCore.QEvent.FocusIn]:
                self.focused_split_view = obj
        return super().eventFilter(obj, event)
            
    def _load_settings(self):
        """Load application settings."""
        # Load recent files from config
        self.recent_files = self.config.get('recent_files', [])
        self._update_recent_files_menu()
        
    def load_data_file(self, filename: str) -> bool:
        """
        Load hyperspectral data file (ENVI or EMIT format).
        
        Args:
            filename: Path to ENVI data file (.bsq, .bil, .bip, .hdr) or EMIT file (.nc)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(filename):
                QtWidgets.QMessageBox.critical(self, "Error", 
                    f"File not found: {filename}")
                return False
            
            # Determine file type and validate accordingly
            if filename.lower().endswith('.nc') and 'EMIT' in os.path.basename(filename).upper():
                # EMIT NetCDF file - no additional validation needed
                data_path = filename
                print(f"Detected EMIT file: {filename}")
            else:
                # ENVI file - validate file pair
                is_valid, data_path, header_path = validate_envi_file_pair(filename)
                
                if not is_valid:
                    QtWidgets.QMessageBox.critical(self, "Error", 
                        f"Invalid ENVI file or missing header: {filename}")
                    return False
                print(f"Detected ENVI file: {data_path}")
                
            # Create a new DataHandler for this file (don't reuse self.data_handler)
            new_data_handler = DataHandler()
            
            # Load data (DataHandler now handles both ENVI and EMIT)
            load_mode = self.config.get('data_loading.default_load_mode', 'memmap')
            load_to_ram = (load_mode == 'ram')
            
            success = new_data_handler.load_envi_data(data_path, load_to_ram)
            
            if not success:
                QtWidgets.QMessageBox.critical(self, "Error", 
                    f"Failed to load data file: {filename}")
                return False
                
            # Update self.data_handler to the new one (for backward compatibility)
            self.data_handler = new_data_handler
                
            # Update UI
            self._update_after_data_load()
            
            # Update recent files
            self.current_file = filename
            self._add_to_recent_files(filename)
            
            # Update window title
            self.setWindowTitle(f"Info_i - {os.path.basename(filename)}")
            
            # Show success message
            if hasattr(self, 'status_bar'):
                info = self.data_handler.get_info()
                
                # Format status message based on file type
                file_format = info.get('file_format', 'unknown')
                if file_format == 'emit':
                    status_msg = (
                        f"Loaded EMIT: {info['shape'][1]}x{info['shape'][0]}x{info['shape'][2]} "
                        f"({info.get('product_level', 'Unknown')})"
                    )
                else:
                    # Handle interleave format safely for ENVI files
                    interleave = info.get('interleave', 'unknown')
                    if isinstance(interleave, str):
                        interleave_str = interleave.upper()
                    elif isinstance(interleave, int):
                        # Convert numeric interleave codes to strings
                        interleave_map = {0: 'BSQ', 1: 'BIL', 2: 'BIP'}
                        interleave_str = interleave_map.get(interleave, f'INTERLEAVE_{interleave}')
                    else:
                        interleave_str = str(interleave).upper()
                    
                    status_msg = (
                        f"Loaded ENVI: {info['shape'][1]}x{info['shape'][0]}x{info['shape'][2]} "
                        f"({interleave_str})"
                    )
                
                self.status_bar.showMessage(status_msg)
                
            return True
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Detailed error information:")
            print(error_details)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
            return False
            
    def _load_file_multi_dataset(self, filename: str) -> bool:
        """
        Load a file into the multi-dataset system.
        
        Args:
            filename: Path to the hyperspectral data file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load data using the original method
            success = self.load_data_file(filename)
            
            if success:
                # Add to DataManager
                dataset_name = os.path.splitext(os.path.basename(filename))[0]
                source_info = {
                    'type': 'original',
                    'file_path': filename,
                    'loaded_at': filename
                }
                
                # Add to data manager
                self.data_manager.add_dataset(dataset_name, self.data_handler, source_info)
                
                # Add tab to tabbed image view
                self.image_view.add_dataset_tab(dataset_name)
                
                # Always update file manager (whether visible or not)
                if self.file_manager:
                    self.file_manager.refresh_datasets()
                    
                # Connections are already set up in _setup_connections(), no need to repeat
                
                # Update window title to show the loaded file
                self.setWindowTitle(f"Info_i - {os.path.basename(filename)} (+ {len(self.data_manager.datasets)} datasets)")
                
                # Update status bar
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(f"Loaded {dataset_name} into multi-dataset system")
                    
                return True
            
            return False
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load file into multi-dataset system: {e}")
            return False
            
    def _setup_dataset_connections(self):
        """Setup connections for file manager events."""
        if self.file_manager:
            # Connect file manager signals
            self.file_manager.dataset_selected.connect(self._on_dataset_selected)
            self.file_manager.dataset_activated.connect(self._on_dataset_activated)
            
    def _on_dataset_selected(self, dataset_name: str):
        """Handle dataset selection from file manager."""
        # Set as active in data manager
        self.data_manager.set_active_dataset(dataset_name)
        
    def _on_dataset_activated(self, dataset_name: str):
        """Handle dataset activation (double-click) from file manager."""
        # Add or activate tab for this dataset
        success = self.image_view.add_dataset_tab(dataset_name)
        if not success:
            # If dataset not in tabbed view, try to set as active tab
            self.image_view.set_active_tab(dataset_name)
            
    def _update_after_data_load(self):
        """Update UI components after successful data loading."""
        # Set wavelengths for spectrum plot
        if self.spectrum_plot:
            self.spectrum_plot.set_wavelengths(self.data_handler.wavelengths)
            # Also update RGB band indicators if bands are already set
            active_view = self._get_active_image_view()
            if active_view:
                r, g, b = active_view.get_rgb_bands()
                self.spectrum_plot.update_rgb_bands(r, g, b)
            
        # Update ROI manager
        self.roi_manager.set_data_handler(self.data_handler)
        
        # Set band limits in active image view
        active_view = self._get_active_image_view()
        if active_view:
            active_view.set_band_limits(self.data_handler.shape[2])
            
            # Set line limits for frame view (use number of rows/lines)
            if hasattr(active_view, 'set_line_limits_for_frame_view'):
                active_view.set_line_limits_for_frame_view(
                    self.data_handler.shape[0]  # Number of lines (rows)
                )
            
            # Set optimal RGB bands
            if self.data_handler.wavelengths is not None:
                r, g, b = estimate_optimal_rgb_bands(self.data_handler.wavelengths)
                active_view.set_rgb_bands(r, g, b)
            else:
                # Use config defaults
                default_bands = self.config.get('display.default_rgb_bands', [29, 19, 9])
                active_view.set_rgb_bands(*default_bands)
                
        # If in split mode, refresh dataset combos in both views
        if self.split_mode and self.split_views:
            for split_view in self.split_views:
                if hasattr(split_view, '_refresh_dataset_combo'):
                    split_view._refresh_dataset_combo()
                
        # Update RGB display
        self._update_rgb_display()
        
        # Force immediate overview update and then again with a small delay
        self._update_overview()
        QtCore.QTimer.singleShot(50, self._update_overview)
        
    def _update_rgb_display(self):
        """Update RGB image display and overview."""
        # Try to get the sender of the signal, otherwise use active view
        sender = self.sender()
        if sender and isinstance(sender, TabbedImageView):
            active_view = sender
        else:
            active_view = self._get_active_image_view()
            
        if not active_view:
            return
            
        try:
            r, g, b = active_view.get_rgb_bands()
            
            # Get stretch percentage from current image view
            stretch_percent = 2.0  # Default
            if hasattr(active_view, 'get_stretch_percent'):
                stretch_percent = active_view.get_stretch_percent()
            
            # Get the dataset for the active view
            # Get the current dataset name from the active view
            dataset_name = None
            if hasattr(active_view, 'get_current_dataset_name'):
                dataset_name = active_view.get_current_dataset_name()
                if dataset_name:
                    # Extract base name if it has a counter
                    dataset_name = dataset_name.split(' (')[0]
            
            if not dataset_name:
                # Fallback to active dataset
                current_dataset = self.data_manager.get_active_dataset()
            else:
                current_dataset = self.data_manager.get_dataset(dataset_name)
                
            if not current_dataset or not current_dataset.is_loaded:
                return
            
            # Get no data value from active view if available
            no_data_value = None
            if hasattr(active_view, 'get_no_data_value') and dataset_name:
                no_data_value = active_view.get_no_data_value(dataset_name)
            
            rgb_image = current_dataset.get_rgb_composite(r, g, b, stretch_percent, no_data_value)
            
            if rgb_image is not None:
                # Apply the same stretching that's already in rgb_image
                active_view.set_image(rgb_image)
                # Also update overview to show same RGB bands
                self._update_overview()
                
            # Update spectrum plot RGB band indicators
            if self.spectrum_plot:
                self.spectrum_plot.update_rgb_bands(r, g, b)
                
        except Exception as e:
            print(f"Error updating RGB display: {e}")
            
    def _set_true_color_rgb_bands(self):
        """Set RGB bands to true color (visible spectrum) or well-spaced default."""
        # Try to get the sender of the signal, otherwise use active view
        sender = self.sender()
        if sender and isinstance(sender, TabbedImageView):
            active_view = sender
        else:
            active_view = self._get_active_image_view()
            
        if not active_view:
            return
            
        # Get the dataset for the active view
        dataset_name = None
        if hasattr(active_view, 'get_current_dataset_name'):
            dataset_name = active_view.get_current_dataset_name()
            if dataset_name:
                # Extract base name if it has a counter
                dataset_name = dataset_name.split(' (')[0]
        
        if not dataset_name:
            # Fallback to active dataset
            current_dataset = self.data_manager.get_active_dataset()
        else:
            current_dataset = self.data_manager.get_dataset(dataset_name)
            
        if not current_dataset or not current_dataset.is_loaded:
            return
            
        try:
            # Get wavelengths from current dataset
            wavelengths = current_dataset.wavelengths
            
            # Get true color RGB bands
            r, g, b = get_true_color_rgb_bands(wavelengths)
            
            # Set the bands in the UI
            active_view.set_rgb_bands(r, g, b)
            
            # Update the display
            self._update_rgb_display()
            
            # Show message in status bar
            if hasattr(self, 'status_bar') and wavelengths is not None:
                r_wl = wavelengths[r] if r < len(wavelengths) else 0
                g_wl = wavelengths[g] if g < len(wavelengths) else 0
                b_wl = wavelengths[b] if b < len(wavelengths) else 0
                
                # Determine if true color or fallback was used
                has_blue = np.min(wavelengths) <= 500
                has_green = np.min(wavelengths) <= 580 and np.max(wavelengths) >= 520
                has_red = np.max(wavelengths) >= 620
                
                if has_blue and has_green and has_red:
                    self.status_bar.showMessage(f"True Color RGB set: R={r}({r_wl:.1f}nm), G={g}({g_wl:.1f}nm), B={b}({b_wl:.1f}nm)")
                else:
                    self.status_bar.showMessage(f"Well-spaced RGB set: R={r}({r_wl:.1f}nm), G={g}({g_wl:.1f}nm), B={b}({b_wl:.1f}nm)")
                    
        except Exception as e:
            print(f"Error setting true color RGB bands: {e}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("Error setting default RGB bands")
                
    def _update_frame_display(self, line_index: int):
        """Update frame view display with spatial-spectral cross-section for the specified line."""
        print(f"[FRAME_VIEW] _update_frame_display called: line_index={line_index}")
        # Try to get the sender of the signal, otherwise use active view
        sender = self.sender()
        print(f"[FRAME_VIEW] Sender: {sender}, type: {type(sender)}")

        # Get the actual ImageView from TabbedImageView
        tabbed_view = None
        if sender and isinstance(sender, TabbedImageView):
            tabbed_view = sender
            print(f"[FRAME_VIEW] Sender is TabbedImageView")
        else:
            tabbed_view = self._get_active_image_view()
            print(f"[FRAME_VIEW] Got TabbedImageView from _get_active_image_view")

        if not tabbed_view:
            print(f"[FRAME_VIEW] No tabbed view")
            return

        # Get the current ImageView from the TabbedImageView
        if hasattr(tabbed_view, 'get_current_image_view'):
            active_view = tabbed_view.get_current_image_view()
            print(f"[FRAME_VIEW] Got current ImageView: {active_view}")
        else:
            print(f"[FRAME_VIEW] TabbedImageView doesn't have get_current_image_view method")
            return

        if not self.data_handler.is_loaded:
            print(f"[FRAME_VIEW] Data handler not loaded")
            return

        if not active_view:
            print(f"[FRAME_VIEW] No active ImageView")
            return

        print(f"[FRAME_VIEW] Proceeding with data_handler and ImageView")

        try:
            # Extract spectra for all spatial positions along the specified line
            line_spectra = self.data_handler.extract_line_spectra(line_index)
            print(f"[FRAME_VIEW] Line spectra extracted: {line_spectra.shape if line_spectra is not None else None}")

            if line_spectra is not None:
                # Display the spatial-spectral heatmap
                if hasattr(active_view, 'set_spatial_spectral_heatmap'):
                    print(f"[FRAME_VIEW] Calling set_spatial_spectral_heatmap with shape {line_spectra.shape}")
                    active_view.set_spatial_spectral_heatmap(line_spectra, line_index)
                    print("[FRAME_VIEW] Heatmap display method called successfully")
                else:
                    print(f"[FRAME_VIEW] Error: ImageView does not have set_spatial_spectral_heatmap method")

                # Update status bar with line info
                if hasattr(self, 'status_bar'):
                    positions, bands = line_spectra.shape
                    self.status_bar.showMessage(f"Frame view: Line {line_index} | {positions} positions × {bands} bands")
            else:
                print("Error: line_spectra is None")
                    
        except Exception as e:
            print(f"Error updating frame display for line {line_index}: {e}")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Error displaying line {line_index}")
                
    def _on_band_changed(self, band_index: int, channel: str):
        """Handle band selection changes from image view."""
        if self.spectrum_plot:
            self.spectrum_plot.update_current_band(band_index, channel)
            
    def _on_dataset_changed(self, dataset_name: str):
        """Handle dataset changes (tab switching)."""
        # Only update if the signal came from the currently active/focused view
        sender = self.sender()
        if self.split_mode:
            # In split mode, only update if sender is the focused view
            if sender != self.focused_split_view:
                return
        else:
            # In single mode, only update if sender is the main view
            if sender != self.image_view:
                return
        
        # Update RGB display with the newly active dataset
        self._update_rgb_display()
        
        # Update spectrum plot wavelengths for the new dataset
        dataset = self.data_manager.get_dataset(dataset_name)
        if dataset and self.spectrum_plot:
            self.spectrum_plot.set_wavelengths(dataset.wavelengths)
            
        # Update ROI manager with new dataset
        if dataset:
            self.roi_manager.set_data_handler(dataset)
            
    def _update_overview(self):
        """Update overview window."""
        if not hasattr(self, 'overview_view'):
            return
            
        if not self.data_handler or not self.data_handler.is_loaded:
            return
            
        try:
            # Create RGB composite for overview using same bands as main view
            r, g, b = self.image_view.get_rgb_bands() if self.image_view else (29, 19, 9)
            
            # Get no data value from current dataset
            no_data_value = None
            if self.image_view and hasattr(self.image_view, 'get_no_data_value'):
                current_dataset = self.image_view.get_current_dataset_name()
                if current_dataset:
                    current_dataset = current_dataset.split(' (')[0]  # Remove counter
                    no_data_value = self.image_view.get_no_data_value(current_dataset)
            
            # Create RGB composite with selected bands
            rgb_image = self.data_handler.get_rgb_composite(r, g, b, 2.0, no_data_value)
            
            if rgb_image is not None:
                # RGB composite generated successfully
                
                # Apply the original working transformation sequence
                flipped = np.fliplr(rgb_image)
                rotated = np.rot90(flipped, k=-1)
                flipped_again = np.fliplr(rotated)
                display_image = np.rot90(flipped_again, k=-2)
                
                # Get the current levels from the main image view to inherit scaling
                main_image_levels = None
                try:
                    current_view = self.image_view.get_current_image_view()
                    if current_view and hasattr(current_view, 'image_widget'):
                        main_image_levels = current_view.image_widget.getLevels()
                except Exception as e:
                    print(f"Could not get main image levels: {e}")
                
                # Set image - inherit levels from main view or use auto levels as fallback
                if main_image_levels is not None:
                    self.overview_view.setImage(display_image, autoRange=True, levels=main_image_levels)
                else:
                    self.overview_view.setImage(display_image, autoRange=True, autoLevels=True)
                
                # Overview image updated
                
                # Force a repaint to ensure immediate display
                self.overview_view.update()
                if hasattr(self.overview_view, 'imageItem') and self.overview_view.imageItem:
                    self.overview_view.imageItem.update()
                QtWidgets.QApplication.processEvents()
            
            # Whether or not RGB composite succeeds, ensure zoom box is properly managed
            # Ensure zoom box is added (only if not already there)
            view = self.overview_view.getView()
            if self.zoom_box not in view.items:
                try:
                    view.addItem(self.zoom_box)
                except:
                    pass
            
            # Update zoom box to show current main view bounds
            self._update_zoom_box_from_main_view()
            
            # Ensure overview is at fixed scale (show entire image)
            self.overview_view.autoRange()
            
            # Make sure zoom box is visible and properly positioned
            QtCore.QTimer.singleShot(100, self._update_zoom_box_from_main_view)
                
        except Exception as e:
            pass
            
    def _on_overview_clicked(self, event):
        """Handle clicks in overview window to center main view."""
        if not self.image_view or not hasattr(self, 'overview_view'):
            return
            
        try:
            # Get click position in overview coordinates
            overview_viewbox = self.overview_view.getView()
            click_pos = overview_viewbox.mapSceneToView(event.pos())
            center_x, center_y = click_pos.x(), click_pos.y()
            
            # Get current main view size to determine new range
            current_view = self.image_view.get_current_image_view()
            if current_view and hasattr(current_view, 'image_widget'):
                main_viewbox = current_view.image_widget.getView() 
                current_range = main_viewbox.viewRange()
                current_width = current_range[0][1] - current_range[0][0]
                current_height = current_range[1][1] - current_range[1][0]
                
                # Calculate new range centered on click position
                new_x_min = center_x - current_width / 2
                new_x_max = center_x + current_width / 2
                new_y_min = center_y - current_height / 2
                new_y_max = center_y + current_height / 2
                
                # Update main view
                main_viewbox.setRange(xRange=[new_x_min, new_x_max], 
                                    yRange=[new_y_min, new_y_max], padding=0)
            
        except Exception as e:
            pass
            
    def _on_pixel_selected_with_id(self, x: int, y: int, view_id: str):
        """Handle pixel selection with view ID from image view."""
        if not self.spectrum_plot:
            return
            
        # Get the dataset for the view that sent this signal
        sender = self.sender()
        dataset_name = None
        
        if sender and isinstance(sender, TabbedImageView):
            if hasattr(sender, 'get_current_dataset_name'):
                dataset_name = sender.get_current_dataset_name()
                if dataset_name:
                    # Extract base name if it has a counter
                    dataset_name = dataset_name.split(' (')[0]
        
        if not dataset_name:
            # Fallback to active dataset
            current_dataset = self.data_manager.get_active_dataset()
        else:
            current_dataset = self.data_manager.get_dataset(dataset_name)
            
        if not current_dataset or not current_dataset.is_loaded:
            return
            
        spectrum = current_dataset.get_pixel_spectrum(x, y)
        if spectrum is not None:
            # Get bad bands information for filtering
            active_bands = None
            if hasattr(sender, 'get_active_bands') and dataset_name:
                active_bands = sender.get_active_bands(dataset_name)
            
            # Pass the view_id to the spectrum plot for proper routing
            self.spectrum_plot.update_pixel_spectrum_with_id(spectrum, x, y, view_id, active_bands, dataset_name)
            
            # Update spectrum plot view selectors with current available views
            self._update_spectrum_view_selectors()
            
            # Update status bar
            if hasattr(self, 'status_bar'):
                metadata_str = self._get_instrument_metadata()
                pixel_info = f"Pixel ({x}, {y}) from {view_id}"
                if metadata_str:
                    status_message = f"{pixel_info} | {metadata_str}"
                else:
                    status_message = pixel_info
                self.status_bar.showMessage(status_message)
    
    def _on_pixel_selected(self, x: int, y: int):
        """Handle pixel selection from image view (legacy)."""
        if not self.spectrum_plot:
            return
            
        # Get the dataset for the view that sent this signal
        sender = self.sender()
        dataset_name = None
        
        if sender and isinstance(sender, TabbedImageView):
            if hasattr(sender, 'get_current_dataset_name'):
                dataset_name = sender.get_current_dataset_name()
                if dataset_name:
                    # Extract base name if it has a counter
                    dataset_name = dataset_name.split(' (')[0]
        
        if not dataset_name:
            # Fallback to active dataset
            current_dataset = self.data_manager.get_active_dataset()
        else:
            current_dataset = self.data_manager.get_dataset(dataset_name)
            
        if not current_dataset or not current_dataset.is_loaded:
            return
            
        spectrum = current_dataset.get_pixel_spectrum(x, y)
        if spectrum is not None:
            # Check if spectrum looks reasonable (not all zeros, has variation)
            is_valid_spectrum = self._validate_spectrum(spectrum)
            validation_status = "✓ Valid" if is_valid_spectrum else "✗ Suspicious"
            
            # Get bad bands information for filtering
            active_bands = None
            if hasattr(self.image_view, 'get_active_bands') and dataset_name:
                active_bands = self.image_view.get_active_bands(dataset_name)
            
            self.spectrum_plot.update_pixel_spectrum(spectrum, x, y, None, active_bands, dataset_name)
            
            # Update status bar with pixel info, validation, and metadata
            if hasattr(self, 'status_bar'):
                metadata_str = self._get_instrument_metadata()
                pixel_info = f"Pixel ({x}, {y}) - Range: [{spectrum.min():.4f}, {spectrum.max():.4f}] - {validation_status}"
                if metadata_str:
                    status_message = f"{pixel_info} | {metadata_str}"
                else:
                    status_message = pixel_info
                self.status_bar.showMessage(status_message)
        else:
            if hasattr(self, 'status_bar'):
                metadata_str = self._get_instrument_metadata()
                no_data_info = f"No spectrum data for pixel ({x}, {y})"
                if metadata_str:
                    status_message = f"{no_data_info} | {metadata_str}"
                else:
                    status_message = no_data_info
                self.status_bar.showMessage(status_message)

    def _on_spectrum_collect_requested(self, x: int, y: int):
        """Handle spectrum collection request from image view."""
        print(f"[MAIN] _on_spectrum_collect_requested called: x={x}, y={y}")

        if not self.spectrum_plot:
            print("[MAIN] No spectrum_plot available")
            return

        # Get the dataset for the view that sent this signal
        sender = self.sender()
        dataset_name = None

        if sender and isinstance(sender, TabbedImageView):
            if hasattr(sender, 'get_current_dataset_name'):
                dataset_name = sender.get_current_dataset_name()
                if dataset_name:
                    # Extract base name if it has a counter
                    dataset_name = dataset_name.split(' (')[0]

        if not dataset_name:
            # Fallback to active dataset
            current_dataset = self.data_manager.get_active_dataset()
            print(f"[MAIN] Using active dataset: {current_dataset}")
        else:
            current_dataset = self.data_manager.get_dataset(dataset_name)
            print(f"[MAIN] Using dataset: {dataset_name}")

        if not current_dataset or not current_dataset.is_loaded:
            print("[MAIN] No dataset loaded or dataset not loaded")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage("No dataset loaded for spectrum collection")
            return

        spectrum = current_dataset.get_pixel_spectrum(x, y)
        print(f"[MAIN] Got spectrum: {spectrum is not None}, shape: {spectrum.shape if spectrum is not None else 'None'}")

        if spectrum is not None:
            # Collect the spectrum
            source_file = getattr(current_dataset, 'filename', 'Unknown')
            print(f"[MAIN] Calling collect_spectrum, source_file={source_file}")
            spectrum_id = self.spectrum_plot.collect_spectrum(
                spectrum=spectrum, x=x, y=y, source_file=source_file
            )
            print(f"[MAIN] Collected spectrum_id: {spectrum_id}")

            if spectrum_id:
                # Update status bar with success message
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage(f"✓ Collected spectrum from pixel ({x}, {y})")

                # Switch to collected spectra tab to show the new spectrum
                if hasattr(self.spectrum_plot, 'tab_widget') and hasattr(self.spectrum_plot, 'collected_spectra_tab'):
                    for i in range(self.spectrum_plot.tab_widget.count()):
                        if self.spectrum_plot.tab_widget.widget(i) == self.spectrum_plot.collected_spectra_tab:
                            self.spectrum_plot.tab_widget.setCurrentIndex(i)
                            print(f"[MAIN] Switched to collected spectra tab at index {i}")
                            break
            else:
                print("[MAIN] collect_spectrum returned None")
                if hasattr(self, 'status_bar'):
                    self.status_bar.showMessage("Failed to collect spectrum")
        else:
            print("[MAIN] spectrum is None - could not get pixel spectrum")
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"No spectrum data available at pixel ({x}, {y})")

    def _update_spectrum_view_selectors(self):
        """Update spectrum plot view selectors with current available image views."""
        if not self.spectrum_plot:
            return
        
        available_views = []
        
        # Add main view tabs
        if self.image_view:
            main_views = self.image_view.get_all_tab_view_ids()
            for view_id, tab_label in main_views:
                available_views.append((view_id, f"Main: {tab_label}"))
        
        # Add split views if in split mode
        if self.split_mode and self.split_views:
            for i, split_view in enumerate(self.split_views):
                split_views_list = split_view.get_all_tab_view_ids()
                for view_id, tab_label in split_views_list:
                    available_views.append((view_id, f"Split {i+1}: {tab_label}"))
        
        # Update spectrum plot selectors
        self.spectrum_plot.update_view_selectors(available_views)
                
    def _get_instrument_metadata(self) -> str:
        """Get instrument metadata string for status bar display."""
        if not self.data_handler.is_loaded:
            return ""
            
        # Get basic info
        info = self.data_handler.get_info()
        
        # Determine instrument
        instrument = "Proprietary"
        if self.data_handler.file_type == 'emit':
            instrument = "EMIT"
        elif self.data_handler.header:
            # Check for known instruments in header
            sensor_type = self.data_handler.header.get('sensor_type', '').upper()
            if 'EMIT' in sensor_type:
                instrument = "EMIT"
            elif 'AVIRIS' in sensor_type:
                instrument = "AVIRIS"
            elif 'HYPERION' in sensor_type:
                instrument = "Hyperion"
            # Add more instrument detection as needed
        
        # Get spectral range
        spectral_range = "Unknown"
        num_bands = info.get('num_bands', 0)
        if self.data_handler.wavelengths is not None and len(self.data_handler.wavelengths) > 0:
            min_wl = int(np.round(self.data_handler.wavelengths[0]))
            max_wl = int(np.round(self.data_handler.wavelengths[-1]))
            spectral_range = f"{min_wl}-{max_wl}nm"
        
        # Get SNR (default to 400:1 if unknown)
        snr = "400:1"
        
        # Get spatial resolution (default to 10cm GSD if unknown)
        spatial_res = "10cm GSD"
        if self.data_handler.file_type == 'emit':
            spatial_res = self.data_handler.header.get('spatial_resolution', '60m GSD')
        elif self.data_handler.header:
            # Check for pixel size or spatial resolution in header
            if 'pixel_size' in self.data_handler.header:
                spatial_res = f"{self.data_handler.header['pixel_size']}m GSD"
            elif 'map_info' in self.data_handler.header:
                # Try to extract pixel size from map info
                map_info = self.data_handler.header['map_info']
                if isinstance(map_info, str) and ',' in map_info:
                    parts = map_info.split(',')
                    if len(parts) > 5:
                        try:
                            pixel_x = float(parts[5])
                            spatial_res = f"{pixel_x}m GSD"
                        except (ValueError, IndexError):
                            pass
        
        return f"Metadata: {instrument} | {spectral_range} | {num_bands} channels | SNR {snr} | {spatial_res}"

    def _validate_spectrum(self, spectrum: np.ndarray) -> bool:
        """Check if spectrum looks reasonable."""
        if spectrum is None or len(spectrum) == 0:
            return False
        
        # Check for all zeros
        if np.all(spectrum == 0):
            return False
        
        # Check for reasonable variation (std > 1% of mean)
        if np.std(spectrum) < 0.01 * np.abs(np.mean(spectrum)):
            return False
        
        # Check for reasonable range (not all negative, not impossibly high)
        if np.all(spectrum < 0) or np.any(spectrum > 10):  # Adjust range based on your data
            return False
        
        return True
            
    def _on_roi_selected(self, roi_data):
        """Handle ROI selection from image view."""        
        if not self.roi_manager or not self.spectrum_plot:
            print("ERROR: roi_manager or spectrum_plot not available")
            return
        
        # Extract data from the new format
        if isinstance(roi_data, dict) and 'roi_definition' in roi_data:
            roi_definition = roi_data['roi_definition']
            create_new_tab = roi_data.get('create_new_tab', True)
        else:
            # Backward compatibility
            roi_definition = roi_data
            create_new_tab = True
            
        # Generate ROI ID
        roi_id = f"roi_{len(self.roi_manager.rois) + 1}"
        
        # Add to ROI manager
        success = self.roi_manager.add_roi(roi_id, roi_definition)
        
        if success:
            # Update ROI list
            self._update_roi_list()
            
            # Get ROI stats
            roi_stats = self.roi_manager.get_roi_stats(roi_id)
            
            if roi_stats:
                if create_new_tab:
                    # Create new ROI tab
                    self.spectrum_plot.add_roi_tab(roi_id, roi_definition, roi_stats)
                else:
                    # Show dialog to select existing tab
                    existing_tabs = self.spectrum_plot.get_existing_roi_tabs()
                    if existing_tabs:
                        self._show_tab_selection_dialog(roi_id, roi_definition, roi_stats, existing_tabs)
                    else:
                        # No existing tabs, create new one
                        self.spectrum_plot.add_roi_tab(roi_id, roi_definition, roi_stats)
            else:
                print(f"ERROR: No ROI stats computed for ROI {roi_id}")
        else:
            print(f"ERROR: Failed to add ROI {roi_id} to manager")
                        
    def _show_tab_selection_dialog(self, roi_id: str, roi_definition: dict, roi_stats, existing_tabs: list):
        """Show dialog to select existing ROI tab."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select ROI Tab")
        dialog.setModal(True)
        
        layout = QtWidgets.QVBoxLayout()
        
        layout.addWidget(QtWidgets.QLabel("Add ROI to which tab?"))
        
        # List of existing tabs
        tab_list = QtWidgets.QListWidget()
        for tab_index, tab_name in existing_tabs:
            tab_list.addItem(tab_name)
        
        # Select first item by default
        if tab_list.count() > 0:
            tab_list.setCurrentRow(0)
            
        layout.addWidget(tab_list)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        new_tab_btn = QtWidgets.QPushButton("Create New Tab")
        new_tab_btn.clicked.connect(lambda: self._handle_tab_selection(dialog, roi_id, roi_definition, roi_stats, None))
        
        use_existing_btn = QtWidgets.QPushButton("Add to Selected Tab")
        use_existing_btn.clicked.connect(lambda: self._handle_tab_selection(dialog, roi_id, roi_definition, roi_stats, tab_list))
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(new_tab_btn)
        button_layout.addWidget(use_existing_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        
        dialog.exec_()
        
    def _handle_tab_selection(self, dialog, roi_id: str, roi_definition: dict, roi_stats, tab_list: QtWidgets.QListWidget = None):
        """Handle tab selection from dialog."""
        dialog.accept()
        
        if tab_list is None or tab_list.currentItem() is None:
            # Create new tab
            self.spectrum_plot.add_roi_tab(roi_id, roi_definition, roi_stats)
        else:
            # Add to existing tab using overlay functionality
            current_row = tab_list.currentRow()
            existing_tabs = self.spectrum_plot.get_existing_roi_tabs()
            
            if current_row >= 0 and current_row < len(existing_tabs):
                tab_index, tab_name = existing_tabs[current_row]
                success = self.spectrum_plot.add_roi_to_existing_tab(roi_id, roi_definition, roi_stats, tab_index)
                
                if not success:
                    print(f"ERROR: Failed to add ROI {roi_id} to existing tab {tab_name}")
                    # Fallback: create new tab
                    self.spectrum_plot.add_roi_tab(roi_id, roi_definition, roi_stats)
                
    def _update_roi_list(self):
        """Update ROI list widget."""
        if not hasattr(self, 'roi_list'):
            return
            
        self.roi_list.clear()
        
        for roi_id in self.roi_manager.get_roi_list():
            roi_info = self.roi_manager.get_roi_info(roi_id)
            if roi_info:
                self.roi_list.addItem(f"{roi_id}: {roi_info['name']}")
                
    def _clear_all_rois(self):
        """Clear all ROIs."""
        self.roi_manager.clear_all_rois()
        
        if self.spectrum_plot:
            self.spectrum_plot.clear_all_spectra()
            
        if self.image_view:
            self.image_view._clear_roi()
            
        self._update_roi_list()
        
    def _open_file_dialog(self):
        """Open file selection dialog."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Hyperspectral Data",
            "", "Hyperspectral Files (*.bsq *.bil *.bip *.bin *.nc);;ENVI Files (*.bsq *.bil *.bip *.bin);;EMIT Files (*.nc);;No Extension;;All Files (*)"
        )
        
        if filename:
            self._load_file_multi_dataset(filename)
    
    def _reload_with_interleave(self):
        """Reload current file with selected interleave format."""
        if not self.current_file:
            QtWidgets.QMessageBox.information(self, "No File", "No file currently loaded to reload.")
            return
            
        # Get selected interleave
        selected_interleave = self.interleave_combo.currentText()
        force_interleave = None if selected_interleave == 'Auto' else selected_interleave
        
        try:
            # Reload with specified interleave
            load_mode = self.config.get('data_loading.default_load_mode', 'memmap')
            load_to_ram = (load_mode == 'ram')
            
            success = self.data_handler.load_envi_data(self.current_file, load_to_ram, force_interleave)
            
            if success:
                # Update UI after successful reload
                self._update_after_data_load()
                
                # Update window title
                interleave_text = f" [{selected_interleave}]" if force_interleave else ""
                self.setWindowTitle(f"Info_i - {os.path.basename(self.current_file)}{interleave_text}")
                
                # Show success message
                if hasattr(self, 'status_bar'):
                    info = self.data_handler.get_info()
                    file_format = info.get('file_format', 'unknown')
                    if file_format == 'envi':
                        status_msg = (
                            f"Reloaded ENVI: {info['shape'][1]}x{info['shape'][0]}x{info['shape'][2]} "
                            f"({info.get('interleave', 'unknown')}){interleave_text}"
                        )
                    else:
                        status_msg = f"Reloaded {file_format.upper()}: {info['shape'][1]}x{info['shape'][0]}x{info['shape'][2]}"
                    
                    self.status_bar.showMessage(status_msg)
                    
                print(f"Successfully reloaded with interleave: {selected_interleave}")
            else:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to reload file with {selected_interleave} interleave")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during reload: {e}")
            
    def _add_to_recent_files(self, filename: str):
        """Add file to recent files list."""
        if filename in self.recent_files:
            self.recent_files.remove(filename)
            
        self.recent_files.insert(0, filename)
        
        # Limit list size
        max_recent = self.config.get('data_loading.max_recent_files', 10)
        self.recent_files = self.recent_files[:max_recent]
        
        # Update config
        self.config.set('recent_files', self.recent_files)
        
        self._update_recent_files_menu()
        
    def _update_recent_files_menu(self):
        """Update recent files menu."""
        if not hasattr(self, 'recent_menu'):
            return

        self.recent_menu.clear()

        # Ensure recent_files is a list
        if self.recent_files is None:
            self.recent_files = []

        for filename in self.recent_files[:5]:  # Show only first 5
            if os.path.exists(filename):
                action = self.recent_menu.addAction(os.path.basename(filename))
                action.triggered.connect(lambda checked, f=filename: self._load_file_multi_dataset(f))
                
        if not self.recent_files:
            self.recent_menu.addAction("No recent files").setEnabled(False)
            
    def _save_project(self):
        """Save current project (ROIs, settings, etc.)."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", "", "Project Files (*.yaml *.json)"
        )
        
        if filename:
            try:
                # Export ROIs and settings
                self.roi_manager.export_rois(filename, 'yaml' if filename.endswith('.yaml') else 'json')
                QtWidgets.QMessageBox.information(self, "Success", f"Project saved to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
                
    def _load_project(self):
        """Load project file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project Files (*.yaml *.json)"
        )
        
        if filename:
            try:
                success = self.roi_manager.import_rois(filename)
                if success:
                    self._update_roi_list()
                    QtWidgets.QMessageBox.information(self, "Success", f"Project loaded from {filename}")
                else:
                    QtWidgets.QMessageBox.critical(self, "Error", "Failed to load project")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load project: {e}")
                
    def _export_rois(self):
        """Export ROI definitions."""
        if not self.roi_manager.rois:
            QtWidgets.QMessageBox.information(self, "Export", "No ROIs to export")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ROIs", "", "YAML Files (*.yaml);;JSON Files (*.json);;ENVI ROI (*.roi)"
        )
        
        if filename:
            try:
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.yaml':
                    format = 'yaml'
                elif ext == '.json':
                    format = 'json'  
                elif ext == '.roi':
                    format = 'envi'
                else:
                    format = 'yaml'
                    
                success = self.roi_manager.export_rois(filename, format)
                if success:
                    QtWidgets.QMessageBox.information(self, "Success", f"ROIs exported to {filename}")
                else:
                    QtWidgets.QMessageBox.critical(self, "Error", "Failed to export ROIs")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to export ROIs: {e}")
                
    def _import_rois(self):
        """Import ROI definitions."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import ROIs", "", "ROI Files (*.yaml *.json *.roi)"
        )
        
        if filename:
            try:
                success = self.roi_manager.import_rois(filename)
                if success:
                    self._update_roi_list()
                    QtWidgets.QMessageBox.information(self, "Success", f"ROIs imported from {filename}")
                else:
                    QtWidgets.QMessageBox.critical(self, "Error", "Failed to import ROIs")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to import ROIs: {e}")
    def _on_zoom_box_changed(self):
        """Handle zoom box changes in overview - update main view."""
        if not hasattr(self, 'zoom_box') or not self.image_view:
            return
            
        try:
            # Get zoom box position and size in overview coordinates
            pos = self.zoom_box.pos()
            size = self.zoom_box.size()
            
            # Map to main image coordinates
            x = int(pos[0])
            y = int(pos[1]) 
            width = int(size[0])
            height = int(size[1])
            
            # Update main image view to show this region
            # Get the current image view from TabbedImageView
            current_view = self.image_view.get_current_image_view()
            if current_view and hasattr(current_view, 'image_widget'):
                main_view = current_view.image_widget.getView()
                main_view.setRange(xRange=[x, x + width], yRange=[y, y + height], padding=0)
            
        except Exception as e:
            pass
            
    def _update_zoom_box_from_main_view(self):
        """Update zoom box position based on main view bounds."""
        if not hasattr(self, 'zoom_box') or not self.image_view:
            return
            
        try:
            # Get current main view range
            view_range = self.image_view.get_view_range()
            
            # Update zoom box position and size
            x = view_range['x']
            y = view_range['y']
            width = view_range['width'] 
            height = view_range['height']
            
            # Temporarily disconnect signal to avoid recursive updates
            self.zoom_box.sigRegionChanged.disconnect()
            self.zoom_box.setPos([x, y], update=False)
            self.zoom_box.setSize([width, height], update=True)
            self.zoom_box.sigRegionChanged.connect(self._on_zoom_box_changed)
            
        except Exception as e:
            pass
    
    def _on_main_view_changed(self, view_range):
        """Handle main view changes - update overview zoom box."""
        self._update_zoom_box_from_main_view()
    
    def _toggle_fullscreen(self, checked: bool):
        """Toggle fullscreen mode."""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
            # Optionally maximize after exiting fullscreen
            if self.config.get('ui.start_maximized', True):
                self.showMaximized()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == QtCore.Qt.Key_Escape and self.isFullScreen():
            # Exit fullscreen on Escape
            self.showNormal()
            if self.config.get('ui.start_maximized', True):
                self.showMaximized()
            # Update menu action state
            if hasattr(self, 'fullscreen_action'):
                self.fullscreen_action.setChecked(False)
        else:
            super().keyPressEvent(event)
    
    def _toggle_overview(self, show: bool):
        """Toggle overview window visibility."""
        if hasattr(self, 'floating_overview'):
            self.floating_overview.setVisible(show)
        elif hasattr(self, 'overview_window'):  # Fallback for old attribute name
            self.overview_window.setVisible(show)
            
    def _toggle_file_manager(self, show: bool):
        """Toggle File Manager visibility."""
        if self.file_manager:
            if show:
                self.file_manager.show()
                # Position it relative to the main window if it hasn't been moved yet
                if self.file_manager.pos() in [QtCore.QPoint(0, 0), QtCore.QPoint(-1, -1)]:
                    # Position relative to main window
                    main_pos = self.pos()
                    self.file_manager.move(main_pos.x() + 50, main_pos.y() + 50)
                # Refresh the dataset list
                try:
                    self.file_manager.refresh_datasets()
                except Exception as e:
                    print(f"Error refreshing datasets: {e}")
                # Bring to front
                self.file_manager.raise_()
                self.file_manager.activateWindow()
            else:
                self.file_manager.hide()
                
    def _toggle_toolbar(self, show: bool):
        """Toggle toolbar visibility."""
        for toolbar in self.findChildren(QtWidgets.QToolBar):
            toolbar.setVisible(show)
            
    def _show_header_viewer(self):
        """Show header file contents in a new spectrum plot tab."""
        print("Header viewer clicked!")  # Debug
        if not self.data_handler or not self.data_handler.is_loaded:
            print("No data loaded")  # Debug
            QtWidgets.QMessageBox.information(self, "No Data", 
                "Please load a data file first to view its header.")
            return
            
        import os  # Move import to top of method
        
        # Get header file path
        header_path = None
        print(f"Looking for header file...")  # Debug
        if hasattr(self.data_handler, 'header_filename') and self.data_handler.header_filename:
            header_path = self.data_handler.header_filename
            print(f"Found header_filename: {header_path}")  # Debug
        elif hasattr(self.data_handler, 'filename') and self.data_handler.filename:
            # Try to find header file based on data file
            data_file = self.data_handler.filename
            # Try common header file patterns
            possible_headers = [
                data_file + '.hdr',
                os.path.splitext(data_file)[0] + '.hdr',
                data_file.replace('.bsq', '.hdr').replace('.bil', '.hdr').replace('.bip', '.hdr')
            ]
            for header_file in possible_headers:
                if os.path.exists(header_file):
                    header_path = header_file
                    break
        
        if not header_path or not os.path.exists(header_path):
            QtWidgets.QMessageBox.warning(self, "Header Not Found", 
                "Could not locate the header file for the current dataset.")
            return
            
        # Read header file contents
        try:
            with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
                header_content = f.read()
                
            # Add header info at the top
            file_info = f"Header File: {header_path}\n"
            file_info += f"Data File: {self.data_handler.filename}\n"
            file_info += "=" * 50 + "\n\n"
            header_content = file_info + header_content
            
            # Add header tab to spectrum plot
            if self.spectrum_plot:
                self.spectrum_plot.add_header_tab(header_content, "Header")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", 
                    "Spectrum plot not available.")
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Reading Header", 
                f"Failed to read header file: {e}")
                
    def _open_nc_viewer(self):
        """Open NetCDF structure viewer window."""
        try:
            nc_viewer = NCViewerWindow(self)
            nc_viewer.exec_()  # Show as modal dialog
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Opening NC Viewer",
                f"Failed to open NetCDF viewer: {str(e)}"
            )

    def _show_sam_dialog(self):
        """Show Spectral Angle Mapper analysis dialog."""
        try:
            if not self.data_manager or not hasattr(self.data_manager, 'datasets') or not self.data_manager.datasets:
                QtWidgets.QMessageBox.warning(self, "No Data", 
                    "No datasets are currently loaded. Please open a hyperspectral dataset first.")
                return
                
            if not self.roi_manager or not hasattr(self.roi_manager, 'rois') or not self.roi_manager.rois:
                QtWidgets.QMessageBox.warning(self, "No ROIs", 
                    "No ROIs are defined. Please create an ROI first to use as reference spectrum.")
                return
                
            # Create and show SAM dialog
            dialog = SAMDialog(self.data_manager, self.roi_manager, self)
            
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                # Get results and add to file system
                result_data = dialog.get_result()
                if result_data:
                    self._process_sam_results(result_data)
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Dialog Error", 
                f"Failed to open SAM dialog: {str(e)}")
                
    def _show_whitened_similarity_dialog(self):
        """Show Whitened Similarity analysis dialog."""
        try:
            if not self.data_manager or not hasattr(self.data_manager, 'datasets') or not self.data_manager.datasets:
                QtWidgets.QMessageBox.warning(self, "No Data", 
                    "No datasets are currently loaded. Please open a hyperspectral dataset first.")
                return
                
            if not self.roi_manager or not hasattr(self.roi_manager, 'rois') or not self.roi_manager.rois:
                QtWidgets.QMessageBox.warning(self, "No ROIs", 
                    "No ROIs are defined. Please create an ROI first to use as reference spectrum.")
                return
                
            # Create and show Whitened Similarity dialog
            dialog = WhitenedSimilarityDialog(self.data_manager, self.roi_manager, self)
            
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                # Get results and add to file system
                result_data = dialog.get_result()
                if result_data:
                    self._process_whitened_similarity_results(result_data)
                    
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Whitened Similarity Dialog Error",
                f"Failed to open Whitened Similarity dialog: {str(e)}")

    def _show_indices_dialog(self):
        """Show Spectral Indices Calculator dialog."""
        try:
            # Get active dataset
            current_dataset = self.data_manager.get_active_dataset()

            if not current_dataset or not current_dataset.is_loaded:
                QtWidgets.QMessageBox.warning(self, "No Data",
                    "No dataset is currently loaded. Please open a hyperspectral dataset first.")
                return

            # Create and show indices dialog
            dialog = IndicesDialog(current_dataset, self)

            # Connect signal to handle calculated indices
            dialog.index_calculated.connect(self._on_index_calculated)

            dialog.exec_()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Indices Dialog Error",
                f"Failed to open Indices Calculator dialog: {str(e)}")

    def _on_index_calculated(self, index_name: str, index_array: np.ndarray):
        """Handle calculated spectral index and add as new dataset."""
        try:
            # Create a simple dataset wrapper for the index
            class IndexDataset:
                def __init__(self, index_array, index_name):
                    self.data = np.expand_dims(index_array, axis=2)  # Add band dimension
                    self.shape = self.data.shape
                    self.is_loaded = True
                    self.filename = f"{index_name}_index"
                    self.wavelengths = None

                def get_band(self, band_idx):
                    """Get a single band."""
                    if band_idx == 0:
                        return self.data[:, :, 0]
                    return None

                def get_pixel_spectrum(self, x, y):
                    """Get pixel value."""
                    if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                        return np.array([self.data[y, x, 0]])
                    return None

                def get_rgb_image(self, stretch_percent=2.0):
                    """Get RGB visualization of index."""
                    band = self.data[:, :, 0]

                    # Normalize to 0-255
                    valid_mask = np.isfinite(band)
                    if not np.any(valid_mask):
                        return np.zeros((*band.shape, 3), dtype=np.uint8)

                    vmin, vmax = np.percentile(band[valid_mask], [stretch_percent, 100-stretch_percent])
                    if vmax == vmin:
                        vmax = vmin + 1

                    normalized = np.clip((band - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

                    # Create grayscale RGB
                    rgb = np.stack([normalized, normalized, normalized], axis=-1)
                    return rgb

            # Create dataset
            index_dataset = IndexDataset(index_array, index_name)
            dataset_name = f"{index_name}_Index"

            # Add to data manager
            source_info = {
                'type': 'Spectral Index',
                'index_name': index_name,
                'creation_time': QtCore.QDateTime.currentDateTime().toString()
            }
            self.data_manager.add_dataset(dataset_name, index_dataset, source_info)

            # Update file manager
            if hasattr(self, 'file_manager') and self.file_manager:
                self.file_manager.refresh_datasets()

            # Add to image view
            if hasattr(self, 'image_view') and self.image_view:
                self.image_view.add_dataset_tab(dataset_name)

            # Show success message
            self.status_bar.showMessage(f"✓ {index_name} index calculated successfully!")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Index Processing Error",
                f"Failed to process calculated index:\n{str(e)}")

    def _process_sam_results(self, result_data):
        """Process and integrate SAM analysis results."""
        try:
            sam_map = result_data['sam_map']
            source_dataset = result_data['source_dataset']
            source_roi = result_data['source_roi']
            output_path = result_data.get('output_path')
            source_handler = result_data['data_handler']
            
            # Create a unique name for the SAM result
            timestamp = QtCore.QDateTime.currentDateTime().toString('hhmmss')
            sam_name = f"SAM_{source_dataset}_{source_roi}_{timestamp}"
            
            if output_path:
                # Write to disk
                self._write_sam_to_disk(sam_map, output_path, source_handler, sam_name)
                # Load the written file as a new dataset
                self.load_data_file(output_path)
            else:
                # Keep in memory - create a virtual dataset
                self._create_sam_memory_dataset(sam_map, sam_name, source_handler, source_dataset, source_roi)
                
            # Show success message
            pixels_processed = np.prod(sam_map.shape)
            valid_pixels = np.sum(~np.isnan(sam_map))
            QtWidgets.QMessageBox.information(self, "SAM Analysis Complete", 
                f"SAM analysis completed successfully!\n\n"
                f"Dataset: {sam_name}\n"
                f"Processed: {valid_pixels:,} of {pixels_processed:,} pixels\n"
                f"Reference: ROI {source_roi} from {source_dataset}\n"
                f"Storage: {'Disk' if output_path else 'Memory'}")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SAM Results Error", 
                f"Failed to process SAM results: {str(e)}")
                
    def _process_whitened_similarity_results(self, result_data):
        """Process and integrate Whitened Similarity analysis results."""
        try:
            similarity_map = result_data['similarity_map']
            source_dataset = result_data['source_dataset']
            source_roi = result_data['source_roi']
            output_path = result_data.get('output_path')
            source_handler = result_data['data_handler']
            similarity_function = result_data['similarity_function']
            
            # Create a unique name for the Whitened Similarity result
            timestamp = QtCore.QDateTime.currentDateTime().toString('hhmmss')
            function_name = "Exp" if similarity_function == 0 else "Dist"
            ws_name = f"WhitenedSim_{function_name}_{source_dataset}_{source_roi}_{timestamp}"
            
            if output_path:
                # Write to disk
                self._write_whitened_similarity_to_disk(similarity_map, output_path, source_handler, ws_name)
                # Load the written file as a new dataset
                self.load_data_file(output_path)
            else:
                # Keep in memory - create a virtual dataset
                self._create_whitened_similarity_memory_dataset(similarity_map, ws_name, source_handler, source_dataset, source_roi)
                
            # Show success message
            pixels_processed = np.prod(similarity_map.shape)
            valid_pixels = np.sum(~np.isnan(similarity_map))
            function_desc = "Exponential (0-1)" if similarity_function == 0 else "Negative Distance"
            QtWidgets.QMessageBox.information(self, "Whitened Similarity Analysis Complete", 
                f"Whitened Similarity analysis completed successfully!\n\n"
                f"Dataset: {ws_name}\n"
                f"Processed: {valid_pixels:,} of {pixels_processed:,} pixels\n"
                f"Reference: ROI {source_roi} from {source_dataset}\n"
                f"Function: {function_desc}\n"
                f"Storage: {'Disk' if output_path else 'Memory'}")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Whitened Similarity Results Error", 
                f"Failed to process Whitened Similarity results: {str(e)}")
                
    def _write_sam_to_disk(self, sam_map, output_path, source_handler, sam_name):
        """Write SAM results to disk in ENVI format."""
        import spectral.io.envi as envi
        
        # Ensure proper file extensions
        if output_path.lower().endswith('.hdr'):
            img_path = output_path[:-4]  # Remove .hdr to get data file path
            header_path = output_path
        else:
            img_path = output_path
            header_path = output_path + '.hdr'
            
        print(f"DEBUG: Writing SAM to disk - Header: {header_path}, Data: {img_path}")
        
        # Prepare data - ensure it's 3D with shape (lines, samples, bands)
        if len(sam_map.shape) == 2:
            sam_data = sam_map[:, :, np.newaxis]  # Add band dimension
        else:
            sam_data = sam_map
            
        print(f"DEBUG: SAM data shape for writing: {sam_data.shape}")
        
        # Create comprehensive ENVI header metadata
        metadata = {
            'description': f'SAM Analysis Result - {sam_name}',
            'samples': int(sam_data.shape[1]),
            'lines': int(sam_data.shape[0]), 
            'bands': int(sam_data.shape[2]),
            'data type': 4,  # float32
            'interleave': 'bsq',
            'byte order': 0,
            'header offset': 0,
            'file type': 'ENVI Standard'
        }
        
        # Copy spatial metadata from source if available
        if hasattr(source_handler, 'metadata') and source_handler.metadata:
            for key in ['map info', 'coordinate system string', 'projection info']:
                if key in source_handler.metadata:
                    metadata[key] = source_handler.metadata[key]
                    
        print(f"DEBUG: ENVI metadata: {metadata}")
        
        try:
            # Write ENVI file (this creates both .hdr and data file)
            envi.save_image(header_path, sam_data, metadata=metadata, force=True)
            print(f"DEBUG: Successfully wrote ENVI file to {header_path}")
        except Exception as e:
            print(f"ERROR writing ENVI file: {e}")
            raise
        
    def _create_sam_memory_dataset(self, sam_map, sam_name, source_handler, source_dataset, source_roi):
        """Create an in-memory dataset for SAM results."""
        # Add SAM result as a new dataset in memory
        # This requires creating a minimal data handler for the SAM map
        
        class SAMDataHandler:
            """Complete data handler for SAM results - compatible with DataHandler interface."""
            def __init__(self, sam_data, source_handler):
                self.data = sam_data[:, :, np.newaxis]  # Add band dimension
                self.shape = self.data.shape
                self.wavelengths = np.array([1.0])  # Single band
                
                # Required DataHandler attributes
                self.is_loaded = True
                self.use_memmap = False
                self.filename = f"SAM_Result_Memory_{id(self)}"
                self.header_filename = None
                self.file_type = 'sam_result'
                self.fill_value = np.nan
                self.spy_file = None
                self.nc_dataset = None
                self.header = {}
                self.interleave = 'bsq'
                self.data_type = np.float32
                
                # Metadata compatible with ENVI format
                self.metadata = {
                    'description': 'SAM Analysis Result',
                    'data type': '4',  # float32
                    'interleave': 'bsq',
                    'bands': '1',
                    'lines': str(sam_data.shape[0]),
                    'samples': str(sam_data.shape[1]),
                    'byte order': '0'
                }
                
                # Copy spatial metadata from source if available
                if hasattr(source_handler, 'metadata') and source_handler.metadata:
                    for key in ['map info', 'coordinate system string']:
                        if key in source_handler.metadata:
                            self.metadata[key] = source_handler.metadata[key]
                            
                # Copy header from source if available  
                if hasattr(source_handler, 'header') and source_handler.header:
                    self.header = source_handler.header.copy()
                    # Override key fields for SAM result
                    self.header.update({
                        'bands': 1,
                        'data type': 4,
                        'lines': sam_data.shape[0], 
                        'samples': sam_data.shape[1],
                        'description': 'SAM Analysis Result'
                    })
                            
            def get_pixel_spectrum(self, x, y):
                """Get pixel spectrum at coordinates (x, y)."""
                if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                    return self.data[y, x, :]
                return None
                
            def get_band_data(self, band_index=0):
                """Get data for a specific band."""
                if band_index == 0 and len(self.shape) >= 3:
                    return self.data[:, :, 0]
                return None
                
            def get_monochromatic_image(self, band_index=0):
                """Get single band data for monochromatic display."""
                if band_index == 0 and len(self.shape) >= 3:
                    return self.data[:, :, 0]  # Return raw 2D array for colormap application
                return None
                
            def get_rgb_composite(self, red_band=0, green_band=0, blue_band=0, 
                                stretch_percent=2.0, no_data_value=None):
                """Generate RGB composite (grayscale for SAM results)."""
                # Return SAM map as grayscale for all RGB channels
                sam_2d = self.data[:, :, 0]
                
                # Handle no data values
                if no_data_value is not None:
                    sam_2d = np.where(np.isclose(sam_2d, no_data_value), np.nan, sam_2d)
                
                # Normalize to 0-255 for display
                valid_data = sam_2d[~np.isnan(sam_2d)]
                if len(valid_data) > 0:
                    # Use percentile stretching
                    low_val = np.percentile(valid_data, stretch_percent)
                    high_val = np.percentile(valid_data, 100 - stretch_percent)
                    
                    normalized = np.clip((sam_2d - low_val) / (high_val - low_val) * 255, 0, 255)
                    normalized = np.where(np.isnan(sam_2d), 0, normalized).astype(np.uint8)
                    
                    # Create RGB image (grayscale)
                    rgb = np.stack([normalized, normalized, normalized], axis=2)
                    return rgb
                    
                return np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
                
            def _normalize_for_display(self, image: np.ndarray, stretch_percent: float = 2.0, no_data_value: float = None) -> np.ndarray:
                """Normalize image for display with contrast stretching."""
                if image.size == 0:
                    return image
                    
                # Handle no data values
                if no_data_value is not None:
                    valid_mask = ~np.isclose(image, no_data_value)
                else:
                    valid_mask = ~np.isnan(image)
                    
                valid_data = image[valid_mask]
                if valid_data.size == 0:
                    return np.zeros_like(image, dtype=np.uint8)
                    
                # Calculate percentile values
                low_val = np.percentile(valid_data, stretch_percent)
                high_val = np.percentile(valid_data, 100 - stretch_percent)
                
                if high_val <= low_val:
                    high_val = low_val + 1e-6  # Avoid division by zero
                    
                # Apply stretching
                stretched = (image - low_val) / (high_val - low_val) * 255
                stretched = np.clip(stretched, 0, 255)
                
                # Set invalid pixels to 0
                stretched[~valid_mask] = 0
                
                return stretched.astype(np.uint8)
                
            def get_info(self):
                """Get dataset information."""
                return {
                    'filename': self.filename,
                    'shape': self.shape,
                    'bands': self.shape[2] if len(self.shape) > 2 else 1,
                    'data_type': 'float32',
                    'interleave': 'bsq',
                    'wavelengths': self.wavelengths.tolist() if self.wavelengths is not None else None,
                    'file_type': self.file_type,
                    'is_loaded': self.is_loaded
                }
                
            def cleanup(self):
                """Clean up resources."""
                self.data = None
                self.is_loaded = False
                
            def extract_line_spectra(self, line_index):
                """Extract spectra for a line (row)."""
                if 0 <= line_index < self.shape[0]:
                    return self.data[line_index, :, :]
                return None
                
            def get_band_by_wavelength(self, target_wavelength):
                """Get band index closest to target wavelength."""
                if self.wavelengths is not None and len(self.wavelengths) > 0:
                    distances = np.abs(self.wavelengths - target_wavelength)
                    best_idx = np.argmin(distances)
                    return (best_idx, self.wavelengths[best_idx])
                return None
        
        # Create dataset entry
        sam_handler = SAMDataHandler(sam_map, source_handler)
        
        # Add to data manager using proper method
        if hasattr(self.data_manager, 'add_dataset'):
            source_info = {
                'type': 'SAM Analysis',
                'source_dataset': source_dataset,
                'source_roi': source_roi,
                'creation_time': QtCore.QDateTime.currentDateTime().toString(),
                'algorithm': 'Spectral Angle Mapper (Cosine Similarity)'
            }
            self.data_manager.add_dataset(sam_name, sam_handler, source_info)
            
            # Update file manager widget if available
            if hasattr(self, 'file_manager') and self.file_manager:
                self.file_manager.refresh_datasets()
                
            # Add as new tab to image view
            if hasattr(self, 'image_view') and self.image_view:
                self.image_view.add_dataset_tab(sam_name)
                
    def _write_whitened_similarity_to_disk(self, whitened_map, output_path, source_handler, whitened_name):
        """Write Whitened Similarity results to disk in ENVI format."""
        import spectral.io.envi as envi
        
        # Ensure proper file extensions
        if output_path.lower().endswith('.hdr'):
            img_path = output_path[:-4]  # Remove .hdr to get data file path
            header_path = output_path
        else:
            img_path = output_path
            header_path = output_path + '.hdr'
            
        print(f"DEBUG: Writing Whitened Similarity to disk - Header: {header_path}, Data: {img_path}")
        
        # Prepare data - ensure it's 3D with shape (lines, samples, bands)
        if len(whitened_map.shape) == 2:
            whitened_data = whitened_map[:, :, np.newaxis]  # Add band dimension
        else:
            whitened_data = whitened_map
            
        print(f"DEBUG: Whitened Similarity data shape for writing: {whitened_data.shape}")
        
        # Create comprehensive ENVI header metadata
        metadata = {
            'description': f'Whitened Similarity Analysis Result - {whitened_name}',
            'samples': int(whitened_data.shape[1]),
            'lines': int(whitened_data.shape[0]), 
            'bands': int(whitened_data.shape[2]),
            'data type': 4,  # float32
            'interleave': 'bsq',
            'byte order': 0,
            'header offset': 0,
            'file type': 'ENVI Standard'
        }
        
        # Copy spatial metadata from source if available
        if hasattr(source_handler, 'metadata') and source_handler.metadata:
            for key in ['map info', 'coordinate system string', 'projection info']:
                if key in source_handler.metadata:
                    metadata[key] = source_handler.metadata[key]
                    
        print(f"DEBUG: ENVI metadata: {metadata}")
        
        try:
            # Write ENVI file (this creates both .hdr and data file)
            envi.save_image(header_path, whitened_data, metadata=metadata, force=True)
            print(f"DEBUG: Successfully wrote ENVI file to {header_path}")
        except Exception as e:
            print(f"ERROR writing ENVI file: {e}")
            raise
        
    def _create_whitened_similarity_memory_dataset(self, whitened_map, whitened_name, source_handler, source_dataset, source_roi):
        """Create an in-memory dataset for Whitened Similarity results."""
        # Add Whitened Similarity result as a new dataset in memory
        # This requires creating a minimal data handler for the whitened map
        
        class WhitenedDataHandler:
            """Complete data handler for Whitened Similarity results - compatible with DataHandler interface."""
            def __init__(self, whitened_data, source_handler):
                self.data = whitened_data[:, :, np.newaxis]  # Add band dimension
                self.shape = self.data.shape
                self.wavelengths = np.array([1.0])  # Single band
                
                # Required DataHandler attributes
                self.is_loaded = True
                self.use_memmap = False
                self.filename = f"Whitened_Similarity_Result_Memory_{id(self)}"
                self.header_filename = None
                self.file_type = 'whitened_similarity_result'
                self.fill_value = np.nan
                self.spy_file = None
                self.nc_dataset = None
                self.header = {}
                self.interleave = 'bsq'
                self.data_type = np.float32
                
                # Metadata compatible with ENVI format
                self.metadata = {
                    'description': 'Whitened Similarity Analysis Result',
                    'data type': '4',  # float32
                    'interleave': 'bsq',
                    'bands': '1',
                    'lines': str(whitened_data.shape[0]),
                    'samples': str(whitened_data.shape[1]),
                    'byte order': '0'
                }
                
                # Copy spatial metadata from source if available
                if hasattr(source_handler, 'metadata') and source_handler.metadata:
                    for key in ['map info', 'coordinate system string']:
                        if key in source_handler.metadata:
                            self.metadata[key] = source_handler.metadata[key]
                            
                # Copy header from source if available  
                if hasattr(source_handler, 'header') and source_handler.header:
                    self.header = source_handler.header.copy()
                    # Override key fields for Whitened Similarity result
                    self.header.update({
                        'bands': 1,
                        'data type': 4,
                        'lines': whitened_data.shape[0], 
                        'samples': whitened_data.shape[1],
                        'description': 'Whitened Similarity Analysis Result'
                    })
                            
            def get_pixel_spectrum(self, x, y):
                """Get pixel spectrum at coordinates (x, y)."""
                if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                    return self.data[y, x, :]
                return None
                
            def get_band_data(self, band_index=0):
                """Get data for a specific band."""
                if band_index == 0 and len(self.shape) >= 3:
                    return self.data[:, :, 0]
                return None
                
            def get_monochromatic_image(self, band_index=0):
                """Get single band data for monochromatic display."""
                if band_index == 0 and len(self.shape) >= 3:
                    return self.data[:, :, 0]  # Return raw 2D array for colormap application
                return None
                
            def get_rgb_composite(self, red_band=0, green_band=0, blue_band=0, 
                                stretch_percent=2.0, no_data_value=None):
                """Generate RGB composite (grayscale for Whitened Similarity results)."""
                # Return whitened map as grayscale for all RGB channels
                whitened_2d = self.data[:, :, 0]
                
                # Handle no data values
                if no_data_value is not None:
                    whitened_2d = np.where(np.isclose(whitened_2d, no_data_value), np.nan, whitened_2d)
                
                # Normalize to 0-255 for display
                valid_data = whitened_2d[~np.isnan(whitened_2d)]
                if len(valid_data) > 0:
                    # Use percentile stretching
                    low_val = np.percentile(valid_data, stretch_percent)
                    high_val = np.percentile(valid_data, 100 - stretch_percent)
                    
                    normalized = np.clip((whitened_2d - low_val) / (high_val - low_val) * 255, 0, 255)
                    normalized = np.where(np.isnan(whitened_2d), 0, normalized).astype(np.uint8)
                    
                    # Create RGB image (grayscale)
                    rgb = np.stack([normalized, normalized, normalized], axis=2)
                    return rgb
                    
                return np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
                
            def _normalize_for_display(self, image: np.ndarray, stretch_percent: float = 2.0, no_data_value: float = None) -> np.ndarray:
                """Normalize image for display with contrast stretching."""
                if image.size == 0:
                    return image
                    
                # Handle no data values
                if no_data_value is not None:
                    valid_mask = ~np.isclose(image, no_data_value)
                else:
                    valid_mask = ~np.isnan(image)
                    
                valid_data = image[valid_mask]
                if valid_data.size == 0:
                    return np.zeros_like(image, dtype=np.uint8)
                    
                # Calculate percentile values
                low_val = np.percentile(valid_data, stretch_percent)
                high_val = np.percentile(valid_data, 100 - stretch_percent)
                
                if high_val <= low_val:
                    high_val = low_val + 1e-6  # Avoid division by zero
                    
                # Apply stretching
                stretched = (image - low_val) / (high_val - low_val) * 255
                stretched = np.clip(stretched, 0, 255)
                
                # Set invalid pixels to 0
                stretched[~valid_mask] = 0
                
                return stretched.astype(np.uint8)
                
            def get_info(self):
                """Get dataset information."""
                return {
                    'filename': self.filename,
                    'shape': self.shape,
                    'bands': self.shape[2] if len(self.shape) > 2 else 1,
                    'data_type': 'float32',
                    'interleave': 'bsq',
                    'wavelengths': self.wavelengths.tolist() if self.wavelengths is not None else None,
                    'file_type': self.file_type,
                    'is_loaded': self.is_loaded
                }
                
            def cleanup(self):
                """Clean up resources."""
                self.data = None
                self.is_loaded = False
                
            def extract_line_spectra(self, line_index):
                """Extract spectra for a line (row)."""
                if 0 <= line_index < self.shape[0]:
                    return self.data[line_index, :, :]
                return None
                
            def get_band_by_wavelength(self, target_wavelength):
                """Get band index closest to target wavelength."""
                if self.wavelengths is not None and len(self.wavelengths) > 0:
                    distances = np.abs(self.wavelengths - target_wavelength)
                    best_idx = np.argmin(distances)
                    return (best_idx, self.wavelengths[best_idx])
                return None
        
        # Create dataset entry
        whitened_handler = WhitenedDataHandler(whitened_map, source_handler)
        
        # Add to data manager using proper method
        if hasattr(self.data_manager, 'add_dataset'):
            source_info = {
                'type': 'Whitened Similarity Analysis',
                'source_dataset': source_dataset,
                'source_roi': source_roi,
                'creation_time': QtCore.QDateTime.currentDateTime().toString(),
                'algorithm': 'Whitened Similarity (Background Statistics & Whitening Transform)'
            }
            self.data_manager.add_dataset(whitened_name, whitened_handler, source_info)
            
            # Update file manager widget if available
            if hasattr(self, 'file_manager') and self.file_manager:
                self.file_manager.refresh_datasets()
                
            # Add as new tab to image view
            if hasattr(self, 'image_view') and self.image_view:
                self.image_view.add_dataset_tab(whitened_name)
            
    def _get_tile_providers(self):
        """Get dictionary of tile map providers and their URLs."""
        return {
            "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "OpenTopoMap": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
            "Stamen Terrain": "https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
            "Stamen Toner": "https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
            "CartoDB Positron": "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            "CartoDB Dark Matter": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            "ESRI World Imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "ESRI World Street Map": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        }

    def _on_provider_changed(self, provider_name):
        """Handle provider selection change."""
        if provider_name == "Custom Image File...":
            self.load_base_map_btn.setText("Load Image File")
        else:
            self.load_base_map_btn.setText("Load Tile Base Map")

    def _load_tile_base_map(self):
        """Load base map from selected provider."""
        provider = self.base_map_provider_combo.currentText()

        if provider == "Custom Image File...":
            self._load_custom_image_file()
            return

        try:
            # Get coordinates and zoom level
            lat = self.center_lat_input.value()
            lon = self.center_lon_input.value()
            zoom = self.zoom_level_input.value()

            self.base_map_label.setText(f"Loading tiles from {provider}...\nLat: {lat:.4f}, Lon: {lon:.4f}, Zoom: {zoom}")
            QtWidgets.QApplication.processEvents()

            # Fetch tiles and stitch them together
            self.base_map_data = self._fetch_and_stitch_tiles(provider, lat, lon, zoom)

            if self.base_map_data is not None:
                self.base_map_label.setText(
                    f"Loaded: {provider}\n"
                    f"Center: ({lat:.4f}, {lon:.4f})\n"
                    f"Zoom: {zoom}, Size: {self.base_map_data.shape[1]}x{self.base_map_data.shape[0]}"
                )

                # Enable controls
                self.base_map_opacity_slider.setEnabled(True)
                self.base_map_visible_checkbox.setEnabled(True)

                # Display the base map
                self._display_base_map()

                self.status_bar.showMessage(f"Base map loaded from {provider}")
            else:
                raise Exception("Failed to fetch tiles")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Base Map",
                f"Failed to load base map:\n{str(e)}\n\nNote: You need an internet connection to load web tiles."
            )
            print(f"Error loading base map: {e}")
            self.base_map_label.setText("Failed to load base map")

    def _load_custom_image_file(self):
        """Load a custom base map image file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Base Map Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)"
        )

        if not filename:
            return

        try:
            # Load the image using various libraries depending on format
            if filename.lower().endswith(('.tif', '.tiff')):
                try:
                    from PIL import Image
                    img = Image.open(filename)
                    self.base_map_data = np.array(img)
                except ImportError:
                    import matplotlib.pyplot as plt
                    img = plt.imread(filename)
                    self.base_map_data = img
            else:
                from PIL import Image
                img = Image.open(filename)
                self.base_map_data = np.array(img)

            # Convert to RGB if necessary
            if len(self.base_map_data.shape) == 2:
                self.base_map_data = np.stack([self.base_map_data] * 3, axis=-1)
            elif self.base_map_data.shape[2] == 4:
                self.base_map_data = self.base_map_data[:, :, :3]

            self.base_map_path = filename
            self.base_map_label.setText(f"Loaded: {os.path.basename(filename)}\nSize: {self.base_map_data.shape[1]}x{self.base_map_data.shape[0]}")

            # Enable controls
            self.base_map_opacity_slider.setEnabled(True)
            self.base_map_visible_checkbox.setEnabled(True)

            # Display the base map
            self._display_base_map()

            self.status_bar.showMessage(f"Base map loaded: {os.path.basename(filename)}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading Base Map",
                f"Failed to load base map:\n{str(e)}"
            )
            print(f"Error loading base map: {e}")

    def _fetch_and_stitch_tiles(self, provider, center_lat, center_lon, zoom, tile_count=3):
        """
        Fetch map tiles and stitch them together.

        Args:
            provider: Name of the tile provider
            center_lat: Center latitude
            center_lon: Center longitude
            zoom: Zoom level (1-19)
            tile_count: Number of tiles in each direction (default 3 for 3x3 grid)

        Returns:
            numpy array with stitched tiles
        """
        import urllib.request
        import io

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL (Pillow) is required for tile fetching. Install with: pip install Pillow")

        # Get tile URL template
        url_template = self.tile_providers.get(provider)
        if not url_template:
            raise ValueError(f"Unknown provider: {provider}")

        # Convert lat/lon to tile coordinates
        center_x, center_y = self._lat_lon_to_tile(center_lat, center_lon, zoom)

        # Calculate tile range
        half_count = tile_count // 2
        tiles = []

        for dy in range(-half_count, half_count + 1):
            row = []
            for dx in range(-half_count, half_count + 1):
                tile_x = int(center_x) + dx
                tile_y = int(center_y) + dy

                # Build URL
                url = url_template.format(z=zoom, x=tile_x, y=tile_y)

                try:
                    # Fetch tile
                    req = urllib.request.Request(url, headers={'User-Agent': 'HyperspectralViewer/1.0'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        tile_data = response.read()
                        tile_img = Image.open(io.BytesIO(tile_data))
                        row.append(np.array(tile_img))
                except Exception as e:
                    print(f"Failed to fetch tile ({tile_x}, {tile_y}): {e}")
                    # Create blank tile
                    row.append(np.zeros((256, 256, 3), dtype=np.uint8))

            tiles.append(row)

        # Stitch tiles together
        stitched_rows = []
        for row in tiles:
            if row:
                stitched_row = np.hstack(row)
                stitched_rows.append(stitched_row)

        if stitched_rows:
            stitched_image = np.vstack(stitched_rows)
            return stitched_image

        return None

    def _lat_lon_to_tile(self, lat, lon, zoom):
        """
        Convert latitude/longitude to tile coordinates.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            zoom: Zoom level

        Returns:
            tuple (tile_x, tile_y) as floats
        """
        import math

        n = 2.0 ** zoom
        tile_x = (lon + 180.0) / 360.0 * n
        lat_rad = math.radians(lat)
        tile_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

        return (tile_x, tile_y)

    def _display_base_map(self):
        """Display or update the base map overlay on the image view."""
        if self.base_map_data is None:
            return

        try:
            # Get the active image view
            active_view = self._get_active_image_view()
            if active_view is None or not hasattr(active_view, 'image_widget'):
                return

            # Remove existing base map item if present
            if self.base_map_item is not None:
                try:
                    active_view.image_widget.removeItem(self.base_map_item)
                except:
                    pass

            # Create new image item for base map
            self.base_map_item = pg.ImageItem(self.base_map_data)

            # Set opacity
            opacity = self.base_map_opacity_slider.value() / 100.0
            self.base_map_item.setOpacity(opacity)

            # Add to view (below the main hyperspectral image)
            active_view.image_widget.addItem(self.base_map_item)
            self.base_map_item.setZValue(-1)  # Place behind main image

            # Set visibility
            self.base_map_item.setVisible(self.base_map_visible_checkbox.isChecked())

        except Exception as e:
            print(f"Error displaying base map: {e}")

    def _update_base_map_opacity(self, value):
        """Update base map opacity."""
        self.base_map_opacity_label.setText(f"{value}%")

        if self.base_map_item is not None:
            opacity = value / 100.0
            self.base_map_item.setOpacity(opacity)

    def _toggle_base_map_visibility(self, state):
        """Toggle base map visibility."""
        if self.base_map_item is not None:
            self.base_map_item.setVisible(state == QtCore.Qt.Checked)

    def _clear_base_map(self):
        """Clear the current base map."""
        if self.base_map_data is None:
            return

        # Remove from view
        if self.base_map_item is not None:
            try:
                active_view = self._get_active_image_view()
                if active_view is not None and hasattr(active_view, 'image_widget'):
                    active_view.image_widget.removeItem(self.base_map_item)
            except:
                pass
            self.base_map_item = None

        # Clear data
        self.base_map_data = None
        self.base_map_path = None

        # Update UI
        self.base_map_label.setText("No base map loaded")
        self.base_map_opacity_slider.setEnabled(False)
        self.base_map_visible_checkbox.setEnabled(False)

        self.status_bar.showMessage("Base map cleared")

    def _get_active_image_view(self):
        """Get the currently active image view."""
        if self.split_mode and self.focused_split_view:
            return self.focused_split_view
        return self.image_view

    def _show_about(self):
        """Show about dialog."""
        QtWidgets.QMessageBox.about(self, "About Info_i",
            "Info_i - Hyperspectral Viewer v1.0\n\n"
            "ENVI Classic-like viewer for hyperspectral data.\n"
            "Features RGB display, live spectral plots, custom ROI tools, "
            "and overview/zoom capabilities.\n\n"
            "Built with Python, PyQt5, and PyQtGraph."
        )
        
    def resizeEvent(self, event):
        """Handle window resize events to reposition overview."""
        super().resizeEvent(event)
        # Reposition overview after a short delay to allow layout to settle
        if hasattr(self, 'floating_overview') and self.floating_overview:
            QtCore.QTimer.singleShot(100, self._position_overview_bottom_left)
            
    def _toggle_split_view(self, orientation: str):
        """Toggle split view mode (horizontal or vertical)."""
        if self.split_mode == orientation:
            # Already in this split mode, close it
            self._close_split_view()
            return
            
        # Close existing split if different orientation
        if self.split_mode:
            self._close_split_view()
            
        self.split_mode = orientation
        
        # Remove single view from layout (but don't delete it)
        self.image_layout.removeWidget(self.image_view)
        
        # Create splitter for split views
        if orientation == 'horizontal':
            self.split_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        else:  # vertical
            self.split_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            
        self.split_splitter.setHandleWidth(5)
        self.split_splitter.setChildrenCollapsible(False)
        
        # Use the existing image view as the first split view (preserves tabs and state)
        self.split_view_1 = self.image_view
        
        # Disconnect original connections to avoid duplicates
        self._disconnect_image_view_connections(self.split_view_1)
        
        # Create only one new TabbedImageView instance
        self.split_view_2 = TabbedImageView(view_id="split_view_2")
        
        # Install event filters to track focus
        self.split_view_1.installEventFilter(self)
        self.split_view_2.installEventFilter(self)
        
        # Set first view as initially focused
        self.focused_split_view = self.split_view_1
        
        # Setup split view connections for both views
        # Even though split_view_1 is the original, we need split-aware connections
        self._setup_split_view_connections(self.split_view_1)
        self._setup_split_view_connections(self.split_view_2)
        
        # Refresh dataset combos in split views
        if hasattr(self.split_view_1, '_refresh_dataset_combo'):
            self.split_view_1._refresh_dataset_combo()
        if hasattr(self.split_view_2, '_refresh_dataset_combo'):
            self.split_view_2._refresh_dataset_combo()
        
        # Add to splitter
        self.split_splitter.addWidget(self.split_view_1)
        self.split_splitter.addWidget(self.split_view_2)
        
        # Set equal sizes
        self.split_splitter.setSizes([400, 400])
        
        # Add splitter to layout
        self.image_layout.addWidget(self.split_splitter)
        
        # Store references
        self.split_views = [self.split_view_1, self.split_view_2]
        
        # Don't automatically copy tabs - let user choose what to display in each pane
                
        # Update status
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Split view enabled: {orientation}")
            
    def _close_split_view(self):
        """Close split view and return to single view mode."""
        if not self.split_mode:
            return
            
        # Remove splitter from layout
        if self.split_splitter:
            self.image_layout.removeWidget(self.split_splitter)
            self.split_splitter.setParent(None)
            self.split_splitter = None
            
        # Clear split view references (but don't clear split_view_1 since it's our original image_view)
        self.split_views = []
        self.split_mode = None
        self.focused_split_view = None
        
        # Remove event filters
        if self.split_view_1:
            self.split_view_1.removeEventFilter(self)
        if self.split_view_2:
            self.split_view_2.removeEventFilter(self)
            
        # split_view_1 is our original image_view, so just restore it to layout
        # split_view_2 can be deleted since it was created for split mode
        self.split_view_2 = None
        
        # Restore original connections for single view mode
        self._setup_image_view_connections(self.image_view)
        
        # Restore single image view (which is split_view_1)
        self.image_layout.addWidget(self.image_view)
        
        # Update status
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage("Split view closed")
            
    def _setup_split_view_connections(self, split_view):
        """Setup connections for a split view instance."""
        if split_view and self.spectrum_plot:
            # In split view, allow any view to update spectrum when clicked
            # The focus tracking will ensure the right dataset is used
            split_view.pixel_selected.connect(self._on_pixel_selected)
            if hasattr(split_view, 'pixel_selected_with_id'):
                split_view.pixel_selected_with_id.connect(self._on_pixel_selected_with_id)

            # Spectrum collection
            if hasattr(split_view, 'spectrum_collect_requested'):
                split_view.spectrum_collect_requested.connect(self._on_spectrum_collect_requested)

            split_view.roi_selected.connect(self._on_roi_selected)
            
            # Zoom changes update overview zoom box
            split_view.zoom_changed.connect(self._on_main_view_changed)
            
            # RGB band changes
            split_view.rgb_bands_changed.connect(self._update_rgb_display)
            
            # Default RGB request
            split_view.default_rgb_requested.connect(self._set_true_color_rgb_bands)
            
            # Frame view line changes
            split_view.line_changed.connect(self._update_frame_display)
            
            # Band selection changes
            split_view.band_changed.connect(self._on_band_changed)
            
            # Dataset changes (tab switching)
            if hasattr(split_view, 'dataset_changed'):
                split_view.dataset_changed.connect(self._on_dataset_changed)
    
    def _toggle_spectrum_split_view(self, enabled: bool):
        """Toggle spectrum split view mode."""
        if self.spectrum_plot and hasattr(self.spectrum_plot, '_toggle_split_view'):
            self.spectrum_plot._toggle_split_view(enabled)

    def closeEvent(self, event):
        """Handle application close."""
        # Save settings
        self.config.save_config()
        event.accept()


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Hyperspectral Viewer')
    parser.add_argument('file', nargs='?', help='Hyperspectral data file to open (ENVI or EMIT format)')
    parser.add_argument('--config', help='Configuration file path')
    args = parser.parse_args()
    
    # Create QApplication
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Info_i")
    app.setApplicationVersion("1.0")
    
    # Set style
    app.setStyle('Fusion')
    
    # Create and show main window
    viewer = HyperspectralViewer()
    
    # Load config file if specified
    if args.config:
        viewer.config.load_config(args.config)
        
    # Load data file if specified
    if args.file:
        viewer._load_file_multi_dataset(args.file)
        
    viewer.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()