"""
CollectedSpectraTab widget for managing and displaying collected pixel spectra.

Provides interface for naming, organizing, and plotting multiple collected spectra.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from typing import Dict, Optional, List
import time
import json

try:
    from .spectrum_collection import SpectrumCollection, CollectedSpectrum
except ImportError:
    from spectrum_collection import SpectrumCollection, CollectedSpectrum


class SpectrumListWidget(QtWidgets.QListWidget):
    """Custom list widget for displaying collected spectra with color indicators."""

    spectrum_selection_changed = QtCore.pyqtSignal(list)  # List of selected spectrum IDs
    spectrum_rename_requested = QtCore.pyqtSignal(str)  # spectrum_id
    spectrum_delete_requested = QtCore.pyqtSignal(str)  # spectrum_id
    spectrum_color_change_requested = QtCore.pyqtSignal(str)  # spectrum_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected_items = self.selectedItems()
        spectrum_ids = []
        for item in selected_items:
            spectrum_id = item.data(QtCore.Qt.UserRole)
            if spectrum_id:
                spectrum_ids.append(spectrum_id)
        self.spectrum_selection_changed.emit(spectrum_ids)

    def _show_context_menu(self, position):
        """Show context menu for spectrum operations."""
        item = self.itemAt(position)
        if item is None:
            return

        spectrum_id = item.data(QtCore.Qt.UserRole)
        if not spectrum_id:
            return

        menu = QtWidgets.QMenu(self)

        # Rename action
        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self.spectrum_rename_requested.emit(spectrum_id))

        # Change color action
        color_action = menu.addAction("Change Color")
        color_action.triggered.connect(lambda: self.spectrum_color_change_requested.emit(spectrum_id))

        menu.addSeparator()

        # Delete action
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self.spectrum_delete_requested.emit(spectrum_id))

        menu.exec_(self.mapToGlobal(position))

    def add_spectrum_item(self, spectrum: CollectedSpectrum):
        """Add a spectrum to the list."""
        item = QtWidgets.QListWidgetItem()
        item.setText(spectrum.get_display_name())
        item.setToolTip(spectrum.get_tooltip())
        item.setData(QtCore.Qt.UserRole, spectrum.id)

        # Create color indicator icon
        color_pixmap = QtGui.QPixmap(16, 16)
        color_pixmap.fill(QtGui.QColor(spectrum.color))
        item.setIcon(QtGui.QIcon(color_pixmap))

        self.addItem(item)

    def update_spectrum_item(self, spectrum: CollectedSpectrum):
        """Update an existing spectrum item."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(QtCore.Qt.UserRole) == spectrum.id:
                item.setText(spectrum.get_display_name())
                item.setToolTip(spectrum.get_tooltip())

                # Update color indicator
                color_pixmap = QtGui.QPixmap(16, 16)
                color_pixmap.fill(QtGui.QColor(spectrum.color))
                item.setIcon(QtGui.QIcon(color_pixmap))
                break

    def remove_spectrum_item(self, spectrum_id: str):
        """Remove a spectrum item from the list."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(QtCore.Qt.UserRole) == spectrum_id:
                self.takeItem(i)
                break


class CollectedSpectraTab(QtWidgets.QWidget):
    """Tab widget for managing collected spectra with plotting and organization."""

    # Signals for main app integration
    spectra_visibility_changed = QtCore.pyqtSignal()  # Emitted when plot visibility changes

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize spectrum collection
        self.collection = SpectrumCollection()
        self.wavelengths = None

        # Plot state
        self.plot_items = {}  # spectrum_id -> plot_item mapping
        self.visible_spectra = set()  # Set of visible spectrum IDs

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout()

        # Header with title and controls
        header_layout = QtWidgets.QHBoxLayout()

        title_label = QtWidgets.QLabel("Collected Spectra")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Collection management buttons
        self.clear_button = QtWidgets.QPushButton("Clear All")
        self.clear_button.setToolTip("Clear all collected spectra")
        self.clear_button.clicked.connect(self._clear_all_spectra)
        header_layout.addWidget(self.clear_button)

        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setToolTip("Export collected spectra to file")
        self.export_button.clicked.connect(self._export_spectra)
        header_layout.addWidget(self.export_button)

        self.import_button = QtWidgets.QPushButton("Import")
        self.import_button.setToolTip("Import collected spectra from file")
        self.import_button.clicked.connect(self._import_spectra)
        header_layout.addWidget(self.import_button)

        layout.addLayout(header_layout)

        # Main content area with splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # Plot area (top)
        plot_widget = pg.PlotWidget()
        plot_widget.setLabel('left', 'Reflectance/Radiance')
        plot_widget.setLabel('bottom', 'Wavelength (nm)')
        plot_widget.showGrid(True, True, alpha=0.3)
        plot_widget.setMouseEnabled(x=True, y=True)
        plot_widget.setMinimumHeight(300)
        plot_widget.setBackground('w')

        # Add crosshair
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        plot_widget.addItem(self.crosshair_h, ignoreBounds=True)

        # Connect mouse events for crosshair
        plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.plot_widget = plot_widget
        splitter.addWidget(plot_widget)

        # Control area (bottom)
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_widget)

        # Spectrum list with visibility controls
        list_header = QtWidgets.QHBoxLayout()
        list_label = QtWidgets.QLabel("Spectrum List:")
        list_label.setStyleSheet("font-weight: bold;")
        list_header.addWidget(list_label)

        list_header.addStretch()

        # Visibility controls
        self.show_all_button = QtWidgets.QPushButton("Show All")
        self.show_all_button.setToolTip("Show all spectra in plot")
        self.show_all_button.clicked.connect(self._show_all_spectra)
        list_header.addWidget(self.show_all_button)

        self.hide_all_button = QtWidgets.QPushButton("Hide All")
        self.hide_all_button.setToolTip("Hide all spectra from plot")
        self.hide_all_button.clicked.connect(self._hide_all_spectra)
        list_header.addWidget(self.hide_all_button)

        control_layout.addLayout(list_header)

        # Spectrum list
        self.spectrum_list = SpectrumListWidget()
        self.spectrum_list.setMaximumHeight(200)
        control_layout.addWidget(self.spectrum_list)

        # Info panel
        self.info_label = QtWidgets.QLabel("No spectra collected")
        self.info_label.setStyleSheet("font-style: italic; color: gray;")
        control_layout.addWidget(self.info_label)

        splitter.addWidget(control_widget)

        # Set splitter sizes (70% plot, 30% controls)
        splitter.setSizes([700, 300])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _setup_connections(self):
        """Setup signal connections."""
        # Collection signals
        self.collection.spectrum_added.connect(self._on_spectrum_added)
        self.collection.spectrum_removed.connect(self._on_spectrum_removed)
        self.collection.spectrum_renamed.connect(self._on_spectrum_renamed)
        self.collection.spectrum_recolored.connect(self._on_spectrum_recolored)
        self.collection.collection_cleared.connect(self._on_collection_cleared)

        # List widget signals
        self.spectrum_list.spectrum_selection_changed.connect(self._on_selection_changed)
        self.spectrum_list.spectrum_rename_requested.connect(self._rename_spectrum)
        self.spectrum_list.spectrum_delete_requested.connect(self._delete_spectrum)
        self.spectrum_list.spectrum_color_change_requested.connect(self._change_spectrum_color)

    def set_wavelengths(self, wavelengths: np.ndarray):
        """Set wavelength values for x-axis."""
        self.wavelengths = wavelengths
        if wavelengths is not None:
            self.plot_widget.setLabel('bottom', f'Wavelength (nm): {wavelengths[0]:.1f} - {wavelengths[-1]:.1f}')
            # Set fixed X-axis range
            self.plot_widget.setXRange(wavelengths[0], wavelengths[-1], padding=0)

    def collect_spectrum(self, spectrum: np.ndarray, x: int, y: int,
                        name: str = None, source_file: str = None) -> str:
        """
        Collect a new spectrum.

        Args:
            spectrum: Spectral values array
            x: Pixel x coordinate
            y: Pixel y coordinate
            name: Optional custom name
            source_file: Optional source file path

        Returns:
            Spectrum ID
        """
        return self.collection.add_spectrum(
            spectrum=spectrum,
            x=x, y=y,
            wavelengths=self.wavelengths,
            name=name,
            source_file=source_file
        )

    def _on_spectrum_added(self, spectrum_id: str):
        """Handle new spectrum added to collection."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if spectrum:
            # Add to list
            self.spectrum_list.add_spectrum_item(spectrum)

            # Plot spectrum (visible by default)
            self._plot_spectrum(spectrum_id)
            self.visible_spectra.add(spectrum_id)

            # Update info
            self._update_info_label()

    def _on_spectrum_removed(self, spectrum_id: str):
        """Handle spectrum removed from collection."""
        # Remove from plot
        self._remove_spectrum_plot(spectrum_id)

        # Remove from list
        self.spectrum_list.remove_spectrum_item(spectrum_id)

        # Update visibility tracking
        self.visible_spectra.discard(spectrum_id)

        # Update info
        self._update_info_label()

    def _on_spectrum_renamed(self, spectrum_id: str, new_name: str):
        """Handle spectrum renamed."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if spectrum:
            self.spectrum_list.update_spectrum_item(spectrum)
            # Update plot item name if it exists
            if spectrum_id in self.plot_items:
                plot_item = self.plot_items[spectrum_id]
                if hasattr(plot_item, 'setName'):
                    plot_item.setName(new_name)

    def _on_spectrum_recolored(self, spectrum_id: str, new_color: str):
        """Handle spectrum recolored."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if spectrum:
            self.spectrum_list.update_spectrum_item(spectrum)
            # Update plot color
            if spectrum_id in self.plot_items:
                self._remove_spectrum_plot(spectrum_id)
                if spectrum_id in self.visible_spectra:
                    self._plot_spectrum(spectrum_id)

    def _on_collection_cleared(self):
        """Handle collection cleared."""
        # Clear plot
        self.plot_widget.clear()
        # Re-add crosshair
        self.plot_widget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plot_widget.addItem(self.crosshair_h, ignoreBounds=True)

        # Clear list
        self.spectrum_list.clear()

        # Clear tracking
        self.plot_items.clear()
        self.visible_spectra.clear()

        # Update info
        self._update_info_label()

    def _plot_spectrum(self, spectrum_id: str):
        """Plot a spectrum."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if not spectrum or spectrum.spectrum is None or self.wavelengths is None:
            return

        # Create plot item
        plot_item = self.plot_widget.plot(
            self.wavelengths, spectrum.spectrum,
            pen=pg.mkPen(spectrum.color, width=2),
            name=spectrum.get_display_name()
        )

        self.plot_items[spectrum_id] = plot_item

    def _remove_spectrum_plot(self, spectrum_id: str):
        """Remove a spectrum from the plot."""
        if spectrum_id in self.plot_items:
            plot_item = self.plot_items[spectrum_id]
            self.plot_widget.removeItem(plot_item)
            del self.plot_items[spectrum_id]

    def _on_selection_changed(self, selected_ids: List[str]):
        """Handle spectrum selection changes."""
        # Could add highlighting or other visual feedback here
        if selected_ids:
            selected_names = []
            for spectrum_id in selected_ids:
                spectrum = self.collection.get_spectrum(spectrum_id)
                if spectrum:
                    selected_names.append(spectrum.get_display_name())
            self.info_label.setText(f"Selected: {', '.join(selected_names)}")
        else:
            self._update_info_label()

    def _show_all_spectra(self):
        """Show all spectra in the plot."""
        for spectrum_id in self.collection.get_spectrum_ids():
            if spectrum_id not in self.visible_spectra:
                self._plot_spectrum(spectrum_id)
                self.visible_spectra.add(spectrum_id)

    def _hide_all_spectra(self):
        """Hide all spectra from the plot."""
        for spectrum_id in list(self.visible_spectra):
            self._remove_spectrum_plot(spectrum_id)
        self.visible_spectra.clear()

    def _clear_all_spectra(self):
        """Clear all collected spectra."""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All Spectra",
            "Are you sure you want to clear all collected spectra?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.collection.clear_all()

    def _rename_spectrum(self, spectrum_id: str):
        """Rename a spectrum."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if not spectrum:
            return

        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Spectrum",
            "Enter new name:", text=spectrum.name
        )

        if ok and new_name.strip():
            self.collection.rename_spectrum(spectrum_id, new_name.strip())

    def _delete_spectrum(self, spectrum_id: str):
        """Delete a spectrum."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if not spectrum:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Delete Spectrum",
            f"Delete spectrum '{spectrum.get_display_name()}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.collection.remove_spectrum(spectrum_id)

    def _change_spectrum_color(self, spectrum_id: str):
        """Change spectrum color."""
        spectrum = self.collection.get_spectrum(spectrum_id)
        if not spectrum:
            return

        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(spectrum.color), self, "Choose Spectrum Color"
        )

        if color.isValid():
            self.collection.recolor_spectrum(spectrum_id, color.name())

    def _update_info_label(self):
        """Update the info label."""
        count = self.collection.get_spectrum_count()
        if count == 0:
            self.info_label.setText("No spectra collected")
        elif count == 1:
            self.info_label.setText("1 spectrum collected")
        else:
            visible_count = len(self.visible_spectra)
            self.info_label.setText(f"{count} spectra collected ({visible_count} visible)")

    def _export_spectra(self):
        """Export collected spectra to file."""
        if self.collection.get_spectrum_count() == 0:
            QtWidgets.QMessageBox.information(self, "Export", "No spectra to export")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Collected Spectra", "",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if filename:
            try:
                if filename.endswith('.csv'):
                    self._export_csv(filename)
                else:
                    self._export_json(filename)
                QtWidgets.QMessageBox.information(self, "Export", f"Spectra exported to {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    def _export_json(self, filename: str):
        """Export to JSON format."""
        data = self.collection.export_to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, filename: str):
        """Export to CSV format."""
        if self.wavelengths is None:
            raise ValueError("No wavelength data available for CSV export")

        with open(filename, 'w') as f:
            # Write header
            header = ["Wavelength"]
            spectra_data = []

            for spectrum_id in self.collection.get_spectrum_ids():
                spectrum = self.collection.get_spectrum(spectrum_id)
                if spectrum and spectrum.spectrum is not None:
                    header.append(spectrum.get_display_name())
                    spectra_data.append(spectrum.spectrum)

            f.write(','.join(header) + '\n')

            # Write data
            for i in range(len(self.wavelengths)):
                row = [str(self.wavelengths[i])]
                for spectrum in spectra_data:
                    if i < len(spectrum):
                        row.append(str(spectrum[i]))
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')

    def _import_spectra(self):
        """Import collected spectra from file."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Collected Spectra", "",
            "JSON Files (*.json)"
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                if self.collection.import_from_dict(data):
                    QtWidgets.QMessageBox.information(self, "Import", "Spectra imported successfully")
                else:
                    QtWidgets.QMessageBox.warning(self, "Import", "Failed to import spectra")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to import: {e}")

    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # Update crosshair
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)