"""
Spectral Angle Mapper (SAM) Dialog

Provides interface for running spectral angle mapper analysis on hyperspectral data.
"""

import os
import tempfile
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PyQt5 import QtWidgets, QtCore, QtGui
import spectral
import spectral.io.envi as envi


class SAMDialog(QtWidgets.QDialog):
    """Dialog for configuring and running Spectral Angle Mapper analysis."""
    
    def __init__(self, data_manager, roi_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.roi_manager = roi_manager
        self.setWindowTitle('Spectral Angle Mapper')
        self.setModal(True)
        self.resize(500, 400)
        
        self._setup_ui()
        self._populate_datasets()
        self._populate_rois()
        
    def _setup_ui(self):
        """Setup the dialog user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Input dataset selection
        input_group = QtWidgets.QGroupBox('Input Dataset')
        input_layout = QtWidgets.QFormLayout()
        
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        input_layout.addRow('Dataset:', self.dataset_combo)
        
        # Dataset info
        self.dataset_info_label = QtWidgets.QLabel('No dataset selected')
        self.dataset_info_label.setStyleSheet('color: gray; font-size: 10px;')
        input_layout.addRow('Info:', self.dataset_info_label)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Reference spectrum selection
        ref_group = QtWidgets.QGroupBox('Reference Spectrum')
        ref_layout = QtWidgets.QFormLayout()
        
        self.roi_combo = QtWidgets.QComboBox()
        self.roi_combo.currentTextChanged.connect(self._on_roi_changed)
        ref_layout.addRow('ROI:', self.roi_combo)
        
        # ROI info
        self.roi_info_label = QtWidgets.QLabel('No ROI selected')
        self.roi_info_label.setStyleSheet('color: gray; font-size: 10px;')
        ref_layout.addRow('Info:', self.roi_info_label)
        
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        # Processing options
        options_group = QtWidgets.QGroupBox('Processing Options')
        options_layout = QtWidgets.QVBoxLayout()
        
        # Storage option
        storage_layout = QtWidgets.QHBoxLayout()
        self.memory_radio = QtWidgets.QRadioButton('Store in memory (default)')
        self.disk_radio = QtWidgets.QRadioButton('Write to disk')
        self.memory_radio.setChecked(True)  # Default to memory
        storage_layout.addWidget(self.memory_radio)
        storage_layout.addWidget(self.disk_radio)
        options_layout.addLayout(storage_layout)
        
        # Output file path (only enabled when disk radio is selected)
        self.output_layout = QtWidgets.QHBoxLayout()
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.setPlaceholderText('Output file path (will auto-generate if empty)')
        self.output_path_edit.setEnabled(False)
        
        self.browse_button = QtWidgets.QPushButton('Browse...')
        self.browse_button.clicked.connect(self._browse_output_path)
        self.browse_button.setEnabled(False)
        
        self.output_layout.addWidget(QtWidgets.QLabel('Output:'))
        self.output_layout.addWidget(self.output_path_edit, 1)
        self.output_layout.addWidget(self.browse_button)
        options_layout.addLayout(self.output_layout)
        
        # Connect radio buttons to enable/disable output path
        self.disk_radio.toggled.connect(self._on_storage_mode_changed)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress and status
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QtWidgets.QLabel('Ready to process')
        self.status_label.setStyleSheet('color: blue; font-size: 10px;')
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.run_button = QtWidgets.QPushButton('Run SAM Analysis')
        self.run_button.clicked.connect(self._run_sam_analysis)
        self.run_button.setEnabled(False)  # Disabled until valid inputs
        
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
    def _populate_datasets(self):
        """Populate the dataset dropdown."""
        self.dataset_combo.clear()
        self.dataset_combo.addItem('Select dataset...', None)
        
        if hasattr(self.data_manager, 'datasets'):
            for dataset_name in self.data_manager.datasets.keys():
                self.dataset_combo.addItem(dataset_name, dataset_name)
                
    def _populate_rois(self):
        """Populate the ROI dropdown."""
        self.roi_combo.clear()
        self.roi_combo.addItem('Select ROI...', None)
        
        if hasattr(self.roi_manager, 'rois') and self.roi_manager.rois:
            for roi_id, roi_data in self.roi_manager.rois.items():
                roi_name = roi_data.get('name', f'ROI {roi_id}')
                self.roi_combo.addItem(roi_name, roi_id)
                
    def _on_dataset_changed(self):
        """Handle dataset selection change."""
        dataset_name = self.dataset_combo.currentData()
        if dataset_name and hasattr(self.data_manager, 'datasets'):
            handler = self.data_manager.datasets.get(dataset_name)
            if handler and hasattr(handler, 'shape'):
                info = f"{handler.shape[0]}×{handler.shape[1]} pixels, {handler.shape[2]} bands"
                if hasattr(handler, 'wavelengths') and handler.wavelengths is not None:
                    wl_range = f"{handler.wavelengths[0]:.1f}-{handler.wavelengths[-1]:.1f} nm"
                    info += f", {wl_range}"
                self.dataset_info_label.setText(info)
            else:
                self.dataset_info_label.setText('Dataset info unavailable')
        else:
            self.dataset_info_label.setText('No dataset selected')
            
        self._validate_inputs()
        
    def _on_roi_changed(self):
        """Handle ROI selection change."""
        roi_id = self.roi_combo.currentData()
        if roi_id and hasattr(self.roi_manager, 'rois') and roi_id in self.roi_manager.rois:
            roi_data = self.roi_manager.rois[roi_id]
            roi_def = roi_data.get('definition', {})
            if roi_def and 'points' in roi_def and roi_def['points']:
                points = roi_def['points']
                roi_type = roi_def.get('type', 'unknown')
                
                if roi_type == 'rectangle' and len(points) >= 2:
                    # For rectangles, calculate dimensions from corner points
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    width = abs(max(x_coords) - min(x_coords)) + 1
                    height = abs(max(y_coords) - min(y_coords)) + 1
                    pixels = width * height
                    self.roi_info_label.setText(f'{roi_type}: {width}×{height} pixels ({pixels} total)')
                else:
                    # For other types, just show point count
                    pixels = len(points)
                    self.roi_info_label.setText(f'{roi_type}: {pixels} pixels')
            else:
                self.roi_info_label.setText('ROI definition unavailable')
        else:
            self.roi_info_label.setText('No ROI selected')
            
        self._validate_inputs()
        
    def _on_storage_mode_changed(self):
        """Handle storage mode radio button changes."""
        disk_mode = self.disk_radio.isChecked()
        self.output_path_edit.setEnabled(disk_mode)
        self.browse_button.setEnabled(disk_mode)
        
    def _browse_output_path(self):
        """Browse for output file path."""
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save SAM Results', '', 
            'ENVI Files (*.hdr);;All Files (*.*)')
        if filename:
            # Ensure .hdr extension
            if not filename.lower().endswith('.hdr'):
                filename += '.hdr'
            self.output_path_edit.setText(filename)
            
    def _validate_inputs(self):
        """Validate current inputs and enable/disable run button."""
        dataset_valid = self.dataset_combo.currentData() is not None
        roi_valid = self.roi_combo.currentData() is not None
        
        self.run_button.setEnabled(dataset_valid and roi_valid)
        
        if dataset_valid and roi_valid:
            self.status_label.setText('Ready to process')
            self.status_label.setStyleSheet('color: blue; font-size: 10px;')
        elif not dataset_valid:
            self.status_label.setText('Please select a dataset')
            self.status_label.setStyleSheet('color: orange; font-size: 10px;')
        elif not roi_valid:
            self.status_label.setText('Please select an ROI')
            self.status_label.setStyleSheet('color: orange; font-size: 10px;')
            
    def _run_sam_analysis(self):
        """Run the SAM analysis."""
        try:
            self.run_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText('Starting SAM analysis...')
            self.status_label.setStyleSheet('color: blue; font-size: 10px;')
            
            # Process events to update UI
            QtWidgets.QApplication.processEvents()
            
            dataset_name = self.dataset_combo.currentData()
            roi_id = self.roi_combo.currentData()
            
            # Get data handler directly
            data_handler = self.data_manager.datasets.get(dataset_name)
            if not data_handler or not hasattr(data_handler, 'shape'):
                raise ValueError(f"Dataset '{dataset_name}' not found or invalid")
            self.progress_bar.setValue(10)
            
            # Get ROI data and calculate reference spectrum
            roi_data = self.roi_manager.rois.get(roi_id)
            if not roi_data:
                raise ValueError(f"ROI '{roi_id}' not found")
                
            self.status_label.setText('Extracting reference spectrum from ROI...')
            QtWidgets.QApplication.processEvents()
            
            reference_spectrum = self._extract_roi_spectrum(data_handler, roi_data)
            self.progress_bar.setValue(30)
            
            # Get active bands (exclude bad bands)
            active_bands = self._get_active_bands(dataset_name)
            if active_bands:
                reference_spectrum = reference_spectrum[active_bands]
                self.status_label.setText(f'Using {len(active_bands)} active bands (excluding bad bands)...')
            else:
                self.status_label.setText('Using all bands...')
            QtWidgets.QApplication.processEvents()
            
            # Run SAM calculation
            self.status_label.setText('Computing SAM similarity map...')
            self.progress_bar.setValue(40)
            QtWidgets.QApplication.processEvents()
            
            sam_result = self._compute_sam(data_handler, reference_spectrum, active_bands)
            self.progress_bar.setValue(80)
            
            # Handle output
            if self.memory_radio.isChecked():
                output_path = None  # Memory only
                self.status_label.setText('Storing results in memory...')
            else:
                output_path = self.output_path_edit.text().strip()
                if not output_path:
                    # Auto-generate filename
                    timestamp = QtCore.QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')
                    output_path = f"sam_result_{dataset_name}_{timestamp}.hdr"
                self.status_label.setText(f'Saving results to {output_path}...')
                
            QtWidgets.QApplication.processEvents()
            
            # Store result information
            self.result_data = {
                'sam_map': sam_result,
                'reference_spectrum': reference_spectrum,
                'source_dataset': dataset_name,
                'source_roi': roi_id,
                'active_bands': active_bands,
                'output_path': output_path,
                'data_handler': data_handler  # Keep reference for wavelengths, etc.
            }
            
            self.progress_bar.setValue(100)
            self.status_label.setText('SAM analysis completed successfully!')
            self.status_label.setStyleSheet('color: green; font-size: 10px;')
            
            # Accept dialog to return results
            QtCore.QTimer.singleShot(500, self.accept)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f'Error: {str(e)}')
            self.status_label.setStyleSheet('color: red; font-size: 10px;')
            self.run_button.setEnabled(True)
            
            QtWidgets.QMessageBox.critical(self, 'SAM Analysis Error', 
                                         f'Failed to run SAM analysis:\n{str(e)}')
                                         
    def _extract_roi_spectrum(self, data_handler, roi_data):
        """Extract reference spectrum from ROI."""
        roi_def = roi_data.get('definition', {})
        if not roi_def or 'points' not in roi_def or not roi_def['points']:
            raise ValueError("ROI has no point information")
        
        points = roi_def['points']
        roi_type = roi_def.get('type', 'unknown')
        
        roi_spectra = []
        
        if roi_type == 'rectangle' and len(points) >= 2:
            # For rectangle, extract all pixels within bounds
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    try:
                        spectrum = data_handler.get_pixel_spectrum(x, y)
                        if spectrum is not None:
                            roi_spectra.append(spectrum)
                    except:
                        continue  # Skip invalid pixels
        else:
            # For other types (point, polygon), extract spectra from specific points
            for x, y in points:
                try:
                    spectrum = data_handler.get_pixel_spectrum(int(x), int(y))
                    if spectrum is not None:
                        roi_spectra.append(spectrum)
                except:
                    continue  # Skip invalid pixels
                    
        if not roi_spectra:
            raise ValueError("No valid spectra found in ROI")
            
        # Calculate mean spectrum
        roi_spectra = np.array(roi_spectra)
        reference_spectrum = np.mean(roi_spectra, axis=0)
        
        return reference_spectrum
        
    def _get_active_bands(self, dataset_name):
        """Get active bands for the dataset (excluding bad bands)."""
        # Check if the main window has the image view with bad bands info
        main_window = self.parent()
        if hasattr(main_window, 'image_view') and hasattr(main_window.image_view, 'get_active_bands'):
            try:
                active_bands = main_window.image_view.get_active_bands(dataset_name)
                if active_bands:
                    return active_bands
            except:
                pass
        return None  # Use all bands
        
    def _compute_sam(self, data_handler, reference_spectrum, active_bands=None):
        """Compute Spectral Angle Mapper similarity map."""
        shape = data_handler.shape
        rows, cols, bands = shape
        
        # Initialize result array
        sam_result = np.zeros((rows, cols), dtype=np.float32)
        
        # Normalize reference spectrum
        ref_norm = np.linalg.norm(reference_spectrum)
        if ref_norm == 0:
            raise ValueError("Reference spectrum has zero magnitude")
            
        reference_spectrum_normalized = reference_spectrum / ref_norm
        
        # Progress tracking
        total_pixels = rows * cols
        processed_pixels = 0
        
        # Process each pixel
        for row in range(rows):
            for col in range(cols):
                try:
                    # Get pixel spectrum
                    pixel_spectrum = data_handler.get_pixel_spectrum(col, row)
                    if pixel_spectrum is None:
                        sam_result[row, col] = np.nan
                        continue
                        
                    # Apply bad bands filtering
                    if active_bands:
                        pixel_spectrum = pixel_spectrum[active_bands]
                        
                    # Normalize pixel spectrum
                    pixel_norm = np.linalg.norm(pixel_spectrum)
                    if pixel_norm == 0:
                        sam_result[row, col] = np.nan
                        continue
                        
                    pixel_spectrum_normalized = pixel_spectrum / pixel_norm
                    
                    # Calculate spectral angle (cosine similarity)
                    # SAM angle = arccos(dot product of normalized spectra)
                    # We'll store cosine similarity (closer to 1 = more similar)
                    cosine_sim = np.dot(reference_spectrum_normalized, pixel_spectrum_normalized)
                    
                    # Clamp to valid range [-1, 1] to avoid numerical errors
                    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                    
                    sam_result[row, col] = cosine_sim
                    
                except Exception as e:
                    sam_result[row, col] = np.nan
                    
                processed_pixels += 1
                
            # Update progress periodically
            if row % max(1, rows // 20) == 0:  # Update 20 times
                progress = 40 + int((processed_pixels / total_pixels) * 40)  # 40-80% range
                self.progress_bar.setValue(progress)
                QtWidgets.QApplication.processEvents()
                
        return sam_result
        
    def get_result(self):
        """Get the analysis result data."""
        return getattr(self, 'result_data', None)