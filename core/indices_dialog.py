"""
Spectral Indices Dialog for calculating vegetation and environmental indices.

Provides a UI for users to select and calculate spectral indices from hyperspectral data.
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import os

# Add indices_code path to system path
indices_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'indices_code', 'hv_vision_utilities')
sys.path.insert(0, indices_path)

from band_identification_combination import *


class IndicesDialog(QtWidgets.QDialog):
    """Dialog for selecting and calculating spectral indices."""

    # Signal emitted when index is calculated
    index_calculated = QtCore.pyqtSignal(str, np.ndarray)  # (index_name, result_array)

    # Define available indices organized by category
    INDICES = {
        "Broadband Greenness": {
            "NDVI": {
                "name": "Normalized Difference Vegetation Index",
                "function": "calculate_ndvi",
                "inputs": ["Red", "NIR"],
                "description": "Classic vegetation index using red and NIR bands"
            },
            "SR": {
                "name": "Simple Ratio Index",
                "function": "calculate_sr",
                "inputs": ["Red", "NIR"],
                "description": "Ratio of NIR to Red reflectance"
            },
            "EVI": {
                "name": "Enhanced Vegetation Index",
                "function": "calculate_evi",
                "inputs": ["Red", "NIR", "Blue"],
                "description": "Improved vegetation index with atmospheric correction"
            },
            "ARVI": {
                "name": "Atmospherically Resistant Vegetation Index",
                "function": "calculate_arvi",
                "inputs": ["Red", "NIR", "Blue"],
                "description": "Vegetation index resistant to atmospheric effects"
            },
            "SG": {
                "name": "Sum Green Index",
                "function": "calculate_sg",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Mean reflectance across 500-600 nm range"
            }
        },
        "Narrowband Greenness": {
            "NDVI705": {
                "name": "Red Edge NDVI",
                "function": "calculate_ndvi705",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Red edge normalized difference (750nm - 705nm)"
            },
            "VOG1": {
                "name": "Vogelmann Red Edge Index 1",
                "function": "calculate_vog1",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Ratio R740/R720"
            },
            "VOG2": {
                "name": "Vogelmann Red Edge Index 2",
                "function": "calculate_vog2",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Complex red edge ratio"
            },
            "VOG3": {
                "name": "Vogelmann Red Edge Index 3",
                "function": "calculate_vog3",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Alternative red edge ratio"
            },
            "REP": {
                "name": "Red Edge Position",
                "function": "calculate_rep",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Wavelength of maximum derivative in red edge"
            }
        },
        "Light Use Efficiency": {
            "PRI": {
                "name": "Photochemical Reflectance Index",
                "function": "calculate_pri",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Indicator of photosynthetic efficiency (531nm - 570nm)"
            },
            "SIPI": {
                "name": "Structure Insensitive Pigment Index",
                "function": "calculate_sipi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Carotenoid to chlorophyll ratio"
            }
        },
        "Canopy Nitrogen": {
            "RGR": {
                "name": "Red Green Ratio",
                "function": "calculate_rgr_ratio",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Ratio of mean red to mean green reflectance"
            },
            "NDNI": {
                "name": "Normalized Difference Nitrogen Index",
                "function": "calculate_ndni",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Nitrogen content indicator (1510nm, 1680nm)"
            }
        },
        "Dry/Senescent Carbon": {
            "NDLI": {
                "name": "Normalized Difference Lignin Index",
                "function": "calculate_ndli",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Lignin content indicator"
            },
            "CAI": {
                "name": "Cellulose Absorption Index",
                "function": "calculate_cai",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Cellulose absorption feature"
            },
            "PSRI": {
                "name": "Plant Senescence Reflectance Index",
                "function": "calculate_psri",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Plant stress and senescence indicator"
            },
            "CRI1": {
                "name": "Carotenoid Reflectance Index 1",
                "function": "calculate_cri1",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Carotenoid content (510nm, 550nm)"
            },
            "CRI2": {
                "name": "Carotenoid Reflectance Index 2",
                "function": "calculate_cri2",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Carotenoid content (510nm, 700nm)"
            }
        },
        "Leaf Pigments": {
            "ARI1": {
                "name": "Anthocyanin Reflectance Index 1",
                "function": "calculate_ari1",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Anthocyanin content (550nm, 700nm)"
            },
            "ARI2": {
                "name": "Anthocyanin Reflectance Index 2",
                "function": "calculate_ari2",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Anthocyanin content with NIR normalization"
            }
        },
        "Canopy Water Content": {
            "WBI": {
                "name": "Water Band Index",
                "function": "calculate_wbi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Water absorption feature (900nm/970nm)"
            },
            "NDWI": {
                "name": "Normalized Difference Water Index",
                "function": "calculate_ndwi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Water content indicator"
            },
            "MSI": {
                "name": "Moisture Stress Index",
                "function": "calculate_msi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Plant water stress indicator"
            },
            "NDII": {
                "name": "Normalized Difference Infrared Index",
                "function": "calculate_ndii",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Canopy water content"
            },
            "TVI": {
                "name": "Triangular Vegetation Index",
                "function": "calculate_tvi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Chlorophyll-sensitive index"
            }
        },
        "Canopy Structure": {
            "TGI": {
                "name": "Triangular Greenness Index",
                "function": "calculate_tgi",
                "inputs": ["Hypercube", "Wavelengths"],
                "description": "Chlorophyll content indicator"
            }
        }
    }

    def __init__(self, data_handler=None, parent=None):
        super().__init__(parent)
        self.data_handler = data_handler
        self.setWindowTitle("Calculate Spectral Indices")
        self.setMinimumSize(700, 600)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info_label = QtWidgets.QLabel(
            "Select spectral indices to calculate from your hyperspectral data.\n"
            "Results will be displayed as new image layers."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Category tabs
        self.tab_widget = QtWidgets.QTabWidget()

        for category, indices in self.INDICES.items():
            tab = self._create_category_tab(category, indices)
            self.tab_widget.addTab(tab, category)

        layout.addWidget(self.tab_widget)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        calculate_btn = QtWidgets.QPushButton("Calculate Selected")
        calculate_btn.clicked.connect(self._calculate_indices)
        button_layout.addWidget(calculate_btn)

        calculate_all_btn = QtWidgets.QPushButton("Calculate All in Category")
        calculate_all_btn.clicked.connect(self._calculate_all_in_category)
        button_layout.addWidget(calculate_all_btn)

        button_layout.addStretch()

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def _create_category_tab(self, category, indices):
        """Create a tab for a category of indices."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        # Scroll area for indices
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

        for index_key, index_info in indices.items():
            index_widget = self._create_index_widget(index_key, index_info)
            scroll_layout.addWidget(index_widget)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        return widget

    def _create_index_widget(self, index_key, index_info):
        """Create a widget for a single index."""
        group = QtWidgets.QGroupBox()
        layout = QtWidgets.QVBoxLayout()

        # Checkbox with index name
        checkbox = QtWidgets.QCheckBox(f"{index_key} - {index_info['name']}")
        checkbox.setObjectName(f"checkbox_{index_key}")
        layout.addWidget(checkbox)

        # Description
        desc_label = QtWidgets.QLabel(index_info['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #FFE4F5; font-size: 10px; margin-left: 20px;")
        layout.addWidget(desc_label)

        # Required inputs
        inputs_label = QtWidgets.QLabel(f"Requires: {', '.join(index_info['inputs'])}")
        inputs_label.setStyleSheet("font-size: 9px; margin-left: 20px; color: #BA68C8;")
        layout.addWidget(inputs_label)

        group.setLayout(layout)
        return group

    def _get_selected_indices(self):
        """Get list of selected indices from current tab."""
        selected = []
        current_tab = self.tab_widget.currentWidget()

        for checkbox in current_tab.findChildren(QtWidgets.QCheckBox):
            if checkbox.isChecked():
                # Extract index key from object name
                index_key = checkbox.objectName().replace("checkbox_", "")
                selected.append(index_key)

        return selected

    def _get_all_indices_in_category(self):
        """Get all indices in current category."""
        category = self.tab_widget.tabText(self.tab_widget.currentIndex())
        return list(self.INDICES[category].keys())

    def _calculate_indices(self):
        """Calculate selected indices."""
        selected = self._get_selected_indices()

        if not selected:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select at least one index to calculate."
            )
            return

        self._run_calculations(selected)

    def _calculate_all_in_category(self):
        """Calculate all indices in current category."""
        indices = self._get_all_indices_in_category()
        self._run_calculations(indices)

    def _run_calculations(self, index_keys):
        """Run index calculations."""
        if not self.data_handler or not self.data_handler.is_loaded:
            QtWidgets.QMessageBox.critical(
                self,
                "No Data",
                "Please load hyperspectral data before calculating indices."
            )
            return

        # Get data in correct format (bands, rows, cols)
        try:
            # Get full hyperspectral cube
            shape = self.data_handler.shape  # (rows, cols, bands)
            reflectance_cube = np.zeros((shape[2], shape[0], shape[1]), dtype=np.float32)

            for band in range(shape[2]):
                band_data = self.data_handler.get_band_data(band)
                if band_data is not None:
                    reflectance_cube[band, :, :] = band_data

            wavelengths = self.data_handler.wavelengths

            if wavelengths is None:
                QtWidgets.QMessageBox.critical(
                    self,
                    "No Wavelength Information",
                    "This dataset does not have wavelength information required for index calculation."
                )
                return

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Data Error",
                f"Error loading hyperspectral data:\n{str(e)}"
            )
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(index_keys))
        self.progress_bar.setValue(0)

        # Calculate each index
        for i, index_key in enumerate(index_keys):
            try:
                self.status_label.setText(f"Calculating {index_key}...")
                QtWidgets.QApplication.processEvents()

                # Find index info
                index_info = None
                category = None
                for cat, indices in self.INDICES.items():
                    if index_key in indices:
                        index_info = indices[index_key]
                        category = cat
                        break

                if not index_info:
                    continue

                # Get calculation function
                func_name = index_info['function']
                func = globals()[func_name]

                # Call with appropriate parameters
                if "Hypercube" in index_info['inputs']:
                    result = func(reflectance_cube, wavelengths)
                else:
                    # Simple band indices (NDVI, SR, EVI, ARVI)
                    # Need to extract specific bands
                    result = self._calculate_simple_index(func_name, reflectance_cube, wavelengths)

                # Emit signal with result
                self.index_calculated.emit(index_key, result)

                self.progress_bar.setValue(i + 1)

            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    f"Error Calculating {index_key}",
                    f"Failed to calculate {index_key}:\n{str(e)}"
                )

        self.status_label.setText(f"✓ Calculated {len(index_keys)} indices successfully!")
        self.progress_bar.setVisible(False)

    def _calculate_simple_index(self, func_name, reflectance_cube, wavelengths):
        """Calculate simple band-based indices (NDVI, SR, EVI, ARVI)."""
        wavelengths = np.array(wavelengths)

        # Find band indices
        red_idx = np.argmin(np.abs(wavelengths - 650))
        nir_idx = np.argmin(np.abs(wavelengths - 850))
        blue_idx = np.argmin(np.abs(wavelengths - 475))

        red_band = reflectance_cube[red_idx, :, :]
        nir_band = reflectance_cube[nir_idx, :, :]
        blue_band = reflectance_cube[blue_idx, :, :]

        # Call appropriate function
        if func_name == "calculate_ndvi":
            return calculate_ndvi(red_band, nir_band)
        elif func_name == "calculate_sr":
            return calculate_sr(red_band, nir_band)
        elif func_name == "calculate_evi":
            return calculate_evi(red_band, nir_band, blue_band)
        elif func_name == "calculate_arvi":
            return calculate_arvi(red_band, nir_band, blue_band)
        else:
            raise ValueError(f"Unknown function: {func_name}")
