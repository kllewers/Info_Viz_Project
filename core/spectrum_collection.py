"""
SpectrumCollection class for managing collected pixel spectra with naming and color coding.

Supports collecting, storing, naming, and organizing multiple pixel spectra for comparison.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from PyQt5 import QtCore
import time
import uuid


class CollectedSpectrum:
    """Individual collected spectrum with metadata."""

    def __init__(self, spectrum: np.ndarray, x: int, y: int, wavelengths: np.ndarray = None,
                 name: str = None, color: str = None, source_file: str = None):
        self.id = str(uuid.uuid4())[:8]  # Short unique ID
        self.spectrum = spectrum.copy() if spectrum is not None else None
        self.x = x
        self.y = y
        self.wavelengths = wavelengths.copy() if wavelengths is not None else None
        self.name = name or f"Pixel ({x}, {y})"
        self.color = color or '#ff0000'  # Default red
        self.source_file = source_file
        self.timestamp = time.time()
        self.notes = ""

    def get_display_name(self) -> str:
        """Get display name for this spectrum."""
        if self.name.startswith("Pixel (") and self.name.endswith(")"):
            # Default name, show coordinates
            return f"{self.name}"
        else:
            # Custom name, show name with coordinates in tooltip
            return self.name

    def get_tooltip(self) -> str:
        """Get tooltip text for this spectrum."""
        tooltip_parts = [
            f"Name: {self.name}",
            f"Location: ({self.x}, {self.y})",
            f"Collected: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}"
        ]
        if self.source_file:
            tooltip_parts.append(f"Source: {self.source_file}")
        if self.notes:
            tooltip_parts.append(f"Notes: {self.notes}")
        return "\n".join(tooltip_parts)


class SpectrumCollection(QtCore.QObject):
    """Collection manager for pixel spectra with naming and organization."""

    # Signals
    spectrum_added = QtCore.pyqtSignal(str)  # spectrum_id
    spectrum_removed = QtCore.pyqtSignal(str)  # spectrum_id
    spectrum_renamed = QtCore.pyqtSignal(str, str)  # spectrum_id, new_name
    spectrum_recolored = QtCore.pyqtSignal(str, str)  # spectrum_id, new_color
    collection_cleared = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectra: Dict[str, CollectedSpectrum] = {}
        self.default_colors = [
            '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
            '#ff8000', '#8000ff', '#00ff80', '#ff0080', '#80ff00', '#0080ff',
            '#ff4000', '#4000ff', '#40ff00', '#00ff40', '#ff0040', '#0040ff'
        ]
        self.color_index = 0

    def add_spectrum(self, spectrum: np.ndarray, x: int, y: int, wavelengths: np.ndarray = None,
                    name: str = None, source_file: str = None) -> str:
        """
        Add a spectrum to the collection.

        Args:
            spectrum: Spectral values array
            x: Pixel x coordinate
            y: Pixel y coordinate
            wavelengths: Wavelength values (optional)
            name: Custom name for the spectrum (optional)
            source_file: Source file path (optional)

        Returns:
            Spectrum ID string
        """
        # Generate next color
        color = self.default_colors[self.color_index % len(self.default_colors)]
        self.color_index += 1

        # Create collected spectrum
        collected = CollectedSpectrum(
            spectrum=spectrum,
            x=x, y=y,
            wavelengths=wavelengths,
            name=name,
            color=color,
            source_file=source_file
        )

        # Store in collection
        self.spectra[collected.id] = collected

        # Emit signal
        self.spectrum_added.emit(collected.id)

        return collected.id

    def remove_spectrum(self, spectrum_id: str) -> bool:
        """
        Remove a spectrum from the collection.

        Args:
            spectrum_id: ID of spectrum to remove

        Returns:
            True if removed successfully
        """
        if spectrum_id in self.spectra:
            del self.spectra[spectrum_id]
            self.spectrum_removed.emit(spectrum_id)
            return True
        return False

    def get_spectrum(self, spectrum_id: str) -> Optional[CollectedSpectrum]:
        """Get a spectrum by ID."""
        return self.spectra.get(spectrum_id)

    def rename_spectrum(self, spectrum_id: str, new_name: str) -> bool:
        """
        Rename a spectrum.

        Args:
            spectrum_id: ID of spectrum to rename
            new_name: New name for the spectrum

        Returns:
            True if renamed successfully
        """
        if spectrum_id in self.spectra:
            old_name = self.spectra[spectrum_id].name
            self.spectra[spectrum_id].name = new_name
            self.spectrum_renamed.emit(spectrum_id, new_name)
            return True
        return False

    def recolor_spectrum(self, spectrum_id: str, new_color: str) -> bool:
        """
        Change the color of a spectrum.

        Args:
            spectrum_id: ID of spectrum to recolor
            new_color: New color (hex string like '#ff0000')

        Returns:
            True if recolored successfully
        """
        if spectrum_id in self.spectra:
            self.spectra[spectrum_id].color = new_color
            self.spectrum_recolored.emit(spectrum_id, new_color)
            return True
        return False

    def clear_all(self):
        """Clear all spectra from the collection."""
        self.spectra.clear()
        self.color_index = 0
        self.collection_cleared.emit()

    def get_all_spectra(self) -> Dict[str, CollectedSpectrum]:
        """Get all spectra in the collection."""
        return self.spectra.copy()

    def get_spectrum_count(self) -> int:
        """Get the number of spectra in the collection."""
        return len(self.spectra)

    def get_spectrum_ids(self) -> List[str]:
        """Get list of all spectrum IDs."""
        return list(self.spectra.keys())

    def export_to_dict(self) -> dict:
        """Export collection to dictionary format for saving."""
        export_data = {
            'version': '1.0',
            'timestamp': time.time(),
            'spectra': []
        }

        for spectrum_id, spectrum in self.spectra.items():
            spectrum_data = {
                'id': spectrum.id,
                'name': spectrum.name,
                'x': spectrum.x,
                'y': spectrum.y,
                'color': spectrum.color,
                'notes': spectrum.notes,
                'timestamp': spectrum.timestamp,
                'source_file': spectrum.source_file,
                'spectrum': spectrum.spectrum.tolist() if spectrum.spectrum is not None else None,
                'wavelengths': spectrum.wavelengths.tolist() if spectrum.wavelengths is not None else None
            }
            export_data['spectra'].append(spectrum_data)

        return export_data

    def import_from_dict(self, data: dict) -> bool:
        """Import collection from dictionary format."""
        try:
            if 'spectra' not in data:
                return False

            self.clear_all()

            for spectrum_data in data['spectra']:
                spectrum_array = None
                if spectrum_data.get('spectrum'):
                    spectrum_array = np.array(spectrum_data['spectrum'])

                wavelengths_array = None
                if spectrum_data.get('wavelengths'):
                    wavelengths_array = np.array(spectrum_data['wavelengths'])

                collected = CollectedSpectrum(
                    spectrum=spectrum_array,
                    x=spectrum_data['x'],
                    y=spectrum_data['y'],
                    wavelengths=wavelengths_array,
                    name=spectrum_data['name'],
                    color=spectrum_data['color'],
                    source_file=spectrum_data.get('source_file')
                )
                collected.id = spectrum_data['id']
                collected.notes = spectrum_data.get('notes', '')
                collected.timestamp = spectrum_data.get('timestamp', time.time())

                self.spectra[collected.id] = collected
                self.spectrum_added.emit(collected.id)

            return True

        except Exception as e:
            print(f"Error importing spectrum collection: {e}")
            return False