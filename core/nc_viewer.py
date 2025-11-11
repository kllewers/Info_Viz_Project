"""
NetCDF Structure Viewer - Standalone window for exploring NetCDF file structure.

Displays groups, variables, dimensions, and attributes in a Panoply-style interface.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from pathlib import Path
import netCDF4 as nc


class NCViewerWindow(QtWidgets.QDialog):
    """Standalone window for browsing NetCDF file structure."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nc_dataset = None
        self.current_file = None

        self.setWindowTitle("NetCDF Structure Viewer")
        self.resize(1000, 700)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar with file loading
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main split panel
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Left panel: Tree view
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabel("NetCDF Structure")
        self.tree_widget.itemClicked.connect(self._on_item_clicked)
        splitter.addWidget(self.tree_widget)

        # Right panel: Details view
        self.details_text = QtWidgets.QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QtGui.QFont("Courier", 10))
        splitter.addWidget(self.details_text)

        # Set splitter proportions (40% tree, 60% details)
        splitter.setSizes([400, 600])

        layout.addWidget(splitter)

        # Status bar
        self.status_label = QtWidgets.QLabel("No file loaded")
        layout.addWidget(self.status_label)

    def _create_toolbar(self):
        """Create toolbar with file operations."""
        toolbar = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        # Open file button
        open_btn = QtWidgets.QPushButton("Open NetCDF File")
        open_btn.clicked.connect(self._open_file)
        toolbar_layout.addWidget(open_btn)

        # File path label
        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray;")
        toolbar_layout.addWidget(self.file_label)

        toolbar_layout.addStretch()

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        toolbar_layout.addWidget(close_btn)

        return toolbar

    def _open_file(self):
        """Open file dialog and load NetCDF file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open NetCDF File",
            "",
            "NetCDF Files (*.nc);;All Files (*)"
        )

        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str):
        """Load and parse NetCDF file structure.

        Args:
            file_path: Path to NetCDF file
        """
        try:
            # Close previous dataset if open
            if self.nc_dataset is not None:
                self.nc_dataset.close()

            # Open new dataset
            self.nc_dataset = nc.Dataset(file_path, 'r')
            self.current_file = file_path

            # Update UI
            self.file_label.setText(Path(file_path).name)
            self.file_label.setStyleSheet("color: black;")
            self.status_label.setText(f"Loaded: {file_path}")

            # Build tree structure
            self._build_tree()

            # Show file summary in details panel
            self._show_file_summary()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to load NetCDF file:\n{str(e)}"
            )

    def _build_tree(self):
        """Build tree widget from NetCDF structure."""
        self.tree_widget.clear()

        if self.nc_dataset is None:
            return

        # Root item
        root_name = Path(self.current_file).name
        root_item = QtWidgets.QTreeWidgetItem(self.tree_widget, [f"📄 {root_name}"])
        root_item.setData(0, QtCore.Qt.UserRole, {'type': 'file', 'obj': self.nc_dataset})

        # Dimensions section
        dims_item = QtWidgets.QTreeWidgetItem(root_item, [f"📐 Dimensions ({len(self.nc_dataset.dimensions)})"])
        dims_item.setData(0, QtCore.Qt.UserRole, {'type': 'dimensions_section', 'obj': self.nc_dataset})
        for dim_name, dim_obj in self.nc_dataset.dimensions.items():
            dim_child = QtWidgets.QTreeWidgetItem(dims_item, [f"{dim_name}: {len(dim_obj)}"])
            dim_child.setData(0, QtCore.Qt.UserRole, {'type': 'dimension', 'name': dim_name, 'obj': dim_obj})

        # Variables section
        vars_item = QtWidgets.QTreeWidgetItem(root_item, [f"📊 Variables ({len(self.nc_dataset.variables)})"])
        vars_item.setData(0, QtCore.Qt.UserRole, {'type': 'variables_section', 'obj': self.nc_dataset})
        for var_name, var_obj in self.nc_dataset.variables.items():
            var_child = QtWidgets.QTreeWidgetItem(
                vars_item,
                [f"📈 {var_name}: {var_obj.shape} {var_obj.dtype}"]
            )
            var_child.setData(0, QtCore.Qt.UserRole, {'type': 'variable', 'name': var_name, 'obj': var_obj})

        # Groups section
        if len(self.nc_dataset.groups) > 0:
            groups_item = QtWidgets.QTreeWidgetItem(root_item, [f"📁 Groups ({len(self.nc_dataset.groups)})"])
            groups_item.setData(0, QtCore.Qt.UserRole, {'type': 'groups_section', 'obj': self.nc_dataset})
            for group_name, group_obj in self.nc_dataset.groups.items():
                self._add_group_recursive(groups_item, group_name, group_obj)

        # Global attributes section
        global_attrs = {attr: self.nc_dataset.getncattr(attr) for attr in self.nc_dataset.ncattrs()}
        attrs_item = QtWidgets.QTreeWidgetItem(root_item, [f"🏷️  Global Attributes ({len(global_attrs)})"])
        attrs_item.setData(0, QtCore.Qt.UserRole, {'type': 'attributes_section', 'obj': self.nc_dataset, 'attrs': global_attrs})

        # Expand root and first level
        root_item.setExpanded(True)

    def _add_group_recursive(self, parent_item, group_name, group_obj):
        """Recursively add group and its contents to tree.

        Args:
            parent_item: Parent tree widget item
            group_name: Name of the group
            group_obj: NetCDF group object
        """
        # Group node
        group_item = QtWidgets.QTreeWidgetItem(parent_item, [f"📂 {group_name}"])
        group_item.setData(0, QtCore.Qt.UserRole, {'type': 'group', 'name': group_name, 'obj': group_obj})

        # Group dimensions
        if len(group_obj.dimensions) > 0:
            dims_item = QtWidgets.QTreeWidgetItem(group_item, [f"📐 Dimensions ({len(group_obj.dimensions)})"])
            dims_item.setData(0, QtCore.Qt.UserRole, {'type': 'dimensions_section', 'obj': group_obj})
            for dim_name, dim_obj in group_obj.dimensions.items():
                dim_child = QtWidgets.QTreeWidgetItem(dims_item, [f"{dim_name}: {len(dim_obj)}"])
                dim_child.setData(0, QtCore.Qt.UserRole, {'type': 'dimension', 'name': dim_name, 'obj': dim_obj})

        # Group variables
        if len(group_obj.variables) > 0:
            vars_item = QtWidgets.QTreeWidgetItem(group_item, [f"📊 Variables ({len(group_obj.variables)})"])
            vars_item.setData(0, QtCore.Qt.UserRole, {'type': 'variables_section', 'obj': group_obj})
            for var_name, var_obj in group_obj.variables.items():
                var_child = QtWidgets.QTreeWidgetItem(
                    vars_item,
                    [f"📈 {var_name}: {var_obj.shape} {var_obj.dtype}"]
                )
                var_child.setData(0, QtCore.Qt.UserRole, {'type': 'variable', 'name': var_name, 'obj': var_obj})

        # Sub-groups (recursive)
        if len(group_obj.groups) > 0:
            subgroups_item = QtWidgets.QTreeWidgetItem(group_item, [f"📁 Sub-groups ({len(group_obj.groups)})"])
            subgroups_item.setData(0, QtCore.Qt.UserRole, {'type': 'groups_section', 'obj': group_obj})
            for subgroup_name, subgroup_obj in group_obj.groups.items():
                self._add_group_recursive(subgroups_item, subgroup_name, subgroup_obj)

        # Group attributes
        group_attrs = {attr: group_obj.getncattr(attr) for attr in group_obj.ncattrs()}
        if len(group_attrs) > 0:
            attrs_item = QtWidgets.QTreeWidgetItem(group_item, [f"🏷️  Attributes ({len(group_attrs)})"])
            attrs_item.setData(0, QtCore.Qt.UserRole, {'type': 'attributes_section', 'obj': group_obj, 'attrs': group_attrs})

    def _on_item_clicked(self, item, column):
        """Handle tree item click - show details in right panel.

        Args:
            item: Clicked tree widget item
            column: Column index
        """
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return

        item_type = data.get('type')

        if item_type == 'file':
            self._show_file_summary()
        elif item_type == 'dimension':
            self._show_dimension_details(data['name'], data['obj'])
        elif item_type == 'variable':
            self._show_variable_details(data['name'], data['obj'])
        elif item_type == 'group':
            self._show_group_details(data['name'], data['obj'])
        elif item_type == 'attributes_section':
            self._show_attributes_details(data['attrs'])
        elif item_type == 'dimensions_section':
            self._show_dimensions_section(data['obj'])
        elif item_type == 'variables_section':
            self._show_variables_section(data['obj'])

    def _show_file_summary(self):
        """Show overall file summary."""
        if self.nc_dataset is None:
            return

        summary = f"NetCDF File: {Path(self.current_file).name}\n"
        summary += "=" * 60 + "\n\n"
        summary += f"File Path: {self.current_file}\n"
        summary += f"Format: {self.nc_dataset.data_model}\n\n"

        summary += f"Root Dimensions: {len(self.nc_dataset.dimensions)}\n"
        summary += f"Root Variables: {len(self.nc_dataset.variables)}\n"
        summary += f"Groups: {len(self.nc_dataset.groups)}\n"
        summary += f"Global Attributes: {len(self.nc_dataset.ncattrs())}\n\n"

        # List dimensions
        if len(self.nc_dataset.dimensions) > 0:
            summary += "Root Dimensions:\n"
            for dim_name, dim_obj in self.nc_dataset.dimensions.items():
                unlimited = " (unlimited)" if dim_obj.isunlimited() else ""
                summary += f"  {dim_name}: {len(dim_obj)}{unlimited}\n"
            summary += "\n"

        # List groups
        if len(self.nc_dataset.groups) > 0:
            summary += "Groups:\n"
            for group_name in self.nc_dataset.groups.keys():
                summary += f"  /{group_name}\n"
            summary += "\n"

        self.details_text.setText(summary)

    def _show_dimension_details(self, name, dim_obj):
        """Show dimension details."""
        details = f"Dimension: {name}\n"
        details += "=" * 60 + "\n\n"
        details += f"Size: {len(dim_obj)}\n"
        details += f"Unlimited: {dim_obj.isunlimited()}\n"

        self.details_text.setText(details)

    def _show_variable_details(self, name, var_obj):
        """Show variable details."""
        details = f"Variable: {name}\n"
        details += "=" * 60 + "\n\n"
        details += f"Shape: {var_obj.shape}\n"
        details += f"Dimensions: {var_obj.dimensions}\n"
        details += f"Data Type: {var_obj.dtype}\n"
        details += f"Size: {var_obj.size} elements\n\n"

        # Chunking info
        if hasattr(var_obj, 'chunking') and var_obj.chunking() != 'contiguous':
            details += f"Chunking: {var_obj.chunking()}\n"
        else:
            details += "Chunking: contiguous\n"

        # Compression info
        if hasattr(var_obj, 'filters'):
            filters = var_obj.filters()
            if filters:
                details += f"Compression: "
                if filters.get('zlib'):
                    details += f"zlib (level {filters.get('complevel', 'unknown')})"
                if filters.get('shuffle'):
                    details += " + shuffle"
                details += "\n"

        # Attributes
        attrs = {attr: var_obj.getncattr(attr) for attr in var_obj.ncattrs()}
        if attrs:
            details += f"\nAttributes ({len(attrs)}):\n"
            for attr_name, attr_val in attrs.items():
                # Format attribute value
                if isinstance(attr_val, (list, tuple)) and len(attr_val) > 5:
                    val_str = f"[{attr_val[0]}, {attr_val[1]}, ..., {attr_val[-1]}] ({len(attr_val)} values)"
                else:
                    val_str = str(attr_val)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                details += f"  {attr_name}: {val_str}\n"

        self.details_text.setText(details)

    def _show_group_details(self, name, group_obj):
        """Show group details."""
        details = f"Group: /{name}\n"
        details += "=" * 60 + "\n\n"
        details += f"Dimensions: {len(group_obj.dimensions)}\n"
        details += f"Variables: {len(group_obj.variables)}\n"
        details += f"Sub-groups: {len(group_obj.groups)}\n"
        details += f"Attributes: {len(group_obj.ncattrs())}\n\n"

        # List contents
        if len(group_obj.dimensions) > 0:
            details += "Dimensions:\n"
            for dim_name, dim_obj in group_obj.dimensions.items():
                unlimited = " (unlimited)" if dim_obj.isunlimited() else ""
                details += f"  {dim_name}: {len(dim_obj)}{unlimited}\n"
            details += "\n"

        if len(group_obj.variables) > 0:
            details += "Variables:\n"
            for var_name, var_obj in group_obj.variables.items():
                details += f"  {var_name}: {var_obj.shape} {var_obj.dtype}\n"
            details += "\n"

        if len(group_obj.groups) > 0:
            details += "Sub-groups:\n"
            for subgroup_name in group_obj.groups.keys():
                details += f"  /{name}/{subgroup_name}\n"
            details += "\n"

        # Attributes
        attrs = {attr: group_obj.getncattr(attr) for attr in group_obj.ncattrs()}
        if attrs:
            details += f"Attributes ({len(attrs)}):\n"
            for attr_name, attr_val in attrs.items():
                val_str = str(attr_val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                details += f"  {attr_name}: {val_str}\n"

        self.details_text.setText(details)

    def _show_attributes_details(self, attrs):
        """Show attributes section details."""
        details = "Attributes\n"
        details += "=" * 60 + "\n\n"

        for attr_name, attr_val in attrs.items():
            val_str = str(attr_val)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            details += f"{attr_name}:\n  {val_str}\n\n"

        self.details_text.setText(details)

    def _show_dimensions_section(self, obj):
        """Show dimensions section summary."""
        details = "Dimensions\n"
        details += "=" * 60 + "\n\n"

        for dim_name, dim_obj in obj.dimensions.items():
            unlimited = " (unlimited)" if dim_obj.isunlimited() else ""
            details += f"{dim_name}: {len(dim_obj)}{unlimited}\n"

        self.details_text.setText(details)

    def _show_variables_section(self, obj):
        """Show variables section summary."""
        details = "Variables\n"
        details += "=" * 60 + "\n\n"

        for var_name, var_obj in obj.variables.items():
            details += f"{var_name}\n"
            details += f"  Shape: {var_obj.shape}\n"
            details += f"  Dimensions: {var_obj.dimensions}\n"
            details += f"  Type: {var_obj.dtype}\n\n"

        self.details_text.setText(details)

    def closeEvent(self, event):
        """Clean up when window is closed."""
        if self.nc_dataset is not None:
            self.nc_dataset.close()
        super().closeEvent(event)
