"""
Integration Test for Crossplot with Well and Manager Classes
=============================================================

This test creates synthetic well data and validates the complete workflow:
1. Creating wells with properties
2. Creating crossplots from Well objects
3. Creating multi-well crossplots from Manager
4. Adding regressions to crossplots
5. Testing all parameter combinations
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from pylog import WellDataManager


def create_synthetic_las_file(filepath, well_name, depth_range=(1000, 2000), n_points=500):
    """Create a synthetic LAS file for testing."""
    np.random.seed(hash(well_name) % 2**32)  # Different data per well

    # Generate depth
    depth = np.linspace(depth_range[0], depth_range[1], n_points)

    # Generate realistic well log data
    # Density (RHOB): 2.0 - 2.8 g/cc
    rhob = 2.3 + 0.2 * np.sin(depth / 100) + np.random.normal(0, 0.05, n_points)

    # Neutron Porosity (NPHI): 0.1 - 0.4
    nphi = 0.25 - 0.3 * (rhob - 2.3) + np.random.normal(0, 0.02, n_points)
    nphi = np.clip(nphi, 0.05, 0.45)

    # Porosity (PHIE): correlated with NPHI
    phie = nphi * 0.9 + np.random.normal(0, 0.01, n_points)
    phie = np.clip(phie, 0, 0.5)

    # Water Saturation (SW): 0.2 - 1.0
    sw = 0.5 - 0.3 * phie + np.random.normal(0, 0.1, n_points)
    sw = np.clip(sw, 0.2, 1.0)

    # Permeability (PERM): log-normal distribution, correlated with porosity
    perm = np.exp(10 * phie + np.random.normal(0, 1, n_points))
    perm = np.clip(perm, 0.1, 10000)

    # Gamma Ray (GR): 20 - 150 API
    gr = 60 + 40 * np.sin(depth / 150) + np.random.normal(0, 10, n_points)
    gr = np.clip(gr, 20, 150)

    # Resistivity (RT): 1 - 1000 ohm.m
    rt = 100 * (1 - sw)**2 + np.random.normal(0, 10, n_points)
    rt = np.clip(rt, 1, 1000)

    # Shale Volume (VSH): 0 - 1
    vsh = (gr - 20) / 130
    vsh = np.clip(vsh, 0, 1)

    # Net Sand Flag (discrete)
    net_sand = ((phie > 0.15) & (vsh < 0.4) & (sw < 0.6)).astype(float)

    # Zone (discrete): 3 zones
    zone = np.zeros(n_points)
    zone[depth < 1333] = 1
    zone[(depth >= 1333) & (depth < 1667)] = 2
    zone[depth >= 1667] = 3

    # Write LAS file
    with open(filepath, 'w') as f:
        # Header
        f.write("~Version Information\n")
        f.write(" VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0\n")
        f.write(" WRAP.                          NO  :   ONE LINE PER DEPTH STEP\n")
        f.write("~Well Information\n")
        f.write(f" WELL.  {well_name:20s}:   WELL NAME\n")
        f.write("~Curve Information\n")
        f.write(" DEPT.M                        :   DEPTH\n")
        f.write(" RHOB.G/CC                     :   BULK DENSITY\n")
        f.write(" NPHI.V/V                      :   NEUTRON POROSITY\n")
        f.write(" PHIE.V/V                      :   EFFECTIVE POROSITY\n")
        f.write(" SW  .V/V                      :   WATER SATURATION\n")
        f.write(" PERM.MD                       :   PERMEABILITY\n")
        f.write(" GR  .API                      :   GAMMA RAY\n")
        f.write(" RT  .OHMM                     :   RESISTIVITY\n")
        f.write(" VSH .V/V                      :   SHALE VOLUME\n")
        f.write(" NetSand.                      :   NET SAND FLAG\n")
        f.write(" Zone.                         :   ZONE NUMBER\n")
        f.write("~Parameter Information\n")
        f.write("~A  DEPT     RHOB     NPHI     PHIE       SW     PERM       GR       RT      VSH  NetSand     Zone\n")

        # Data
        for i in range(n_points):
            f.write(f"{depth[i]:8.2f} {rhob[i]:8.4f} {nphi[i]:8.4f} {phie[i]:8.4f} ")
            f.write(f"{sw[i]:8.4f} {perm[i]:8.2f} {gr[i]:8.2f} {rt[i]:8.2f} ")
            f.write(f"{vsh[i]:8.4f} {net_sand[i]:8.0f} {zone[i]:8.0f}\n")


class IntegrationTests:
    """Integration tests for Crossplot functionality."""

    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.temp_dir = tempfile.mkdtemp()
        self.manager = None

    def setup_test_data(self):
        """Create synthetic wells for testing."""
        print("\n" + "=" * 70)
        print("SETUP: Creating Synthetic Test Data")
        print("=" * 70)

        # Create 3 synthetic wells
        well_names = ["Well_A", "Well_B", "Well_C"]

        for well_name in well_names:
            las_path = os.path.join(self.temp_dir, f"{well_name}.las")
            create_synthetic_las_file(las_path, well_name)
            print(f"  Created: {well_name}.las")

        # Load into manager
        self.manager = WellDataManager()
        for well_name in well_names:
            las_path = os.path.join(self.temp_dir, f"{well_name}.las")
            self.manager.load_las(las_path)
            print(f"  Loaded: {well_name}")

        print(f"\n  ✓ Created {len(well_names)} synthetic wells")
        print(f"  ✓ Each well has 500 data points")
        print(f"  ✓ Properties: RHOB, NPHI, PHIE, SW, PERM, GR, RT, VSH, NetSand, Zone")

    def test_basic_well_crossplot(self):
        """Test basic crossplot from Well object."""
        print("\n1. Basic Well Crossplot Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            # Create simple crossplot
            plot = well.Crossplot(x="RHOB", y="NPHI", title="Basic Crossplot Test")

            # Check plot was created
            if plot.fig is None:
                plot.plot()

            if plot.fig is not None and plot.ax is not None:
                print("  ✓ Crossplot created successfully")
                print(f"  ✓ Figure size: {plot.figsize}")
                print(f"  ✓ Data points plotted: {len(plot._prepare_data())}")
                self.tests_passed += 1

                # Save plot
                save_path = os.path.join(self.temp_dir, "basic_crossplot.png")
                plot.save(save_path)
                plot.close()
                print(f"  ✓ Plot saved: {os.path.basename(save_path)}")
            else:
                print("  ✗ FAILED: Plot not created")
                self.tests_failed += 1

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_crossplot_with_color_mapping(self):
        """Test crossplot with color by depth."""
        print("\n2. Color Mapping Test (depth)")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(
                x="PHIE",
                y="SW",
                color="depth",
                colortemplate="viridis",
                color_range=[1000, 2000],
                title="Color by Depth"
            )
            plot.plot()

            if plot.colorbar is not None:
                print("  ✓ Colorbar created")
                self.tests_passed += 1
            else:
                print("  ✗ FAILED: Colorbar not created")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "color_depth.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_crossplot_with_property_color(self):
        """Test crossplot with color by property."""
        print("\n3. Color Mapping Test (property)")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(
                x="PHIE",
                y="PERM",
                color="VSH",
                colortemplate="RdYlGn_r",
                title="Color by VSH"
            )
            plot.plot()

            if plot.colorbar is not None:
                print("  ✓ Property color mapping successful")
                self.tests_passed += 1
            else:
                print("  ✗ FAILED: Colorbar not created")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "color_property.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_crossplot_with_size_mapping(self):
        """Test crossplot with size mapping."""
        print("\n4. Size Mapping Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(
                x="PHIE",
                y="SW",
                size="PERM",
                size_range=(20, 200),
                color="depth",
                title="Size by Permeability"
            )
            plot.plot()

            print("  ✓ Size mapping successful")
            self.tests_passed += 1

            save_path = os.path.join(self.temp_dir, "size_mapping.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_log_scale(self):
        """Test logarithmic scales."""
        print("\n5. Logarithmic Scale Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(
                x="PERM",
                y="PHIE",
                x_log=True,
                title="Log Scale X-axis"
            )
            plot.plot()

            if plot.ax.get_xscale() == 'log':
                print("  ✓ X-axis logarithmic scale applied")
                self.tests_passed += 1
            else:
                print("  ✗ FAILED: Log scale not applied")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "log_scale.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_linear_regression(self):
        """Test adding linear regression."""
        print("\n6. Linear Regression Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(x="RHOB", y="NPHI", title="With Linear Regression")
            plot.plot()
            plot.add_regression("linear", line_color="red", line_width=2)

            if "linear" in plot.regressions:
                reg = plot.regressions["linear"]
                print(f"  ✓ Linear regression added")
                print(f"    Equation: {reg.equation()}")
                print(f"    R²: {reg.r_squared:.4f}")
                print(f"    RMSE: {reg.rmse:.4f}")
                self.tests_passed += 1
            else:
                print("  ✗ FAILED: Regression not added")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "linear_regression.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_multiple_regressions(self):
        """Test adding multiple regressions."""
        print("\n7. Multiple Regressions Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(x="PHIE", y="SW", title="Multiple Regressions")
            plot.plot()

            plot.add_regression("linear", line_color="red")
            plot.add_regression("polynomial", degree=2, line_color="blue")
            plot.add_regression("polynomial", degree=3, line_color="green", name="cubic")

            if len(plot.regressions) == 3:
                print(f"  ✓ Added 3 regressions")
                for name, reg in plot.regressions.items():
                    print(f"    {name}: R² = {reg.r_squared:.4f}")
                self.tests_passed += 1
            else:
                print(f"  ✗ FAILED: Expected 3 regressions, got {len(plot.regressions)}")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "multiple_regressions.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_multiwell_crossplot(self):
        """Test multi-well crossplot from Manager."""
        print("\n8. Multi-Well Crossplot Test")
        print("-" * 50)

        try:
            plot = self.manager.Crossplot(
                x="PHIE",
                y="SW",
                shape="well",
                color="depth",
                title="Multi-Well Crossplot"
            )
            plot.plot()

            data = plot._prepare_data()
            unique_wells = data['well'].nunique()

            if unique_wells == 3:
                print(f"  ✓ Plotted data from {unique_wells} wells")
                print(f"  ✓ Total data points: {len(data)}")
                self.tests_passed += 1
            else:
                print(f"  ✗ FAILED: Expected 3 wells, got {unique_wells}")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "multiwell.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_multiwell_specific_wells(self):
        """Test multi-well crossplot with specific wells."""
        print("\n9. Multi-Well Specific Wells Test")
        print("-" * 50)

        try:
            plot = self.manager.Crossplot(
                x="RHOB",
                y="NPHI",
                wells=["Well_A", "Well_B"],
                shape="well",
                title="Wells A and B Only"
            )
            plot.plot()

            data = plot._prepare_data()
            unique_wells = data['well'].nunique()

            if unique_wells == 2:
                print(f"  ✓ Plotted specific wells only ({unique_wells} wells)")
                self.tests_passed += 1
            else:
                print(f"  ✗ FAILED: Expected 2 wells, got {unique_wells}")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "multiwell_specific.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_depth_filtering(self):
        """Test depth range filtering."""
        print("\n10. Depth Range Filtering Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(
                x="PHIE",
                y="SW",
                depth_range=(1200, 1800),
                title="Depth Filtered (1200-1800m)"
            )
            plot.plot()

            data = plot._prepare_data()
            depth_min = data['depth'].min()
            depth_max = data['depth'].max()

            if 1200 <= depth_min and depth_max <= 1800:
                print(f"  ✓ Depth filtering applied")
                print(f"    Depth range: {depth_min:.0f} - {depth_max:.0f} m")
                print(f"    Data points: {len(data)}")
                self.tests_passed += 1
            else:
                print(f"  ✗ FAILED: Depth range not filtered correctly")
                self.tests_failed += 1

            save_path = os.path.join(self.temp_dir, "depth_filtered.png")
            plot.save(save_path)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_all_parameters_combined(self):
        """Test combining all parameters."""
        print("\n11. All Parameters Combined Test")
        print("-" * 50)

        try:
            plot = self.manager.Crossplot(
                x="PHIE",
                y="SW",
                wells=["Well_A", "Well_B"],
                shape="well",
                color="VSH",
                size="PERM",
                colortemplate="viridis",
                color_range=[0, 0.5],
                size_range=(30, 200),
                title="All Features Combined",
                xlabel="Effective Porosity",
                ylabel="Water Saturation",
                figsize=(12, 10),
                dpi=100,
                marker_alpha=0.7,
                edge_color="black",
                edge_width=0.5,
                grid=True,
                depth_range=(1000, 1500),
                show_colorbar=True,
                show_legend=True
            )
            plot.plot()
            plot.add_regression("linear", line_color="red")

            print("  ✓ All parameters applied successfully")
            self.tests_passed += 1

            save_path = os.path.join(self.temp_dir, "all_parameters.png")
            plot.save(save_path, dpi=150)
            plot.close()
            print(f"  ✓ Plot saved: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def test_regression_prediction(self):
        """Test using regression for predictions."""
        print("\n12. Regression Prediction Test")
        print("-" * 50)

        try:
            well = self.manager.well_Well_A

            plot = well.Crossplot(x="RHOB", y="NPHI")
            plot.plot()
            plot.add_regression("linear")

            reg = plot.regressions["linear"]

            # Test predictions
            test_values = np.array([2.2, 2.3, 2.4, 2.5])
            predictions = reg(test_values)

            print(f"  ✓ Regression prediction successful")
            print(f"    Input (RHOB):  {test_values}")
            print(f"    Output (NPHI): {predictions}")
            self.tests_passed += 1

            plot.close()

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            self.tests_failed += 1

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            # List generated files
            files = os.listdir(self.temp_dir)
            print(f"\n  Generated {len(files)} test files:")
            for f in sorted(files):
                if f.endswith('.png'):
                    print(f"    - {f}")

            # Cleanup
            shutil.rmtree(self.temp_dir)
            print(f"\n  ✓ Cleaned up temporary directory")
        except Exception as e:
            print(f"  Warning: Could not clean up: {e}")

    def run_all(self):
        """Run all integration tests."""
        print("\n" + "=" * 70)
        print("CROSSPLOT INTEGRATION TESTS")
        print("=" * 70)

        self.setup_test_data()

        self.test_basic_well_crossplot()
        self.test_crossplot_with_color_mapping()
        self.test_crossplot_with_property_color()
        self.test_crossplot_with_size_mapping()
        self.test_log_scale()
        self.test_linear_regression()
        self.test_multiple_regressions()
        self.test_multiwell_crossplot()
        self.test_multiwell_specific_wells()
        self.test_depth_filtering()
        self.test_all_parameters_combined()
        self.test_regression_prediction()

        self.cleanup()

        return self.tests_passed, self.tests_failed


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = IntegrationTests()
    passed, failed = tests.run_all()

    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Success Rate: {100 * passed / (passed + failed):.1f}%")
    print("=" * 70)

    exit(0 if failed == 0 else 1)
