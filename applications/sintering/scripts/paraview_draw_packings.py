from paraview.simple import *
import argparse
import library
import os
import math
import numpy as np
import re

parser = argparse.ArgumentParser(description='Generate output for a series of packings')
parser.add_argument("-f", "--files", dest="files", nargs='+', required=True, help="Source VTU filenames, can be defined as masks")
parser.add_argument("-o", "--output", type=str, required=False, help="Output folder", default=None)
parser.add_argument("-p", "--mask-particles", type=str, required=False, help="Particles mask regexp", default="my_particles_(.*)")
parser.add_argument("-g", "--mask-grid", type=str, required=False, help="Grid mask regexp", default="my_grid_(.*)")
parser.add_argument("-w", "--width", type=int, required=False, help="Image width", default=2000)
parser.add_argument("-t", "--height", type=int, required=False, help="Image height", default=2000)
parser.add_argument("-x", "--x-rotation", type=int, required=False, help="Rotation about x-axis", default=0)
parser.add_argument("-y", "--y-rotation", type=int, required=False, help="Rotation about y-axis", default=35)
parser.add_argument("-z", "--z-rotation", type=int, required=False, help="Rotation about z-axis", default=0)
parser.add_argument("-r", "--resolution", type=int, required=False, help="Spheres resolution", default=32)
parser.add_argument("-s", "--scale-factor", type=int, required=False, help="Glyph scale factor", default=2)
parser.add_argument("-d", "--scale-field", type=str, required=False, help="Glyph scale field", default="radius")
parser.add_argument("-e", "--edge-width", type=int, required=False, help="Bounding box edge width", default=10)
parser.add_argument("-a", "--opacity", type=float, required=False, help="Bounding box opacity", default=0.3)
parser.add_argument("-v", "--advanced", action='store_true', help="Advanced rendering", required=False, default=False)

args = parser.parse_args()

# Get all files to process
vtu_files = library.get_solutions(args.files)

if not vtu_files:
    raise Exception("The files list is empty, nothing to plot")

# Sort vtu files
pattern_particles = re.compile(args.mask_particles)
pattern_grid = re.compile(args.mask_grid)

vtu_pairs = {}
for file in vtu_files:
    fname = os.path.splitext(os.path.basename(file))[0]
    mp = pattern_particles.match(fname)
    mg = pattern_grid.match(fname)

    if mp and mg:
        raise Exception("Something is not OK with naming of your files")

    if mp:
        pack_name = mp.group(1)
        if not pack_name in vtu_pairs:
            vtu_pairs[pack_name] = {"grid": None, "particles": None}
        vtu_pairs[pack_name]["particles"] = file

    if mg:
        pack_name = mg.group(1)
        if not pack_name in vtu_pairs:
            vtu_pairs[pack_name] = {"grid": None, "particles": None}
        vtu_pairs[pack_name]["grid"] = file

# Check if all pairs have packing at least
for key, data in vtu_pairs.items():
    if not data["particles"]:
        raise Exception("No particles file for packing {}".format(key))
    if not data["grid"]:
        raise Exception("No grid file for packing {}".format(key))

# Output folder path
if args.output:
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    output_folder = args.output
else:
    output_folder = None

# Track the largest bounding box
max_extent = -1
largest_vtu_key = None

# Store glyphs for later rendering
scenes_list = {}

# Create a render view
renderView = CreateView('RenderView')

def rotate_camera_around_x(position, focal_point, angle_degrees):
    """Rotate camera position around the X-axis relative to focal point."""
    angle_rad = math.radians(angle_degrees)
    
    # Translate position to origin-centered coordinates
    dx = position[0] - focal_point[0]
    dy = position[1] - focal_point[1]
    dz = position[2] - focal_point[2]

    # Apply Y-axis rotation
    new_dy = dy * math.cos(angle_rad) + dz * math.sin(angle_rad)
    new_dz = -dy * math.sin(angle_rad) + dz * math.cos(angle_rad)

    # Translate back
    new_position = [
        position[0],  # X stays the same
        focal_point[1] + new_dy,
        focal_point[2] + new_dz
    ]
    return new_position

def rotate_camera_around_y(position, focal_point, angle_degrees):
    """Rotate camera position around the Y-axis relative to focal point."""
    angle_rad = math.radians(angle_degrees)
    
    # Translate position to origin-centered coordinates
    dx = position[0] - focal_point[0]
    dy = position[1] - focal_point[1]
    dz = position[2] - focal_point[2]

    # Apply Y-axis rotation
    new_dx = dx * math.cos(angle_rad) + dz * math.sin(angle_rad)
    new_dz = -dx * math.sin(angle_rad) + dz * math.cos(angle_rad)

    # Translate back
    new_position = [
        focal_point[0] + new_dx,
        position[1],  # Y stays the same
        focal_point[2] + new_dz
    ]
    return new_position

def rotate_camera_around_z(position, focal_point, angle_degrees):
    """Rotate camera position around the Z-axis relative to focal point."""
    angle_rad = math.radians(angle_degrees)
    
    # Translate position to origin-centered coordinates
    dx = position[0] - focal_point[0]
    dy = position[1] - focal_point[1]
    dz = position[2] - focal_point[2]

    # Apply Y-axis rotation
    new_dx = dx * math.cos(angle_rad) + dy * math.sin(angle_rad)
    new_dy = -dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

    # Translate back
    new_position = [
        focal_point[0] + new_dx,
        focal_point[1] + new_dy,
        position[2],  # Z stays the same
    ]
    return new_position

def compute_bounding_box_with_radius(data_source):
    """Compute axis-aligned bounding box expanded by 'radius'."""
    data_source.UpdatePipeline()
    info = servermanager.Fetch(data_source)
    
    points = np.array(info.GetPoints().GetData())
    radius_array = np.array(info.GetPointData().GetArray("radius"))
    
    # Expand each point by its radius
    min_coords = np.min(points - radius_array[:, None], axis=0)
    max_coords = np.max(points + radius_array[:, None], axis=0)
    
    return min_coords, max_coords

for key, data in vtu_pairs.items():
    file_particles = data["particles"]
    file_grid = data["grid"]

    # Read VTU files
    particles = XMLUnstructuredGridReader(FileName=[file_particles])
    grid = XMLUnstructuredGridReader(FileName=[file_grid])

    # Compute bounding box with radius
    try:
        min_coords, max_coords = compute_bounding_box_with_radius(particles)
    except Exception as e:
        print(f"Skipping {key}: {e}")
        continue

    extent_vector = max_coords - min_coords
    bbox_diagonal = np.linalg.norm(extent_vector)

    # Apply Glyph filter
    glyph = Glyph(Input=particles, GlyphType='Sphere')
    glyph.ScaleArray = args.scale_field
    glyph.ScaleFactor = args.scale_factor
    glyph.GlyphMode = 'All Points'
    
    # Set sphere resolution
    glyph.GlyphType.ThetaResolution = args.resolution
    glyph.GlyphType.PhiResolution = args.resolution

    # Extract edges
    edges = ExtractEdges(Input=grid)

    scenes_list[key] = (glyph, grid, edges)

    if bbox_diagonal > max_extent:
        max_extent = bbox_diagonal
        largest_vtu_key = key

# Show packing
def show_packing(key):
    display_particles = Show(scenes_list[key][0], renderView)
    #display_particles.DiffuseColor = [0, 0, 1]
    display_grid = Show(scenes_list[key][1], renderView)
    display_grid.Opacity = args.opacity
    display_edges = Show(scenes_list[key][2], renderView)
    display_edges.DiffuseColor = [0, 0, 0]  # black edges
    display_edges.LineWidth = args.edge_width

# Show packing
def hide_packing(key):
    Hide(scenes_list[key][0], renderView)
    Hide(scenes_list[key][1], renderView)
    Hide(scenes_list[key][2], renderView)

# Hide axes
renderView.OrientationAxesVisibility = 0

# Set view to fit the largest bounding box
show_packing(largest_vtu_key)
renderView.ResetCamera()

# Optional: tighten zoom (lower view angle = more zoomed out)
renderView.CameraViewAngle = 30  # default is ~30–45

# Rotate around Y axis by the given degrees
rotated_position = rotate_camera_around_x(
    renderView.CameraPosition,
    renderView.CameraFocalPoint,
    args.x_rotation
)
renderView.CameraPosition = rotated_position

# Rotate around Y axis by the given degrees
rotated_position = rotate_camera_around_y(
    renderView.CameraPosition,
    renderView.CameraFocalPoint,
    args.y_rotation
)
renderView.CameraPosition = rotated_position

# Rotate around Z axis by the given degrees
rotated_position = rotate_camera_around_z(
    renderView.CameraPosition,
    renderView.CameraFocalPoint,
    args.z_rotation
)
renderView.CameraPosition = rotated_position

# Save camera settings
camera_position = renderView.CameraPosition
camera_focal_point = renderView.CameraFocalPoint
camera_view_up = renderView.CameraViewUp

# Enable ray tracing and ambient occlusion
renderView.BackEnd = 'OSPRay raycaster'
if args.advanced:
    renderView.EnableRayTracing = 1
    renderView.Shadows = 0  # Optional: turn off shadows if not needed
    renderView.UseAmbientOcclusion = 1
    #renderView.AmbientSamples = 10  # Typical range: 5–20

# Enable tone mapping
renderView.UseToneMapping = 1
#renderView.ToneMappingType = 3 # Clamp = 0, Reinhard = 1, Exponential = 2, GenericFilmic = 3, NeutralPBR = 4
renderView.Exposure = 1.0  # Higher = brighter; try values like 0.8–1.5

# Function to output a packing
def output_packing(key):
    data = vtu_pairs[key]
    file_particles = data["particles"]
    file_grid = data["grid"]

    Render()
    image_name = key + ".png"

    local_output_folder = output_folder
    if not local_output_folder:
        local_output_folder = os.path.dirname(file_particles)

    SaveScreenshot(os.path.join(local_output_folder, image_name), renderView, TransparentBackground=1, ImageResolution=[args.width, args.height])

# Output at first the largest one
output_packing(largest_vtu_key)

hide_packing(largest_vtu_key)

# Render and save images with consistent camera settings
for key, data in scenes_list.items():
    if key == largest_vtu_key:
        continue

    show_packing(key)

    renderView.CameraPosition = camera_position
    renderView.CameraFocalPoint = camera_focal_point
    renderView.CameraViewUp = camera_view_up
    
    output_packing(key)

    hide_packing(key)

Delete(renderView)
