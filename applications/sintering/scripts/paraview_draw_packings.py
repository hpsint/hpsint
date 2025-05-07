from paraview.simple import *
import os
import math
import numpy as np

# Path to VTU files
vtu_folder = "c:\\Work\\HZG\\TUM\\clouds\\"
output_folder = "c:\\Work\\HZG\\TUM\\clouds\\img"

vtu_files = [os.path.join(vtu_folder, f) for f in os.listdir(vtu_folder) if f.endswith(".vtu")]

# Track the largest bounding box
max_extent = -1
largest_glyph = None
largest_vtu = None

# Store glyphs for later rendering
glyphs_list = []

# Create a render view
renderView = CreateView('RenderView')

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

for file in vtu_files:
    # Read VTU file
    particles = XMLUnstructuredGridReader(FileName=[file])
    
    # Compute bounding box with radius
    try:
        min_coords, max_coords = compute_bounding_box_with_radius(particles)
    except Exception as e:
        print(f"Skipping {file}: {e}")
        continue

    extent_vector = max_coords - min_coords
    bbox_diagonal = np.linalg.norm(extent_vector)
    
    # Apply Glyph filter
    glyph = Glyph(Input=particles, GlyphType='Sphere')
    glyph.ScaleArray = 'radius'
    glyph.ScaleFactor = 2.0
    glyph.GlyphMode = 'All Points'
    
    # Set sphere resolution
    glyph.GlyphType.ThetaResolution = 32
    glyph.GlyphType.PhiResolution = 32

    ss = Sphere(ThetaResolution=1000, PhiResolution=500) 

    #print(dir(glyph))
    #exit()
    
    glyphs_list.append((file, glyph))
    
    if bbox_diagonal > max_extent:
        max_extent = bbox_diagonal
        largest_glyph = glyph
        largest_vtu = file

# Hide axes
renderView.OrientationAxesVisibility = 0

# Set view to fit the largest bounding box
Show(largest_glyph, renderView)
renderView.ResetCamera()

# Optional: tighten zoom (lower view angle = more zoomed out)
renderView.CameraViewAngle = 30  # default is ~30–45

# Rotate around Y axis by the given degrees, for example
angle_deg = 35
rotated_position = rotate_camera_around_y(
    renderView.CameraPosition,
    renderView.CameraFocalPoint,
    angle_deg
)
renderView.CameraPosition = rotated_position

# Save camera settings
camera_position = renderView.CameraPosition
camera_focal_point = renderView.CameraFocalPoint
camera_view_up = renderView.CameraViewUp

# Function to output a packing
def output_packing(filename):
    Render()
    image_name = os.path.splitext(os.path.basename(filename))[0] + ".png"
    SaveScreenshot(os.path.join(output_folder, image_name), renderView, TransparentBackground=1, ImageResolution=[2000, 2000])

# Output at first the largest one
output_packing(largest_vtu)

Hide(largest_glyph, renderView)

# Render and save images with consistent camera settings
for file, glyph in glyphs_list:
    if file == largest_vtu:
        continue

    Show(glyph, renderView)
    renderView.CameraPosition = camera_position
    renderView.CameraFocalPoint = camera_focal_point
    renderView.CameraViewUp = camera_view_up
    
    output_packing(file)

    Hide(glyph, renderView)

Delete(renderView)
