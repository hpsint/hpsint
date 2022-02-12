# Script that visualizes the contours of an arbitray number
# of particles (described by eta_*=0.5).  
#
# based on a trace generated using paraview version 5.9.1

#### import the simple module from the paraview
from paraview.simple import *

###############################################
# begin MS
###############################################

# generate isolines

def plot_contour(solution0vtu, renderView1, name):
    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=solution0vtu)
    
    # Properties modified on contour1
    contour1.ContourBy = ['POINTS',  name]
    contour1.Isosurfaces = [0.5]
    # show data in view
    contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')
    
    # get color transfer function/color map for 'eta0'
    eta0LUT = GetColorTransferFunction(name)
    
    # trace defaults for the display properties.
    contour1Display.Representation = 'Surface'
    contour1Display.ColorArrayName = ['POINTS', name]
    contour1Display.LookupTable = eta0LUT
    contour1Display.SelectTCoordArray = 'None'
    contour1Display.SelectNormalArray = 'None'
    contour1Display.SelectTangentArray = 'None'
    contour1Display.OSPRayScaleArray = name
    contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    contour1Display.SelectOrientationVectors = 'None'
    contour1Display.ScaleFactor = 1.499977743625641
    contour1Display.SelectScaleArray = name
    contour1Display.GlyphType = 'Arrow'
    contour1Display.GlyphTableIndexArray = name
    contour1Display.GaussianRadius = 0.07499888718128205
    contour1Display.SetScaleArray = ['POINTS', name]
    contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
    contour1Display.OpacityArray = ['POINTS', name]
    contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
    contour1Display.DataAxesGrid = 'GridAxesRepresentation'
    contour1Display.PolarAxes = 'PolarAxesRepresentation'
    
    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
    
    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
    
    
    # show color bar/color legend
    contour1Display.SetScalarBarVisibility(renderView1, True)
    
    # update the view to ensure updated data information
    renderView1.Update()
    
    # get opacity transfer function/opacity map for 'eta0'
    eta0PWF = GetOpacityTransferFunction(name)
    
    # turn off scalar coloring
    ColorBy(contour1Display, None)
    
    # Hide the scalar bar for this color map if no visible data is colored by it.
    HideScalarBarIfNotNeeded(eta0LUT, renderView1)
    
    # change representation type
    contour1Display.SetRepresentationType('Wireframe')
    
    # Properties modified on contour1Display
    contour1Display.LineWidth = 5.0
    
    # set active source
    SetActiveSource(solution0vtu)
    
    # show data in view
    solution0vtuDisplay = Show(solution0vtu, renderView1, 'UnstructuredGridRepresentation')
    
    # show color bar/color legend
    solution0vtuDisplay.SetScalarBarVisibility(renderView1, True)


###############################################
# end MS
###############################################



#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
solution0vtu = GetActiveSource()

# set active source
SetActiveSource(solution0vtu)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
solution0vtuDisplay = Show(solution0vtu, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
solution0vtuDisplay.Representation = 'Surface'
solution0vtuDisplay.ColorArrayName = [None, '']
solution0vtuDisplay.SelectTCoordArray = 'None'
solution0vtuDisplay.SelectNormalArray = 'None'
solution0vtuDisplay.SelectTangentArray = 'None'
solution0vtuDisplay.OSPRayScaleArray = 'aux_00'
solution0vtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
solution0vtuDisplay.SelectOrientationVectors = 'None'
solution0vtuDisplay.ScaleFactor = 3.75
solution0vtuDisplay.SelectScaleArray = 'None'
solution0vtuDisplay.GlyphType = 'Arrow'
solution0vtuDisplay.GlyphTableIndexArray = 'None'
solution0vtuDisplay.GaussianRadius = 0.1875
solution0vtuDisplay.SetScaleArray = ['POINTS', 'aux_00']
solution0vtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
solution0vtuDisplay.OpacityArray = ['POINTS', 'aux_00']
solution0vtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
solution0vtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
solution0vtuDisplay.PolarAxes = 'PolarAxesRepresentation'
solution0vtuDisplay.ScalarOpacityUnitDistance = 2.743426973760503
solution0vtuDisplay.OpacityArrayName = ['POINTS', 'aux_00']

# set scalar coloring
ColorBy(solution0vtuDisplay, ('POINTS', 'c'))

# rescale color and/or opacity maps used to include current data range
solution0vtuDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
solution0vtuDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'c'
cLUT = GetColorTransferFunction('c')

# get opacity transfer function/opacity map for 'c'
cPWF = GetOpacityTransferFunction('c')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
cLUT.ApplyPreset('Cold and Hot', True)

# change representation type
solution0vtuDisplay.SetRepresentationType('Surface With Edges')

# get color legend/bar for cLUT in view renderView1
cLUTColorBar = GetScalarBar(cLUT, renderView1)

# Properties modified on solution0vtu
solution0vtu.TimeArray = 'None'

# show data in view
solution0vtuDisplay = Show(solution0vtu, renderView1, 'UnstructuredGridRepresentation')

# update the view to ensure updated data information
renderView1.Update()


###############################################
# begin MS
###############################################
for key in GetActiveSource().PointData.keys():
    if(key.startswith('eta')):
        plot_contour(solution0vtu, renderView1, key)
###############################################
# end by MS
###############################################


