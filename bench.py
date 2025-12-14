import argparse
import logging
import sys

import cadquery as cq
import numpy as np
import svgpathtools

from utils import *

from OCP.StdFail import StdFail_NotDone
from more_itertools import powerset

logger = logging.getLogger(__name__)

#
# CadQuery
#

cadquery_extra = 0.005


class SolidSelector(cq.Selector):
    """
    Custom selector to filter objects based on containment within a solid.
    """

    def __init__(self, solid) -> None:
        super().__init__()
        self._solid = solid  # Reference solid boundary

    def filter(self, object_list):
        """
        Filters objects, keeping only those entirely inside the solid.

        Args:
            object_list: List of geometry objects to evaluate
        """

        def is_enclosed(obj):
            # Check if both start and end points of the object are within bounds
            points = [obj.startPoint(), obj.endPoint()]
            return all(self._solid.isInside(pt) for pt in points)

        return filter(is_enclosed, object_list)


#
# Plate
#


def fillet_size(length):
    """Calculates a standard fillet radius based on part length."""
    return length / 5


def try_fillet(body_o, workspace, radius):
    """
    Attempts to apply a fillet to various edge combinations until successful.

    Args:
        body_o: Original solid body fallback
        workspace: CadQuery workspace containing target edges
        radius: Fillet radius to apply
    """
    # Iterate through the powerset of edges in reverse (most edges first)
    for x in reversed(list(powerset(range(workspace.size())))[1:]):
        try:
            # Create a selection of edges based on current power set index
            selection = workspace.newObject([workspace.objects[i] for i in x])
            return selection.fillet(radius)
        except StdFail_NotDone as _:
            pass

    # Fallback if no edge combination can be filleted
    logger.exception(f"Can't create base fillet for column {workspace}")
    return body_o


def is_point_inside(point, path):
    """
    Checks if a complex point is inside an svgpathtools.Path using Shapely.

    Args:
        point (complex): The point to test.
        path (svgpathtools.Path): The closed path boundary.

    Returns:
        bool: True if the point is inside or on the boundary.
    """
    # 1. Convert the svgpathtools Path (via sampled points) into a Shapely Polygon.
    points = create_path(path, sub_div=20)
    shapely_poly = shapely.geometry.Polygon(points)

    # 2. Convert the complex point to a Shapely Point.
    p = shapely.geometry.Point(point.real, point.imag)

    # 3. Check for containment (covers includes the boundary).
    return shapely_poly.covers(p)


def get_segments(outline, named_overlapped_areas, default_area=None):
    """
    Subdivides an outline into segments based on its intersections with overlapping areas.

    Args:
        outline: (path, attributes) tuple.
        named_overlapped_areas: List of (name, (path, attributes)) to check for overlap.
        default_area: Label for segments not covered by named areas.
    """

    def get_cuts(zone):
        """
        Finds parameter values (T1) on the outline where it intersects a zone.

        Args:
            zone: (path, attributes) tuple representing the intersecting region.
        """
        # Find points where outline and zone boundaries cross
        intersections = [T1 for T1, T2 in find_intersections(outline[0], zone[0], samples=int(outline[0].length()))]

        # We assume zones are convex or simple, resulting in exactly two crossings
        assert len(intersections) == 2, "Expected entry and exit intersection points"

        return intersections

    # Collect and sort unique parameter values (cuts) along the outline path
    cuts = sorted(list(set(sum([get_cuts(area) for _, area in named_overlapped_areas], []))))

    segments = []
    for i in range(len(cuts)):
        # Define the segment boundaries (using wrap-around index i-1 for start)
        start, end = cuts[i - 1], cuts[i]

        # Calculate midpoint, handling cases where the path wraps from t=1 back to t=0
        mid = (start + end + 1) / 2 if start > end else (start + end) / 2
        if mid >= 1:
            mid -= 1

        # Find which named zones contain the current segment's midpoint
        areas = {name for name, (path, attrs) in named_overlapped_areas
                 if is_point_inside(outline[0].point(mid), path)}

        assert len(areas) < 2
        area = next(iter(areas)) if areas else default_area
        segments.append((area, (start, end)))

    return segments


def get_screw_position(pcb_outline, screw):
    """
    Calculates the 3D position and normal vector for a screw mount.

    Args:
        pcb_outline: List containing the SVG path of the PCB outline at index 0
        screw: List containing the path representing screw alignment at index 0
    """
    # 1. Locate the intersection on the PCB outline
    (T1, _, _), _ = pcb_outline[0].intersect(screw[0], justonemode=True)

    # 2. Get coordinates (p) at path parameter T1
    p = pcb_outline[0].point(T1)

    # 3. Derive the direction vector (v) from the screw path segment
    v = np.subtract(screw[0].start, screw[0].end)
    v = v / np.linalg.norm(v)  # Normalize to unit vector

    # 4. Correct orientation
    # Ensure the vector points outward relative to the coordinate origin
    plus = np.linalg.norm(p + v) > np.linalg.norm(p)
    if not plus:
        v *= -1

    return euclid_to_tuple(p, True), euclid_to_tuple(v, True)


def create_plate(body_o, pcb_outline, args):
    """
    Generates a mounting plate with screw holes based on the PCB outline.

    Args:
        body_o: Main solid body reference
        pcb_outline: List containing the SVG path of the PCB outline
        args: Design parameters (screw diameter, wall thickness, etc.)

    Returns:
        tuple: (CadQuery object of the plate, list of generated SVG paths)
    """
    bottom_draw = []

    # 1. Calculate boundaries based on screw size and wall thickness
    offset = max(args.box_screw_diameter, args.wall_thickness)
    screw_outline = offset_curve(pcb_outline[0], args.wall_play + offset)
    screw_bbox = screw_outline.bbox()

    # 2. Define the inner boundary path for the plate
    offset_inline_bbox = (
        screw_bbox[0] - offset, screw_bbox[1] + offset,
        screw_bbox[2] - offset, screw_bbox[3] + offset
    )
    offset_inline = bbox2path(*offset_inline_bbox, offset if args.extra_fillet > 1 else 0)
    bottom_draw.append(offset_inline)

    # 3. Define the outer boundary path for the plate
    offset_outline_bbox = (
        screw_bbox[0] - (offset + args.box_wall_thickness),
        screw_bbox[1] + (offset + args.box_wall_thickness),
        screw_bbox[2] - (offset + args.box_wall_thickness),
        screw_bbox[3] + (offset + args.box_wall_thickness)
    )
    offset_outline = bbox2path(*offset_outline_bbox,
                               offset + args.box_wall_thickness if args.extra_fillet > 1 else 0)
    bottom_draw.append(offset_outline)

    # 4. Create the solid plate and fillet the top edges
    result = to_wire(offset_outline, body_o)
    result = result.extrude(args.bottom, True).faces("+Z").fillet(args.box_wall_thickness)

    # 5. Generate screw holes at the bounding box corners
    result = result.faces("<Z").workplane()
    corners = [
        (screw_bbox[0], screw_bbox[2]), (screw_bbox[0], screw_bbox[3]),
        (screw_bbox[1], screw_bbox[2]), (screw_bbox[1], screw_bbox[3])
    ]

    for x, y in corners:
        # Create physical hole in the CAD object
        result = result.moveTo(x, y).hole(args.box_screw_diameter, args.bottom - args.box_screw_offset)
        # Record the hole path for the 2D export
        bottom_draw.append(circle_path(x, y, args.box_screw_diameter / 2))

    # 6. Translate the final object to the correct relative position
    result = result.translate((0, 0, -(args.bottom + args.under_space))).faces(">Z").workplane()

    return result, bottom_draw


def create_wall(body_o, pcb_outline, fingers, args):
    """
    Creates a protective wall around the PCB with specific cutouts for fingers/connectors.

    Args:
        body_o: Main solid body
        pcb_outline: List containing the SVG path of the PCB outline
        fingers: List of paths indicating where the wall should be cut
        args: Parameters for wall height, thickness, play, and fillets
    """

    # 1. Define inner and outer boundaries based on wall thickness and play
    inner_outline = offset_curve(pcb_outline[0], args.wall_play)
    outer_outline = offset_curve(pcb_outline[0], args.wall_play + args.wall_thickness)

    # 2. Extrude boundaries to create the wall base shape
    wall_height_total = args.under_space + args.pcb_thickness + args.wall_height
    wall_o = to_wire(outer_outline, body_o).extrude(wall_height_total, False)
    wall_o = to_wire(inner_outline, wall_o).extrude(wall_height_total, "cut")

    # 3. Process cutouts (fingers) along the wall
    for _, segment in filter(lambda x: x[0], get_segments(pcb_outline, [(True, finger) for finger in fingers])):
        # Create a closed path loop to subtract material for connector access
        path_segment = pcb_outline[0].cropped(segment[0], segment[1])
        cut_outline_1 = offset_curve(path_segment, args.wall_play + args.wall_thickness * 2)
        cut_outline_2 = path_segment.reversed()

        # Combine segments into a manifold path for extrusion
        cut_outline = svgpathtools.Path(
            *(cut_outline_2[:] + [svgpathtools.Line(cut_outline_2.end, cut_outline_1.start)] + cut_outline_1[:]) +
             [svgpathtools.Line(cut_outline_1.end, cut_outline_2.start)]
        )

        # Create the subtractive solid for the finger cutout
        wall_o = to_wire(cut_outline, wall_o).extrude(wall_height_total * 2, "cut")

    # 4. Tag base edges for later filleting and combine with main body
    wall_o.edges("<Z").tag("wall_base")
    body_o = body_o.union(wall_o)

    # 5. Apply optional fillets to top and base edges
    if args.extra_fillet > 3:
        # Base fillet for structural strength
        body_o = body_o.edges(tag="wall_base").fillet(args.wall_fillet)
        # Top fillet for handling comfort
        body_o = body_o.edges(">Z").fillet(args.wall_thickness / 3)
    return body_o


def create_screw(body_o, pcb_outline, screw, args):
    """
    Generates screw mounting geometry including the boss and the clearance hole.

    Args:
        body_o: The main solid body to be modified
        pcb_outline: The SVG path object of the PCB
        screw: The path object defining the screw location/orientation
        args: Namespace containing design parameters (diameters, wall thickness, etc.)
    """
    # 1. Retrieve 3D coordinates and orientation vector
    position, vector = get_screw_position(pcb_outline, screw)

    # 2. Adjust position for PCB thickness and user-defined offset
    position = np.array((
        position[0],
        position[1],
        position[2] + args.pcb_thickness + args.fixture_screw_offset
    ))

    # 3. Define the start point for the screw boss (extrusion)
    extra_position = position + np.array(vector) * (args.play + args.wall_thickness)

    # 4. Create the Screw Boss (Additive)
    # Define a custom workplane perpendicular to the normal vector
    plane_screw = cq.Workplane(cq.Plane(
        origin=tuple(extra_position),
        xDir=tuple(np.cross(vector, [0, 0, 1])),
        normal=vector
    ))
    screw_boss = plane_screw.circle(args.fixture_screw_diameter / 2) \
        .extrude(args.fixture_screw_extra, False)
    body_o = body_o.union(screw_boss)

    # 5. Create the Clearance Hole (Subtractive)
    plane_hole = cq.Workplane(cq.Plane(
        origin=tuple(position),
        xDir=tuple(np.cross(vector, [0, 0, 1])),
        normal=vector
    ))
    hole_o = plane_hole.circle(args.fixture_screw_hole / 2) \
        .extrude(args.play + args.wall_thickness + args.fixture_screw_extra, False)
    body_o = body_o.cut(hole_o)

    # 6. Apply fillets to the boss edges if requested
    if args.extra_fillet > 2:
        # Fillet the front edge (screw entry side)
        body_o = body_o.edges(SolidSelector(screw_boss.translate(tuple(np.array(vector))).solids().val())) \
            .fillet((args.fixture_screw_diameter - args.fixture_screw_hole) / 4 - cadquery_extra)

        # Fillet the base (attachment to the body)
        body_o = body_o.edges(SolidSelector(screw_boss.translate(tuple(-np.array(vector))).solids().val())) \
            .fillet(fillet_size(args.fixture_screw_diameter))

    return body_o


def create_column(body_o, path, length, fillet=0):
    """
    Extrudes a column from an SVG path, unions it to the body, and applies fillets.

    Args:
        body_o: The main solid body to modify
        path: SVG path defining the column profile
        length: Extrusion distance
        fillet: Fillet radius for the column edges
    """
    # 1. Convert path to wire and extrude to create the column solid
    column_o = to_wire(path, body_o).extrude(length, False)

    # 2. Add the column to the main body
    body_o = body_o.union(column_o)

    # 3. Apply fillets if requested
    if fillet > 0:
        # Fillet the 'top' edges
        # SolidSelector finds edges belonging to the translated reference solid
        body_o = body_o.edges(SolidSelector(column_o.translate((0, 0, 1)).solids().val())) \
            .fillet(fillet)

        # Fillet the 'base' edges (attachment point)
        # Uses 'try_fillet' fallback logic to handle potentially complex edge geometry
        body_o = try_fillet(
            body_o,
            body_o.edges(SolidSelector(column_o.translate((0, 0, -1)).solids().val())),
            min(fillet_size(length), fillet)
        )

    return body_o


class ShapeTypes(object):
    Outline = style_filter(stroke='#00ff00')
    Support = style_filter(stroke='#ff0000')
    Hole = style_filter(stroke='#000000')
    Filler = style_filter(stroke='#0000ff')
    Screw = style_filter(stroke='#ff00ff')
    Finger = style_filter(stroke='#00ffff')


def generate(args):
    """
    Main orchestration function: processes SVG layers and builds the 3D enclosure.

    Args:
        args: Namespace containing file paths and design dimensions.
    """
    # 1. Load SVG and normalize scale/alignment
    paths, attributes = svg2paths(args.svg)
    elements = list(zip(paths, attributes))
    pcb_outline = list(filter(ShapeTypes.Outline, elements))[0]

    # Rescale all geometry based on the board outline reference
    paths = center_rescale(pcb_outline, paths, args.scale, mirror_y=not args.mirrored)
    elements = list(zip(paths, attributes))
    pcb_outline = list(filter(ShapeTypes.Outline, elements))[0]
    pcb_outline = (closed_path_sanitizing(pcb_outline[0]), pcb_outline[1])

    # 2. Initialize the base solid (Mounting Plate)
    body_o = cq.Workplane("front")
    body_o, bottom_draw = create_plate(body_o, pcb_outline, args)

    # Cache the workplane state to ensure all components reference the same origin
    plane = body_o.plane

    # 3. Build the outer walls with finger cutouts
    fingers = list(filter(ShapeTypes.Finger, elements))
    body_o.plane = plane
    body_o = create_wall(body_o, pcb_outline, fingers, args)

    # 4. Generate Support Pillars (Additive)
    supports = list(filter(ShapeTypes.Support, elements))
    for support in supports:
        body_o.plane = plane
        path = closed_path_sanitizing(support[0])
        body_o = create_column(body_o, path, args.under_space,
                               args.support_fillet if args.extra_fillet > 2 else 0)

    # 5. Generate Fillers (Additive going through PCB holes)
    fillers = list(filter(ShapeTypes.Filler, elements))
    for filler in fillers:
        body_o.plane = plane
        path = closed_path_sanitizing(filler[0])
        path = offset_curve(path, -args.play)
        body_o = create_column(body_o, path,
                               args.under_space + args.pcb_thickness + args.filler_height,
                               args.filler_fillet if args.extra_fillet > 2 else 0)

    # 6. Process Holes (Subtractive features/Pins)
    holes = list(filter(ShapeTypes.Hole, elements))
    for hole in holes:
        body_o.plane = plane
        if len(hole[0]) == 1:  # Handle circles/points
            x1, x2, y1, y2 = hole[0].bbox()
            body_o = body_o.moveTo((x1 + x2) / 2, (y1 + y2) / 2) \
                .circle(args.pin_diameter / 2) \
                .extrude(args.bottom, "cut", both=True)
        else:  # Handle complex cutout paths
            path = closed_path_sanitizing(hole[0])
            body_o = to_wire(path, body_o).extrude(args.bottom, "cut", both=True)

    # 7. Generate Screw Mounts (Additive Boss + Subtractive Hole)
    screws = list(filter(ShapeTypes.Screw, elements))
    for screw in screws:
        body_o.plane = plane
        body_o = create_screw(body_o, pcb_outline, screw, args)

    # 8. Export result to both SVG (2D drawing) and CAD (3D STEP/STL)
    svgpathtools.wsvg(bottom_draw, filename=f'{args.output}.svg')
    cq.exporters.export(body_o, args.output)

    # 9. Optional: GUI Display
    try:
        show_object(body_o, name='plate')
    except BaseException:
        pass

    print(f"Exported to {args.output}")


def _main(argv=sys.argv):
    def auto_int(x):
        return int(x, 0)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(prog=argv[0],
                                     description='TestBenchGenerator: Parametric PCB Test Fixture Generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count', default=0,
                        help="Increases log verbosity (e.g., from INFO to DEBUG) for each occurrence.")

    # PCB & Base Plate Parameters
    parser.add_argument('-t', '--pcb-thickness', default=1.6, type=float,
                        help="Thickness of the PCB in mm.")
    parser.add_argument('-b', '--bottom', default=8.0, type=float,
                        help="Thickness (in mm) of the main bench base plate.")
    parser.add_argument('-u', '--under-space', default=5.0, type=float,
                        help="Vertical clearance (in mm) between the bottom surface of the PCB and the top surface of the bench base plate.")

    # Pogo Pin / Hole Parameters
    parser.add_argument('-d', '--pin-diameter', default=0.95, type=float,
                        help="Diameter (in mm) for the pogo pin holes (spring-loaded contacts).")

    # Side Wall Parameters
    parser.add_argument('--wall-thickness', default=2.0, type=float,
                        help="Thickness (in mm) of the protective side walls surrounding the PCB.")
    parser.add_argument('--wall-fillet', default=2.0, type=float,
                        help="Fillet radius (in mm) applied to the corners of the bench side walls.")
    parser.add_argument('--wall-height', default=5.0, type=float,
                        help="Height (in mm) of the protective side walls.")
    parser.add_argument('--wall-play', default=0.4, type=float,
                        help="Lateral clearance (in mm) between the PCB perimeter and the surrounding bench walls (X, Y axes).")

    # Guide Pin (Filler) Parameters
    parser.add_argument('--filler-height', default=5.0, type=float,
                        help="Height (in mm) of the guide pins/columns (Fillers, Blue SVG) protruding through PCB holes.")

    # Fixture Clamping Mechanism Parameters
    parser.add_argument('--fixture-screw-hole', default=2.8, type=float,
                        help="Diameter (in mm) for the hole that mounts the fixture clamping mechanism (e.g., ballscrew).")
    parser.add_argument('--fixture-screw-diameter', default=5.0, type=float,
                        help="Outer diameter (in mm) of the boss/standoff feature generated to support the clamping mechanism.")
    parser.add_argument('--fixture-screw-offset', default=0.5, type=float,
                        help="Z-offset (in mm) of the clamp's contact surface relative to the top surface of the PCB.")
    parser.add_argument('--fixture-screw-extra', default=3.0, type=float,
                        help="Extra wall thickness (in mm) added to reinforce the fixture screw boss/standoff.")
    # Fillet Parameters
    parser.add_argument('--support-fillet', default=1.0, type=float,
                        help="Fillet radius (in mm) applied to the top edges of PCB support features.")
    parser.add_argument('--filler-fillet', default=0.5, type=float,
                        help="Fillet radius (in mm) applied to the top edges of filler/guide pins.")
    parser.add_argument('--extra-fillet', default=4, type=int,
                        help="Enables or sets the size/iteration of extra, secondary fillets applied to global features.")

    # Test Bench Mounting Parameters
    parser.add_argument('--box-wall-thickness', default=2.0, type=float,
                        help="Thickness (in mm) of the walls/standoffs used for mounting the entire test bench assembly.")
    parser.add_argument('--box-screw-diameter', default=2.45, type=float,
                        help="Diameter (in mm) of the screw holes for mounting the entire test bench assembly.")
    parser.add_argument('--box-screw-offset', default=1.00, type=float,
                        help="Depth (in mm) of the counterbore or screw head cavity relative to the top of the mounting plate.")

    # Global Parameters
    parser.add_argument('-z', '--scale', default=0.264583, type=float,
                        help="Scaling factor to convert SVG coordinates to working units (mm).")
    parser.add_argument('-p', '--play', default=0.15, type=float,
                        help="Mechanical clearance (tolerance) applied between manufactured parts/features in mm.")
    parser.add_argument('-m', '--mirrored', default=False, action='store_true',
                        help="If set, mirrors the SVG geometry along the X-axis (top-bottom flip).")

    # File Arguments
    parser.add_argument('svg',
                        help="Path to the PCB SVG input outline file.")
    parser.add_argument('output',
                        help="Path to the output file (e.g., `.step` or `.stl`).")

    # Parse
    args, unknown_args = parser.parse_known_args(argv[1:])

    # Set logging level
    logging.getLogger().setLevel(max(3 - args.verbose_count, 0) * 10)

    generate(args)

    return 0


# ------------------------------------------------------------------------------

def main():
    try:
        sys.exit(_main(sys.argv))
    except Exception as e:
        logger.exception(e)
        sys.exit(-1)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    main()
