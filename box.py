import argparse
import logging
import math
import sys

import cadquery as cq
import numpy as np
import svgpathtools

from utils import *

logger = logging.getLogger(__name__)


class ShapeTypes(object):
    Outline = style_filter(stroke='#000000')
    Anchor = style_filter(stroke='#ff0000')
    Clip = style_filter(stroke='#00ff00')


def generate_profiles(args):
    """
    Generates 2D cross-sectional paths (W, Z, S, C) for structural components
    based on input geometry (angles, thickness, play).

    Args:
        args (object): Configuration arguments.
    """

    # 1. Validation checks to ensure physical feasibility and correct alignment.
    assert args.wall_extra > 0, "You must have some extra wall"
    if not args.junction_flipped:
        assert args.pcb_thickness / 2 + args.pcb_z >= args.junction_height / 2, "The upper PCB support is on the wrong part. Change the pcb offset"
    if args.junction_flipped:
        assert -args.pcb_thickness / 2 + args.pcb_z <= -args.junction_height / 2, "The lower PCB support is on the wrong part. Change the pcb offset"

    # 2. GEOMETRIC CALCULATIONS

    # Calculate vertical step for Z-profile based on the junction angle.
    z_center_height = math.tan(args.z_angle / 180 * math.pi) * args.junction_width
    assert z_center_height * 2 <= args.junction_height

    # Calculate horizontal chamfer distance for C-profile.
    c_center = (args.junction_height / 2) / math.tan(args.c_angle / 180 * math.pi)

    # Calculate total bounding box for the profiles including thickness and play.
    width = (args.wall_thickness * 2) + args.play + z_center_height
    height = args.junction_height + args.play + args.wall_extra * 2

    # Determine PCB support dimensions based on support angle and wall offsets.
    pcb_support_width = args.pcb_cover_offset + args.wall_offset
    pcb_support_height = math.tan(args.pcb_support_angle / 180 * math.pi) * pcb_support_width

    # Determine internal clearance width for the PCB profile.
    pcb_clearance_width = -args.wall_offset + args.pcb_clearance_offset

    # Define the offset as half of the intended mechanical play.
    offset = args.play / 2

    # 3. HELPER FUNCTION

    def points_to_path(pts):
        """
        Converts a list of (x, y) coordinate tuples into an svgpathtools.Path
        composed solely of connecting Line segments.

        Args:
            pts (list): A list of (float, float) tuples representing the vertices.

        Returns:
            svgpathtools.Path: A path object consisting of Line segments connecting
                               each point sequentially.
        """
        # Convert (x, y) tuples to complex numbers (Euclidean points).
        pts = list(map(tuple_to_euclid, pts))

        # Build Path by creating a Line segment between each adjacent pair of points.
        return svgpathtools.Path(*[svgpathtools.Line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)])

    def get_block(width, height, line: svgpathtools.Path, wall_offset, offset, extra=0.0):
        """
        Creates a closed 'block' path by offsetting a line and closing it
        with rectangular segments based on the wall full size (width, height).

        Args:
            width (float): The target width of the wall.
            height (float): The target height of the wall.
            line (svgpathtools.Path): The base path to offset.
            offset (float): The distance for the offset curve. (Must be non-zero).
            extra (float, optional): Extra dimension added to the wall for margins. Defaults to 0.0.

        Returns:
            svgpathtools.Path: The new, closed profile block path.
        """

        # Calculate the offset curve.
        path = offset_curve(line, offset)

        if offset > 0:
            # Close the shape by wrapping around the bottom corners of the bounding box.
            path.append(svgpathtools.Line(path[-1].end, complex(wall_offset + width + extra, -height / 2 - extra)))
            path.append(svgpathtools.Line(path[-1].end, complex(wall_offset - extra, -height / 2 - extra)))
            path.append(svgpathtools.Line(path[-1].end, path[0].start))

        elif offset < 0:
            # Close the shape by wrapping around the top corners of the bounding box.
            path.append(svgpathtools.Line(path[-1].end, complex(wall_offset + width + extra, height / 2 + extra)))
            path.append(svgpathtools.Line(path[-1].end, complex(wall_offset - extra, height / 2 + extra)))
            path.append(svgpathtools.Line(path[-1].end, path[0].start))

        else:
            raise RuntimeError("Can't do block if offset 0")

        return path

    # 4. W-PROFILE (Full Wall)
    w_pts = [
        (args.wall_offset, height / 2),
        (args.wall_offset + width, height / 2),
        (args.wall_offset + width, -height / 2),
        (args.wall_offset, -height / 2),
    ]
    w_path = points_to_path(w_pts)

    # 5. Z-PROFILE (Interlocking Step)
    z_pts = [
        (args.wall_offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset, 0 + z_center_height / 2),
        (args.wall_offset + args.wall_thickness + offset, 0 - z_center_height / 2),
        (args.wall_offset + args.wall_thickness + offset, -height / 2 + args.wall_extra + offset),
        (args.wall_offset + width, -height / 2 + args.wall_extra + offset),
    ]
    z_path = points_to_path(z_pts)

    # Create inner and outer offset blocks.
    z_paths = [
        get_block(width, height, z_path, args.wall_offset, args.play / 2),
        get_block(width, height, z_path, args.wall_offset, -args.play / 2)
    ]

    # 6. S-PROFILE (Simple Notch)
    s_pts = [
        (args.wall_offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset, -height / 2 + args.wall_extra + offset),
        (args.wall_offset + width, -height / 2 + args.wall_extra + offset),
    ]
    s_path = points_to_path(s_pts)

    # Create inner and outer offset blocks.
    s_paths = [
        get_block(width, height, s_path, args.wall_offset, args.play / 2),
        get_block(width, height, s_path, args.wall_offset, -args.play / 2)
    ]

    # 7. C-PROFILE (Chamfered Edge)
    c_pts = [
        (args.wall_offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset, height / 2 - args.wall_extra - offset),
        (args.wall_offset + width - args.wall_thickness - offset - c_center, 0),
        (args.wall_offset + width - args.wall_thickness - offset, -height / 2 + args.wall_extra + offset),
        (args.wall_offset + width, -height / 2 + args.wall_extra + offset),
    ]
    c_path = points_to_path(c_pts)

    # Create inner and outer offset blocks.
    c_paths = [
        get_block(width, height, c_path, args.wall_offset, args.play / 2),
        get_block(width, height, c_path, args.wall_offset, -args.play / 2)
    ]

    # 8. T-PROFILE (Triangle support)
    if pcb_support_width > 0:
        t_pts = [
            (-args.pcb_cover_offset, 0.0),
            (args.wall_offset, 0.0),
            (args.wall_offset, pcb_support_height),
        ]
        t_path = points_to_path(t_pts)
    else:
        t_path = None

    # 9. P-PROFILE (PCB slot)
    if pcb_clearance_width > 0:
        p_pts = [
            (args.wall_offset, args.pcb_thickness / 2 + args.pcb_z),
            (args.pcb_clearance_offset, args.pcb_thickness / 2 + args.pcb_z),
            (args.pcb_clearance_offset, -args.pcb_thickness / 2 + args.pcb_z),
            (args.wall_offset, -args.pcb_thickness / 2 + args.pcb_z),
        ]
        p_path = points_to_path(p_pts)
    else:
        p_path = None

    return w_path, z_paths, s_paths, c_paths, t_path, p_path


def get_outline_segments(args):
    """
    Processes SVG paths: scales, sanitizes, offsets the main outline,
    and segments the result based on intersections with Anchor and Clip regions.

    Args:
        args (object): Configuration arguments.
    """

    # 1. Load paths and identify the initial outline.
    paths, attributes = svgpathtools.svg2paths(args.svg)
    elements = list(zip(paths, attributes))
    outline = list(filter(ShapeTypes.Outline, elements))[0]

    # 2. Apply scaling
    if args.scale != 1.0:
        paths = center_rescale(outline, paths, args.scale)
        elements = list(zip(paths, attributes))
        outline = list(filter(ShapeTypes.Outline, elements))[0]

    # 3. Sanitize geometry
    outline = (close_path_sanitizing(outline[0]), outline[1])

    # 4. Filter for Anchor and Clip zones to perform segmentation
    interest_types = [ShapeTypes.Anchor, ShapeTypes.Clip]
    areas = sum([list(map(lambda x: (st, x), filter(st, elements))) for st in interest_types], [])

    segments = get_segments(outline, areas, ShapeTypes.Outline)

    return outline, segments


def get_plane(path, position=0, flipped=False):
    """
    Creates a CadQuery Workplane perpendicular (normal) to the path at a given position.

    Args:
        path (svgpathtools.Path): The path object defining the curve.
        position (float, optional): The parameter 't' (0 to 1) along the path
                                    where the plane should be placed. Defaults to 0 (start).

    Returns:
        cq.Workplane: A CadQuery Workplane object aligned normal to the path.
    """
    # 1. Get the point (p) and the unit normal vector (v) at the given position 't'.
    p = path.point(position)
    v = path.normal(position)

    # 2. Ensure the normal vector points 'outward' (away from the origin) in a consistent direction.
    # Check if the length of (p + v) is greater than the length of p. If not, the normal is pointing
    # 'inward' or towards the origin, and must be reversed.
    plus = np.linalg.norm(p + v) > np.linalg.norm(p)
    if not plus:
        v *= - 1

    # 3. Convert complex coordinates to tuples for CadQuery (which uses (x, y, z)).
    position_tuple = euclid_to_tuple(p, True)
    vector_tuple = euclid_to_tuple(v, True)

    # 4. Construct the CadQuery Plane.
    # origin: The point on the path.
    # xDir: The normal vector (v).
    # normal: The vector perpendicular to the path AND the x-z plane (for 2D input paths).
    #         np.cross(vector, [0, 0, 1/-1]) provides the y-axis direction for the plane.
    z_axis = -1 if flipped else 1
    wp = cq.Workplane(cq.Plane(
        origin=tuple(position_tuple),
        xDir=vector_tuple,
        normal=tuple(np.cross(vector_tuple, [0, 0, z_axis]))
    ))
    return wp


def generate_pcb(profile, outline):
    path = outline[0]

    # 1. Create a 3D wire from the path and position the profile
    # on the normal plane at the start of the sweep path.
    outline_wire = to_wire(path, closed=False)
    profile_path = to_wire(profile, get_plane(path))

    # 2. Generate the PCB geometry by sweeping the profile along the wire.
    # Uses Frenet frame and transformed transition for accurate orientation.
    return profile_path.sweep(outline_wire, isFrenet=True, transition="transformed", combine=False)


def generate_pcb_support(profile, outline, pcb_thickness, shift, upper):
    path = outline[0]

    # 1. Create the sweep wire from the path and align the profile profile on the plane.
    # The normal plane orientation is flipped based on the 'upper' boolean flag.
    outline_wire = to_wire(path, closed=False)
    profile_path = to_wire(profile, get_plane(path, flipped=False if upper else True))

    # 2. Sweep the profile along the wire using the Frenet frame.
    obj = profile_path.sweep(outline_wire, isFrenet=True, transition="transformed", combine=False)

    # 3. Translate the resulting geometry along the Z-axis.
    # Logic: Offset by 'shift' plus half the thickness directed up (1) or down (-1).
    return obj.translate((0, 0, shift + (pcb_thickness / 2) * (1 if upper else -1)))


def generate_wall(paths, outline, segments, play, inner, flipped=False):
    """
    Generates a 3D wall object by sweeping selected 2D profiles along the
    segmented 2D outline path.

    Args:
        paths (tuple): The predefined 2D cross-section paths (w_path, z_path, s_path, c_path).
        outline (tuple): The processed 2D offset outline (svgpathtools.Path, attributes).
        segments (list): Classified segments of the outline, [(type, (start_t, end_t))].
        play (float): The total clearance/gap distance.
        inner (bool): If True, indicates an inner wall generation, which affects sweep play adjustment direction.

    Returns:
        cq.Workplane: A Workplane containing the final combined 3D Solid wall object.
    """
    z_path, s_path, c_path = paths

    # Calculate 't' parameter shift (normalized distance) for half the clearance.
    play_ratio = (play / 2) / outline[0].length()
    # Factor determines if adjustment shortens (inner=-1) or lengthens (inner=1) the sweep path.
    play_factor = 1 if inner else -1

    body_o = None

    # Iterate through each segment (type, (start t, end t)).
    for segment in segments:
        type, (start, end) = segment

        # 1. Select the correct cross-section profile.
        if type == ShapeTypes.Outline:
            type_str = "Outline"
            profile = s_path
        elif type == ShapeTypes.Clip:
            type_str = "Clip"
            profile = c_path
        elif type == ShapeTypes.Anchor:
            type_str = "Anchor"
            profile = z_path
        else:
            raise RuntimeError("Unsupported segment type")

        # 2. Adjust segment endpoints for clearance.
        if type == ShapeTypes.Outline:
            # Shorten/Lengthen the sweep path.
            start += play_ratio * play_factor
            end -= play_ratio * play_factor
        else:
            # Lengthen/Shorten the sweep path.
            start -= play_ratio * play_factor
            end += play_ratio * play_factor

        logger.debug(f"Segment type: {type_str} start: {start} end: {end}")

        # 3. Crop the outline path (handling wrap-around).
        if start > end:
            part = svgpathtools.Path(*(outline[0].cropped(start, 1)[:] + outline[0].cropped(0, end)[:]))
        else:
            part = outline[0].cropped(start, end)

        # 4. Perform the sweep (2D Profile along 3D Path).
        outline_wire = to_wire(part, closed=False)
        profile_path = to_wire(profile, get_plane(part, flipped=flipped))  # Profile placed on normal plane at start.

        sweep_result = profile_path.sweep(outline_wire, isFrenet=True, transition="transformed", combine=False)

        # 5. Union the result with the main body.
        body_o = sweep_result if body_o is None else body_o.union(sweep_result)

    return body_o


def generate(args):
    """
    Main function to generate the final 3D printable object.

    It prepares the 2D outline, generates 2D profiles, sweeps to create
    inner/outer walls, combines them, and exports the solid.

    Args:
        args (object): Configuration arguments.
    """

    # 1. Prepare and segment the 2D outline for the sweep path.
    outline, segments = get_outline_segments(args)

    # 2. Generate required 2D cross-sectional profiles (inner/outer variations).
    w_path, z_paths, s_paths, c_paths, t_path, p_path = generate_profiles(args)

    # 3. Generate the INNER 3D wall structure using index [0] profiles.
    inner = generate_wall((z_paths[0], s_paths[0], c_paths[0]), outline, segments, args.play,
                          True,
                          args.junction_flipped)

    # 4. Generate the OUTER 3D wall structure using index [1] profiles.
    outer = generate_wall((z_paths[1], s_paths[1], c_paths[1]), outline, segments, args.play,
                          False,
                          args.junction_flipped)

    # 5. Process PCB logic: subtract PCB volume and generate upper/lower supports.
    if not args.disable_pcb:
        # Subtract the main PCB profile from both wall structures.
        if p_path is not None:
            pcb = generate_pcb(p_path, outline)
            outer = outer.cut(pcb)
            inner = inner.cut(pcb)

        # Generate upper and lower supports and union them based on junction orientation.
        if t_path is not None:
            pcb_up = generate_pcb_support(t_path, outline, args.pcb_thickness, args.pcb_z, True)
            pcb_down = generate_pcb_support(t_path, outline, args.pcb_thickness, args.pcb_z, False)
            if args.junction_flipped:
                outer = outer.union(pcb_down)
                inner = inner.union(pcb_up)
            else:
                outer = outer.union(pcb_up)
                inner = inner.union(pcb_down)

    # 6. Combine wall components and apply a coordinate mirror to correct orientation.
    final_combined = outer.val() + inner.val()
    final_combined = final_combined.mirror("XZ")

    # 7. Export the final solid and optionally show in GUI.
    cq.exporters.export([final_combined], args.output)

    try:
        show_object(final_combined, name='wall')
    except BaseException:
        pass

    print(f"Exported to {args.output}")


def _main(argv=sys.argv):
    def auto_int(x):
        return int(x, 0)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG,
                        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(prog=argv[0], description='CBox: Parametric PCB Enclosure Generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count', default=0,
                        help="Increases log verbosity (e.g., from INFO to DEBUG) for each occurrence.")

    # PCB Parameters
    parser.add_argument('-t', '--pcb-thickness', default=1.6, type=float,
                        help="Thickness of the PCB in mm.")
    parser.add_argument('-z', '--pcb-z', default=1.2, type=float,
                        help="Vertical offset (Z-axis) of the PCB's center plane relative to the junction center in mm.")
    parser.add_argument('-w', '--pcb-clearance-offset', default=0.15, type=float,
                        help="Clearance distance (in mm) between the PCB outline and the surrounding enclosure walls/slots.")
    parser.add_argument('-u', '--pcb-cover-offset', default=0.4, type=float,
                        help="Distance (in mm) the internal support lip extends over the surface of the PCB edge.")
    parser.add_argument('--pcb-support-angle', default=45.0, type=float,
                        help="Angle (in degrees) of the slanted internal support lip holding the PCB.")
    parser.add_argument('--disable-pcb', default=False, action='store_true',
                        help="If set, prevents the generation of internal PCB features (slots, supports).")

    # Wall Parameters
    parser.add_argument('-n', '--wall-thickness', default=0.44 * 2, type=float,
                        help="Minimum wall thickness (in mm). Should be a multiple of the 3D printer's nozzle width.")
    parser.add_argument('-o', '--wall-offset', default=-0.4, type=float,
                        help="Offset distance (in mm) applied to the PCB outline to determine the inner wall perimeter. Negative values move the wall inward.")
    parser.add_argument('-m', '--wall-extra', default=1.5, type=float,
                        help="Extra wall height (in mm) extending above and below the main junction line.")

    # Junction/Interlocking Parameters
    parser.add_argument('-y', '--junction-height', default=2.0, type=float,
                        help="Total height (in mm) of the section between the two interlocking box halves.")
    parser.add_argument('-x', '--junction-width', default=1.0, type=float,
                        help="Horizontal width (in mm) of the section between the two interlocking box halves.")
    parser.add_argument('-f', '--junction-flipped', default=False, action='store_true',
                        help="If set, flips the junction profile, reversing the inner/outer lip placement between the box halves.")
    parser.add_argument('--z-angle', default=30.0, type=float,
                        help="Angle (in degrees) used for the slanted edge of the Z-profile junction.")
    parser.add_argument('--c-angle', default=60.0, type=float,
                        help="Angle (in degrees) used for the chamfered edge of the C-profile junction.")

    # Global Parameters
    parser.add_argument('-s', '--scale', default=0.264583, type=float,
                        help="Scaling factor to convert SVG coordinates (e.g., pixels) to working units (mm). Default assumes 96 DPI.")
    parser.add_argument('-p', '--play', default=0.10, type=float,
                        help="Mechanical clearance (tolerance) applied between interlocking parts/features in mm.")

    # File Arguments
    parser.add_argument('svg',
                        help="Path to the PCB SVG input file.")
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
