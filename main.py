import argparse
import logging
import sys

import shapely
import cadquery as cq
import numpy as np
import svgpathtools

from warnings import warn

from OCP.StdFail import StdFail_NotDone
from more_itertools import powerset

logger = logging.getLogger(__name__)


#
# Math
#


def euclid_to_tuple(p, dim3=False):
    """
    Converts a complex number representing a 2D point into a coordinate tuple.

    Args:
        p: Complex number (x + yj)
        dim3: If True, returns an (x, y, z) tuple; if False, returns (x, y)
    """
    x, y = p.real, p.imag

    if dim3:
        return x, y, 0
    return x, y


def tuple_to_euclid(p):
    """
    Converts a coordinate tuple into a complex number (Euclidean point).

    Args:
        p: Tuple or list containing at least (x, y)
    """
    # p[0] is X, p[1] is Y; returns x + yj
    return p[0] + p[1] * 1j


#
# SVG
#


# =========================================================================
# PATCH: svgpathtools.Arc.point Accuracy Fix
# This entire block patches the Arc.point method to ensure exact coordinates
# are returned at t=0.0 and t=1.0, resolving potential floating-point errors.
# =========================================================================

# Preserve the original method before patching it.
original_arc_point = svgpathtools.Arc.point


def my_custom_arc_point(self, t):
    """
    Custom Arc.point method to ensure exact coordinates are returned
    at the start (t=0.0) and end (t=1.0) points, mitigating floating-point errors.
    """
    # Return exact start point if t is 0.0
    if t == 0.0:
        return self.start

    # Return exact end point if t is 1.0
    if t == 1.0:
        return self.end

    # For intermediate points, use the original calculation
    return original_arc_point(self, t)


# Apply the custom method to override the original one.
svgpathtools.Arc.point = my_custom_arc_point


# =========================================================================
# END PATCH
# =========================================================================

def bbox2path(xmin, xmax, ymin, ymax, r=0):
    """
    Converts bounding box coordinates into an svgpathtools path object.

    Args:
        xmin, xmax: Horizontal boundaries
        ymin, ymax: Vertical boundaries
        r: Corner radius (rx and ry)
    """
    # Define rectangle geometry for the SVG parser
    rect_params = dict(
        x=xmin,
        y=ymin,
        width=xmax - xmin,
        height=ymax - ymin,
        rx=r,
        ry=r
    )

    # Generate the 'd' path string from the dictionary
    path_d = svgpathtools.svg_to_paths.rect2pathd(rect_params)

    # Return parsed path object
    return svgpathtools.parser.parse_path(path_d)


def center_rescale(ref, paths, scale=1.0):
    """
    Centers a list of paths relative to the bounding box of a reference path,
    then scales them by a given factor.

    Args:
        ref (list): A list containing the reference path (at index 0) used to calculate the centering offset.
        paths (list): A list of paths (e.g., svgpathtools objects) to be transformed.
        scale (float, optional): The factor by which to scale the paths. Defaults to 1.0 (no scaling).

    Returns:
        list: The list of centered and scaled paths.
    """
    # 1. Determine the center point of the reference path's bounding box (bbox).

    # Get the bounding box of the reference path (xmin, xmax, ymin, ymax)
    bbox = ref[0].bbox()

    # Calculate the center coordinates
    center_x = (bbox[0] + bbox[1]) / 2
    center_y = (bbox[2] + bbox[3]) / 2

    # 2. Translate the paths to center them at the origin (0, 0).
    # The offset is the negative of the calculated center point.
    paths = [p.translated(-tuple_to_euclid((center_x, center_y))) for p in paths]

    # 3. Scale the centered paths by the given factor.
    paths = [p.scaled(scale) for p in paths]

    return paths


def create_path(path, sub_div=100, endpoint=False):
    """
    Samples points along an svgpathtools.Path and returns them as a list of (x, y) tuples.

    Args:
        path (svgpathtools.Path): The path object, which is a list of segments.
        sub_div (int, optional): The number of points to sample per non-linear segment. Defaults to 100.
        endpoint (bool, optional): If True, ensures the very last point of the final segment is included. Defaults to False.

    Returns:
        list: A list of (x, y) coordinate tuples representing the path's outline.
    """
    ps = []

    # Iterate through each segment (s) in the path
    for j, s in enumerate(path):

        # Determine number of samples: 2 for Lines, sub_div for all other curves.
        num_samples = 2 if isinstance(s, svgpathtools.Line) else sub_div

        # Determine if endpoint should be included: only on the very last segment AND if 'endpoint' is True.
        include_endpoint = (j == len(path) - 1 and endpoint)

        # Generate parameter 'i' values (0 to 1) for sampling
        for i in np.linspace(0, 1, num_samples, endpoint=include_endpoint):
            p = s.point(i)
            ps.append(euclid_to_tuple(p))

    return ps


def get_angle(v1, v2):
    """
    Calculates the signed angle between two 2D vectors (v1 and v2) in radians.

    The sign of the angle is determined by the 2D cross product to indicate
    the direction of rotation from v1 to v2.

    Args:
        v1 (np.ndarray): The starting vector (2D).
        v2 (np.ndarray): The target vector (2D).

    Returns:
        float: The signed angle in radians.
    """
    # 1. Normalize both input vectors to unit length.
    # This ensures the dot product (v1 . v2) equals cos(angle).
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Check for near-identical vectors to prevent calculation errors and return 0 immediately.
    if np.isclose(v1, v2).all():
        return 0

    # 2. Calculate the unsigned angle (x) using the arccos of the dot product.
    # np.dot(v1, v2) is equivalent to cos(angle) for unit vectors.
    x = np.arccos(np.dot(v1, v2))

    # 3. Determine the sign of the angle using the 2D cross product.
    # In 2D (assuming (x, y) vectors), the cross product (v2_x * v1_y - v2_y * v1_x)
    # gives the z-component of the 3D cross product.
    # If the z-component is negative (v1 -> v2 is clockwise), return -x (negative angle).
    # Otherwise (counter-clockwise), return x (positive angle).
    return x if np.cross(v2, v1) < 0 else -x


def is_path_clockwise(path):
    """
    Checks if a closed path is oriented clockwise using a signed area summation method.

    The method samples points along the path and sums a quantity related to the
    turn angle at each vertex, weighted by the length of the adjacent edges.
    A negative final sum typically indicates a clockwise winding.

    Args:
        path (svgpathtools.Path): A closed path object.

    Returns:
        bool: True if the path is clockwise, False otherwise (counter-clockwise).

    Raises:
        Exception: If the input path is not closed.
    """
    # 1. Validation: Ensure the path is closed (start point equals end point).
    if not path.isclosed():
        raise Exception("Should be a closed path")

    # 2. Preparation: Sample the continuous path into a list of discrete vertices.
    # sub_div=20 samples per segment for accuracy.
    path = create_path(path, sub_div=20)

    acc = 0  # Accumulator for the signed sum.

    # 3. Iteration: Loop through every vertex (i+1) on the path to calculate the turn angle.
    # The loop runs from i=-1 to len(path)-3, covering the entire path due to wrapping.
    for i in range(-1, len(path) - 2):

        # Define the two vectors forming the angle at vertex path[i+1]:
        # v1: Vector from current vertex (path[i+0]) to next vertex (path[i+1])
        v1 = np.subtract(path[i + 1], path[i + 0])
        # v2: Vector from next vertex (path[i+1]) to the vertex after that (path[i+2])
        v2 = np.subtract(path[i + 2], path[i + 1])

        # Calculate the sum of the lengths of the two vectors (used for weighting).
        s_sum = np.linalg.norm(v1) + np.linalg.norm(v2)

        # Get the signed angle (radians) between the two vectors.
        angle = get_angle(v1, v2)

        # Accumulate the weighted angle:
        # Positive angle (counter-clockwise turn) -> add the length sum.
        if angle > 0:
            acc += s_sum
        # Negative angle (clockwise turn) -> subtract the length sum.
        elif angle < 0:
            acc -= s_sum

    # 4. Result: A negative total accumulation indicates a net clockwise winding.
    return acc < 0


def close_path_sanitizing(path, clockwise=False):
    """
    Sanitizes, reorders, closes, and standardizes the winding direction
    of a list of svgpathtools segments intended to form a closed path.

    Args:
        path (list or svgpathtools.Path): A list of path segments.
        clockwise (bool, optional): If True, the resulting path will be oriented
                                    clockwise. If False, it will be counter-clockwise.
                                    Defaults to False (counter-clockwise).

    Returns:
        svgpathtools.Path: A single, closed, and correctly oriented Path object.

    Raises:
        Exception: If segments cannot be ordered to form a closed loop.
    """
    # 1. INITIALIZATION and COPY
    # Create a new svgpathtools.Path object from the input segments.
    path = svgpathtools.Path(*path)

    # 2. SANITIZATION: Remove Zero-Length Segments
    # Filter out segments where the distance between start and end points is effectively zero.
    path = svgpathtools.Path(*list(filter(
        lambda p: not np.isclose(np.linalg.norm(p.end - p.start), 0), path
    )))

    # 3. SANITIZATION: Replace Degenerate Cubic Bezier Curves
    def replace_path(p):
        if isinstance(p, svgpathtools.CubicBezier):
            # Convert complex points to (x, y) tuples for easy comparison
            pts = np.asarray([euclid_to_tuple(point) for point in [p.start, p.control1, p.control2, p.end]])

            # Check if all points form a vertical or horizontal line
            if np.all(pts[:, 1] == pts[:, 1][0]) or np.all(pts[:, 0] == pts[:, 0][0]):
                warn(f'Replacing degenerate CubicBezier with a Line: CubicBezier(start={p.start},' +
                     f' control1={p.control1}, control2={p.control2}, end={p.end})' +
                     f' --> Line(start={p.start}, end={p.end})')
                return svgpathtools.Line(p.start, p.end)
        return p

    # Apply the replacement function to all segments
    path = svgpathtools.Path(*[replace_path(p) for p in path])

    # 4. SEGMENT JOINING AND REORDERING
    # Loop to find the segment that connects to path[i].end and place it at path[i+1].
    for i in range(-1, len(path) - 1):
        found = False

        # Search subsequent segments (j) for a connection point
        for j in range(i + 1, len(path)):

            # Case 1: Direct connection found (path[i].end == path[j].start)
            if np.isclose(path[i].end, path[j].start):
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap into place
                found = True
                break

            # Case 2: Reversed connection found (path[i].end == path[j].end)
            if np.isclose(path[i].end, path[j].end):
                path[j] = path[j].reversed()  # Reverse the segment
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap into place
                found = True
                break

        if not found:
            raise Exception("Can't close path: No segment found to follow index %d" % i)

        # Coalesce endpoints to fix floating-point gaps
        path[i].end = path[i + 1].start = (path[i].end + path[i + 1].start) / 2

    path.closed = True
    path = svgpathtools.Path(*path)

    # 5. ORIENTATION STANDARDIZATION

    should_reverse = is_path_clockwise(path) != clockwise
    return path.reversed() if should_reverse else path


# From https://github.com/meerk40t/svgelements/blame/master/svgelements/svgelements.py
def find_intersections(segment1, segment2, samples=50, ta=(0.0, 1.0, None), tb=(0.0, 1.0, None),
                       depth=0, enhancements=2, enhance_samples=50):
    """
    Calculate intersections by linearized polyline intersections with enhancements.
    We calculate probable intersections by linearizing our segment into `sample` polylines
    we then find those intersecting segments and the range of t where those intersections
    could have occurred and then subdivide those segments in a series of enhancements to
    find their intersections with increased precision.

    This code is fast, but it could fail by both finding a rare phantom intersection (if there
    is a low or no enhancements) or by failing to find a real intersection. Because the polylines
    approximation did not intersect in the base case.

    At a resolution of about 1e-15 the intersection calculations become unstable and intersection
    candidates can duplicate or become lost. We terminate at that point and give the last best
    guess.

    :param segment1:
    :param segment2:
    :param samples:
    :param ta:
    :param tb:
    :param depth:
    :param enhancements:
    :param enhance_samples:
    :return:
    """
    if depth == 0:
        # Quick Fail. There are no intersections without overlapping quick bounds
        try:
            s1x = [e.real for e in segment1.bpoints()]
            s2x = [e.real for e in segment2.bpoints()]
            if min(s1x) > max(s2x) or max(s1x) < min(s2x):
                return
            s1y = [e.imag for e in segment1.bpoints()]
            s2y = [e.imag for e in segment2.bpoints()]
            if min(s1y) > max(s2y) or max(s1y) < min(s2y):
                return
        except AttributeError:
            pass
    assert (samples >= 2)
    a = np.linspace(ta[0], ta[1], num=samples)
    b = np.linspace(tb[0], tb[1], num=samples)
    step_a = a[1] - a[0]
    step_b = b[1] - b[0]
    j = segment1.points(a) if hasattr(segment1, 'points') else [segment1.point(i) for i in a]
    k = segment2.points(b) if hasattr(segment2, 'points') else [segment2.point(i) for i in b]

    ax1, bx1 = np.meshgrid(np.real(j[:-1]), np.real(k[:-1]))
    ax2, bx2 = np.meshgrid(np.real(j[1:]), np.real(k[1:]))
    ay1, by1 = np.meshgrid(np.imag(j[:-1]), np.imag(k[:-1]))
    ay2, by2 = np.meshgrid(np.imag(j[1:]), np.imag(k[1:]))

    denom = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    qa = (bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)
    qb = (ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)
    hits = np.dstack(
        (
            denom != 0,  # Cannot be parallel.
            np.sign(denom) == np.sign(qa),  # D and Qa must have same sign.
            np.sign(denom) == np.sign(qb),  # D and Qb must have same sign.
            abs(denom) >= abs(qa),  # D >= Qa (else not between 0 - 1)
            abs(denom) >= abs(qb),  # D >= Qb (else not between 0 - 1)
        )
    ).all(axis=2)

    where_hit = np.argwhere(hits)
    if len(where_hit) != 1 and step_a < 1e-10:
        # We're hits are becoming unstable give last best value.
        if ta[2] is not None and tb[2] is not None:
            yield ta[2], tb[2]
        return

    # Calculate the t values for the intersections
    ta_hit = qa[hits] / denom[hits]
    tb_hit = qb[hits] / denom[hits]

    for i, hit in enumerate(where_hit):

        at = ta[0] + float(hit[1]) * step_a  # Zoomed min+segment intersected.
        bt = tb[0] + float(hit[0]) * step_b
        a_fractional = ta_hit[i] * step_a  # Fractional guess within intersected segment
        b_fractional = tb_hit[i] * step_b
        if depth == enhancements:
            # We've enhanced as good as we can, yield the current + segment t-value to our answer
            yield at + a_fractional, bt + b_fractional
        else:
            yield from find_intersections(
                segment1,
                segment2,
                ta=(at, at + step_a, at + a_fractional),
                tb=(bt, bt + step_b, bt + b_fractional),
                samples=enhance_samples,
                depth=depth + 1,
                enhancements=enhancements,
                enhance_samples=enhance_samples,
            )


def offset_curve(path, offset_distance):
    """
    Calculates a piecewise-linear approximation of the 'parallel' offset curve
    at a given distance from the input path.

    This function first calculates the offset points for each segment based on
    segment normals and then handles corner connections. Finally, it checks for
    and trims self-intersections that occur when offsetting tight curves
    (especially for inner offsets).

    Args:
        path (svgpathtools.Path): A closed path object.
        offset_distance (float): The perpendicular distance to offset the curve.
                                 Positive distance is typically outer offset;
                                 Negative distance is inner offset.

    Returns:
        svgpathtools.Path or None: The calculated offset path, or None if an
                                   inner offset is too large for the bounding box.
    """

    # Determine the sign for corner calculation consistency.
    sign = 1 if offset_distance >= 0 else -1

    # 1. INNER OFFSET BOUNDING BOX CHECK

    # If the offset is inner (negative), check if the distance is too large
    # relative to the path's bounding box (bbox). If the offset is greater
    # than half the path's width or height, the path will likely disappear
    # or collapse, so return None.
    if offset_distance < 0:
        xmin, xmax, ymin, ymax = path.bbox()
        if -offset_distance * 2 >= abs(xmin - xmax) or -offset_distance * 2 >= abs(ymin - ymax):
            warn("Inner offset too large: path collapses to zero size.")
            return None

    # 2. HELPER FUNCTION: Calculate the corner connection vector

    def new_vector(va1, va2):
        """
        Calculates the vector required to connect the offset endpoints of two
        adjacent segments (the 'corner' vector).

        Args:
            va1 (complex): Normal vector of the end of the first segment.
            va2 (complex): Normal vector of the start of the second segment.

        Returns:
            complex: The corrected corner offset vector.
        """
        # Sum the two normal vectors (va1 and va2)
        va3 = va1 + va2
        # Normalize the sum vector to get the direction of the corner bisector
        va3 /= np.linalg.norm(va3)

        # Calculate the scalar distance (d) needed to move along the bisector (va3).
        # This corrects for the angle between va1 and va2, ensuring the final
        # point is offset by exactly 'offset_distance' perpendicular to the original segments.
        # The formula uses the half-angle property derived from geometry.
        d = offset_distance / np.sqrt((1 + np.dot([va1.real, va1.imag], [va2.real, va2.imag])) / 2)

        # Scale the bisector vector by the corrected distance
        va3 *= d
        return va3

    # 3. OFFSET SEGMENTS

    new_path = []
    # Iterate through segments, linking seg1 -> seg2 -> seg3 to handle the corners around seg2.
    for i in range(len(path)):
        seg1 = path[i - 1]  # Previous segment
        seg2 = path[i]  # Current segment being offset
        seg3 = path[(i + 1) % len(path)]  # Next segment (handles wrap-around)

        # v1: Corner vector for the start of seg2 (connects seg1.end normal and seg2.start normal)
        v1 = new_vector(seg1.normal(1), seg2.normal(0))
        # v2: Corner vector for the end of seg2 (connects seg2.end normal and seg3.start normal)
        v2 = new_vector(seg2.normal(1), seg3.normal(0))

        # Calculate the new offset start and end points of the segment
        start = seg2.start + v1
        end = seg2.end + v2

        # Create the new segment based on the original type
        if isinstance(seg2, svgpathtools.Line):
            new_path.append(svgpathtools.Line(start, end))

        elif isinstance(seg2, svgpathtools.Arc):
            # Arcs are offset by changing the radius (r)
            r = offset_distance
            new_path.append(
                svgpathtools.Arc(start, seg2.radius + tuple_to_euclid((r, r)), seg2.rotation, seg2.large_arc,
                                 seg2.sweep, end))

        elif isinstance(seg2, svgpathtools.CubicBezier):
            # Cubic Bezier curves require control point adjustments.
            # a and b are adjustment vectors for the control points, based on corner vectors v1 and v2,
            # scaled by a constant (0.552...) typically used for approximating circular arcs with Beziers.
            a = v1 + (v2 / np.linalg.norm(v2) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            b = v2 + (v1 / np.linalg.norm(v1) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            new_path.append(svgpathtools.CubicBezier(start, seg2.control1 + a, seg2.control2 + b, end))

        else:
            raise Exception("Not supported: {0}".format(type(seg2)))

    # 4. SELF-INTERSECTION TRIMMING

    # This loop detects and trims self-intersections that commonly occur on the
    # inner offset of tight corners or small paths. The process iteratively
    # crops intersecting segments and removes the segments that fall between the intersections.

    i = 0
    while i < len(new_path):
        j = i + 2
        # Only check intersections up to half a path length away (optimization)
        while j < i + len(new_path) / 2 + 1:
            k = j % len(new_path)  # Wrap index k

            # Find intersections between segment i and segment k
            r = list(find_intersections(new_path[i], new_path[k]))

            if len(r) == 1:
                # Intersection found: Trim the segments to the intersection point
                new_path[i] = new_path[i].cropped(0, r[0][0])  # Trim seg i end
                new_path[k] = new_path[k].cropped(r[0][1], 1)  # Trim seg k start

                # Remove intermediate segments (j up to k) that cause the overlap loop.
                # 'a' handles the wrap-around when the loop involves the last segment.
                a = max(j - len(new_path), 0)
                new_path = new_path[a:i + 1] + new_path[j:]

                # Reset indices to re-check after trimming/removal
                i = i - a
                j = i + 2
            else:
                j += 1
        i += 1

    # 5. FINALIZATION

    # Convert the list of new segments back into a single Path object.
    offset_path = svgpathtools.Path(*new_path)
    return offset_path


def circle_path(x, y, r=0):
    """
    Converts circle data to an svgpathtools path object.

    Args:
        x: Center X coordinate
        y: Center Y coordinate
        r: Circle radius
    """
    # Create geometric definition
    params = dict(cx=x, cy=y, r=r)

    # Generate SVG 'd' path string
    path_d = svgpathtools.svg_to_paths.ellipse2pathd(params)

    return svgpathtools.parser.parse_path(path_d)


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


def to_wire(path, plane=None, closed=True):
    """
    Converts an svgpathtools.Path into a CadQuery Wire object.

    Args:
        path (svgpathtools.Path): Path to convert.
        plane (cq.Workplane, optional): Starting workplane.
        closed (bool, optional): If True, closes the wire.

    Returns:
        cq.Workplane: Workplane containing the constructed Wire.
    """

    if plane is None:
        plane = cq.Workplane("front")

    # Start the workplane at the beginning of the path.
    plane = plane.moveTo(*euclid_to_tuple(path.start))

    for seg in path:
        # Map svgpathtools segments to CadQuery methods.
        if isinstance(seg, svgpathtools.Line):
            plane = plane.lineTo(*euclid_to_tuple(seg.end))
        elif isinstance(seg, svgpathtools.Arc):
            # Use threePointArc via the midpoint.
            plane = plane.threePointArc(euclid_to_tuple(seg.point(0.5)), euclid_to_tuple(seg.end))
        elif isinstance(seg, svgpathtools.CubicBezier):
            # Approximate Beziers using a spline through two intermediate points.
            plane = plane.spline(
                [euclid_to_tuple(seg.point(1 / 3)), euclid_to_tuple(seg.point(2 / 3)), euclid_to_tuple(seg.end)],
                includeCurrent=True)
        else:
            raise Exception("Not supported: {0}".format(type(seg)))

    # Close the wire if requested.
    if closed:
        plane = plane.close()

    return plane.wire(True)


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
    wall_outer_o = to_wire(outer_outline, body_o).extrude(wall_height_total, False)
    wall_inner_o = to_wire(inner_outline, body_o).extrude(wall_height_total, False)

    # 3. Hollow out the wall
    wall_o = wall_outer_o.cut(wall_inner_o)

    # 4. Process cutouts (fingers) along the wall
    for _, segment in filter(lambda x: x[0], get_segments(pcb_outline, [(True, finger) for finger in fingers])):
        # Create a closed path loop to subtract material for connector access
        cut_outline_1 = offset_curve(pcb_outline[0].cropped(segment[0], segment[1]),
                                     args.wall_play + args.wall_thickness * 2)
        cut_outline_2 = pcb_outline[0].cropped(segment[0], segment[1]).reversed()

        # Combine segments into a manifold path for extrusion
        cut_outline = svgpathtools.Path(
            *(cut_outline_2[:] + [svgpathtools.Line(cut_outline_2.end, cut_outline_1.start)] + cut_outline_1[:]) +
             [svgpathtools.Line(cut_outline_1.end, cut_outline_2.start)]
        )

        # Create the subtractive solid for the finger cutout
        cut_o = to_wire(cut_outline, body_o).extrude(wall_height_total * 2, False)
        wall_o = wall_o.cut(cut_o)

    # 5. Tag base edges for later filleting and combine with main body
    wall_o.edges("<Z").tag("wall_base")
    body_o = body_o.union(wall_o)

    # 6. Apply optional fillets to top and base edges
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


def style_filter(**kwargs):
    """
    Creates a function to filter SVG elements by checking if all given
    CSS style 'key:value' pairs exist in the element's 'style' attribute.
    """
    # Convert kwargs into required 'key:value' strings.
    filters = ['{0}:{1}'.format(a, b) for (a, b) in kwargs.items()]

    def fct(element):
        path, attributes = element
        # Check if ALL required style filters are present.
        for f in filters:
            # Fails if 'style' is missing or the required 'key:value' string is not found.
            if 'style' not in attributes or attributes['style'].find(f) < 0:
                return False
        return True

    return fct


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
    paths, attributes = svgpathtools.svg2paths(args.svg)
    elements = list(zip(paths, attributes))
    pcb_outline = list(filter(ShapeTypes.Outline, elements))[0]

    # Rescale all geometry based on the board outline reference
    paths = center_rescale(pcb_outline, paths, args.scale)
    elements = list(zip(paths, attributes))
    pcb_outline = list(filter(ShapeTypes.Outline, elements))[0]
    pcb_outline = (close_path_sanitizing(pcb_outline[0]), pcb_outline[1])

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
        path = close_path_sanitizing(support[0])
        body_o = create_column(body_o, path, args.under_space,
                               args.support_fillet if args.extra_fillet > 2 else 0)

    # 5. Generate Fillers (Additive going through PCB holes)
    fillers = list(filter(ShapeTypes.Filler, elements))
    for filler in fillers:
        body_o.plane = plane
        path = close_path_sanitizing(filler[0])
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
            path = close_path_sanitizing(hole[0])
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
    parser = argparse.ArgumentParser(prog=argv[0], description='TestBenchGenerator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose_count', action='count', default=0,
                        help="increases log verbosity for each occurrence.")

    parser.add_argument('-t', '--pcb-thickness', default=1.6, type=float,
                        help="PCB thickness")

    parser.add_argument('-d', '--pin-diameter', default=0.95, type=float,
                        help="Spring touch diameter")

    parser.add_argument('--wall-thickness', default=2.0, type=float,
                        help="Bench walls thickness")
    parser.add_argument('--wall-fillet', default=2.0, type=float,
                        help="Bench walls fillet radius")
    parser.add_argument('--wall-height', default=5.0, type=float,
                        help="Bench walls height")
    parser.add_argument('--wall-play', default=0.4, type=float,
                        help="Play between PCB and bench (X, Y) only")

    parser.add_argument('--filler-height', default=5.0, type=float,
                        help="Bench walls height")
    """
    parser.add_argument('--feet-thickness', default=2.0, type=float,
                        help="Bench feet thickness")
    parser.add_argument('--feet-height', default=25.0, type=float,
                        help="Bench feet height")
    """

    parser.add_argument('-b', '--bottom', default=8.0, type=float,
                        help="Bench bottom thickness")
    parser.add_argument('-u', '--under-space', default=5.0, type=float,
                        help="Space between under PCB")

    parser.add_argument('--fixture-screw-hole', default=2.8, type=float,
                        help="Fixture screw hole diameter")
    parser.add_argument('--fixture-screw-diameter', default=5.0, type=float,
                        help="Fixture screw outer diameter")
    parser.add_argument('--fixture-screw-offset', default=0.5, type=float,
                        help="Fixture screw Z offset compare to PCB top")
    parser.add_argument('--fixture-screw-extra', default=3.0, type=float,
                        help="Fixture screw extra wall thickness")

    parser.add_argument('--box-wall-thickness', default=2.0, type=float,
                        help="Space between under PCB")
    parser.add_argument('--box-screw-diameter', default=2.45, type=float,
                        help="Box screw diameter")
    parser.add_argument('--box-screw-offset', default=1.00, type=float,
                        help="SBox screw offset compare to the top of the plate")

    # Fillets
    parser.add_argument('--support-fillet', default=1.0, type=float,
                        help="Support top fillet")
    parser.add_argument('--filler-fillet', default=0.5, type=float,
                        help="Filler top fillet")
    parser.add_argument('--extra-fillet', default=4, type=int,
                        help="Enable extra fillet")

    parser.add_argument('-z', '--scale', default=0.264583, type=float,
                        help="svg scaling")

    parser.add_argument('-p', '--play', default=0.15, type=float,
                        help="Other plays")

    parser.add_argument('svg',
                        help="PCB SVG input outline")
    parser.add_argument('output',
                        help="PCB output file")

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
