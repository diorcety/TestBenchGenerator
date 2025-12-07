import shapely
import cadquery as cq
import numpy as np
import svgpathtools

from warnings import warn


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
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Check for near-identical vectors to prevent calculation errors.
    if np.isclose(v1, v2).all():
        return 0

    # 2. Calculate the unsigned angle (x) using the arccos of the dot product.
    x = np.arccos(np.dot(v1, v2))

    # 3. Manual 2D Cross Product (Determinant)
    # Logic: (v2.x * v1.y) - (v2.y * v1.x)
    cross_z = v2[0] * v1[1] - v2[1] * v1[0]

    # 4. Determine the sign of the angle using the 2D cross product.
    # If the cross product is negative (v1 -> v2 is clockwise), return -x.
    # Otherwise (counter-clockwise), return x.
    return x if cross_z < 0 else -x


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


#
# CadQuery
#

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
