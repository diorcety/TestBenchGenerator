import argparse
import logging
import sys

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
    if dim3:
        return np.real(p), np.imag(p), 0
    else:
        return np.real(p), np.imag(p)


def tuple_to_euclid(p):
    return p[0] + p[1] * 1j


#
# SVG
#


def bbox(paths):
    bbs = [p.bbox() for p in paths]
    xmins, xmaxs, ymins, ymaxs = list(zip(*bbs))
    xmin = min(xmins)
    xmax = max(xmaxs)
    ymin = min(ymins)
    ymax = max(ymaxs)
    return xmin, xmax, ymin, ymax


def bbox2path(xmin, xmax, ymin, ymax, r=0):
    p = svgpathtools.parser.parse_path(
        svgpathtools.svg_to_paths.rect2pathd(dict(x=xmin, y=ymin, width=xmax - xmin, height=ymax - ymin, rx=r, ry=r)))
    return p


def center_rescale(paths, scale=1.0):
    pcb_bbox = bbox(paths)
    center_x = (pcb_bbox[0] + pcb_bbox[1]) / 2
    center_y = (pcb_bbox[2] + pcb_bbox[3]) / 2

    paths = [p.translated(-tuple_to_euclid((center_x, center_y))) for p in paths]
    paths = [p.scaled(scale) for p in paths]
    return paths


def create_path(path, sub_div=100, endpoint=False):
    ps = []
    for j, s in enumerate(path):
        for i in np.linspace(0, 1, sub_div if not isinstance(s, svgpathtools.Line) else 2,
                             endpoint=(j == len(path) - 1 and endpoint)):
            p = s.point(i)
            ps.append(euclid_to_tuple(p))
    return ps


def get_angle(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    if np.isclose(v1, v2).all():
        return 0
    x = np.arccos(np.dot(v1, v2))
    return x if np.cross(v2, v1) < 0 else -x


def is_path_clockwise(path):
    if not path.isclosed():
        raise Exception("Should be a closed path")
    path = create_path(path, sub_div=20)

    acc = 0
    for i in range(-1, len(path) - 2):
        v1 = np.subtract(path[i + 1], path[i + 0])
        v2 = np.subtract(path[i + 2], path[i + 1])
        s_sum = np.linalg.norm(v1) + np.linalg.norm(v2)
        angle = get_angle(v1, v2)
        if angle > 0:
            acc += s_sum
        elif angle < 0:
            acc -= s_sum

    return acc < 0


def close_path_sanitizing(path):
    # Create a copy
    path = svgpathtools.Path(*path)

    # Remove zero length segments
    path = svgpathtools.Path(*list(filter(lambda p: not np.isclose(np.linalg.norm(p.end - p.start), 0), path)))

    # Replace degenerate CubicBezier
    def replace_path(p):
        if isinstance(p, svgpathtools.CubicBezier):
            pts = np.asarray([euclid_to_tuple(point) for point in [p.start, p.control1, p.control2, p.end]])
            if np.all(pts[:, 1] == pts[:, 1][0]) or np.all(pts[:, 0] == pts[:, 0][0]):
                warn(f'Replacing degenerate CubicBezier with a Line: CubicBezier(start={p.start},' +
                     f' control1={p.control1}, control2={p.control2}, end={p.end})' +
                     f' --> Line(start={p.start}, end={p.end})')
                return svgpathtools.Line(p.start, p.end)
        return p

    path = svgpathtools.Path(*[replace_path(p) for p in path])

    # Join segments
    for i in range(-1, len(path) - 1):
        found = False
        for j in range(i + 1, len(path)):
            if np.isclose(path[i].end, path[j].start):
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap
                found = True
                break
            if np.isclose(path[i].end, path[j].end):
                path[j] = path[j].reversed()
                if j != i + 1:
                    path[i + 1], path[j] = path[j], path[i + 1]  # Swap
                found = True
                break
        if not found:
            raise Exception("Can't close path")
        path[i].end = path[i + 1].start = (path[i].end + path[i + 1].start) / 2
    path.closed = True
    path = svgpathtools.Path(*path)

    return path.reversed() if is_path_clockwise(path) else path


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
    """Takes in a Path object, `path`, and a distance,
    `offset_distance`, and outputs a piecewise-linear approximation
    of the 'parallel' offset curve."""
    sign = 1 if offset_distance >= 0 else -1

    # No inner?
    if offset_distance < 0:
        xmin, xmax, ymin, ymax = path.bbox()
        if -offset_distance * 2 >= abs(xmin - xmax) or -offset_distance * 2 >= abs(ymin - ymax):
            return None

    def new_vector(va1, va2):
        va3 = va1 + va2
        va3 /= np.linalg.norm(va3)
        d = offset_distance / np.sqrt((1 + np.dot([va1.real, va1.imag], [va2.real, va2.imag])) / 2)
        va3 *= d
        return va3

    new_path = []
    for i in range(len(path)):
        seg1 = path[i - 1]
        seg2 = path[i]
        seg3 = path[(i + 1) % len(path)]

        v1 = new_vector(seg1.normal(1), seg2.normal(0))
        v2 = new_vector(seg2.normal(1), seg3.normal(0))

        start = seg2.start + v1
        end = seg2.end + v2
        if isinstance(seg2, svgpathtools.Line):
            new_path.append(svgpathtools.Line(start, end))
        elif isinstance(seg2, svgpathtools.Arc):
            r = offset_distance
            new_path.append(
                svgpathtools.Arc(start, seg2.radius + tuple_to_euclid((r, r)), seg2.rotation, seg2.large_arc,
                                 seg2.sweep, end))
        elif isinstance(seg2, svgpathtools.CubicBezier):
            a = v1 + (v2 / np.linalg.norm(v2) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            b = v2 + (v1 / np.linalg.norm(v1) * (offset_distance * 0.5522847498 * sign)) if offset_distance != 0 else 0
            new_path.append(svgpathtools.CubicBezier(start, seg2.control1 + a, seg2.control2 + b, end))
        else:
            raise Exception("Not supported: {0}".format(type(seg2)))

    i = 0
    while i < len(new_path):
        j = i + 2
        while j < i + len(new_path) / 2 + 1:
            k = j % len(new_path)
            r = list(find_intersections(new_path[i], new_path[k], samples=50))
            if len(r) == 1:
                new_path[i] = new_path[i].cropped(0, r[0][0])
                new_path[k] = new_path[k].cropped(r[0][1], 1)
                a = max(j - len(new_path), 0)
                new_path = new_path[a:i + 1] + new_path[j:]
                i = i - a
                j = i + 2
            else:
                j += 1
        i += 1

    offset_path = svgpathtools.Path(*new_path)
    return offset_path


def circle_path(x, y, r=0):
    p = svgpathtools.parser.parse_path(
        svgpathtools.svg_to_paths.ellipse2pathd(dict(cx=x, cy=y, r=r)))
    return p


#
# CadQuery
#

cadquery_extra = 0.005


class SolidSelector(cq.Selector):
    def __init__(self, solid) -> None:
        super().__init__()
        self._solid = solid

    def filter(self, object_list):
        def fct(x):
            t = [self._solid.isInside(y) for y in [x.startPoint(), x.endPoint()]]
            return all(t)

        return filter(fct, object_list)


def to_wire(path, plane=cq.Workplane("front"), closed=True):
    plane = plane.moveTo(*euclid_to_tuple(path.start))
    for seg in path:
        if isinstance(seg, svgpathtools.Line):
            plane = plane.lineTo(*euclid_to_tuple(seg.end))
        elif isinstance(seg, svgpathtools.Arc):
            plane = plane.threePointArc(euclid_to_tuple(seg.point(0.5)), euclid_to_tuple(seg.end))
        elif isinstance(seg, svgpathtools.CubicBezier):
            plane = plane.spline(
                [euclid_to_tuple(seg.point(1 / 3)), euclid_to_tuple(seg.point(2 / 3)), euclid_to_tuple(seg.end)],
                includeCurrent=True)
        else:
            raise Exception("Not supported: {0}".format(type(seg)))
    if closed:
        plane = plane.close()
    return plane.wire(True)


#
# Plate
#


def fillet_size(length):
    return length / 5


def try_fillet(body_o, workspace, radius):
    for x in reversed(list(powerset(range(workspace.size())))[1:]):
        try:
            selection = workspace.newObject([workspace.objects[i] for i in x])
            return selection.fillet(radius)
        except StdFail_NotDone as _:
            pass
    logger.exception(f"Can't create base fillet for column {workspace}")
    return body_o


def get_path_segments(pcb_outline, fingers):
    def get_positions(finger):
        intersections = pcb_outline[0].intersect(finger[0])
        assert len(intersections) == 2
        return sorted([T1 for (T1, seg1, t1), (_, _, _) in intersections])

    positions = [get_positions(finger) for finger in fingers]
    return positions


def get_screw_position(pcb_outline, screw):
    x = pcb_outline[0].intersect(screw[0], justonemode=True)
    (T1, _, _), _ = x
    p = pcb_outline[0].point(T1)
    v = np.subtract(screw[0].start, screw[0].end)
    v = v / np.linalg.norm(v)
    plus = np.linalg.norm(p + v) > np.linalg.norm(p)
    if not plus:
        v *= - 1
    return euclid_to_tuple(p, True), euclid_to_tuple(v, True)


def create_plate(body_o, pcb_outline, args):
    bottom_draw = []
    offset = max(args.box_screw_diameter, args.wall_thickness)
    screw_outline = offset_curve(pcb_outline[0], args.wall_play + offset)
    screw_bbox = screw_outline.bbox()

    # Plate
    offset_inline_bbox = (screw_bbox[0] - offset,
                          screw_bbox[1] + offset,
                          screw_bbox[2] - offset,
                          screw_bbox[3] + offset)
    offset_inline = bbox2path(*offset_inline_bbox,
                              offset if args.extra_fillet > 1 else 0)
    bottom_draw.append(offset_inline)
    offset_outline_bbox = (screw_bbox[0] - (offset + args.box_wall_thickness),
                           screw_bbox[1] + (offset + args.box_wall_thickness),
                           screw_bbox[2] - (offset + args.box_wall_thickness),
                           screw_bbox[3] + (offset + args.box_wall_thickness))
    offset_outline = bbox2path(*offset_outline_bbox,
                               offset + args.box_wall_thickness if args.extra_fillet > 1 else 0)
    bottom_draw.append(offset_outline)
    result = to_wire(offset_outline, body_o)
    result = result.extrude(args.bottom, True).faces("+Z").fillet(args.box_wall_thickness)

    # Screw holes
    result = result.faces("<Z").workplane()
    for p in [(screw_bbox[0], screw_bbox[2]), (screw_bbox[0], screw_bbox[3]), (screw_bbox[1], screw_bbox[2]),
              (screw_bbox[1], screw_bbox[3])]:
        result = result.moveTo(p[0], p[1]).hole(args.box_screw_diameter, args.bottom - args.box_screw_offset)
        bottom_draw.append(circle_path(p[0], p[1], args.box_screw_diameter / 2))

    result = result.translate((0, 0, -(args.bottom + args.under_space))).faces(">Z").workplane()

    return result, bottom_draw


def create_wall(body_o, pcb_outline, fingers, args):
    inner_outline = offset_curve(pcb_outline[0], args.wall_play)
    outer_outline = offset_curve(pcb_outline[0], args.wall_play + args.wall_thickness)

    wall_outer_o = to_wire(outer_outline, body_o).extrude(args.under_space + args.pcb_thickness + args.wall_height,
                                                          False)
    # wall_outer_o.edges("<Z").tag("wall_base")
    wall_inner_o = to_wire(inner_outline, body_o).extrude(args.under_space + args.pcb_thickness + args.wall_height,
                                                          False)
    wall_o = wall_outer_o.cut(wall_inner_o)
    for segment in get_path_segments(pcb_outline, fingers):
        cut_outline_1 = offset_curve(pcb_outline[0].cropped(segment[0], segment[1]),
                                     args.wall_play + args.wall_thickness * 2)
        cut_outline_2 = pcb_outline[0].cropped(segment[0], segment[1]).reversed()
        cut_outline = svgpathtools.Path(
            *(cut_outline_2[:] + [svgpathtools.Line(cut_outline_2.end, cut_outline_1.start)] + cut_outline_1[:]) + [
                svgpathtools.Line(cut_outline_1.end, cut_outline_2.start)])
        cut_o = to_wire(cut_outline, body_o).extrude(args.under_space + args.pcb_thickness + args.wall_height * 2,
                                                     False)
        wall_o = wall_o.cut(cut_o)
    wall_o.edges("<Z").tag("wall_base")
    body_o = body_o.union(wall_o)
    if args.extra_fillet > 3:
        body_o = body_o.edges(tag="wall_base").fillet(args.wall_fillet)
        body_o = body_o.edges(">Z").fillet(args.wall_thickness / 3)
    return body_o


def create_screw(body_o, pcb_outline, screw, args):
    position, vector = get_screw_position(pcb_outline, screw)
    position = np.array((position[0], position[1], position[2] + args.pcb_thickness + args.fixture_screw_offset))
    extra_position = position + np.array(vector) * (args.play + args.wall_thickness)

    plane = cq.Workplane(cq.Plane(origin=tuple(extra_position), xDir=tuple(np.cross(vector, [0, 0, 1])), normal=vector))
    screw = plane.circle(args.fixture_screw_diameter / 2) \
        .extrude(args.fixture_screw_extra, False)
    body_o = body_o.union(screw)

    plane = cq.Workplane(cq.Plane(origin=tuple(position), xDir=tuple(np.cross(vector, [0, 0, 1])), normal=vector))
    hole_o = plane.circle(args.fixture_screw_hole / 2) \
        .extrude(args.play + args.wall_thickness + args.fixture_screw_extra, False)
    body_o = body_o.cut(hole_o)

    if args.extra_fillet > 2:
        body_o = body_o.edges(SolidSelector(screw.translate(tuple(np.array(vector))).solids().val())) \
            .fillet((args.fixture_screw_diameter - args.fixture_screw_hole) / 4 - cadquery_extra)
        body_o = body_o.edges(SolidSelector(screw.translate(tuple(-np.array(vector))).solids().val())) \
            .fillet(fillet_size(args.fixture_screw_diameter))
    return body_o


def create_column(body_o, path, length, fillet=0):
    column_o = to_wire(path, body_o).extrude(length, False)
    body_o = body_o.union(column_o)
    if fillet > 0:
        body_o = body_o.edges(SolidSelector(column_o.translate((0, 0, 1)).solids().val())) \
            .fillet(fillet)
        body_o = try_fillet(body_o, body_o.edges(SolidSelector(column_o.translate((0, 0, -1)).solids().val())),
                            min(fillet_size(length), fillet))
    return body_o


def style_filter(**kwargs):
    filters = ['{0}:{1}'.format(a, b) for (a, b) in kwargs.items()]

    def fct(element):
        path, attributes = element
        for f in filters:
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
    paths, attributes = svgpathtools.svg2paths(args.svg)
    paths = center_rescale(paths, args.scale)
    elements = list(zip(paths, attributes))

    pcb_outline = list(filter(ShapeTypes.Outline, elements))[0]
    pcb_outline = (close_path_sanitizing(pcb_outline[0]), pcb_outline[1])

    body_o = cq.Workplane("front")
    body_o, bottom_draw = create_plate(body_o, pcb_outline, args)
    plane = body_o.plane

    fingers = list(filter(ShapeTypes.Finger, elements))
    body_o.plane = plane
    body_o = create_wall(body_o, pcb_outline, fingers, args)

    supports = list(filter(ShapeTypes.Support, elements))
    for support in supports:
        body_o.plane = plane
        path = close_path_sanitizing(support[0])
        body_o = create_column(body_o, path, args.under_space, args.support_fillet if args.extra_fillet > 2 else 0)

    fillers = list(filter(ShapeTypes.Filler, elements))
    for filler in fillers:
        body_o.plane = plane
        path = close_path_sanitizing(filler[0])
        path = offset_curve(path, -args.play)
        body_o = create_column(body_o, path, args.under_space + args.pcb_thickness + args.filler_height,
                               args.filler_fillet if args.extra_fillet > 2 else 0)

    holes = list(filter(ShapeTypes.Hole, elements))
    for hole in holes:
        body_o.plane = plane
        if len(hole[0]) == 1:
            x1, x2, y1, y2 = hole[0].bbox()
            body_o = body_o.moveTo((x1 + x2) / 2, (y1 + y2) / 2) \
                .circle(args.pin_diameter / 2) \
                .extrude(args.bottom, "cut", both=True)
        else:
            path = close_path_sanitizing(hole[0])
            body_o = to_wire(path, body_o).extrude(args.bottom, "cut", both=True)

    screws = list(filter(ShapeTypes.Screw, elements))
    for screw in screws:
        body_o.plane = plane
        body_o = create_screw(body_o, pcb_outline, screw, args)

    svgpathtools.wsvg(bottom_draw, filename=f'{args.output}.svg')
    cq.exporters.export(body_o, args.output)
    try:
        show_object(body_o, name='plate')
    except BaseException:
        pass


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