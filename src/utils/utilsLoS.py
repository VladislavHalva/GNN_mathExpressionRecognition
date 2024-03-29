# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import numpy as np
from math import sqrt, acos, pi
from shapely.geometry import Polygon, Point


def rect_point_distance(rect, p):
    """
    Calculates rectangle to point distance.
    :param rect: list of tuples of vertices coordinates
    :param p: tuple of x,y coordinates
    :return: Euclidean distance
    """
    rect = Polygon(rect)
    point = Point(p[0], p[1])
    return point.distance(rect)


def sort_components_by_distance(pov, components, skip_id=None):
    """
    Sort components in ascending order of their bounding boxes of from point of view.
    :param pov: tuple x,y coordinates of point of view
    :param components: list of components
    :param skip_id: id of component that shall be skipped (most likely source component id)
    :return: array of sorted indices of components
    """
    distances = []
    for i in range(len(components)):
        min_comp_dist = rect_point_distance(components[i]['bbox'], pov)
        distances.append(min_comp_dist)
    # get order of items from lowest distance and possibly skip item with given id (mostly self)
    distances_sorted_indices = np.argsort(np.asarray(distances)).tolist()
    if skip_id is not None and skip_id in distances_sorted_indices:
        distances_sorted_indices.remove(skip_id)
    return distances_sorted_indices


def get_blocking_view_angles_range(pov, points):
    """
    Calculates bounding angles of point of view, that is blocked by rectangle specified by given points
    :param pov: x,y coordinates of point of view
    :param points: list of rectangle vertices points
    :return: tuple of bounding angles (blocked range of view)
    """
    angles = []
    horizon_vec = (1.0, 0.0)
    horizon_vec_norm = 1.0

    first_quad = False
    fourth_quad = False

    # calculate angle of each points direction vector to horizontal pane
    for point in points:
        dir_vec = (point[0] - pov[0], point[1] - pov[1])
        dir_vec_norm = sqrt(pow(dir_vec[0], 2) + pow(dir_vec[1], 2))

        vec_prod = dir_vec[0] * horizon_vec[0] + dir_vec[1] * horizon_vec[1]

        if horizon_vec_norm + dir_vec_norm != 0:
            angle_rad = acos(
                vec_prod / (horizon_vec_norm + dir_vec_norm)
            )
        else:
            angle_rad = 0

        if point[1] > pov[1]:
            angle_rad = 2.0 * pi - angle_rad
        angle_deg = angle_rad * 180.0 / pi

        angles.append(angle_deg)

        if angle_deg <= 90.0:
            first_quad = True
        if angle_deg >= 270.0:
            fourth_quad = True

    # if the angles are in first and fourth quadrant the view angle range
    # has to be calculated differently, because it passes the 0/360 degrees edge
    if first_quad and fourth_quad:
        min_angle = 0.0
        max_angle = 360.0
        for angle in angles:
            if 90.0 >= angle > min_angle:
                min_angle = angle
            if 270.0 <= angle < max_angle:
                max_angle = angle
    else:
        min_angle = np.min(np.asarray(angles))
        max_angle = np.max(np.asarray(angles))

    return [min_angle, max_angle]


def do_intervals_overlap(i1_start, i1_end, i2_start, i2_end):
    """
    True if given interval overlap, False otherwise.
    :param i1_start: interval 1 start point
    :param i1_end: interval 1 end point
    :param i2_start: interval 2 start point
    :param i2_end: interval 2 end point
    :return:
    """
    if i2_start <= 90.0 and i2_end >= 270.0:
        # first and fourth quadrant
        if i2_end >= i1_end >= i1_start >= i2_start or i2_start <= i1_start <= i1_end <= i2_end:
            return False
        return True
    else:
        if i1_start <= i1_end <= i2_start <= i2_end or i2_start <= i2_end <= i1_start <= i2_end:
            return False
        return True


def is_interval_contained_in_view_interval(i1_start, i1_end, i2_start, i2_end):
    """
    True if interval 2 is fully contained if interval 1.
    :param i1_start: interval 1 start point
    :param i1_end: interval 1 end point
    :param i2_start: interval 2 start point
    :param i2_end: interval 2 end point
    :return:
    """
    if i1_start <= i2_start <= i2_end <= i1_end:
        return True
    return False


def is_component_visible(unblocked_view, component_view_angle):
    """
    Calculate whether components view angle (range of view) is completely
    visible, given the unblocked view
    :param unblocked_view: list of tuples of unblocked view intervals
    :param component_view_angle: tuple of bounding angles of component view angle
    :return: True if visible, False otherwise
    """
    for unblocked_view_interval in unblocked_view:
        # component is completely visible
        if is_interval_contained_in_view_interval(
                unblocked_view_interval[0], unblocked_view_interval[1],
                component_view_angle[0], component_view_angle[1]):
            return True
    if component_view_angle[0] <= 90.0 and component_view_angle[1] >= 270.0:
        # special processing for first/fourth quadrant
        from_zero_view = None
        to_360_view = None
        for unblocked_view_interval in unblocked_view:
            if unblocked_view_interval[0] == 0:
                from_zero_view = unblocked_view_interval
            if unblocked_view_interval[1] == 360.0:
                to_360_view = unblocked_view_interval
        if from_zero_view is not None and to_360_view is not None:
            if from_zero_view[1] > component_view_angle[0] and to_360_view[0] < component_view_angle[1]:
                return True
    return False


def substract_intervals(i_src_start, i_src_end, i_sub_start, i_sub_end):
    """
    Returns interval/s created as subtraction of interval i_sub from interval i_src.
    :param i_src_start: source interval start point
    :param i_src_end: source interval end point
    :param i_sub_start: subtracted interval start point
    :param i_sub_end: subtracted interval end point
    :return:
    """
    # in notation ||| means source interval, --- mean subtracted interval
    # and +++ mean overlap of both

    if i_sub_start <= 90.0 and i_sub_end >= 270.0:
        # first and fourth quadrant
        if i_src_start <= i_sub_start <= i_sub_end <= i_src_end:
            # +++|||+++
            return [[i_sub_start, i_sub_end]]
        elif i_src_start <= i_sub_start <= i_src_end <= i_sub_end:
            # ---+++|||
            return [[i_sub_start, i_src_end]]
        elif i_sub_start <= i_src_start <= i_sub_end <= i_src_end:
            # |||+++---
            return [[i_src_start, i_sub_end]]
        elif i_src_start <= i_src_end <= i_sub_start <= i_sub_end:
            # ---+++---
            return []
        elif i_sub_start <= i_sub_end <= i_src_start <= i_src_end:
            # ---+++---
            return []
        else:
            # do not intersect |||--- or ---|||
            return [[i_src_start, i_src_end]]
    else:
        if i_src_start <= i_sub_start <= i_src_end <= i_sub_end:
            # source left, sub right and touch or intersect: |||+++---
            return [[i_src_start, i_sub_start]]
        elif i_sub_start <= i_src_start <= i_sub_end <= i_src_end:
            # source right, sub left and touch or intersect: ---+++|||
            return [[i_sub_end, i_src_end]]
        elif i_src_start <= i_sub_start <= i_sub_end <= i_src_end:
            # sub completely inside source: |||+++|||
            return [[i_src_start, i_sub_start], [i_sub_end, i_src_end]]
        elif i_sub_start <= i_src_start <= i_src_end <= i_sub_end:
            # source completely inside sub: ---+++---
            return []
        else:
            # do not intersect: |||--- or ---|||
            return [[i_src_start, i_src_end]]


def block_range_in_view_section(view_section, blocking_view):
    """
    Blocks a portion of view interval by blocking view interval.
    :param view_section: tuple of bounding angles : view section to be blocked
    :param blocking_view: tuple of bounding angles : blocked view section
    :return:
    """
    if not do_intervals_overlap(view_section[0], view_section[1], blocking_view[0], blocking_view[1]):
        return [view_section]
    else:
        return substract_intervals(view_section[0], view_section[1], blocking_view[0], blocking_view[1])


def block_range_in_view_sections(view_sections, blocking_view):
    """
    Block range of view in unblocked view sections list.
    :param view_sections: list of tuples - intervals of unblocked view portions
    :param blocking_view: tuple of bounding angles of blocked interval
    :return:
    """
    global_blocked_view_sections = []
    for view_section in view_sections:
        blocked_view_sections = block_range_in_view_section(view_section, blocking_view)
        global_blocked_view_sections.extend(blocked_view_sections)
    return global_blocked_view_sections


def edge_in_edges_undirected(edges, edge_start, edge_end):
    """
    Determines whether edge is already in edges list - regardless of direction.
    :param edges: list of edges
    :param edge_start: processed edge start point
    :param edge_end: processed edge end point
    :return: True if already is in edges list, False otherwise
    """
    for edge in edges:
        if edge[0] == edge_start and edge[1] == edge_end or edge[0] == edge_end and edge[1] == edge_start:
            return True
    return False
