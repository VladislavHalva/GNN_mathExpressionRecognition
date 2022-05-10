import pyvisgraph as vg
import os
import pickle
import random
from math import sqrt, ceil
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2 as cv
import imutils
import numpy as np
import scipy.ndimage
import torch
import imghdr
import logging
from shapely.geometry import Polygon, LineString
from matplotlib import pyplot as plt
from shapely.ops import nearest_points
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
from networkx.drawing.nx_pydot import graphviz_layout

from src.data.GPairData import GPairData
from src.data.LatexVocab import LatexVocab
from src.definitions.MathMLAnnotationType import MathMLAnnotationType
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.ItemLoadError import ItemLoadError
from src.utils.utils import mathml_unicode_to_latex_label


class CrohmeDataset(Dataset):
    def __init__(self, images_root, inkmls_root, tokenizer, components_shape=(32, 32), tmp_path=None, substitute_terms=False, img_transform=False):
        self.images_root = images_root
        self.inkmls_root = inkmls_root
        self.tokenizer = tokenizer
        self.components_shape = components_shape
        self.edge_features = 19
        self.items = []
        self.tmp_path = tmp_path
        self.substitute_terms = substitute_terms
        self.img_transform= img_transform

        if not os.path.exists(self.images_root):
            raise FileNotFoundError('Images directory not found')
        if not os.path.exists(self.inkmls_root):
            raise FileNotFoundError('Inkmls directory not found')

        logging.info('Loading data...')

        for subdir, _, files in os.walk(images_root):
            for file in files:
                image_file = file
                image_path = os.path.join(subdir, image_file)
                if imghdr.what(image_path) is not None:
                    file_name = '.'.join(file.split('.')[:-1])
                    relative_path = os.path.relpath(subdir, images_root)

                    inkml_file = file_name + '.inkml'
                    inkml_path = os.path.join(inkmls_root, relative_path, inkml_file)
                    if os.path.exists(inkml_path):
                        self.items.append([image_path, inkml_path, file_name])

        logging.info(f"Dataset: {len(self.items)} item found")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        try:
            item = self.items[idx]
            image_path, inkml_path, file_name = item

            if self.tmp_path:
                tmp_file_path = os.path.join(self.tmp_path, file_name + '.tmp')

            if self.tmp_path and os.path.isfile(tmp_file_path):
                # load temp if exists
                with open(tmp_file_path, 'rb') as tmp_file:
                    data = pickle.load(tmp_file)
                    return data
            else:
                # build data if temp not exists
                # get source graph - LoS
                x, edge_index, edge_attr, los_components = self.get_src_item(image_path)
                # get tgt graph - SLT, and LaTeX ground-truth
                gt, gt_ml, tgt_y, tgt_edge_index, tgt_edge_type, tgt_edge_relation, comp_symbols = self.get_tgt_item(image_path, inkml_path, los_components)
                # build dataset item
                data = GPairData(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    gt=gt, gt_ml=gt_ml, tgt_y=tgt_y, tgt_edge_index=tgt_edge_index,
                    tgt_edge_type=tgt_edge_type, tgt_edge_relation=tgt_edge_relation,
                    comp_symbols=comp_symbols
                )
                data.filename = file_name

                if self.tmp_path:
                    # save for next epoch/train run
                    if not os.path.isfile(tmp_file_path):
                        with open(tmp_file_path, 'wb') as tmp_file:
                            pickle.dump(data, tmp_file)

                return data
        except Exception as e:
            logging.debug(e)
            return self.__getitem__(random.randrange(0, self.__len__()))

    def get_src_item(self, image_path):
        # extract components and build LoS graph
        components, components_mask = self.extract_components_from_image(image_path)
        edges, edge_features = self.get_line_of_sight_edges(components, components_mask)
        # self.draw_los(image_path, components, edges)

        # plot components
        # cols = 4
        # rows = ceil(len(components) / cols)
        # row = 0
        # col = 0
        # _, axs = plt.subplots(rows, cols, figsize=(32, 32))
        # axs = axs.flatten()
        # for component, ax in zip(components, axs):
        #     image = component['image']
        #     ax.imshow(image)
        # plt.show()

        # BUILD PyG GRAPH DATA ELEMENT
        # input components images
        component_images = [component['image'] for component in components]
        component_images = np.array(component_images)

        # image rotation code - backup
        if self.img_transform:
            for i, component_image in enumerate(component_images):
                rotation_angle = float(random.randint(-30, 30))
                component_image = imutils.rotate(component_image, angle=rotation_angle)
                component_images[i] = component_image

        x = torch.tensor(
            component_images,
            dtype=torch.float)
        x = torch.unsqueeze(x, 1)

        # input edges - and make undirected - first in one direction, than backward
        edge_index = torch.tensor(
            [edge_idx['components'] for edge_idx in edges] +
            [[edge_idx['components'][1], edge_idx['components'][0]] for edge_idx in edges],
            dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        # input edges attributes - also create backward edge attributes (from the forward ones)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        edge_attr = self.add_backward_edge_attr(edge_attr)

        if edge_index.size(0) == 0:
            # prevent error in case of empty edge set
            # add empty list of desired shape
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_features), dtype=torch.float)

        return x, edge_index, edge_attr, components

    def get_tgt_item(self, image_path, inkml_path, los_components=None):
        # extract ground truth latex sentence from inkml
        gt_latex = self.get_latex_from_inkml(inkml_path)
        gt_latex = LatexVocab.split_to_tokens(gt_latex)
        gt_latex_tokens = self.tokenizer.encode(gt_latex)
        gt = torch.tensor(gt_latex_tokens.ids, dtype=torch.long)

        # build target symbol layout tree
        tgt_y, tgt_edge_index, tgt_edge_type, tgt_edge_relation, gt_from_mathml, comp_symbols = self.get_slt(image_path, inkml_path, los_components)

        tgt_y = torch.tensor(tgt_y, dtype=torch.long)
        tgt_edge_index = torch.tensor(tgt_edge_index, dtype=torch.long)
        tgt_edge_type = torch.tensor(tgt_edge_type, dtype=torch.long)
        tgt_edge_relation = torch.tensor(tgt_edge_relation, dtype=torch.long)
        tgt_edge_index = tgt_edge_index.t().contiguous()

        gt_from_mathml = " ".join(gt_from_mathml)
        gt_from_mathml = self.tokenizer.encode(gt_from_mathml)
        gt_from_mathml = torch.tensor(gt_from_mathml.ids, dtype=torch.long)

        return gt, gt_from_mathml, tgt_y, tgt_edge_index, tgt_edge_type, tgt_edge_relation, comp_symbols

    def add_padding_to_component(self, component):
        # get the bigger of images sizes
        new_size = max(component.shape[0], component.shape[1])
        # create empty array of background
        padded = np.full((new_size, new_size), 255, dtype=np.uint8)
        # compute center offset to insert original component image
        y_center = (new_size - component.shape[0]) // 2
        x_center = (new_size - component.shape[1]) // 2
        # copy original image to center of new bg image
        padded[y_center:y_center + component.shape[0], x_center:x_center + component.shape[1]] = component
        return padded

    def extract_components_from_image(self, imagepath):
        img = cv.imread(imagepath, cv.IMREAD_GRAYSCALE)
        _, bin_img = cv.threshold(img, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        inv_img = 255 - bin_img
        # connected components analysis
        num_labels, labels, stats, centroid = cv.connectedComponentsWithStats(
            image=inv_img, connectivity=8, ltype=cv.CV_32S)
        # extract separate components - 0 is background
        components = []
        for i in range(1, num_labels):
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]
            (cX, cY) = centroid[i]
            # mask processed component
            component_mask = (labels != i).astype("uint8") * 255
            # extract component area - bounding box
            component = component_mask[y:y + h, x:x + w]
            # add padding if aspect ratio is too high --> resize would totally ruin the image
            aspect_ratio = w / h
            if aspect_ratio > 2 or aspect_ratio < 0.5:
                component = self.add_padding_to_component(component)
            # resize to desired shape
            component = cv.resize(component, self.components_shape, interpolation=cv.INTER_AREA)
            # and binarize again - if INTER_NEAREST is used, gaps appear within symbols
            _, component = cv.threshold(component, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # invert so that bg is zero
            component = 255 - component
            components.append({
                'image': component,
                'bbox': [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h)
                ],
                'centroid': (cX, cY)
            })
        # shift components by one in mask -> so that they match their ids in components list
        components_mask = labels - 1
        return components, components_mask

    def get_line_of_sight_edges(self, components, components_mask):
        # getting edge-set
        edges = []
        # enumerate all possible edges - undirected
        for i in range(len(components)):
            startpoint = (int(components[i]['centroid'][0]),
                          int(components[i]['centroid'][1]))
            for j in range(i + 1, len(components)):
                endpoint = (int(components[j]['centroid'][0]),
                            int(components[j]['centroid'][1]))

                # test if line collides with some other component
                some_collision = False
                for c_idx in [x for x in range(len(components)) if x != i and x != j]:
                    c_bbox = components[c_idx]['bbox']
                    collides = self.line_and_rect_intersect([startpoint, endpoint], c_bbox)

                    if collides:
                        some_collision = True
                        break
                # no collision - append to final edge-set
                if not some_collision:
                    edges.append({
                        'components': [i, j],
                        'start': startpoint,
                        'end': endpoint
                    })

        # extracting edge features
        edge_features = []
        for edge in edges:
            # get corresponding components
            component1_id = edge['components'][0]
            component2_id = edge['components'][1]
            component1 = components[component1_id]
            component2 = components[component2_id]
            # find minimal and maximal distance of object strokes
            component2_mask = (components_mask != component2_id).astype('uint8') * 255
            component2_dist = cv.distanceTransform(component2_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
            component1_mask = (components_mask == component1_id).astype('uint8')
            components_dist = component2_dist * component1_mask
            nonzero_mask = (components_dist != 0).astype('uint8')
            min_stroke_dist, max_stroke_dist, _, _ = cv.minMaxLoc(components_dist, nonzero_mask)
            # get centroids distance
            centroids_distance = sqrt(
                pow(component1['centroid'][0] - component2['centroid'][0], 2) +
                pow(component1['centroid'][1] - component2['centroid'][1], 2)
            )
            centroids_distance_h = abs(component1['centroid'][1] - component2['centroid'][1])
            centroids_distance_v = abs(component1['centroid'][0] - component2['centroid'][0])
            # get bbox area and size ratios
            bbox1_w = component1['bbox'][1][0] - component1['bbox'][0][0]
            bbox1_h = component1['bbox'][3][1] - component1['bbox'][0][1]
            bbox2_w = component2['bbox'][1][0] - component2['bbox'][0][0]
            bbox2_h = component2['bbox'][3][1] - component2['bbox'][0][1]
            bbox1_diagonal = sqrt(pow(bbox1_w, 2) + pow(bbox1_h, 2))
            bbox2_diagonal = sqrt(pow(bbox2_w, 2) + pow(bbox2_h, 2))
            bbox1_area = bbox1_w * bbox1_h
            bbox2_area = bbox2_w * bbox2_h
            bbox_area_ratio = bbox1_area / bbox2_area
            bbox_w_ratio = bbox1_w / bbox2_w
            bbox_h_ratio = bbox1_h / bbox2_h
            bbox_diagonal_ratio = bbox1_diagonal / bbox2_diagonal
            if bbox1_area > bbox2_area:
                bbox_l_to_union_ratio = bbox1_area / (bbox1_area + bbox2_area)
            else:
                bbox_l_to_union_ratio = bbox2_area / (bbox1_area + bbox2_area)
            # append edge features with features extracted for current edge
            edge_features.append([
                min_stroke_dist, max_stroke_dist,
                centroids_distance, centroids_distance_h, centroids_distance_v,
                bbox_area_ratio, bbox_l_to_union_ratio,
                bbox_w_ratio, bbox_h_ratio,
                bbox_diagonal_ratio, bbox_area_ratio,
                component1['bbox'][0][0], component1['bbox'][0][1],
                component1['bbox'][2][0], component1['bbox'][2][1],
                component2['bbox'][0][0], component2['bbox'][0][1],
                component2['bbox'][2][0], component2['bbox'][2][1]
            ])
        return edges, edge_features

    def add_backward_edge_attr(self, edge_attr):
        """
        Transforms extracted edge features in one direction, so that they
        match the values that would be calculated for opposite direction.
        Without new extraction - only "inverts" the values in appropriate way.
        :param edge_attr: edge attributes for directed edges
        :return: edge_attr for edges in both directions
        """
        bw_edge_attr = torch.clone(edge_attr)
        if bw_edge_attr.size(0) > 0:
            # if there are some attributes only - prevent IndexError
            bw_edge_attr[:, 5] = torch.pow(bw_edge_attr[:, 5], -1)
            bw_edge_attr[:, 7] = torch.pow(bw_edge_attr[:, 7], -1)
            bw_edge_attr[:, 8] = torch.pow(bw_edge_attr[:, 8], -1)
            bw_edge_attr[:, 9] = torch.pow(bw_edge_attr[:, 9], -1)
            bw_edge_attr[:, 10] = torch.pow(bw_edge_attr[:, 10], -1)
            bw_edge_attr[:, 11], bw_edge_attr[:, 15] = bw_edge_attr[:, 15], bw_edge_attr[:, 11]
            bw_edge_attr[:, 12], bw_edge_attr[:, 16] = bw_edge_attr[:, 16], bw_edge_attr[:, 12]
            bw_edge_attr[:, 13], bw_edge_attr[:, 17] = bw_edge_attr[:, 17], bw_edge_attr[:, 13]
            bw_edge_attr[:, 14], bw_edge_attr[:, 18] = bw_edge_attr[:, 18], bw_edge_attr[:, 14]

        return torch.cat([edge_attr, bw_edge_attr], dim=0)

    def line_and_rect_intersect(self, line_bounds, rect_corners):
        line_start = line_bounds[0]
        line_end = line_bounds[1]
        rect = Polygon([
            [rect_corners[0][0], rect_corners[0][1]],
            [rect_corners[1][0], rect_corners[1][1]],
            [rect_corners[2][0], rect_corners[2][1]],
            [rect_corners[3][0], rect_corners[3][1]]
        ])

        line = LineString([line_start, line_end])
        return line.intersects(rect)

    def draw_los(self, imagepath, components, edges):
        img = cv.imread(imagepath)
        for i in range(len(components)):
            topleft = components[i]['bbox'][0]
            bottomright = components[i]['bbox'][2]
            (cX, cY) = components[i]['centroid']
            cv.rectangle(img, topleft, bottomright, (0, 255, 0), 1)
            cv.circle(img, (int(cX), int(cY)), 2, (0, 0, 255), -1)

        for edge in edges:
            (sX, sY) = edge['start']
            (eX, eY) = edge['end']
            cv.line(img, (int(sX), int(sY)), (int(eX), int(eY)), (255, 0, 0), 1)

        plt.imshow(img)
        plt.show()

    def get_latex_from_inkml(self, filepath):
        if not os.path.isfile(filepath) and Path(filepath).suffix != '.inkml':
            logging.debug("Inkml file does not exists: " + filepath)
            return ""

        doc_namespace = '{http://www.w3.org/2003/InkML}'
        tree = ET.parse(filepath)
        root = tree.getroot()

        try:
            latex_gt = root.find(doc_namespace + 'annotation[@type="truth"]').text
            latex_gt = latex_gt.replace('$', '')
            return latex_gt
        except AttributeError:
            # element not found
            logging.debug("Inkml file does not contain latex groundtruth: " + filepath)
            return ""

    def mathml_dfs(self, xml_ns, mathml_ns, root):
        s, r = [], []
        sequence = []

        if root.tag in [mathml_ns + 'math', mathml_ns + 'mrow']:
            # just connect all children to row (right relation)
            linearly_connect = []
            for child in root:
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                sequence.extend(sequence_a)
                linearly_connect.append(subtree_root_id)
            for src, tgt in zip(linearly_connect, linearly_connect[1:]):
                r.append({
                    'src_id': src,
                    'tgt_id': tgt,
                    'type': SrtEdgeTypes.RIGHT
                })
            return s, r, sequence, linearly_connect[0]
        elif root.tag == mathml_ns + 'msqrt':
            # append \sqrt symbol and connect its first child as inside
            # all children connect to row - linear (just like mrow)
            sqrt_symbol_id = root.attrib.get(xml_ns + 'id')
            s.append({
                'id': sqrt_symbol_id,
                'symbol': r"\sqrt"
            })
            sequence.extend([r"\sqrt", "{"])
            linearly_connect = []
            for i, child in enumerate(root):
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                sequence.extend(sequence_a)
                linearly_connect.append(subtree_root_id)
                if i == 0:
                    r.append({
                        'src_id': sqrt_symbol_id,
                        'tgt_id': subtree_root_id,
                        'type': SrtEdgeTypes.INSIDE
                    })
            sequence.append("}")
            for src, tgt in zip(linearly_connect, linearly_connect[1:]):
                r.append({
                    'src_id': src,
                    'tgt_id': tgt,
                    'type': SrtEdgeTypes.RIGHT
                })
            return s, r, sequence, sqrt_symbol_id
        elif root.tag == mathml_ns + 'mroot':
            # process subtrees, add sqrt symbol and add base inside and index above
            sqrt_symbol_id = root.attrib.get(xml_ns + 'id')
            s.append({
                'id': sqrt_symbol_id,
                'symbol': r"\sqrt"
            })
            sequence.append(r"\sqrt")
            basis_sequence = None
            root_sequence = None
            for i, child in enumerate(root):
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                if i == 0:
                    r.append({
                        'src_id': sqrt_symbol_id,
                        'tgt_id': subtree_root_id,
                        'type': SrtEdgeTypes.INSIDE
                    })
                    basis_sequence = sequence_a
                elif i == 1:
                    r.append({
                        'src_id': sqrt_symbol_id,
                        'tgt_id': subtree_root_id,
                        'type': SrtEdgeTypes.ABOVE
                    })
                    root_sequence = sequence_a

                if root_sequence:
                    sequence.append('[')
                    sequence.extend(root_sequence)
                    sequence.append(']')
                sequence.append('{')
                sequence.extend(basis_sequence)
                sequence.append('}')
            return s, r, sequence, sqrt_symbol_id
        elif root.tag in [mathml_ns + 'msub', mathml_ns + 'msup', mathml_ns + 'munder', mathml_ns + 'mover']:
            # process subtrees and add sub/superscript or over/under connection between their roots
            basis_id = None
            script_id = None
            for i, child in enumerate(root):
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                if i == 0:
                    basis_id = subtree_root_id
                    basis_sequence = sequence_a
                elif i == 1:
                    script_id = subtree_root_id
                    script_sequence = sequence_a
            if not basis_id or not script_id:
                logging.debug('MathML sub/superscript syntax error')
                return s, r, [], root.attrib.get(xml_ns + 'id')

            sequence.extend(basis_sequence)
            if root.tag == mathml_ns + 'msub':
                relation_type = SrtEdgeTypes.SUBSCRIPT
                sequence.extend(["_", "{"])
            elif root.tag == mathml_ns + 'msup':
                relation_type = SrtEdgeTypes.SUPERSCRIPT
                sequence.extend(["^", "{"])
            elif root.tag == mathml_ns + 'munder':
                relation_type = SrtEdgeTypes.BELOW
                sequence.extend([r"\underset", "{"])
            else:
                relation_type = SrtEdgeTypes.ABOVE
                sequence.extend([r"\overset", "{"])
            sequence.extend(script_sequence)
            sequence.append("}")

            r.append({
                'src_id': basis_id,
                'tgt_id': script_id,
                'type': relation_type
            })
            return s, r, sequence, basis_id
        elif root.tag in [mathml_ns + 'msubsup', mathml_ns + 'munderover']:
            # process subtrees and add sub+superscript/under+over connection between their roots
            basis_id = None
            subscript_id = None
            superscript_id = None
            for i, child in enumerate(root):
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                if i == 0:
                    basis_id = subtree_root_id
                    basis_sequence = sequence_a
                elif i == 1:
                    subscript_id = subtree_root_id
                    subscript_sequence = sequence_a
                elif i == 2:
                    superscript_id = subtree_root_id
                    superscript_sequence = sequence_a
            if not basis_id:
                logging.debug('MathML sub+superscript syntax error')
                return s, r, [], root.attrib.get(xml_ns + 'id')

            if root.tag == mathml_ns + 'msubsup':
                relation1 = SrtEdgeTypes.SUBSCRIPT
                relation2 = SrtEdgeTypes.SUPERSCRIPT
            else:
                relation1 = SrtEdgeTypes.BELOW
                relation2 = SrtEdgeTypes.ABOVE

            if root.tag == mathml_ns + 'msubsup':
                sequence.extend(basis_sequence)
                if subscript_id:
                    sequence.extend(["_", "{"])
                    sequence.extend(subscript_sequence)
                    sequence.append("}")
                if superscript_id:
                    sequence.extend(["^", "{"])
                    sequence.extend(superscript_sequence)
                    sequence.append("}")
            else:
                sequence.extend(basis_sequence)
                if subscript_id:
                    sequence.extend([r"\underset", "{"])
                    sequence.extend(subscript_sequence)
                    sequence.append("}")
                if superscript_id:
                    sequence.extend([r"\overset", "{"])
                    sequence.extend(superscript_sequence)
                    sequence.append("}")

            if subscript_id:
                r.append({
                    'src_id': basis_id,
                    'tgt_id': subscript_id,
                    'type': relation1
                })
            if superscript_id:
                r.append({
                    'src_id': basis_id,
                    'tgt_id': superscript_id,
                    'type': relation2
                })
            return s, r, sequence, basis_id
        elif root.tag == mathml_ns + 'mfrac':
            # process subtrees, add \frac symbol and add above/below
            # relation to numerator/denominator
            frac_symbol_id = root.attrib.get(xml_ns + 'id')
            s.append({
                'id': frac_symbol_id,
                'symbol': r"\frac"
            })
            sequence.append(r"\frac")

            numerator_root_id = None
            denominator_root_id = None
            for i, child in enumerate(root):
                s_a, r_a, sequence_a, subtree_root_id = self.mathml_dfs(xml_ns, mathml_ns, child)
                s.extend(s_a)
                r.extend(r_a)
                if i == 0:
                    numerator_root_id = subtree_root_id
                    numerator_sequence = sequence_a
                elif i == 1:
                    denominator_root_id = subtree_root_id
                    denominator_sequence = sequence_a
            if not numerator_root_id or not denominator_root_id:
                logging.debug('MathML fraction syntax error')
                if not numerator_root_id and not denominator_root_id:
                    sequence.extend(["{", "}", "{", "}"])
                elif not numerator_root_id:
                    sequence.extend(["{", "}", "{"])
                    sequence.extend(denominator_sequence)
                    sequence.append("}")
                elif not denominator_sequence:
                    sequence.append("{")
                    sequence.extend(numerator_sequence)
                    sequence.extend(["}", "{", "}"])
                return s, r, sequence, root.attrib.get(xml_ns + 'id')
            r.append({
                'src_id': frac_symbol_id,
                'tgt_id': numerator_root_id,
                'type': SrtEdgeTypes.ABOVE
            })
            r.append({
                'src_id': frac_symbol_id,
                'tgt_id': denominator_root_id,
                'type': SrtEdgeTypes.BELOW
            })
            sequence.append("{")
            sequence.extend(numerator_sequence)
            sequence.extend(["}", "{"])
            sequence.extend(denominator_sequence)
            sequence.append("}")
            return s, r, sequence, frac_symbol_id
        elif root.tag in [mathml_ns + 'mi', mathml_ns + 'mn', mathml_ns + 'mo', mathml_ns + 'mtext',
                          mathml_ns + 'mspace', mathml_ns + 'ms']:
            id = root.attrib.get(xml_ns + 'id')
            if self.substitute_terms:
                if root.tag in [mathml_ns + 'mi', mathml_ns + 'mn', mathml_ns + 'mtext']:
                    s.append({
                        'id': id,
                        'symbol': '<TERM>'
                    })
                    sequence = ['<TERM>']
                else:
                    s.append({
                        'id': id,
                        'symbol': root.text
                    })
                    sequence = [root.text]
            else:
                s.append({
                    'id': id,
                    'symbol': root.text
                })
                sequence = [root.text]
            return s, r, sequence, id
        else:
            raise ItemLoadError('unknown MathML element: ' + root.tag)

    def parse_traces_inkml(self, ns, root):
        traces = []
        for trace_node in root.findall(ns + 'trace'):
            trace_id = trace_node.get('id')
            trace_def = trace_node.text
            trace_def = trace_def.split(',')
            trace_points = []
            for trace_def_elem in trace_def:
                point_coords = trace_def_elem.split(' ')
                point_coords = [coord.strip() for coord in point_coords]
                point_coords = list(filter(lambda coords: coords != '', point_coords))
                point_x = float(point_coords[0])
                point_y = float(point_coords[1])
                trace_points.append([point_x, point_y])
            traces.append({
                'id': trace_id,
                'points': trace_points
            })

        for i, trace in enumerate(traces):
            min_x = float('inf')
            min_y = float('inf')
            max_x = 0
            max_y = 0
            for point in trace['points']:
                min_x = min(min_x, point[0])
                min_y = min(min_y, point[1])
                max_x = max(max_x, point[0])
                max_y = max(max_y, point[1])
            traces[i]['bbox'] = [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y)
            ]

        return traces

    def parse_tracegroups_inkml(self, ns, root):
        tracegroups_list = []
        expression_group = root.find(ns + 'traceGroup')
        if expression_group is None:
            return []
        tracegroups = expression_group.findall(ns + 'traceGroup')
        for tracegroup in tracegroups:
            gr_symbol_id = tracegroup.find(ns + 'annotationXML')
            if gr_symbol_id is not None:
                gr_symbol_id = gr_symbol_id.get('href')
            gr_symbol_text = tracegroup.find(ns + 'annotation[@type="truth"]')
            if gr_symbol_text is not None:
                gr_symbol_text = gr_symbol_text.text
            traces_list = []
            traces = tracegroup.findall(ns + 'traceView')
            for trace in traces:
                traces_list.append(trace.get('traceDataRef'))
            tracegroups_list.append({
                'symbol_id': gr_symbol_id,
                'symbol_text': gr_symbol_text,
                'traces': traces_list
            })
        return tracegroups_list

    def get_tracegroups_bboxes(self, traces, tracegroups):
        for i, tracegroup in enumerate(tracegroups):
            max_x = 0
            max_y = 0
            min_x = float('inf')
            min_y = float('inf')
            trace_ids = tracegroup['traces']
            some_trace_found = False
            for trace_id in trace_ids:
                trace = next((trace for trace in traces if trace['id'] == trace_id), None)
                if trace is not None:
                    some_trace_found = True
                    trace_min_x = min(trace['bbox'][0][0], trace['bbox'][1][0], trace['bbox'][2][0], trace['bbox'][3][0])
                    trace_max_x = max(trace['bbox'][0][0], trace['bbox'][1][0], trace['bbox'][2][0], trace['bbox'][3][0])
                    trace_min_y = min(trace['bbox'][0][1], trace['bbox'][1][1], trace['bbox'][2][1], trace['bbox'][3][1])
                    trace_max_y = max(trace['bbox'][0][1], trace['bbox'][1][1], trace['bbox'][2][1], trace['bbox'][3][1])
                    max_x = max(max_x, trace_max_x)
                    min_x = min(min_x, trace_min_x)
                    max_y = max(max_y, trace_max_y)
                    min_y = min(min_y, trace_min_y)

            if some_trace_found:
                tracegroups[i]['bbox'] = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y)
                ]
            else:
                tracegroups[i]['bbox'] = None
        return tracegroups

    def parse_inkml(self, inkml_path):
        if not os.path.isfile(inkml_path) and Path(inkml_path).suffix != '.inkml':
            raise ItemLoadError("Inkml file does not exists: " + inkml_path)

        # define namespaces
        xml_namespace = '{http://www.w3.org/XML/1998/namespace}'
        doc_namespace = '{http://www.w3.org/2003/InkML}'
        mathml_namespace = '{http://www.w3.org/1998/Math/MathML}'

        # parse inkml file and get root
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        # get mathml annotation section and determine type
        annotation_mathml_content = root.find(
            doc_namespace + 'annotationXML[@type="truth"][@encoding="Content-MathML"]')
        annotation_mathml_presentation = root.find(
            doc_namespace + 'annotationXML[@type="truth"][@encoding="Presentation-MathML"]')
        if annotation_mathml_content:
            annotation_type = MathMLAnnotationType.CONTENT
            annotation_mathml = annotation_mathml_content
        elif annotation_mathml_presentation:
            annotation_type = MathMLAnnotationType.PRESENTATION
            annotation_mathml = annotation_mathml_presentation
        else:
            raise ItemLoadError("Inkml file does not contain MathML annotation: " + inkml_path)

        # find math definition root
        if annotation_type == MathMLAnnotationType.CONTENT:
            math_root = annotation_mathml.find(mathml_namespace + 'math')
        else:
            math_root = annotation_mathml.find(doc_namespace + 'math')
        if not math_root:
            raise ItemLoadError("Inkml file does not contain math description root: " + inkml_path)

        # parse expression and identify symbols and relations

        try:
            # different namespaces in various types of annotation
            if annotation_type == MathMLAnnotationType.CONTENT:
                s, r, seq, _ = self.mathml_dfs(xml_namespace, mathml_namespace, math_root)
            else:
                s, r, seq, _ = self.mathml_dfs(xml_namespace, doc_namespace, math_root)
        except AttributeError as e:
            raise ItemLoadError(e)
        except ValueError as e:
            raise ItemLoadError(e)

        # identify all traces included in expression
        traces = self.parse_traces_inkml(doc_namespace, root)

        # parse trace-groups corresponding to separate symbols
        tracegroups = self.parse_tracegroups_inkml(doc_namespace, root)
        tracegroups = self.get_tracegroups_bboxes(traces, tracegroups)

        return s, r, seq, tracegroups

    def rescale_tracegroup_coordinates(self, image_path, tracegroups):
        img = cv.imread(image_path)
        img_shape = img.shape
        # size of rendered image
        img_w = img_shape[0]
        img_h = img_shape[1]
        # get size image from original traces coordinates
        traces_max_x = 0
        traces_max_y = 0
        traces_min_x = float('inf')
        traces_min_y = float('inf')
        for tg in tracegroups:
            if tg['bbox'] is not None:
                traces_max_x = max(traces_max_x, tg['bbox'][2][0])
                traces_max_y = max(traces_max_y, tg['bbox'][2][1])
                traces_min_x = min(traces_min_x, tg['bbox'][0][0])
                traces_min_y = min(traces_min_y, tg['bbox'][0][1])
        traces_w = traces_max_x - traces_min_x
        traces_h = traces_max_y - traces_min_y
        # traces to image width ration used as scaling coefficient
        # there is a padding of 0.04*width padding in image
        # small value is added to prevent division by zero
        w_ratio = img_w / ((traces_w + traces_w * 0.04) + 0.00001)
        # expected image height if padding to create square was not added
        expected_height = traces_h * w_ratio
        # padding above formula
        padding_top = (img_h - expected_height) / 2
        for tg_idx in range(len(tracegroups)):
            if tracegroups[tg_idx]['bbox'] is not None:
                for bb_idx in range(4):
                    # shift formula to coordinate system start (take padding in account)
                    tracegroups[tg_idx]['bbox'][bb_idx] = (
                        tracegroups[tg_idx]['bbox'][bb_idx][0] - traces_min_x + (traces_w * 0.02),
                        tracegroups[tg_idx]['bbox'][bb_idx][1] - traces_min_y
                    )
                    # rescale coordinates according to img/traces width ratio and shift vertically to center (padding)
                    tracegroups[tg_idx]['bbox'][bb_idx] = (
                        round(tracegroups[tg_idx]['bbox'][bb_idx][0] * w_ratio),
                        round(tracegroups[tg_idx]['bbox'][bb_idx][1] * w_ratio + padding_top)
                    )
        return tracegroups

    def get_slt(self, image_path, inkml_path, los_components=None):
        symbols, relations, sequence, tracegroups = self.parse_inkml(inkml_path)
        for symbol in symbols:
            symbol['symbol'] = mathml_unicode_to_latex_label(symbol['symbol'])
        for i, symbol in enumerate(sequence):
            sequence[i] = mathml_unicode_to_latex_label(symbol, skip_curly_brackets=True)
        # tokenize symbols
        unk_token_id = self.tokenizer.encode('[UNK]', add_special_tokens=False).ids[0]
        x = [self.tokenizer.encode(s['symbol'], add_special_tokens=False).ids for s in symbols]
        x = [x_i[0] if len(x_i) == 1 else unk_token_id for x_i in x]

        # construct attention ground-truths to LoS input graph nodes
        if los_components is None:
            comp_symbols = None
        else:
            # img = cv.imread(image_path)
            # rescale tracegroups coordinates
            tracegroups = self.rescale_tracegroup_coordinates(image_path, tracegroups)
            # assign symbols to tracegroups
            symbols_bbox_polygons = []
            for symbol in symbols:
                tg = next((tg for tg in tracegroups if tg['symbol_id'] == symbol['id']), None)
                if tg is not None and tg['bbox'] is not None:
                    symbols_bbox_polygons.append(Polygon(tg['bbox']))
                    # topleft = tg['bbox'][0]
                    # bottomright = tg['bbox'][2]
                    # cv.rectangle(img, topleft, bottomright, (255, 0, 0), 1)
                else:
                    symbols_bbox_polygons.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))

            # match most probable symbol to each los component - biggest intersection of bounding boxes
            los_components_symbols = []
            for los_component in los_components:
                los_bbox_polygon = Polygon(los_component['bbox'])
                # topleft = los_component['bbox'][0]
                # bottomright = los_component['bbox'][2]
                # cv.rectangle(img, topleft, bottomright, (0, 255, 0), 1)
                intersection_areas = []
                # intersections = []
                for symbol_polygon in symbols_bbox_polygons:
                    intersection = los_bbox_polygon.intersection(symbol_polygon)
                    # intersections.append(intersection)
                    intersection_areas.append(intersection.area)
                symbol_id = np.argmax(intersection_areas)
                los_components_symbols.append(symbol_id)
                # intersection = intersections[symbol_id]
                # topleft = (round(intersection.bounds[0]), round(intersection.bounds[1]))
                # bottomright = (round(intersection.bounds[2]), round(intersection.bounds[3]))
                # cv.rectangle(img, topleft, bottomright, (0, 0, 255), 1)

            comp_symbols = torch.tensor(los_components_symbols, dtype=torch.long)

            # plt.imshow(img)
            # plt.show()

        # generate end child nodes
        end_nodes, end_edge_index = self.get_end_child_nodes(len(x))

        # append nodes with end children
        x.extend(end_nodes)

        edge_index = []
        edge_type = []
        edge_relation = []

        # build basic SLT graph on symbols and relations given by MathML
        for relation in relations:
            src_arr_id = next((i for i, x in enumerate(symbols) if x['id'] == relation['src_id']), None)
            tgt_arr_id = next((i for i, x in enumerate(symbols) if x['id'] == relation['tgt_id']), None)
            edge_index.append([src_arr_id, tgt_arr_id])
            edge_type.append(SltEdgeTypes.PARENT_CHILD)
            edge_relation.append(relation['type'])

        # append graph with edges to end child nodes
        edge_index.extend(end_edge_index)
        edge_type.extend(SltEdgeTypes.PARENT_CHILD for _ in end_edge_index)
        edge_relation.extend(SrtEdgeTypes.TO_ENDNODE for _ in end_edge_index)

        # self.draw_slt(symbols, x, edge_index, edge_type, edge_relation, False)

        # get grandparent and left brother edges and self loops
        gp_edges = self.get_gp_edges(edge_index)
        bro_edges = self.get_bro_edges(edge_index)
        self_edges = self.get_self_edges(len(x))

        # append gp and bro edges and self loops
        edge_index.extend(gp_edges)
        edge_type.extend([SltEdgeTypes.GRANDPARENT_GRANDCHILD for _ in gp_edges])
        edge_relation.extend(SrtEdgeTypes.UNDEFINED for _ in gp_edges)
        edge_index.extend(bro_edges)
        edge_type.extend(SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER for _ in bro_edges)
        edge_relation.extend(SrtEdgeTypes.UNDEFINED for _ in bro_edges)
        edge_index.extend(self_edges)
        edge_type.extend(SltEdgeTypes.CURRENT_CURRENT for _ in self_edges)
        edge_relation.extend(SrtEdgeTypes.UNDEFINED for _ in self_edges)

        # self.draw_slt(symbols, x, edge_index, edge_type, edge_relation, include_end_nodes=True)

        return x, edge_index, edge_type, edge_relation, sequence, comp_symbols

    def get_tree_root(self, edge_index):
        # find the root node as the only one who does not
        # play the role of a target node of any edge
        root_candidates = [edge[0] for edge in edge_index]
        root_candidates = list(set(root_candidates))

        for edge in edge_index:
            tgt_id = edge[1]
            if tgt_id in root_candidates:
                root_candidates.remove(tgt_id)

        if len(root_candidates) != 1:
            return None

        root = root_candidates[0]
        return root

    def get_end_child_nodes(self, nodes_count):
        eos_token_id = self.tokenizer.encode('[EOS]', add_special_tokens=False).ids[0]
        end_nodes = [eos_token_id for _ in range(nodes_count)]
        end_edge_index = [[i, nodes_count + i] for i in range(nodes_count)]
        return end_nodes, end_edge_index

    def get_gp_edges(self, edge_index):
        root = self.get_tree_root(edge_index)
        gp_edges = self.dfs_gp_edges_idenitification(root, None, edge_index)
        return gp_edges

    def dfs_gp_edges_idenitification(self, root_idx, root_parent_idx, edge_index):
        children = [edge[1] for edge in edge_index if edge[0] == root_idx]

        gp_edges = []

        if len(children) == 0:
            # leaf
            return gp_edges
        else:
            # add edges from roots parent to roots children
            if root_parent_idx is not None:
                for child in children:
                    gp_edges.append([root_parent_idx, child])

            # recursively continue with subtree
            for child in children:
                gp_edges.extend(
                    self.dfs_gp_edges_idenitification(child, root_idx, edge_index)
                )

            # return merged list of gp edges
            return gp_edges

    def get_bro_edges(self, edge_index):
        bro_edges = []
        root = self.get_tree_root(edge_index)
        root_children = [edge[1] for edge in edge_index if edge[0] == root]

        # BFS traversal to identify left siblings edges
        # init
        prev_node = root
        visited = [root]
        queue = root_children
        parents = {root: None}
        for root_child in root_children:
            parents[root_child] = root

        # traverse tree
        while queue:
            node = queue.pop(0)
            if parents[prev_node] == parents[node]:
                bro_edges.append([prev_node, node])

            prev_node = node
            visited.append(node)
            node_children = [edge[1] for edge in edge_index if edge[0] == node]
            for node_child in node_children:
                if node_child not in visited:
                    queue.append(node_child)
                    parents[node_child] = node

        return bro_edges

    def get_self_edges(self, nodes_count):
        self_loop_edges = [[i, i] for i in range(nodes_count)]
        return self_loop_edges

    def draw_slt(self, symbols, x, edge_index, edge_type, edge_relation, include_end_nodes=False):
        x_indices = list(range(len(x)))
        x_indices = torch.tensor(x_indices, dtype=torch.float)

        symbols = [(str(i) + '. ' + symbol['symbol']) for i, symbol in enumerate(symbols)]

        pc_edges = []
        for i, edge in enumerate(edge_index):
            if edge_type[i] == SltEdgeTypes.PARENT_CHILD:
                if include_end_nodes or edge_relation[i] != SrtEdgeTypes.TO_ENDNODE:
                    pc_edges.append(edge)

        edge_index = torch.tensor(pc_edges, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        data = Data(x=x_indices, edge_index=edge_index)

        labeldict = {}
        for i, x_i in enumerate(symbols):
            labeldict[i] = symbols[i]

        G = to_networkx(data=data, to_undirected=False)
        G = nx.relabel_nodes(G, labeldict)

        if not include_end_nodes:
            for idx in range(len(x_indices)):
                if idx >= len(symbols):
                    G.remove_node(idx)

        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True)
        plt.draw()
        plt.show()