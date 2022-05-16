# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import os
import pickle
import random
from math import sqrt
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2 as cv
import numpy as np
import torch
import imghdr
import logging
from shapely.geometry import Polygon, LineString
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx
from networkx.drawing.nx_pydot import graphviz_layout

from src.data.GMathData import GMathData
from src.data.LatexVocab import LatexVocab
from src.definitions.MathMLAnnotationType import MathMLAnnotationType
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.ItemLoadError import ItemLoadError
from src.utils.utils import mathml_unicode_to_latex_label
from src.utils.utilsLoS import sort_components_by_distance, get_blocking_view_angles_range, \
    is_component_visible, block_range_in_view_sections, edge_in_edges_undirected


class CrohmeDataset(Dataset):
    """
    Dataset representation used by DataLoader to fetch data.
    """
    def __init__(self,
                 images_root,
                 inkmls_root,
                 tokenizer,
                 components_shape=(32, 32),
                 tmp_path=None,
                 substitute_terms=False,
                 transform=None):
        """
        :param images_root: path to folder with source images
        :param inkmls_root: path to folder with groudtruth InkML files
        :param tokenizer: huggingface tokenizer object to tokenize symbols
        :param components_shape: desired shape of components images, default (32, 32)
        :param tmp_path: path to folder where temporary data file will be stored
            used to create each data-element only once during training
        :param substitute_terms: whether to substitute identifiers, numbers and text elements in dataset with special tokens
        :param transform: data transformation
        """
        self.images_root = images_root
        self.inkmls_root = inkmls_root
        self.tokenizer = tokenizer
        self.components_shape = components_shape
        self.edge_features = 10
        self.items = []
        self.tmp_path = tmp_path
        self.substitute_terms = substitute_terms
        self.transform = transform

        if not os.path.exists(self.images_root):
            raise FileNotFoundError('Images directory not found')
        if not os.path.exists(self.inkmls_root):
            raise FileNotFoundError('Inkmls directory not found')

        logging.info('Loading data...')

        # create datalist - check whether both image and inkml files exists
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
        """
        Returns dataset element specified by idx
        :param idx: item id
        :return: data object
        """
        try:
            item = self.items[idx]
            image_path, inkml_path, file_name = item

            # fetch element data from tmp file if exist
            if self.tmp_path:
                tmp_file_path = os.path.join(self.tmp_path, file_name + '.tmp')
            if self.tmp_path and os.path.isfile(tmp_file_path):
                # load temp if exists
                with open(tmp_file_path, 'rb') as tmp_file:
                    data = pickle.load(tmp_file)
                    if self.transform:
                        data = self.transform(data)
                    return data
            else:
                # build data if tmp does not exists
                # get source graph - LoS
                x, edge_index, edge_attr, los_components = self.get_src_item(image_path)
                # get tgt graph - SLT, and LaTeX ground-truth
                gt, gt_ml, tgt_y, tgt_edge_index, tgt_edge_type, tgt_edge_relation, comp_symbols = self.get_tgt_item(image_path, inkml_path, los_components)
                # build dataset item object
                data = GMathData(
                    x=x, edge_index=edge_index, edge_attr=edge_attr,
                    gt=gt, gt_ml=gt_ml, tgt_y=tgt_y, tgt_edge_index=tgt_edge_index,
                    tgt_edge_type=tgt_edge_type, tgt_edge_relation=tgt_edge_relation,
                    comp_symbols=comp_symbols
                )
                data.filename = file_name

                # save tmp data representation for next iterations
                if self.tmp_path:
                    if not os.path.isfile(tmp_file_path):
                        with open(tmp_file_path, 'wb') as tmp_file:
                            pickle.dump(data, tmp_file)

                # data transfomation if defined
                if self.transform:
                    data = self.transform(data)
                return data
        except Exception as e:
            # if error while creating item occurred - fetch another random element instead
            logging.debug(e)
            return self.__getitem__(random.randrange(0, self.__len__()))

    def get_src_item(self, image_path):
        """
        Creates source LoS graph from image
        :param image_path: source image filepath
        :return:
            x, edge_index, edge_attr,
            components: list of component object including identifiers and bounding boxes
        """
        # extract components and build LoS graph
        components, component_images, components_mask = self.extract_components_from_image(image_path)
        edges = self.get_line_of_sight_edges(components)
        edge_features = self.compute_los_edge_features(edges, components, components_mask)
        # self.draw_los(image_path, components, edges)

        # build nodes
        component_images = np.array(component_images)
        x = torch.tensor(
            component_images,
            dtype=torch.double)
        x = torch.unsqueeze(x, 1)

        # build edges - and make undirected - first in one direction, than backward
        edge_index = torch.tensor(
            [edge_idx['components'] for edge_idx in edges] +
            [[edge_idx['components'][1], edge_idx['components'][0]] for edge_idx in edges],
            dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        # input edges attributes - also create backward edge attributes (from the forward ones)
        edge_attr = torch.tensor(edge_features, dtype=torch.double)
        edge_attr = self.add_backward_edge_attr(edge_attr)

        if edge_index.size(0) == 0:
            # prevent error in case of empty edge set
            # add empty list of desired shape
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_features), dtype=torch.double)

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
        component_images = []
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
            component_images.append(component)
            components.append({
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
        return components, component_images, components_mask

    def get_line_of_sight_edges(self, components):
        edges = []
        for i, component in enumerate(components):
            # set free view angle of component to 360 degrees
            unblocked_view = [[0.0, 360.0]]
            c_i_center = component['centroid']
            # get other components sorted by ascending distance from component i
            other_components_order = sort_components_by_distance(c_i_center, components, i)
            for j in other_components_order:
                c_j = components[j]
                c_j_center = c_j['centroid']
                # get the view angles range that this component shadows/covers
                blocking_view = get_blocking_view_angles_range(c_i_center, c_j['bbox'])
                # if the components is fully visible from centroid, add visibility edge
                is_visible = is_component_visible(unblocked_view, blocking_view)
                if is_visible:
                    if not edge_in_edges_undirected([edge['components'] for edge in edges], i, j):
                        # append edge if component visible and edge does not exist yet
                        edges.append({
                            'components': [i, j],
                            'start': c_i_center,
                            'end': c_j_center
                        })
                # update free view angle by shadowing the currently processed component angle range
                unblocked_view = block_range_in_view_sections(unblocked_view, blocking_view)

        return edges

    def compute_los_edge_features(self, edges, components, components_mask):
        """
        Calculates geometric features of edges
        :param edges: list of edges
        :param components: list of components
        :param components_mask: image, where each components area has value of its index in components list
        :return: list of lists of edge features per each edge
        """
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
                bbox_diagonal_ratio
            ])
        return edge_features

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

        return torch.cat([edge_attr, bw_edge_attr], dim=0)

    def line_and_rect_intersect(self, line_bounds, rect_corners):
        """
        :param line_bounds: start and endpoint of and edge - list
        :param rect_corners: list of rectangles corner points
        :return: True if intersect, False otherwise
        """
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
        """
            Plots Line of Sight graph
        :param imagepath: source image path
        :param components: components list
        :param edges: edges list
        """
        # draw components bounding boxes
        img = cv.imread(imagepath)
        for i in range(len(components)):
            topleft = components[i]['bbox'][0]
            bottomright = components[i]['bbox'][2]
            (cX, cY) = components[i]['centroid']
            cv.rectangle(img, topleft, bottomright, (0, 255, 0), 1)
            cv.circle(img, (int(cX), int(cY)), 2, (0, 0, 255), -1)
        # draw edges
        for edge in edges:
            (sX, sY) = edge['start']
            (eX, eY) = edge['end']
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            cv.line(img, (int(sX), int(sY)), (int(eX), int(eY)), color, 1)
        plt.imshow(img)
        plt.show()

    def get_latex_from_inkml(self, filepath):
        """
        Returns LaTeX anotation from InkML file.
        :param filepath: InkML filepath
        :return: LaTeX string
        """
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
        """
        Recursive function performing depth first search traversal through MathML notation and parses the defined formula.
        :param xml_ns: used xml namespace
        :param mathml_ns: used matlml namespace
        :param root: current subtree root element
        :return: symbols, relations and serialized sequence of symbols in subtree given by root
        """
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

            if root.tag == mathml_ns + 'msub':
                relation_type = SrtEdgeTypes.SUBSCRIPT
                sequence.extend(basis_sequence)
                sequence.extend(["_", "{"])
                sequence.extend(script_sequence)
                sequence.append("}")
            elif root.tag == mathml_ns + 'msup':
                relation_type = SrtEdgeTypes.SUPERSCRIPT
                sequence.extend(basis_sequence)
                sequence.extend(["^", "{"])
                sequence.extend(script_sequence)
                sequence.append("}")
            elif root.tag == mathml_ns + 'munder':
                relation_type = SrtEdgeTypes.BELOW
                sequence.extend([r"\underset", "{"])
                sequence.extend(script_sequence)
                sequence.extend(['}', '{'])
                sequence.extend(basis_sequence)
                sequence.append("}")
            else:
                relation_type = SrtEdgeTypes.ABOVE
                sequence.extend([r"\overset", "{"])
                sequence.extend(script_sequence)
                sequence.extend(['}', '{'])
                sequence.extend(basis_sequence)
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
                if subscript_id and superscript_id:
                    sequence.extend([r"\overset", "{"])
                    sequence.extend(superscript_sequence)
                    sequence.extend(['}', '{'])
                    sequence.extend([r"\underset", "{"])
                    sequence.extend(subscript_sequence)
                    sequence.extend(['}', '{'])
                    sequence.extend(basis_sequence)
                    sequence.extend(['}', '}'])
                elif subscript_id:
                    sequence.extend([r"\underset", "{"])
                    sequence.extend(subscript_sequence)
                    sequence.extend(['}', '{'])
                    sequence.extend(basis_sequence)
                    sequence.append("}")
                elif superscript_id:
                    sequence.extend([r"\overset", "{"])
                    sequence.extend(superscript_sequence)
                    sequence.extend(['}', '{'])
                    sequence.extend(basis_sequence)
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
                if root.tag == mathml_ns + 'mn':
                    s.append({
                        'id': id,
                        'symbol': '<NUM>'
                    })
                    sequence = ['<NUM>']
                elif root.tag == mathml_ns + 'mi':
                    s.append({
                        'id': id,
                        'symbol': '<ID>'
                    })
                    sequence = ['<ID>']
                elif root.tag == mathml_ns + 'mtext':
                    s.append({
                        'id': id,
                        'symbol': '<TEXT>'
                    })
                    sequence = ['<TEXT>']
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
        """
        Parses traces ground truth from InkML file and returns their list with computed bounding boxes.
        :param ns: elements namespace
        :param root: traces definition root element
        :return: list of traces
        """
        traces = []
        # get traces
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
        # calculate bounding boxes
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
        """
        Parses tracegroups definition from InkML
        :param ns: elements namespace
        :param root: definition root element
        :return: list of tracegroups
        """
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
        """
        Matches traces with their tracegroups and calculates bounding boxes for whole tracegroups.
        :param traces: list of traces
        :param tracegroups: list of tracegroups
        :return: list of tracegroups with bounding boxes
        """
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
        """
        Extracts information from InkML file - symbols and relations from MathML GT, traces, tracegroups
        :param inkml_path: InkML filepath
        :return: SLT symbols, SLT relations, traces, tracegroups
        """
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
        """
        Rescale tracegroup coordinates parsed from InkML file so that they would
        match positions in rendered source image. There is some padding on all sides and the whole image is padded to
        square shape.
        :param image_path: source image filepath
        :param tracegroups: list of tracegroups
        :return: list of tracegroups with rescaled coordinates of bounding boxes
        """
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
        """
        Build Symbol Layout Tree (SLT) from InkML file and returns mapping of target graph nodes and source graph nodes
        :param image_path: source image filepath
        :param inkml_path: InkML GT filepath
        :param los_components: list of source graph components
        :return:
            x, edge_index, edge_type, edge_relation - target graph
            sequence - sequence of LaTeX symbols
            comp_symbols - source components to target symbols mapping
        """
        symbols, relations, sequence, tracegroups = self.parse_inkml(inkml_path)
        for symbol in symbols:
            # convert mathml unicode symbols to latex label for defined set of expressions
            symbol['symbol'] = mathml_unicode_to_latex_label(symbol['symbol'])
        for i, symbol in enumerate(sequence):
            # convert mathml unicode symbols to latex label for defined set of expressions
            sequence[i] = mathml_unicode_to_latex_label(symbol, skip_curly_brackets=True)
        # tokenize symbols
        unk_token_id = self.tokenizer.encode('[UNK]', add_special_tokens=False).ids[0]
        x = [self.tokenizer.encode(s['symbol'], add_special_tokens=False).ids for s in symbols]
        x = [x_i[0] if len(x_i) == 1 else unk_token_id for x_i in x]

        # construct attention ground-truths to LoS input graph nodes
        if los_components is None:
            comp_symbols = None
        else:
            tracegroups = self.rescale_tracegroup_coordinates(image_path, tracegroups)
            # assign symbols to tracegroups
            symbols_bbox_polygons = []
            for symbol in symbols:
                # get tracegroup for each SLT graph node
                tg = next((tg for tg in tracegroups if tg['symbol_id'] == symbol['id']), None)
                if tg is not None and tg['bbox'] is not None:
                    symbols_bbox_polygons.append(Polygon(tg['bbox']))
                else:
                    symbols_bbox_polygons.append(Polygon([(0, 0), (0, 0), (0, 0), (0, 0)]))

            # match most probable symbol to each los component - biggest intersection of bounding boxes
            los_components_symbols = []
            for los_component in los_components:
                los_bbox_polygon = Polygon(los_component['bbox'])
                intersection_areas = []
                for symbol_polygon in symbols_bbox_polygons:
                    intersection = los_bbox_polygon.intersection(symbol_polygon)
                    intersection_areas.append(intersection.area)
                symbol_id = np.argmax(intersection_areas)
                los_components_symbols.append(symbol_id)

            comp_symbols = torch.tensor(los_components_symbols, dtype=torch.long)

        # generate end leaf nodes
        end_nodes, end_edge_index = self.get_end_leaf_nodes(len(x))
        # append end leaf nodes
        x.extend(end_nodes)
        # init edges
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

    def get_end_leaf_nodes(self, nodes_count):
        """
        Creates end leaf - level termination node for each node in graph.
        :param nodes_count: Number of nodes in graph.
        :return: end leaf nodes and edges connecting them to parents
        """
        eos_token_id = self.tokenizer.encode('[EOS]', add_special_tokens=False).ids[0]
        end_nodes = [eos_token_id for _ in range(nodes_count)]
        end_edge_index = [[i, nodes_count + i] for i in range(nodes_count)]
        return end_nodes, end_edge_index

    def get_gp_edges(self, edge_index):
        """
        Creates edges for grandparent-grandchild relation in extSLT.
        :param edge_index: edge index of SLT graph
        :return: list of grandparent edges
        """
        root = self.get_tree_root(edge_index)
        gp_edges = self.dfs_gp_edges_idenitification(root, None, edge_index)
        return gp_edges

    def dfs_gp_edges_idenitification(self, root_idx, root_parent_idx, edge_index):
        """
        Depth first search traversal to indentify grandparent nodes in graph.
        :param root_idx: subtree root
        :param root_parent_idx: subtree root parent
        :param edge_index: graph edge index
        :return: list of grandparent edges in subtree
        """
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
        """
        Creates edges for leftbrother-rightbrother relation in extSLT.
        Performs Breadth first traversal of graph.
        :param edge_index: edge index of SLT graph
        :return: list of leftbrother edges
        """
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
        """
        Creates self loop edges for each node in graph
        :param nodes_count: count of nodes in graph
        :return: list of edges
        """
        self_loop_edges = [[i, i] for i in range(nodes_count)]
        return self_loop_edges

    def draw_slt(self, symbols, x, edge_index, edge_type, edge_relation, include_end_nodes=False):
        """
        Plots SLT graph using networkx library
        :param symbols: list of symbols
        :param x: list of symbol features
        :param edge_index: edge index
        :param edge_type: edge types list
        :param edge_relation: edge relations list
        :param include_end_nodes: whether to include end leaf nodes in plot
        :return:
        """
        x_indices = list(range(len(x)))
        x_indices = torch.tensor(x_indices, dtype=torch.double)

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