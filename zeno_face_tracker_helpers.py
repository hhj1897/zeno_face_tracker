import os
import cv2
import dlib
import numpy as np
import xml.etree.cElementTree as ET


def save_pts(pts_path, pts, number_of_decimals=3):
    with open(pts_path, 'w') as pts_file:
        pts_file.write('version: 1\n')
        pts_file.write('n_points: %d\n{\n' % pts.shape[0])
        line_template = '%%.%df %%.%df\n' % (number_of_decimals, number_of_decimals)
        for idx in range(pts.shape[0]):
            pts_file.write(line_template % (pts[idx, 0], pts[idx, 1]))
        pts_file.write('}\n')


def load_pts(pts_path):
    with open(pts_path) as pts_file:
        pts_file_content = pts_file.read().replace('\r', ' ').replace('\n', ' ')
        pts_file_content = pts_file_content[pts_file_content.find('{') + 1:
                                            pts_file_content.find('}')]
        return np.array([float(x) for x in pts_file_content.split(' ') if
                         len(x) > 0]).reshape(-1, 2)


def save_annotation_job(job_path, items, number_of_landmarks):
    root = ET.Element('FaceDataset')
    root.set('xmlns', 'https://github.com/luigivieira/Facial-Landmarks-Annotation-Tool')
    root.set('numberOfFeatures', '%d' % number_of_landmarks)
    samples = ET.SubElement(root, 'Samples')
    job_parent_folder = os.path.dirname(os.path.realpath(job_path))
    for item in items:
        sample = ET.SubElement(samples, 'Sample')
        image_path = os.path.relpath(item[0], job_parent_folder)
        sample.set('fileName', image_path)
        features = ET.SubElement(sample, 'Features')
        for pt_idx, pt in enumerate(item[1]):
            feature = ET.SubElement(features, 'Feature')
            feature.set('x', '%.3f' % pt[0])
            feature.set('y', '%.3f' % pt[1])
            feature.set('id', '%d' % pt_idx)
            connections = ET.SubElement(feature, 'Connections')
            if item[1].shape[0] == 68:
                # The 68 facial landmark markup
                if pt_idx == 30:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '33')
                elif pt_idx == 41:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '36')
                elif pt_idx == 47:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '42')
                elif pt_idx == 59:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '48')
                elif pt_idx == 67:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '60')
                elif pt_idx != 16 and pt_idx != 21 and pt_idx != 26 and pt_idx != 35:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '%d' % (pt_idx + 1))
            elif item[1].shape[0] == 4:
                # Assuming this is a bounding box
                if pt_idx == 3:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '0')
                else:
                    target = ET.SubElement(connections, 'Target')
                    target.set('id', '%d' % (pt_idx + 1))
    ET.ElementTree(root).write(job_path, encoding='UTF-8', xml_declaration=True)


def load_annotation_job(job_path, number_of_landmarks=None):
    job_folder = os.path.realpath(os.path.dirname(job_path))
    root = ET.parse(job_path).getroot()
    if number_of_landmarks is None:
        number_of_landmarks = int(root.attrib['numberOfFeatures'])
    if root.tag[0] == '{':
        namespace = root.tag[1:].split('}')[0]
    else:
        namespace = None

    def form_xml_tag(tag):
        if namespace is None or namespace == '':
            return tag
        else:
            return '{%s}%s' % (namespace, tag)

    items = []
    for sample in root.findall('%s/%s' % (form_xml_tag('Samples'), form_xml_tag('Sample'))):
        item = {}
        if os.name == 'nt':
            image_file_path = sample.attrib['fileName'].replace('/', '\\')
        else:
            image_file_path = sample.attrib['fileName'].replace('\\', '/')
        if os.path.isabs(image_file_path):
            item['image_path'] = image_file_path
        else:
            item['image_path'] = os.path.realpath(os.path.join(job_folder, image_file_path))
        item['facial_landmarks'] = -np.ones((number_of_landmarks, 2), dtype=float)
        for feature in sample.findall('%s/%s' % (form_xml_tag('Features'), form_xml_tag('Feature'))):
            index = int(feature.attrib['id'])
            if 0 <= index < number_of_landmarks:
                item['facial_landmarks'][index, 0] = float(feature.attrib['x'])
                item['facial_landmarks'][index, 1] = float(feature.attrib['y'])
        items.append(item)

    return items


def align_landmarks(landmarks, mean_shape, anchors):
    anchors = list(set([x for x in anchors if 0 <= x < min(landmarks.shape[0], mean_shape.shape[0])]))
    transform = cv2.estimateRigidTransform(landmarks[anchors], mean_shape[anchors], False)
    return (np.matmul(landmarks, transform[:, 0:2].T) +
            np.tile(transform[:, 2].T, (landmarks.shape[0], 1))), transform


def run_multiple_detectors(detectors, image, overlap_threshold, upsample_num_times=0, adjust_threshold=0.0):
    face_boxes = dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times, adjust_threshold)
    if len(face_boxes[1]) > 0:
        sorted_indices = sorted(range(len(face_boxes[1])), key=lambda k: face_boxes[1][k], reverse=True)
        filtered_face_boxes = [face_boxes[0][sorted_indices[0]]]
        filtered_confidences = [face_boxes[1][sorted_indices[0]]]
        filtered_detector_indices = [face_boxes[2][sorted_indices[0]]]
        filtered_face_box_sizes = [max(1e-6, float(filtered_face_boxes[0].area()))]
        for idx in sorted_indices[1:]:
            not_overlapping = True
            current_face_box_size = max(1e-6, float(face_boxes[0][idx].area()))
            for idx2, consolidated_face_box in enumerate(filtered_face_boxes):
                intersection_area = float(consolidated_face_box.intersect(face_boxes[0][idx]).area())
                overlap = intersection_area / min(filtered_face_box_sizes[idx2], current_face_box_size)
                if overlap > overlap_threshold:
                    not_overlapping = False
                    break
            if not_overlapping:
                filtered_face_boxes.append(face_boxes[0][idx])
                filtered_confidences.append(face_boxes[1][idx])
                filtered_detector_indices.append(face_boxes[2][idx])
                filtered_face_box_sizes.append(current_face_box_size)
        return filtered_face_boxes, filtered_confidences, filtered_detector_indices
    else:
        return [], [], []


def compute_rigid_alignment_parameters(source, destination):
    assert source.shape == destination.shape and source.shape[1] == 2
    a = np.zeros((4, 4), np.float)
    b = np.zeros((4, 1), np.float)
    for idx in range(source.shape[0]):
        a[0, 0] += np.dot(source[idx], source[idx])
        a[0, 2] += source[idx, 0]
        a[0, 3] += source[idx, 1]
        b[0] += np.dot(source[idx], destination[idx])
        b[1] += np.dot(source[idx], [destination[idx, 1], -destination[idx, 0]])
        b[2] += destination[idx, 0]
        b[3] += destination[idx, 1]
    a[1, 1] = a[0, 0]
    a[3, 0] = a[0, 3]
    a[1, 2] = a[2, 1] = -a[0, 3]
    a[1, 3] = a[3, 1] = a[2, 0] = a[0, 2]
    a[2, 2] = a[3, 3] = source.shape[0]
    return tuple(np.dot(np.linalg.pinv(a), b).T[0].tolist())


def apply_rigid_alignment_parameters(source, scos, ssin, transx, transy):
    return np.dot(source, [[scos, ssin], [-ssin, scos]]) + [transx, transy]
