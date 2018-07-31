import numpy as np


def save_pts(pts_path, pts):
    with open(pts_path, 'w') as pts_file:
        pts_file.write('version: 1\n')
        pts_file.write('n_points: %d\n{\n' % pts.shape[0])
        for idx in range(pts.shape[0]):
            pts_file.write('%.3f %.3f\n' % (pts[idx, 0], pts[idx, 1]))
        pts_file.write('}\n')


def load_pts(pts_path):
    with open(pts_path) as pts_file:
        pts_file_content = pts_file.read().replace('\r', ' ').replace('\n', ' ')
        pts_file_content = pts_file_content[pts_file_content.find('{') + 1:
                                            pts_file_content.find('}')]
        return np.array([float(x) for x in pts_file_content.split(' ') if
                         len(x) > 0]).reshape(-1, 2)
