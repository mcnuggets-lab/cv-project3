import numpy as np
import numpy.linalg as la
import cv2  # pip install opencv-python
import imutils  # pip install imutils
from matplotlib import pyplot as plt
from collections import namedtuple


class SvmInterface(object):
    def __init__(self, img=None):
        self.img = img

        # projection matrix
        self.P = np.zeros((3, 4))
        self.alphax = None
        self.alphay = None
        self.alphaz = None

        # homography matrices for the coordinate planes
        self.Hxy = np.zeros((3, 3))
        self.Hxz = np.zeros((3, 3))
        self.Hyz = np.zeros((3, 3))

        # vanishing points for the 3 coordinate axes
        self.vx = None
        self.vy = None
        self.vz = None

        # vanishing line
        self.lxy = None
        self.lxz = None
        self.lyz = None

        # 3D scene origin, reference plane and z-point for calculating homography and scales
        self.origin = None
        self.x_correspondence = None
        self.y_correspondence = None
        self.z_correspondence = None

        # camera center
        self.camera_center = None

    @staticmethod
    def normalize2D(pt):
        """
        Helper function to normalize a 2D point
        :param pt: A 2D-point, presumably lie on the image
        :return: A normalized 2D point, in projective coordinate (x, y, 1) (or point at infinity)
        """
        if len(pt) == 2:
            return np.array([*pt, 1])
        elif len(pt) == 3:
            return np.array(pt) / pt[2] if pt[2] != 0 else np.array(pt)
        else:
            raise ValueError("Input must be a 2D vector or a 2D projective vector.")

    @staticmethod
    def normalize3D(pt):
        """
        Helper function to normalize a 3D point
        :param pt: A 3D-point, presumably space coordinate
        :return: A normalized 3D point, in projective coordinate (x, y, z, 1) (or point at infinity)
        """
        if len(pt) == 3:
            return np.array([*pt, 1])
        elif len(pt) == 4:
            return np.array(pt) / pt[3] if pt[3] != 0 else np.array(pt)
        else:
            raise ValueError("Input must be a 3D vector or a 3D projective vector.")

    @staticmethod
    def denormalize2D(pt):
        """
        Helper function to denormalize a 2D projective (finite) point
        :param pt: A projective 2D-point, presumably lie on the image
        :return: A denormalized 2D point (x, y)
        """
        if len(pt) == 3 and pt[2] != 0:
            return (np.array(pt) / pt[2])[:-1]
        else:
            raise ValueError("Input must be a 2D projective vector that is not a point at infinity.")

    @staticmethod
    def denormalize3D(pt):
        """
        Helper function to denormalize a 3D projective (finite) point
        :param pt: A projective 3D-point, presumably space coordinates
        :return: A denormalized 3D point (x, y, z)
        """
        if len(pt) == 4 and pt[3] != 0:
            return (np.array(pt) / pt[3])[:-1]
        else:
            raise ValueError("Input must be a 3D vector or a 3D projective vector.")

    def _bob_collins_naive(self, lines):
        """
        Compute the vanishing points given a set of parallel lines (at least 2).
        FOR DEBUG PURPOSE. NO NORMALIZATION.
        :param lines: the set of parallel lines
        :return: The vanishing point corresponding to the lines.
        """
        if len(lines) < 2:
            raise ValueError("Please specify at least 2 parallel lines for computing vanishing points.")

        lls = np.zeros((lines.shape[0], 2, 3))
        lls[:, :, :2] = lines
        lls[:, :, 2] = 1

        M = np.zeros((3, 3))
        for p1, p2 in lls:
            e = np.cross(p1, p2)
            M += np.array([[e[0] * e[0], e[1] * e[0], e[2] * e[0]],
                           [e[0] * e[1], e[1] * e[1], e[2] * e[1]],
                           [e[0] * e[2], e[1] * e[2], e[2] * e[2]]])
        evals, evecs = la.eig(M)
        vanishing_point = evecs[:, np.argmin(evals)]
        return vanishing_point

    def bob_collins(self, lines):
        """
        Compute the vanishing points given a set of parallel lines (at least 2) in 2D images.
        :param lines: the set of parallel lines
        :return: The vanishing point corresponding to the lines.
        """
        if len(lines) < 2:
            raise ValueError("Please specify at least 2 parallel lines for computing vanishing points.")

        # perform Hartley normalization
        mean_points = lines.mean(axis=(0, 1))
        lls = np.zeros((lines.shape[0], 2, 3))
        lls[:, :, :2] = lines - mean_points
        lls_max = lls[:, :, :2].max()
        lls[:, :, 2] = lls_max

        M = np.zeros((3, 3))
        for p1, p2 in lls:
            e = np.cross(p1, p2)
            M += np.array([[e[0] * e[0], e[1] * e[0], e[2] * e[0]],
                           [e[0] * e[1], e[1] * e[1], e[2] * e[1]],
                           [e[0] * e[2], e[1] * e[2], e[2] * e[2]]])
        evals, evecs = la.eig(M)
        vanishing_point = evecs[:, np.argmin(evals)]
        if vanishing_point[2] != 0:
            vanishing_point[2] = vanishing_point[2] / lls_max
            vanishing_point = vanishing_point / vanishing_point[2]
            vanishing_point[:2] = vanishing_point[:2] + mean_points
        return vanishing_point

    def calculate_vanishing_line(self):
        if self.vx is None or self.vy is None or self.vz is None:
            raise Exception("Please compute vanishing points before computing vanishing line.")
        self.lxy = np.cross(self.vx, self.vy)
        self.lxy = self.lxy / la.norm(self.lxy)
        self.lxz = np.cross(self.vx, self.vz)
        self.lxz = self.lxz / la.norm(self.lxz)
        self.lyz = np.cross(self.vy, self.vz)
        self.lyz = self.lyz / la.norm(self.lyz)

        # also update the projection matrix
        self.P[:, 3] = self.lxy

        return np.copy(self.lxy)

    def _compute_homography_naive(self, point_pairs):
        """
        Given at least 4 pairs of point correspondence (on a reference plane), compute the homography.
        FOR DEBUG PURPOSE. NO NORMALIZATION.
        :param point_pairs: pairs of point correspondence, of the form (2D image point, 3D planar point).
                            The planar points are assumed to lie on the reference plane (z=0).
        :return: The homography matrix
        """
        if len(point_pairs) < 4:
            raise ValueError("Please specify at least 4 point correspondences for computing homography.")
        A = np.zeros((2 * len(point_pairs), 9))
        A[::2, 0:2] = point_pairs[:, 0, :]
        A[::2, 2] = 1
        A[1::2, 3:5] = point_pairs[:, 0, :]
        A[1::2, 5] = 1
        A[:, 8] = -point_pairs[:, 1, :].ravel()
        A[::2, 6:8] = point_pairs[:, 0, :]
        A[1::2, 6:8] = point_pairs[:, 0, :]
        A[:, 6:8] = A[:, 6:8] * A[:, 8].reshape(-1, 1)

        evals, evecs = la.eig(A.T @ A)
        hom_matrix = evecs[:, np.argmin(evals)].reshape(3, 3)

        # update info for further use
        # self.ground_pts = point_pairs[:, 0, :]
        # self.H = np.copy(hom_matrix)

        return hom_matrix

    def compute_homography(self, point_pairs):
        """
        Given at least 4 pairs of point correspondence (on a reference plane), compute the homography.
        :param point_pairs: pairs of point correspondence, of the form (3D planar point, 2D image point).
                            The planar points are assumed to lie on the reference plane (z=0).
        :return: The homography matrix
        """
        if len(point_pairs) < 4:
            raise ValueError("Please specify at least 4 point correspondences for computing homography.")

        # perform Hartley normalization
        source = point_pairs[:, 0, :]
        target = point_pairs[:, 1, :]
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        source_T = np.eye(3)
        source_T[0:2, 2] = -source_mean
        target_T = np.eye(3)
        target_T[0:2, 2] = -target_mean
        source = source - source_mean
        target = target - target_mean
        source_max = source.max()
        target_max = target.max()
        source_S = np.diag([1 / source_max, 1 / source_max, 1])
        target_S = np.diag([1 / target_max, 1 / target_max, 1])
        source = source / source_max
        target = target / target_max

        A = np.zeros((2 * len(point_pairs), 9))
        A[::2, 0:2] = source
        A[::2, 2] = 1
        A[1::2, 3:5] = source
        A[1::2, 5] = 1
        A[:, 8] = -target.ravel()
        A[::2, 6:8] = source
        A[1::2, 6:8] = source
        A[:, 6:8] = A[:, 6:8] * A[:, 8].reshape(-1, 1)

        evals, evecs = la.eig(A.T @ A)
        hom_matrix = evecs[:, np.argmin(evals)].reshape(3, 3)
        hom_matrix = la.inv(target_T) @ la.inv(target_S) @ hom_matrix @ source_S @ source_T

        return hom_matrix

    def set_ground_homography(self, corrs, use_vanishing_pt=False):
        """
        Set 4 ground points for calculating homography and (3 columns of) projection matrix
        :param corrs: pairs of point correspondence, of the form (3D ground point, 2D img point), i.e. (X, Y, 0, 1) -> (x, y, 1)
                      The first one is assumed to be the origin.
        :return: The ground homography
        """
        self.origin = self.normalize2D(corrs[0][1])
        H = self.compute_homography(corrs)
        H = H / H[2, 2]
        if use_vanishing_pt and (self.vx is not None) and (self.vy is not None):
            self.alphax = la.norm(H[:, 0]) / la.norm(self.vx)
            self.alphay = la.norm(H[:, 1]) / la.norm(self.vy)
            H[:, 0] = self.alphax * self.vx
            H[:, 1] = self.alphay * self.vy
            self.Hxy = H
        else:
            self.Hxy = H
            self.vx = H[:, 0] / H[2, 0]
            self.vy = H[:, 1] / H[2, 1]
            self.lxy = np.cross(self.vx, self.vy)
            self.P[:, [0, 1, 3]] = H.copy()
            # calculate scales as ratios of columns of projection matrix and the vanishing point
            self.alphax = (H[0, 0] / self.vx[0] + H[1, 0] / self.vx[1] + H[2, 0] / self.vx[2]) / 3
            self.alphay = (H[0, 1] / self.vy[0] + H[1, 1] / self.vy[1] + H[2, 1] / self.vy[2]) / 3
        return H.copy()

    def set_alpha_z(self, corrs):
        """
        Set 2 points that have the same xy-coordinates in space to calculate alpha_z.
        :param corrs: pairs of point correspondence, of the form (3D space point, 2D img point), i.e. (X, Y, Z, 1) -> (x, y, 1)
                      The 2 points should only differ in the Z-coordinates.
        :return: alpha_z
        """
        if np.sum(self.Hxy) == 0 or self.vz is None:
            raise Exception("Please compute ground homography and z-vanishing point before setting alpha_z.")
        b = self.normalize2D(corrs[0][1])
        t = self.normalize2D(corrs[1][1])
        z = corrs[0][0][2]
        delta_z = corrs[1][0][2] - corrs[0][0][2]
        norm_bt = la.norm(np.cross(b, t))
        numerator = -la.det(self.Hxy) * norm_bt
        denominator = delta_z * np.dot(np.cross(self.Hxy[:, 0], self.Hxy[:, 1]), b) * la.norm(np.cross(self.vz, t)) \
                        + z * np.dot(np.cross(self.Hxy[:, 0], self.Hxy[:, 1]), self.vz) * norm_bt
        alpha_z = numerator / denominator
        self.alphaz = alpha_z
        self.P[:, 2] = alpha_z * self.vz

        # update the rest of internal data
        self.Hyz = self.P[:, [1, 2, 3]]
        self.Hxz = self.P[:, [0, 2, 3]]
        self.lxz = np.cross(self.vx, self.vz)
        self.lyz = np.cross(self.vy, self.vz)

        return alpha_z

    def compute_camera_center(self):
        """
        Compute the camera center using the projection matrix, using the decomposition given in Hartley-Zisserman.
        Ref: http://ksimek.github.io/2012/08/14/decompose/
        :return: the camera center, in 3D (non-homogeneous) world coordinates
        """
        if np.linalg.matrix_rank(self.P) != 3:
            raise ValueError("Please compute the projection matrix and all necessary parameters before computing the camera center.")
        M = self.P[:, [0, 1, 2]]
        c = la.lstsq(M, -self.P[:, 3], rcond=None)
        return c[0]

    def texture_map(self, pts, width=None, height=None):
        """
        Transform a selected (tilted) rectangle in the image, and transform it to a rectangle by some homography.
        This is used for saving the texture, not to calculate the 3D coordinates.
        :param pts: The 4 points of a rectangle in the image, in the order top_left, top_right, bottom_right, bottom_left.
        :return: The warped image
        """
        pts = pts.astype("float32")
        (tl, tr, br, bl) = pts

        # if width or height is not provided, try to calculate it from the source image, usually not correct though...
        if width is None:
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width = max(int(widthA), int(widthB))

        if height is None:
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            height = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        # get the perspective transform matrix for warping
        corrs = np.zeros((4, 2, 2))
        corrs[:, 0, :] = pts
        corrs[:, 1, :] = dst
        M = self.compute_homography(corrs)
        warped_img = cv2.warpPerspective(self.img, M, (width, height))

        return warped_img

    def same_z_plane(self, ref_pt, pt):
        """
        Given a point with known 3D-coordinates as reference point, calculate the 3D-coordinates of another point
        that has the same z-coordinates as the reference point.
        :param ref_pt: A point whose 3D-coordinates are known, of the form (3D-coordaintes, 2D image point)
        :param pt: A point that has the same xy-coordinates as the reference point, calculate its 3D-coordinates.
                   Input of the form (2D-coordinates)
        :return: The 3D-coordinates of the point to calculate
        """
        ref_pt_3d = self.normalize3D(ref_pt[0])
        ref_pt_img = self.normalize2D(ref_pt[1])
        ref_height = ref_pt_3d[2]
        pt_img = self.normalize2D(pt)
        H = self.P[:, [0, 1, 3]]
        H[:, 2] = H[:, 2] + self.P[:, 2] * ref_height
        pt_3d = np.dot(la.inv(H), pt_img)
        pt_3d = pt_3d / pt_3d[2]
        return np.array([*pt_3d[:2], ref_height, 1])

    def same_xy(self, ref_pt, pt):
        """
        Given a point with known 3D-coordinates as reference point, calculate the 3D-coordinates of another point.
        See the slides for measuring height for the meaning of symbols.
        that has the same xy-coordinates as the reference point.
        :param ref_pt: A point whose 3D-coordinates are known, of the form (3D-coordaintes, 2D image point)
        :param pt: A point that has the same xy-coordinates as the reference point, calculate its 3D-coordinates.
                   Input of the form (2D-coordinates)
        :return: The 3D-coordinates of the point to calculate
        """
        ref_pt_3d = self.normalize3D(ref_pt[0])
        b = self.normalize2D(ref_pt[1])
        t = self.normalize2D(pt)
        z = ref_pt[0][2]
        norm_bt = la.norm(np.cross(b, t))
        numerator = -la.det(self.P[:, [0, 1, 3]]) * norm_bt \
                        - self.alphaz * z * np.dot(np.cross(self.P[:, 0], self.P[:, 1]), self.vz) * norm_bt
        denominator = self.alphaz * np.dot(np.cross(self.P[:, 0], self.P[:, 1]), b) * la.norm(np.cross(self.vz, t))
        delta_z = numerator / denominator

        return np.array([*ref_pt_3d[:2], z + delta_z, 1])


if __name__ == "__main__":
    img = plt.imread("images/sony_clie.800x600.bmp")
    # plt.imshow(img)
    # plt.show()
    svm = SvmInterface(img)

    ref_pts = np.array([
        [391, 542],
        [606, 431],
        [174, 366],
        [382, 277],  # should be hidden
        [393, 399],
        [618, 290],
        [162, 227],
        [379, 139]
    ])

    print("Vanishing points:")
    xlines = []
    xline1 = (ref_pts[0], ref_pts[1])
    xlines.append(xline1)
    xline2 = (ref_pts[2], ref_pts[3])
    xlines.append(xline2)
    xline3 = (ref_pts[4], ref_pts[5])
    xlines.append(xline3)
    xline4 = (ref_pts[6], ref_pts[7])
    xlines.append(xline4)
    xlines = np.array(xlines)
    vx = svm.bob_collins(xlines)
    svm.vx = vx
    print(vx)

    ylines = []
    yline1 = (ref_pts[0], ref_pts[2])
    ylines.append(yline1)
    yline2 = (ref_pts[1], ref_pts[3])
    ylines.append(yline2)
    yline3 = (ref_pts[4], ref_pts[6])
    ylines.append(yline3)
    yline4 = (ref_pts[5], ref_pts[7])
    ylines.append(yline4)
    ylines = np.array(ylines)
    vy = svm.bob_collins(ylines)
    svm.vy = vy
    print(vy)

    zlines = []
    zline1 = (ref_pts[0], ref_pts[4])
    zlines.append(zline1)
    zline2 = (ref_pts[1], ref_pts[5])
    zlines.append(zline2)
    zline3 = (ref_pts[2], ref_pts[6])
    zlines.append(zline3)
    zline4 = (ref_pts[3], ref_pts[7])
    zlines.append(zline4)
    zlines = np.array(zlines)
    vz = svm.bob_collins(zlines)
    svm.vz = vz
    print(vz)
    print()

    svm.calculate_vanishing_line()

    # test for correspondence
    correspondence = np.array([
        ([0, 0], ref_pts[0].copy()),
        ([300, 0], ref_pts[1].copy()),
        ([0, 400], ref_pts[2].copy()),
        ([300, 400], ref_pts[3].copy())
    ], dtype="float32")
    H1 = svm.set_ground_homography(correspondence, use_vanishing_pt=True)
    H2 = svm.set_ground_homography(correspondence, use_vanishing_pt=False)

    z_corrs = [([0, 0, 0], ref_pts[0].copy()),
               ((0, 0, 183), ref_pts[4].copy())]
    alpha_z = svm.set_alpha_z(z_corrs)

    # texture testing
    pts = []
    # pts.append([387, 143])
    # pts.append([609, 281])
    # pts.append([385, 391])
    # pts.append([169, 231])
    pts.append(ref_pts[7])
    pts.append(ref_pts[5])
    pts.append(ref_pts[4])
    pts.append(ref_pts[6])
    pts = np.array(pts).astype("float32")
    texture = svm.texture_map(pts, width=400, height=300)
    # plt.imshow(texture)
    # plt.show()

    print("same_z_plane:")
    z_ref_pt = ([0, 0, 183], ref_pts[4])
    z_pt = ref_pts[7]
    z_pt_3d = svm.same_z_plane(z_ref_pt, z_pt)
    print(z_pt_3d)

    print("same_xy:")
    xy_ref_pt = ([0, 0, 0], ref_pts[0])
    xy_pt = ref_pts[4]
    xy_pt_3d = svm.same_xy(xy_ref_pt, xy_pt)
    print(xy_pt_3d)



