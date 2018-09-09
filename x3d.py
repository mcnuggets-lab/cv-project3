from lxml import etree
from lxml.builder import ElementMaker

import numpy as np

E = ElementMaker()

X3D = E.X3D
SCENE = E.Scene
SHAPE = E.Shape
APPEARANCE = E.Appearance
MATERIAL = E.Material
IMAGETEXTURE = E.ImageTexture
TRANSFORM = E.Transform
COORDINATE = E.Coordinate
TEXTURECOORDINATE = E.TextureCoordinate

INDEXEDFACESET = E.IndexedFaceSet
BOX = E.Box
SPHERE = E.Sphere
CONE = E.Cone


def output_x3d(filename, shapes):
    xmlsub = [s.to_xml for s in shapes]
    xml = X3D(
            SCENE(
                *xmlsub
            )
        )
    xml_tree = etree.ElementTree(xml)
    xml_tree.write(filename, encoding='utf-8', pretty_print=True)
    return xml


class X3DShape(object):
    def __init__(self):
        pass

    def to_xml(self):
        raise NotImplementedError("Must implement to_xml in subclasses.")


class X3DCube(X3DShape):
    """
    An X3D cube.
    """
    def __init__(self, texture={}, material={}, center=[0, 0, 0], size=[1, 1, 1]):
        super(X3DCube).__init__()
        self.image_texture = texture
        self.material = material
        self.center = center
        self.size = np.array(size)

    def to_xml(self):
        appearance = []
        if self.image_texture:
            appearance.append(IMAGETEXTURE(**self.image_texture))
        if self.material:
            appearance.append(MATERIAL(**self.material))

        xml = TRANSFORM(
            SHAPE(
                APPEARANCE(
                    *appearance
                ),
                BOX(
                    size=" ".join(str(n) for n in self.size)
                )
            ),
            translation=" ".join(str(n) for n in self.center)
        )
        return xml


class X3DSphere(X3DShape):
    """
    An X3D cube.
    """
    def __init__(self, texture={}, material={}, center=[0, 0, 0], radius=1):
        super(X3DSphere).__init__()
        self.image_texture = texture
        self.material = material
        self.center = center
        self.radius = radius

    def to_xml(self):
        appearance = []
        if self.image_texture:
            appearance.append(IMAGETEXTURE(**self.image_texture))
        if self.material:
            appearance.append(MATERIAL(**self.material))

        xml = TRANSFORM(
            SHAPE(
                APPEARANCE(
                    *appearance
                ),
                SPHERE(
                    radius=str(self.radius)
                )
            ),
            translation=" ".join(str(n) for n in self.center)
        )
        return xml


class X3DIndexedFaceSet(X3DShape):
    """
    An X3D IndexedFaceSet. Currently only very primitive.
    """
    def __init__(self, texture={}, material={}, coordinate=None, texture_coordinate=None):
        super(X3DIndexedFaceSet).__init__()
        self.image_texture = texture
        self.material = material
        self.coordinate = coordinate
        self.texture_coordinate = texture_coordinate
        self.coordinate_index = np.array([0, 1, 2, 3, -1])
        self.texture_coordinate_index = np.array([0, 1, 2, 3, -1])

    def to_xml(self):
        appearance = []
        if self.image_texture:
            appearance.append(IMAGETEXTURE(**self.image_texture))
        if self.material:
            appearance.append(MATERIAL(**self.material))

        xml = SHAPE(
            APPEARANCE(
                *appearance
            ),
            INDEXEDFACESET(
                COORDINATE(point=" ".join(str(n) for n in self.coordinate.ravel())),
                TEXTURECOORDINATE(point=" ".join(str(n) for n in self.texture_coordinate.ravel())),
                solid='false',
                coordIndex=" ".join(str(n) for n in self.coordinate_index.ravel()),
                texCoordIndex=" ".join(str(n) for n in self.texture_coordinate_index.ravel()),
            )
        )
        return xml


if __name__ == "__main__":
    shapes = []

    shape1 = X3DIndexedFaceSet(texture={"url": "floor.gif"})
    coord1 = np.array([[0, 0, 0], [100, 0, 0], [100, 300, 0], [0, 300, 0]])
    texcoord1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    shape1.coordinate = coord1
    shape1.texture_coordinate = texcoord1
    shapes.append(shape1)

    shape2 = X3DIndexedFaceSet(texture={"url": "io.gif"})  #0 200 0 0 200 180 50 158 180 50 158 0
    coord2 = np.array([[0, 200, 0], [0, 200, 180], [50, 158, 180], [50, 158, 0]])
    texcoord2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    shape2.coordinate = coord2
    shape2.texture_coordinate = texcoord2
    shapes.append(shape2)

    # shape3 = X3DCube(material={"diffuseColor": "1 0 0"})
    # shapes.append(shape3)

    output_x3d("./x3d/sample.x3d", shapes)


