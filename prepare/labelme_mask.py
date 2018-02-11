import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree


def get_mask_from_labelme_xml(xml_path):
    e = xml.etree.ElementTree.parse(xml_path)
    imagesize = e.find('imagesize')
    height = int(imagesize.find('nrows').text)
    width = int(imagesize.find('ncols').text)

    img = Image.new('L', (width, height), 0)

    for obj in e.findall('object'):
        polygon = []
        p = obj.find('polygon')
        for pt in p.findall('pt'):
            x = int(pt.find('x').text)
            y = int(pt.find('y').text)
            polygon.append((x, y))
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

    mask = np.array(img)

    return mask
