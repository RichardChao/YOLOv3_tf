import requests
import json
import xml.etree.cElementTree as ET
from requests.adapters import HTTPAdapter
import os

def indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem

def write_to_xml(key, val):
    # ET.SubElement(root, "field1", name="blah").text = "some value1"
    # ET.SubElement(root, "field2", name="asdfasd").text = "some vlaue2"

    root = ET.Element("annotation")

    folder = ET.SubElement(root, "folder")
    folder.text = str(key[0])

    filename = ET.SubElement(root, "filename")
    filename.text = str(key[1])

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = 'ILSVRC_2014'

    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(val['width'])
    height = ET.SubElement(size, "height")
    height.text = str(val['height'])
    depth = ET.SubElement(size, "depth")
    depth.text = '3'

    ET.SubElement(root, "segmented").text = '0'
    for synsets_key, synsets_val in val['synsets'].items():
        for o in synsets_val:
            obj = ET.SubElement(root, "object")
            name = ET.SubElement(obj, "name").text = str(synsets_key)
            pose = ET.SubElement(obj, "pose").text = 'Unspecified'
            truncated = ET.SubElement(obj, "truncated").text = '0'
            difficult = ET.SubElement(obj, "difficult").text = '0'
            bndbox = ET.SubElement(obj, "bndbox")

            ET.SubElement(bndbox, "xmax").text = str(o['right'])
            ET.SubElement(bndbox, "xmin").text = str(o['left'])
            ET.SubElement(bndbox, "ymax").text = str(o['top'])
            ET.SubElement(bndbox, "ymin").text = str(o['bottom'])

    indent(root)
    tree = ET.ElementTree(root)
    filename = "LSVRC2013_annotation/{}/{}.xml".format(key[0], key[1])
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    tree.write(filename)


def get_data(cls):
    print('request:', cls)
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=5))
    r = s.get('http://www.image-net.org/challenges/LSVRC/2014/ui/api/bbox_api.php?type=1&classes%5B%5D={}'.format(cls))
    # r = requests.get(
        # 'http://www.image-net.org/challenges/LSVRC/2015/ui/api/bbox_api.php?type=0&classes%5B%5D={}'.format(cls))

    json_acceptable_string = r.text.replace("'", "\"")
    res = json.loads(json_acceptable_string)
    for res_key, res_val in res.items():
        res_key = res_key.split('/')
        print('write_to_xml, folder:{}, key:{}, val:{}'.format(res_key[0], res_key[1], res_val))
        write_to_xml(res_key, res_val)


for cls in range(1, 201):
    get_data(cls)
