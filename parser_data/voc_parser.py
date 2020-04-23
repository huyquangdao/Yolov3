from base.parser import BaseParser
import numpy as np
import xml.etree.ElementTree as ET
from utils.data_utils import get_voc_names

class VocParser(BaseParser):

    def __init__(self, type_name, file_dir, name_dir):
        super(VocParser,self).__init__(type_name,file_dir,name_dir)
        self.voc_names = get_voc_names(self.name_dir)


    def process_one(self, xml_file):

          tree = ET.parse(xml_file)
          root = tree.getroot()

          list_with_all_boxes = []

          labels = []

          for boxes in root.iter('object'):

              filename = root.find('filename').text

              name = boxes.find('name').text

              label = self.voc_names[name]

              ymin, xmin, ymax, xmax =  None, None, None, None

              for box in boxes.findall("bndbox"):

                  ymin = float(box.find("ymin").text)
                  xmin = float(box.find("xmin").text)
                  ymax = float(box.find("ymax").text)
                  xmax = float(box.find("xmax").text)

              list_with_single_boxes = [xmin, ymin, xmax, ymax]
              list_with_all_boxes.append(list_with_single_boxes)
              labels.append(label)

          return filename, np.array(labels), np.array(list_with_all_boxes)
