import csv
import xml.etree.ElementTree as ET
from xml import etree
import argparse
import os


parser = argparse.ArgumentParser(description='convert TSV to xml output')
parser.add_argument('tsv_path', metavar='tsv-path', help='path to cohort chronologies')
parser.add_argument('xml_output_path', metavar='xml_output_path', help='path to cohort vocabulary')


def tsv_to_xml(tsv_path, output_dir):
    def add_text_child(parent, name, text):
        child = ET.SubElement(parent, name)
        child.text = text
        return child

    os.makedirs(output_dir, exist_ok=True)

    with open(tsv_path, 'rt') as tsv_file:
        tsv = csv.reader(tsv_file, delimiter='\t')

        # Skip header row
        next(tsv)

        for i, (subject_id, visit_id, hadm_id, date, category, description, note_text) in enumerate(tsv):
            record = ET.Element('record', category=category, description=description)
            add_text_child(record, 'subject_id', subject_id)
            add_text_child(record, 'visit_id', visit_id)
            add_text_child(record, 'hadm_id', hadm_id)
            add_text_child(record, 'date', date)
            add_text_child(record, 'text', note_text)

            with open(os.path.join(output_dir, 'report-%d.xml' % (i + 1)), 'wb') as xml_file:
                tree = ET.ElementTree(record)
                tree.write(xml_file, encoding='UTF-8', xml_declaration=True)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    tsv_to_xml(args.tsv_path, args.xml_output_path)







