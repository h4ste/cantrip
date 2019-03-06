import argparse
import csv
import os

parser = argparse.ArgumentParser(description='convert TSV to xml output')
parser.add_argument('tsv_path', metavar='tsv-path', help='path to cohort chronologies')
parser.add_argument('xml_output_path', metavar='xml_output_path', help='path to cohort vocabulary')




def tsv_to_xml(tsv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(tsv_path, 'rt') as tsv_file:
        tsv = csv.reader(tsv_file, delimiter='\t')

        # Skip header row
        next(tsv)

        for i, (subject_id, visit_id, hadm_id, date, category, description, note_text) in enumerate(tsv):
            with open(os.path.join(output_dir, 'report-%d.xml' % (i + 1)), 'wb') as xml_file:
                write_record_to_xml(subject_id, visit_id, hadm_id, date, category, description, note_text, xml_file)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    tsv_to_xml(args.tsv_path, args.xml_output_path)
