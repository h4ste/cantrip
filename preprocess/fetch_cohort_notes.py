import argparse
import importlib
import os
import sys
import nio

try:
    from tqdm import trange, tqdm
except ImportError:
    print('Package \'tqdm\' not installed. Falling back to simple progress display.')
    from mock_tqdm import trange, tqdm

from getpass import getuser, getpass

_cohorts = {
    'pneumonia': """SELECT subject_id, hadm_id
                      FROM DIAGNOSES_ICD
                     WHERE icd9_code LIKE '482%' OR icd9_code = '99731' OR icd9_code = '99732'""",

    'bed-sores': """SELECT subject_id, hadm_id
                      FROM DIAGNOSES_ICD
                     WHERE icd9_code LIKE '707%'""",

    'from-sql': None,

    'from-sql-file': None
}

parser = argparse.ArgumentParser(description='extract notes for the given population')
parser.add_argument('cohort', choices=_cohorts, help='cohort to extract notes from; '
                                                    'if \'from-sql\', the next '
                                                    'positional argument must be a SQL query specifying the '
                                                    'subject_ids and hadm_ids for the cohort; '
                                                    'if \'from-sql-file\', the next positional argument must be the '
                                                    'name of or path to a file containing a SQL query specifying the '
                                                    'subject_ids and hadm_ids for the cohort')
parser.add_argument('server', metavar='mimic-server', help='address of mimic3 server to connect to')
parser.add_argument('output_dir', metavar='output-dir', help='destination folder in which mimic notes will be saved')
parser.add_argument('--database', type=str, default='mimic3', help='name of the mimic3 database')
parser.add_argument('--db-api2-impl', type=str, default='mysql.connector',
                    help='name of package to use for accessing the database: package must implement python\'s DB-API2'
                         'specification (see PEP 249). For example, to use MySQL specify \'mysql.connector\' and to use'
                         'PostgreSQL specify \'psycopg2\'')
parser.add_argument('--user', type=str, help='username to access mimic3; '
                                             'if not provided will check the environment variable MIMIC_USER or '
                                             'use the current user\'s name as given by the OS')
parser.add_argument('--password', type=str, help='password to access mimic3; '
                                                 'if not provided, will check the environment variable MIMIC_PASS or '
                                                 'prompt the user')
parser.add_argument('--no-temporary', dest='use_temporary', default=True, action='store_false',
                    help='this scripts defaults to storing MIMIC visits in a temporary table; '
                         'if --no-temporary is passed, visits a permanent table will created at user.visits')


_xml_format = r'''<?xml version='1.0' encoding='UTF-8'?>
<report>
<category>{category}</category>
<description>{description}</description>
<subject_id>{subject_id}</subject_id>
<visit_id>{visit_id}</visit_id>
<hadm_id>{hadm_id}</hadm_id>
<date>{date}</date>
<text><![CDATA[{text}]]></text>
</report>
'''


def write_note_as_xml(subject_id, visit_id, hadm_id, date, category, description, text, xml_file):
    print(
        _xml_format.format(
            subject_id=subject_id,
            visit_id=visit_id,
            hadm_id=hadm_id,
            date=date,
            category=category,
            description=description,
            text=text
        ),
        file=xml_file)


def _fetch_notes(user, cohort, cursor):
    cursor.execute('SET @last_date    = null;')
    cursor.execute('SET @last_hadm_id = null;')
    cursor.execute('SET @note_number  = 1;')

    notes_sql = """
    SELECT subject_id, visit_id, hadm_id,
           `date`,
           @note_number AS note_number,
           category AS note_category,
           description AS note_description,
           `text` AS note_text
      FROM (SELECT subject_id, visit_id, hadm_id,
                   DATE(chartdate) AS `date`,
                   category,
                   description,
                   `text`
              FROM {user}.visits
                   INNER JOIN ({cohort}) AS c
                   USING (subject_id, hadm_id)
    
                   INNER JOIN NOTEEVENTS AS n
                   USING (subject_id, hadm_id)
             ORDER BY subject_id ASC, visit_id ASC, hadm_id ASC, chartdate ASC, IFNULL(charttime, n.ROW_ID) ASC
           ) AS t
     WHERE (@note_number := IF(hadm_id != @last_hadm_id OR `date` != @last_date, 1, @note_number + 1)) IS NOT NULL
       AND (@last_hadm_id := hadm_id) IS NOT NULL
       AND (@last_date := `date`) IS NOT NULL;
    """.format(user=user, cohort=cohort)

    print('Fetching cohort notes with:', notes_sql)
    cursor.execute(notes_sql)


def _make_visits(user, cursor, use_temporary=True):
    cursor.execute('SET @last_discharge  = null;')
    cursor.execute('SET @last_subject_id = null;')
    cursor.execute('SET @visit_id        = 1;')

    visits_sql = """
    CREATE {table_prefix} TABLE IF NOT EXISTS {user}.visits(
           subject_id INT,
           hadm_id INT,
           visit_id INT,
           PRIMARY KEY (subject_id, hadm_id),
           KEY (visit_id)) AS
    SELECT subject_id,
           hadm_id,
           @visit_id AS visit_id
      FROM (SELECT *
              FROM ADMISSIONS
             ORDER BY subject_id ASC, hadm_id ASC, admittime ASC, dischtime ASC
           ) AS t
     WHERE -- Update visit ID
           (@visit_id := CASE
                         -- Assign a new visit ID for each new subject
                         WHEN subject_id != @last_subject_id THEN @visit_id + 1
                         -- Assign a new visit ID when elapsed days since last chart_date is > 35
                         WHEN DATEDIFF(admittime, @last_discharge) > 35 THEN @visit_id + 1
                         -- Use the same visit ID
                         ELSE @visit_id
                         END) IS NOT NULL
           -- Update last subject ID
       AND (@last_subject_id := subject_id) IS NOT NULL
           -- Update last date
       AND (@last_discharge := dischtime) IS NOT NULL;
    """.format(user=user, table_prefix='TEMPORARY ' if use_temporary else '')

    print('Creating {table_status} visits table at {user}.visits with: {sql}'.format(
        user=user,
        sql=visits_sql,
        table_status='TEMPORARY' if use_temporary else 'PERMANENT'))
    cursor.execute(visits_sql)


def fetch_and_write_mimic_notes(output_dir, cohort, db_api2_impl, server, database, user=None, password=None,
                                use_temporary=True, **kwargs):
    try:
        dbapi2 = importlib.import_module(db_api2_impl)
    except ImportError as e:
        print('Failed to import db-api2 driver: are the right packages installed (e.g., mysql-connector-python or '
              'psycopg2)?')
        raise e

    user = user or os.environ.get('MIMIC_USER') or getuser()

    mimic = dbapi2.connect(host=server,
                           database=database,
                           user=user,
                           password=password or os.environ.get('MIMIC_PASS') or getpass())

    # Create visits table
    print('Creating (or re-using) temporary table to hold hospital visits')
    cursor = mimic.cursor()
    _make_visits(user, cursor, use_temporary)
    # cursor.close()

    _fetch_notes(user, cohort, cursor)

    print('Writing notes as XML')
    for i, (subject_id, visit_id, hadm_id, date, note_idx, category, description, text) in enumerate(
            tqdm(cursor, desc='Writing XML')):

        # Creating parent directories if needed
        visit_dir = os.path.join(output_dir, str(subject_id), str(visit_id), str(hadm_id))
        nio.make_dirs_quiet(visit_dir)

        # Write the XML
        xml_file_name = '{date}.{note_idx}.report.xml'.format(
            date=date,
            note_idx=note_idx,
        )
        with open(os.path.join(visit_dir, xml_file_name), 'w') as xml_file:
            write_note_as_xml(subject_id, visit_id, hadm_id, date, category, description, text, xml_file)

    cursor.close()
    mimic.close()


def main():
    args, argv = parser.parse_known_args()

    if args.cohort == 'from-sql':
        if len(argv) < 1:
            parser.error('must specify SQL query specifying the cohort to fetch')
        else:
            args.cohort = argv[1]
    elif args.cohort == 'from-sql-file':
        if len(argv) < 1:
            parser.error('must specify path to file containing SQL query specifying the cohort to fetch')
        else:
            with open(argv[1], 'r') as sql_file:
                args.cohort = sql_file.read()
    else:
        args.cohort = _cohorts[args.cohort]

    sys.exit(fetch_and_write_mimic_notes(**vars(args)))


if __name__ == '__main__':
    main()
