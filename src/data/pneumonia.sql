SELECT subject_id, visit_id, hadm_id, DATE(chartdate) AS `date`,
       `category`,
       description,
       note
  FROM goodwintrr.visits
       INNER JOIN (SELECT subject_id, hadm_id
                     FROM DIAGNOSES_ICD
                    WHERE icd9_code LIKE '482%' OR icd9_code = '99731' OR icd9_code = '99732') AS d
       USING (subject_id, hadm_id)

       INNER JOIN NOTEEVENTS
       USING (subject_id, hadm_id)
 WHERE category IN ('Discharge summary', 'Radiology', 'Nursing/other', 'Physician', 'Nursing', 'General')
 GROUP BY subject_id, visit_id, hadm_id, `date`;
