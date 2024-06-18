## Provided
- images in the DICOM format
- a spreadsheet indicating an assignment of each case to one of four groups
- annotation boxes, and
- a spreadsheet that provides an additional organization of patients/studies/views.

## Overview
First Cancer Patients
DBT-P00107
DBT-P00538

## Challenges
- [ ] Requirement for high resolution.
- [ ] Large dataset(1.5 tb train).
- [x] Paths incorrect
- [x] Inconsistent # image per patient
  - [ ] Different patients have different numbers of images. 1, 2, 3 , 4, 5, 6, 7, 8, 9, 10, 11, 12


## TODO
- [ ] Build model
- [ ] Deploy server
- [ ] Build web Dicom viewer
- [ ] Send Dicom from client to server
- [ ] Get prognosis from server using model


## Web Server
  - Rocket
  - https://actix.rs/

## Dicom Viewer
- https://github.com/OHIF/Viewers
3k
- https://github.com/ivmartel/dwv
1.6k
- https://github.com/nroduit/Weasis?tab=readme-ov-file
771
- https://github.com/FNNDSC/ami
706

## Resources/References

- [Standard Views](https://radiopaedia.org/articles/mammography-views?lang=us)
- [Deep Learning Based Methods for Breast Cancer Diagnosis: A Systematic Review and Future Direction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9818155/)
- [Dataset](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/)
- [Duke Writeup](https://sites.duke.edu/mazurowski/resources/digital-breast-tomosynthesis-database/)
