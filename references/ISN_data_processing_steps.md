# ISN data processing steps

Jonathan Bower
jonathan.bower@unh.edu

---

## Data Generation

1. SOC data extraction 
Functions: /soc_tools/get_data.get_data
- hist data:    `me_show input_file.lohb-N -x header -x mtype=lo_triple_H > output_file.txt`
    - Currently using /id-bas4/ which has correct estep separation
- DE tof data:  `lo_de_tof -h -t tofsH input_file.lotf-N > output_file.txt`
    - id-leops
- tof2_e2:      `me_show input_file.lohb-2 -x header -x mtype=c6 > output_file.txt`
    - id_bas4

## Data processing
- Bulk sequence of processing steps 2-10 in: `pyBEX.lazy_histograms()`
- Total database processing using: `pyBEX.ISN_data.generate_isn_data`

2. Data import to pandas df

Functions:
- good_times:       `pyBEX.tools.import_good_times(file)`
- notspun times:    `pyBEX.tools.import_good_times(file)`
- histograms:       `pyBEX.tools.load_df(data_base_location,estep)`
- DE:               `pyBEX.tools.load_df(data_base_location,estep)`
- tof2_e2_dat:      `pyBEX.tools.load_df(data_base_location,2)`

3. Square the histogram data by spin
Functions: `pyBEX.hist.square_spinterp`

4. Define DE species, apply checksum, select triples

5. Bin the DE to time/phase bins of histograms

6. Generate good_times/notspun_time masks
Functions: `pyBEX.tools.mask_good_times`

7. Generate tof2 rate filter
Functions: `tof_2_filt`

8. compile and apply logical masks
- Good times
- notspun times
- tof2 rate filter

9. Integrate to 512 spins using 8 consecutive 64 spin binns
Functions: `pybex.mat_df.mat_df.accum_bins`

10. Drop masked periods (na values) and outliers

## Data Output
11. Format df to desired output and print to csv
Functions: `pyBEX.ISN_data.hist_to_csv`
- inputs mat_df data struct and outputs csv file in ISN_data format

## Data Testing
12. Test the time bin locations after binning to mat_df
Functions: `pyBEX.ISN_data.mat_df_test`
