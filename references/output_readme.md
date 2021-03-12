# IBEX-Lo Hydrogen Trial CSV Data

--

Jonathan Bower, 
EoS Space Science Center, UNH
jonathan.bower@unh.edu

--

This is a first attempt at a reprocessing of the ISN Hydrogen data 

Problems to be addressed:
**
- need verification on which file to use when there are redundant orbit arc files [named new,old,base]
- There are occational times where DE counts are larger than hist counts
    - could be caused by dividing a spin between two time bins, or by using the incorrect redundant file
**

## Data Set Itterations:
IBEX_ISN_H_trial_1:
- Initial generated data set

IBEX_ISN_H_trial_2:

[11/30/2020]


- [Solved:] still need to implement the despun times mask
    - Despun times combined with goodtimes filter applied 

IBEX_ISN_H_trial_3:

[12/4/2020]
- Fixed NEP rounding error, causing some angle bins to be 5 deg instead of 6

IBEX_ISN_H_trial_4:

[1/15/2021]
- Exposure time updated to account for true 64 spin bins and stepping. 
    - Previosly, exposure times were relative to single spin total integration time. 
- Time bin start/stop edges also updated to account for 64 spins. 
    - Start/Stop times defined as spin time center +- the uniform dt used in rebinning. 
- Fixed bug incorrectly applying not-spun filter
- Includes Orbits where no specific goodtimes are declared
    - In both the normal goodtime list and not-spun list

IBEX_ISN_H_trial_5:

[1/20/2021]
- Time bin edges redefined as center point of each consecutive spin time center.
    - Updated to prevent time bin edges from overlapping when using assuming constant dt
- Orbits where no not-spun times are declared are removed. 
- Good_time ranges excluding any NEP bins between [60deg <= NEP bins <= 120deg ] not included


IBEX_ISN_H_trial_6:

[2/16/2021]
- Partialy full 512 spin bins removed
- Updated binning giving more accurate time information
- ToF2 Rate filter applied


## Data Processing steps

1. Histogram Bins and ToF data are pulled from the ID-bas4 orbit folder on the SOC using:
     - TOF data: 
        ``$ lo_de_tof -h -t tofsH INPUT_FILENAME > OUTPUT_FILENAME``
     - Hist Data: 
        `` $ me_show INPUT_FILENAME -x header -x mtype=lo_triple_H >OUTPUT_FILENAME``
2. ToF and Histogram data are loaded and concatenated into Pandas DataFrames.
3. Particle species (H/O/Z) is determined for each of the ToF observation
     - First ToF data is filtered for:
        - abs(tof0+tof3-tof2-tof1)<1
        - type == 0000 [picks only golden triples]
        - tof3 >= 0
     - Species type is selected according to LUT14:
        - H:{tof2:[13,40], (tof0 + tof3/2): [30,70]},
        - O: {tof2:[75,200], (tof0 + tof3/2):[150,300]}
4. GoodTimes Filter is applied to Estep(1,2,3)
     - Filters for time and NEP angle
     **Data is processed through energy step 8, but goodtime filter is ony applied to first three steps
5. H histogram data is rebinned in phase and time and stacked into a matrix. 
     - Phase bins: 61 bin edges linearly spaced between 0 and 1
     - time bins: (Total ellapsed time)/(average spin time) bins linearly spaced between (min(time)-average_spin_time/2) and (max(time)+average_spin_time/2)
     - Ephemaries and time data are averaged over each time bin
     - Calculate the total integrated time per time bin:
        - Add up the number of individual histogram bins that fall into each time bin and multiply by the average dt for each histogram bin (a little less than .5 sec)
        - This value changes because we are slicing out times and nep angles
6. The ToF H data is binned into identical histogram bins as those outlined in step 5. 
7. ToF and Hist histograms are accumulated over 8 spins through stacking of the histograms.

## Data Naming and File System

The data files are sorted by year and orbit. A file is created for each energy step for each arc of each orbit.

``IBEX_Lo_o####@_E#_HB_and_DE_report.csv``
``####``: orbit number (int)
``@``:  orbit arc (a/b)
``E#``: Energy step

## Data Structure

Each file essentially represents two matrices of both the histogram H data and binned ToF H data. To adhere to heritage data formating, the matices are unstacked, so for every 61 rows one spin phase of histogram bins are shown.

**columns**
1. ``bin_center(NEP)``: North ecliptic pole look direction of IBEX-lo calculated from the phase angle
        ``NEP = 360*(Phase - .5) + 3 ``[rebased within range (0,360)]
2. ``center_time(GPS)``: Center of the time bin [GPS(s)]
3. ``center_time(YYYY-MM-DD DD:HH:MM:SS)``: Datetime of the center of the time bin calculated from the GPS time
4. ``HB_counts``: Histogram bin counts 
5. ``bin_center(NEP)``: copy of row 1. kept in for heritage formatting
6. ``DE_counts``: Binned direct event counts
7. ``Orbit:#``: Orbit number fractional with ``[.0:a,.5:b]`` (not repeating with stacked spin bins)

**Rows**
1. Column labels, non-repeating
2. Time Bin Data, Repeating every 61 rows with cols:
    - Col(5) tart_time [GPS(s)]: earliest histogram bin that falls within the 8 spin accumulated time bin
    - Col(6) end_time [GPS(s)]:  latest histogram bin that fals within the 8 spin accumulated time bin
    - Col(7) total_time [s]: Total accumulated time period that falls within the time bin. (function of the number of hist bins that fall within the 8 spins see Processign #5)
3. Unstacked DE and Hist data. Repeating every 61 rows [3:62].  