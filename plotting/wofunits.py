class wofunits:
    
    """
    
    Dictionaries of WoFS variable information for ease of label and directory path management, colorbar limit guidance, and 
    sensitivity normalization.
    
    To use for labels, for example, write: "from wofunits import wofunits", then to call, write "wofunits.labels['var_string_here']"
    
    """

    paths = {'wofs_path': '/Volumes/faletti_backup/WOFS_output', # WoFS data path
             'wofs_path_3d': '/Volumes/faletti_backup/WOFS_output_3d', # 3D WoFS data path
             'sens_path': '/Volumes/faletti_backup/sens_out', # sensitivity data path
             'resp_path': '/Users/williamfaletti/Documents/python/thesis/wofs_code/respout_center', # storm-centered response function data path
             'mrms_data_path': '/Volumes/faletti_backup/MRMS_data', # path to MRMS data
             'sm_path': '/Volumes/faletti_backup/wofs_obs_motion', # storm motion CSV path
             'wofs_locs_path': '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state', # path to modeled storm center gridpoints
             'mrms_locs_path': '/Users/williamfaletti/Desktop', # path to MRMS observed storm center gridpoints
             'mapping_path': '/Users/williamfaletti/Documents/python/thesis/mapping/us_counties_states', # path to shapefiles
             'outplot_path': '/Users/williamfaletti/Documents/python/thesis/wofs_code/paper_figures' # plot save path
            }

    
    labels = {     # formatted unit strings for each WoFS variable
        
    'T2': 'K',
    'TD2': 'K',
    'THE2': 'K',
    'U10': 'm s$^{-1}$',
    'V10': 'm s$^{-1}$',
    'WND_MAG10': 'm s$^{-1}$',
    'T500': 'K',
    'TD500': 'K', 
    'U500': 'm s$^{-1}$',
    'V500': 'm s$^{-1}$',
    'WMAG500': 'm s$^{-1}$',
    'T700': 'K',
    'TD700': 'K', 
    'U700': 'm s$^{-1}$',
    'V700': 'm s$^{-1}$',
    'WMAG700': 'm s$^{-1}$', 
    'T850': 'K',
    'TD850': 'K',
    'U850': 'm s$^{-1}$',
    'V850':'m s$^{-1}$', 
    'WMAG850': 'm s$^{-1}$',
    'U10_SR': 'm s$^{-1}$',
    'V10_SR': 'm s$^{-1}$',
    'WND_MAG10_SR': 'm s$^{-1}$',
    'U500_SR': 'm s$^{-1}$',
    'V500_SR': 'm s$^{-1}$',
    'WMAG500_SR': 'm s$^{-1}$',
    'U700_SR': 'm s$^{-1}$',
    'V700_SR': 'm s$^{-1}$',
    'WMAG700_SR': 'm s$^{-1}$', 
    'U850_SR': 'm s$^{-1}$',
    'V850_SR':'m s$^{-1}$', 
    'WMAG850_SR': 'm$^2$ s$^{-1}$',
    'SBCAPE': 'J kg$^{-1}$',
    'SBCIN': 'J kg$^{-1}$',
    'MLCAPE': 'J kg$^{-1}$',
    'MLCIN': 'J kg$^{-1}$',
    'MUCAPE': 'J kg$^{-1}$',
    'MUCIN': 'J kg$^{-1}$',
    'SRH1': 'm$^2$ s$^{-2}$',
    'SRH3': 'm$^2$ s$^{-2}$',
    'SRH1_TRUE': 'm$^2$ s$^{-2}$',
    'SRH3_TRUE': 'm$^2$ s$^{-2}$',
    'SHEAR_U1': 'm s$^{-1}$',
    'SHEAR_V1': 'm s$^{-1}$',
    'SHEAR_TOT1': 'm s$^{-1}$',
    'SHEAR_U6': 'm s$^{-1}$',
    'SHEAR_V6': 'm s$^{-1}$',
    'SHEAR_TOT6': 'm s$^{-1}$',
    'REFD_MAX': 'dBZ',
    'REFL_1km': 'dBZ',
    'UH_HRSWT25_WRF': 'm$^2$ s$^{-2}$', 
    'UH_30SWT25_WRF': 'm$^2$ s$^{-2}$',
    'UH_15SWT25_WRF': 'm$^2$ s$^{-2}$',
    'UH_SWT25_WRF': 'm$^2$ s$^{-2}$',
    'WZ_02': 's$^{-1}$',
    'WZ_25': 's$^{-1}$',
    'UH_02': 'm$^2$ s$^{-2}$',
    'UH_25': 'm$^2$ s$^{-2}$',
    'LLUP': 'm s$^{-1}$', 
    'UH_SWT02': 'm$^2$ s$^{-2}$',
    'UH_SWT25': 'm$^2$ s$^{-2}$',
    'WZ_SWT02': 's$^{-1}$',
    'WZ_SWT25': 's$^{-1}$',
    'CREF_SWT': 'dBZ',
    'UH_HRSWT02': 'm$^2$ s$^{-2}$',
    'UH_HRSWT25': 'm$^2$ s$^{-2}$',
    'WZ_HRSWT02': 's$^{-1}$',
    'WZ_HRSWT25': 's$^{-1}$',
    'CREF_HRSWT': 'dBZ',
    'UH_30SWT02': 'm$^2$ s$^{-2}$',
    'UH_30SWT25': 'm$^2$ s$^{-2}$',
    'WZ_30SWT02': 's$^{-1}$',
    'WZ_30SWT25': 's$^{-1}$',
    'CREF_30SWT': 'dBZ',
    'UH_15SWT02': 'm$^2$ s$^{-2}$',
    'UH_15SWT25': 'm$^2$ s$^{-2}$',
    'WZ_15SWT02': 's$^{-1}$',
    'WZ_15SWT25': 's$^{-1}$',
    'CREF_15SWT': 'dBZ',
    'WSPD10MAX': 'm s$^{-1}$',
    'W_UP_MAX': 'm s$^{-1}$',
    'W_DN_MAX': 'm s$^{-1}$',
    'UP_HELI_MAX': 'm$^2$ s$^{-2}$',
    'W_MEAN': 'm s$^{-1}$',
    'UH25_05MIN': 'm$^2$ s$^{-2}$',
    'UH25_15MIN': 'm$^2$ s$^{-2}$',
    'UH25_30MIN': 'm$^2$ s$^{-2}$',
    'UH25_60MIN': 'm$^2$ s$^{-2}$',
    'UH02_05MIN': 'm$^2$ s$^{-2}$',
    'UH02_15MIN': 'm$^2$ s$^{-2}$',
    'UH02_30MIN': 'm$^2$ s$^{-2}$',
    'UH02_60MIN': 'm$^2$ s$^{-2}$',
    'WZ25_05MIN': 's$^{-1}$',
    'WZ25_15MIN': 's$^{-1}$',
    'WZ25_30MIN': 's$^{-1}$',
    'WZ25_60MIN': 's$^{-1}$',
    'WZ02_05MIN': 's$^{-1}$',
    'WZ02_15MIN': 's$^{-1}$',
    'WZ02_30MIN': 's$^{-1}$',
    'WZ02_60MIN':'s$^{-1}$',

    'U10-SR': 'm s$^{-1}$',
    'V10-SR': 'm s$^{-1}$',
    'WND-MAG10-SR': 'm s$^{-1}$',
    'U500-SR': 'm s$^{-1}$',
    'V500-SR': 'm s$^{-1}$',
    'WMAG500-SR': 'm s$^{-1}$',
    'U700-SR': 'm s$^{-1}$',
    'V700-SR': 'm s$^{-1}$',
    'WMAG700-SR': 'm$^2$ s$^{-1}$', 
    'U850-SR': 'm s$^{-1}$',
    'V850-SR':'m s$^{-1}$', 
    'WMAG850-SR': 'm$^2$ s$^{-1}$',
    'SHEAR-U1': 'm s$^{-1}$',
    'SHEAR-V1': 'm s$^{-1}$',
    'SHEAR-TOT1': 'm s$^{-1}$',
    'SHEAR-U6': 'm s$^{-1}$',
    'SHEAR-V6': 'm s$^{-1}$',
    'SHEAR-TOT6': 'm s$^{-1}$',
    'SRH1-TRUE': 'm$^2$ s$^{-2}$',
    'SRH3-TRUE': 'm$^2$ s$^{-2}$',
    'REFD-MAX': 'dBZ',
    'REFL-1km': 'dBZ',
    'UH-HRSWT25-WRF': 'm$^2$ s$^{-2}$', 
    'UH-30SWT25-WRF': 'm$^2$ s$^{-2}$',
    'UH-15SWT25-WRF': 'm$^2$ s$^{-2}$',
    'UH-SWT25-WRF': 'm$^2$ s$^{-2}$',
    'WZ-02': 's$^{-1}$',
    'WZ-25': 's$^{-1}$',
    'UH-02': 'm$^2$ s$^{-2}$',
    'UH-25': 'm$^2$ s$^{-2}$',
    'UH-SWT02': 'm$^2$ s$^{-2}$',
    'UH-SWT25': 'm$^2$ s$^{-2}$',
    'WZ-SWT02': 's$^{-1}$',
    'WZ-SWT25': 's$^{-1}$',
    'CREF-SWT': 'dBZ',
    'UH-HRSWT02': 'm$^2$ s$^{-2}$',
    'UH-HRSWT25': 'm$^2$ s$^{-2}$',
    'WZ-HRSWT02': 's$^{-1}$',
    'WZ-HRSWT25': 's$^{-1}$',
    'CREF-HRSWT': 'dBZ',
    'UH-30SWT02': 'm$^2$ s$^{-2}$',
    'UH-30SWT25': 'm$^2$ s$^{-2}$',
    'WZ-30SWT02': 's$^{-1}$',
    'WZ-30SWT25': 's$^{-1}$',
    'CREF-30SWT': 'dBZ',
    'UH-15SWT02': 'm$^2$ s$^{-2}$',
    'UH-15SWT25': 'm$^2$ s$^{-2}$',
    'WZ-15SWT02': 's$^{-1}$',
    'WZ-15SWT25': 's$^{-1}$',
    'CREF-15SWT': 'dBZ',
    'W-UP-MAX': 'm s$^{-1}$',
    'W-DN-MAX': 'm s$^{-1}$',
    'UP-HELI-MAX': 'm$^2$ s$^{-2}$',
    'W-MEAN': 'm s$^{-1}$',
    'UH25-05MIN': 'm$^2$ s$^{-2}$',
    'UH25-15MIN': 'm$^2$ s$^{-2}$',
    'UH25-30MIN': 'm$^2$ s$^{-2}$',
    'UH25-60MIN': 'm$^2$ s$^{-2}$',
    'UH02-05MIN': 'm$^2$ s$^{-2}$',
    'UH02-15MIN': 'm$^2$ s$^{-2}$',
    'UH02-30MIN': 'm$^2$ s$^{-2}$',
    'UH02-60MIN': 'm$^2$ s$^{-2}$',
    'WZ25-05MIN': 's$^{-1}$',
    'WZ25-15MIN': 's$^{-1}$',
    'WZ25-30MIN': 's$^{-1}$',
    'WZ25-60MIN': 's$^{-1}$',
    'WZ02-05MIN': 's$^{-1}$',
    'WZ02-15MIN': 's$^{-1}$',
    'WZ02-30MIN': 's$^{-1}$',
    'WZ02-60MIN':'s$^{-1}$'
    }
    
  
    
    climlos = {     # lower colorbar limits (2.5th percentile for each variable)
    
    'T2' :  286.0,
    'TD2' :  271.0,
    'U10' :  -7.0,
    'V10' :  -7.0,
    'WND_MAG10' :  4.0,
    'T500' :  259.0,
    'TD500' :  237.0,
    'U500' :  6.0,
    'V500' :  16.0,
    'WMAG500' :  18.0,
    'T700' :  278.0,
    'TD700' :  262.0,
    'U700' :  -2.0,
    'V700' :  7.0,
    'WMAG700' :  10.0,
    'T850' :  284.0,
    'TD850' :  269.0,
    'U850' :  -11.0,
    'V850' :  -7.0,
    'WMAG850' :  6.0,
    'SBCAPE' :  0.0,
    'MLCAPE' :  0.0,
    'SBCIN' :  -106.0,
    'MLCIN' :  -118.0,
    'SRH1' :  20.0,
    'SRH3' :  57.0,
    'SHEAR_U1' :  -4.0,
    'SHEAR_V1' :  3.0,
    'SHEAR_TOT1' :  5.0,
    'SHEAR_U6' :  8.0,
    'SHEAR_V6' :  9.0,
    'SHEAR_TOT6' :  18.0,
    
    }

    climhis = {
    
    'T2':  303.0,
    'TD2':  296.0,
    'U10' :  5.0,
    'V10' :  12.0,
    'WND_MAG10' :  13.0,
    'T500' :  265.0,
    'TD500' :  262.0,
    'U500' :  23.0,
    'V500' :  28.0,
    'WMAG500' :  35.0,
    'T700' :  285.0,
    'TD700' :  280.0,
    'U700' :  15.0,
    'V700' :  24.0,
    'WMAG700' :  26.0,
    'T850' :  301.0,
    'TD850' :  291.0,
    'U850' :  10.0,
    'V850' :  26.0,
    'WMAG850' :  26.0,
    'SBCAPE' :  5239.0,
    'MLCAPE' :  3852.0,
    'SBCIN' :  0.0,
    'MLCIN' :  0.0,
    'SRH1' :  440.0,
    'SRH3' :  565.0,
    'SHEAR_U1' :  10.0,
    'SHEAR_V1' :  20.0,
    'SHEAR_TOT1' :  21.0,
    'SHEAR_U6' :  28.0,
    'SHEAR_V6' :  36.0,
    'SHEAR_TOT6' :  42.0

    }
    
    normlos = {     # lower bound for sample climatology (10th percentile for each variable)
    
    'T2' :  286.0,
    'TD2' :  271.0,
    'U10' :  -7.0,
    'V10' :  -7.0,
    'WND_MAG10' :  4.0,
    'T500' :  259.0,
    'TD500' :  237.0,
    'U500' :  6.0,
    'V500' :  16.0,
    'WMAG500' :  18.0,
    'T700' :  278.0,
    'TD700' :  262.0,
    'U700' :  -2.0,
    'V700' :  7.0,
    'WMAG700' :  10.0,
    'T850' :  284.0,
    'TD850' :  269.0,
    'U850' :  -11.0,
    'V850' :  -7.0,
    'WMAG850' :  6.0,
    'SBCAPE' :  0.0,
    'MLCAPE' :  0.0,
    'SBCIN' :  -106.0,
    'MLCIN' :  -118.0,
    'SRH1' :  20.0,
    'SRH3' :  57.0,
    'SHEAR_U1' :  -4.0,
    'SHEAR_V1' :  3.0,
    'SHEAR_TOT1' :  5.0,
    'SHEAR_U6' :  8.0,
    'SHEAR_V6' :  9.0,
    'SHEAR_TOT6' :  18.0
    
    }

    normhis = {     # upper bound for sample climatology (90th percentile for each variable)
    
    'T2' :  302.0,
    'TD2' :  296.0,
    'U10' :  2.0,
    'V10' :  11.0,
    'WND_MAG10' :  12.0,
    'T500' :  264.0,
    'TD500' :  259.0,
    'U500' :  21.0,
    'V500' :  27.0,
    'WMAG500' :  32.0,
    'T700' :  285.0,
    'TD700' :  277.0,
    'U700' :  13.0,
    'V700' :  22.0,
    'WMAG700' :  24.0,
    'T850' :  296.0,
    'TD850' :  290.0,
    'U850' :  8.0,
    'V850' :  23.0,
    'WMAG850' :  23.0,
    'SBCAPE' :  4674.0,
    'MLCAPE' :  3507.0,
    'SBCIN' :  -0.0,
    'MLCIN' :  -1.0,
    'SRH1' :  320.0,
    'SRH3' :  411.0,
    'SHEAR_U1' :  8.0,
    'SHEAR_V1' :  17.0,
    'SHEAR_TOT1' :  18.0,
    'SHEAR_U6' :  25.0,
    'SHEAR_V6' :  32.0,
    'SHEAR_TOT6' :  39.0

    }
    
    
    resplimlos = {   # lower colorbar limits (subjectively chosen based upon range of typical values)
    
    'UH':  20.0,
    'WZ':  0.001,
    'REF':  10.0
        
    }
    
    resplimhis = {   # upper colorbar limits (subjectively chosen based upon range of typical values)
    
    'UH':  250.0,
    'WZ':  0.008,
    'REF':  70.0
        
    }
    
    schemeidx = {
    
    'ysu': [0, 1, 6, 7, 12, 13, 22, 23, 28, 29, 34, 35],
    'myj': [2, 3, 8, 9, 14, 15, 20, 21, 26, 27, 32, 33],
    'mynn': [4, 5, 10, 11, 16, 17, 18, 19, 24, 25, 30, 31],
        
    'scheme_list': ['YSU', 'YSU', 'MYJ', 'MYJ', 'MYNN', 'MYNN', 'YSU', 'YSU', 
            'MYJ', 'MYJ', 'MYNN', 'MYNN', 'YSU', 'YSU', 'MYJ', 'MYJ', 
            'MYNN', 'MYNN', 'MYNN', 'MYNN', 'MYJ', 'MYJ', 'YSU', 'YSU', 
            'MYNN', 'MYNN', 'MYJ', 'MYJ', 'YSU', 'YSU', 'MYNN', 'MYNN', 
            'MYJ', 'MYJ', 'YSU', 'YSU']
        
    }

    
    rankidx = {

    '201905172200': {
        'UH25-05MIN': [15,  1,  8, 29, 23,  6, 10,  2, 26, 28, 32, 13, 22,  3, 35,  7, 20,
                             21, 27,  5,  9,  0,  4, 17, 25, 33, 14, 31, 24, 16, 19, 18, 30, 12,
                             34, 11],
        'UH25-30MIN': [15,  1,  8, 23, 29,  6, 26, 35,  3, 10, 20,  9, 32,  2, 28, 21, 13,
                             34, 25, 22,  5,  0,  7,  4, 30, 27, 31, 24, 17, 14, 16, 33, 19, 18,
                             12, 11],
        'UH25_05MIN': [15,  1,  8, 29, 23,  6, 10,  2, 26, 28, 32, 13, 22,  3, 35,  7, 20,
                             21, 27,  5,  9,  0,  4, 17, 25, 33, 14, 31, 24, 16, 19, 18, 30, 12,
                             34, 11],
        'UH25_30MIN': [15,  1,  8, 23, 29,  6, 26, 35,  3, 10, 20,  9, 32,  2, 28, 21, 13,
                             34, 25, 22,  5,  0,  7,  4, 30, 27, 31, 24, 17, 14, 16, 33, 19, 18,
                             12, 11]
        },
        
    '201905202030': {
        'UH25-05MIN': [27, 35, 21, 14, 28,  7,  5,  0, 20, 33, 12,  9, 32,  6, 16, 30, 25,
                             26,  1, 24,  8, 29, 34, 18, 22, 13, 15, 19, 23,  4, 10, 31, 17,  3,
                             2, 11],
        'UH25-30MIN': [27, 34, 21, 35,  8,  7, 14, 26, 28,  5,  9,  6, 33,  0, 20, 30, 32,
                             12,  1, 16, 25, 24, 13,  4, 29, 11, 15, 22,  2, 18,  3, 31, 10, 19,
                             23, 17],
        'UH25_05MIN': [27, 35, 21, 14, 28,  7,  5,  0, 20, 33, 12,  9, 32,  6, 16, 30, 25,
                             26,  1, 24,  8, 29, 34, 18, 22, 13, 15, 19, 23,  4, 10, 31, 17,  3,
                             2, 11],
        'UH25_30MIN': [27, 34, 21, 35,  8,  7, 14, 26, 28,  5,  9,  6, 33,  0, 20, 30, 32,
                             12,  1, 16, 25, 24, 13,  4, 29, 11, 15, 22,  2, 18,  3, 31, 10, 19,
                             23, 17]
        },
        
    '201905262000': {
        'UH25-05MIN': [ 6,  1, 34,  2,  7,  0, 20, 25, 18,  5, 24, 10, 19, 12,  3, 31, 11,
                              4, 17, 33, 23, 35, 15, 21, 13,  9, 26, 16, 28, 29, 30,  8, 32, 14,
                              22, 27],
        'UH25-30MIN': [34,  2,  5, 20,  7,  6,  0, 10, 24, 19, 25,  1,  3, 31, 17,  4, 18,
                             11, 12, 23, 33, 13, 26, 35, 15,  9, 30, 27, 32, 29, 14, 22, 28, 21,
                             16,  8],
        'UH25_05MIN': [ 6,  1, 34,  2,  7,  0, 20, 25, 18,  5, 24, 10, 19, 12,  3, 31, 11,
                              4, 17, 33, 23, 35, 15, 21, 13,  9, 26, 16, 28, 29, 30,  8, 32, 14,
                              22, 27],
        'UH25_30MIN': [34,  2,  5, 20,  7,  6,  0, 10, 24, 19, 25,  1,  3, 31, 17,  4, 18,
                             11, 12, 23, 33, 13, 26, 35, 15,  9, 30, 27, 32, 29, 14, 22, 28, 21,
                             16,  8]
        },
        
    '201905282230': {
        'UH25-05MIN': [ 9, 20,  1, 31,  8, 28, 14, 13, 10,  5,  4, 18, 29,  7, 24, 25, 11,
                              26, 12,  0,  6, 27, 15,  2,  3, 23, 21, 19, 33, 16, 30, 32, 34, 35,
                              17, 22],
        'UH25-30MIN': [ 19,  9, 31, 20, 23, 10, 14, 25, 28, 27,  8, 13,  1, 12,  5,  7, 21,
                              6, 24, 15, 26,  4, 29, 18,  2, 30, 11, 35,  0, 22, 16,  3, 32, 33,
                              17, 34],
        'UH25_05MIN': [ 9, 20,  1, 31,  8, 28, 14, 13, 10,  5,  4, 18, 29,  7, 24, 25, 11,
                              26, 12,  0,  6, 27, 15,  2,  3, 23, 21, 19, 33, 16, 30, 32, 34, 35,
                              17, 22],
        'UH25_30MIN': [ 19,  9, 31, 20, 23, 10, 14, 25, 28, 27,  8, 13,  1, 12,  5,  7, 21,
                              6, 24, 15, 26,  4, 29, 18,  2, 30, 11, 35,  0, 22, 16,  3, 32, 33,
                              17, 34]
        },

    }

    
    mapping = {   # county/state shapefiles and specs for plotting them with cartopy

    'shapefiles': [f'''{paths['mapping_path']}/cb_2018_us_county_20m.shp''',
                    f'''{paths['mapping_path']}/cb_2018_us_state_20m.shp'''],
    'edgecolor': ['gray', 'k'],
    'lw': [0.5, 1.4]
    
    }