import numpy as np
import pandas as pd
import xarray as xr
import os
import glob
import datetime
import matplotlib.pyplot as plt
import haversine

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from wofunits import wofunits
from metpy.units import units

from metpy.plots import ctables
import matplotlib.colors as colors

drive_dir = '/Volumes/faletti_backup'

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(
                        n=cmap.name, a=minval,b=maxval),cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def case_sel(case):
    """
    Sets case specs given a date/time string.
    
    To use, type: 
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    Variables:
    -----------
    case: string specifying WoFS run initialization time (format: 'YYYYMMDDHHmm')
    
    """
    
    if case == '201905172100':
        wofs_casedir = f'{drive_dir}/WOFS_output/wofs_20190517_2100'
        file_latlons = f'{wofs_casedir}/wofs_i201905172100_v201905172115.nc'
        file_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state/max_gridpoint_storm_centering_201905172100.csv'
        file_resp_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/response_csv/centered_response_201905172100.csv'
        file_mrms_maxima = '/Users/williamfaletti/Desktop/mrms_maxima_indices_517'
        files_wofs = sorted(glob.glob(f'{drive_dir}/WOFS_output/wofs_20190517_2100/wofs_i*'))[30:43]
        state_times = np.arange(datetime.datetime(2019,5,17,22,30), datetime.datetime(2019,5,18,0,31), 
                      datetime.timedelta(minutes=15)).astype(datetime.datetime)
        resptime = '2019-05-18_00_30_00'
    
    if case == '201905172200':
        wofs_casedir = f'{drive_dir}/WOFS_output/wofs_20190517_2200'
        file_latlons = f'{wofs_casedir}/wofs_i201905172200_v201905172215.nc'
        file_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state/max_gridpoint_storm_centering_201905172200.csv'
        file_resp_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/response_csv/centered_response_201905172200.csv'
        file_mrms_maxima = '/Users/williamfaletti/Desktop/mrms_maxima_indices_517'
        files_wofs = sorted(glob.glob(f'{wofs_casedir}/wofs_i*'))[6:19]
        state_times = np.arange(datetime.datetime(2019,5,17,22,30), datetime.datetime(2019,5,18,0,31), 
                      datetime.timedelta(minutes=15)).astype(datetime.datetime)
        resptime = '2019-05-18_00_30_00'
        
    if case == '201905202030':
        wofs_casedir = f'{drive_dir}/WOFS_output/wofs_20190520_2030'
        file_latlons = f'{wofs_casedir}/wofs_i201905202030_v201905202045.nc'
        file_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state/max_gridpoint_storm_centering_20190520.csv'
        file_resp_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/response_csv/centered_response_20190520.csv'
        file_mrms_maxima = '/Users/williamfaletti/Desktop/mrms_maxima_indices_520'
        files_wofs = sorted(glob.glob(f'{wofs_casedir}/wofs_i*'))[12:25]
        state_times = np.arange(datetime.datetime(2019,5,20,21,0), datetime.datetime(2019,5,20,22,31), 
                      datetime.timedelta(minutes=15)).astype(datetime.datetime)
        resptime = '2019-05-20_22_30_00'
        
    if case == '201905262000':
        wofs_casedir = f'{drive_dir}/WOFS_output/wofs_20190526_2000'
        file_latlons = f'{wofs_casedir}/wofs_i201905262000_v201905262015.nc'
        file_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state/max_gridpoint_storm_centering_20190526.csv'
        file_resp_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/response_csv/centered_response_20190526.csv'
        file_mrms_maxima = '/Users/williamfaletti/Desktop/mrms_maxima_indices_526'
        files_wofs = sorted(glob.glob(f'{wofs_casedir}/wofs_i*'))[0:13]
        state_times = np.arange(datetime.datetime(2019,5,26,20,30), datetime.datetime(2019,5,26,21,31), 
                      datetime.timedelta(minutes=15)).astype(datetime.datetime)
        resptime = '2019-05-26_21_30_00'
        
    if case == '201905282230':
        wofs_casedir = f'{drive_dir}/WOFS_output/wofs_20190528_2230'
        file_latlons = f'{wofs_casedir}/wofs_i201905282230_v201905282245.nc'
        file_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/centers_state/max_gridpoint_storm_centering_20190528.csv'
        file_resp_coords = '/Users/williamfaletti/Documents/python/thesis/wofs_code/response_csv/centered_response_20190528.csv'
        file_mrms_maxima = '/Users/williamfaletti/Desktop/mrms_maxima_indices_528'
        files_wofs = sorted(glob.glob(f'{wofs_casedir}/wofs_i*'))[18:31]
        state_times = np.arange(datetime.datetime(2019,5,28,23,0), datetime.datetime(2019,5,29,1,1), 
                      datetime.timedelta(minutes=15)).astype(datetime.datetime)
        resptime = '2019-05-29_01_00_00'

    return wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime


def create_coords_df(file_coords, file_mrms_maxima):
    """
    Creates a pandas dataframe of gridpoint center nxs and nys at intervals provided in a csv file.
    
    Variables:
    ----------
    file_coords: Path to CSV file containing (ny,nx) coordinates to a storm center.
    file_mrms_maxima: Path to file containing local maxima in MRMS azimuthal wind shear.
    
    """
    
    df = pd.read_csv(file_coords, header=0)
    ds_maxima_mrms = xr.open_dataset(file_mrms_maxima)
    
    valid_times = df.columns.values.tolist()[1:]
    
    df = df.rename(columns={'Unnamed: 0': 'Member'}).set_index('Member')
    
    
    for i in range(len(valid_times)):
        #print(df.iloc[:,i].str.replace('(', '').str.replace(')', ''))
        col = df.iloc[:,i].str.replace('(', '').str.replace(')', '')
        col = col.apply(lambda x: pd.Series(str(x).split(',')))
        col = col.rename(columns={0:f'{valid_times[i]} nx',
                                  1:f'{valid_times[i]} ny'})
        
        df = pd.merge(df, col, on='Member')
        
    df = df.iloc[:,len(valid_times):].astype(int)
    
    
    df_nx = df.iloc[:,1::2] + ds_maxima_mrms.attrs['EW_displacement']
    df_ny = df.iloc[:,0::2] + ds_maxima_mrms.attrs['NS_displacement']
    
    df = pd.concat([df_ny, df_nx], axis=1)
    df = df[list(sum(zip(df_ny.columns, df_nx.columns), ()))]

    return df



def varvis_c(file, df, time, mem, var, thres=2):
    """
    Returns a plot of a given variable within a given number of gridpoints of a storm center.
    This is to visualize if storm centering code composited WoFS files correctly. 
    
    Variables:
    ----------
    file:  WoFS file (dims: mem x time x south_north x west_east)
    df:    Pandas DataFrame containing WoFS storm centers through time (nrows x ncols).
    time:  6-digit date/time string in UTC (ddHHmm format, ex: '180000' for 0000 UTC on the 18th).
    mem:   WoFS member number (not its index) to be viewed.
    var:   String corresponding to the variable to plot.
    thres: Distance threshold (in # of gridpoints) from center gridpoint for axis limits. Each gridpoint is 3 km across.
           Default of 2 sets domain size to 15 km x 15 km, or roughly the typical size of a large mesocyclone.
        
    """
    mem = mem-1 # zero indexing WoFS member
    
    ds = xr.open_dataset(f'{file[:-9]}{time}.nc')
    
    fig = plt.figure(figsize=(8,6.5))
    
    center_ny = df[f'{time[2:]} UTC nx'].values[mem]
    center_nx = df[f'{time[2:]} UTC ny'].values[mem]

    cb = plt.pcolormesh(ds[var][mem, 0, center_ny - thres:center_ny + (thres+1), center_nx - thres:center_nx + (thres+1) ])
    
    plt.xlabel('East-West Distance (km)')
    plt.ylabel('North-South Distance (km)')
    plt.title(f'Storm-Centered {var} | {time[2:]} UTC | Mem {str(mem+1).zfill(2)}')
    
    plt.xticks(ticks=np.arange(0, thres*2+1+1, 1), labels=np.arange(0, thres*2+1+1, 1)*3)
    plt.yticks(ticks=np.arange(0, thres*2+1+1, 1), labels=np.arange(0, thres*2+1+1, 1)*3)

    plt.colorbar(cb)
    
    plt.show()


def create_mrmscenter_df(file_coords, file_mrms_maxima, buffer=30):
    
    """
    Returns a single-row pandas dataframe with coordinates of MRMS storm centers in storm composite files.
    
    Variables:
    ----------
    file_coords: Path to CSV file containing (ny,nx) coordinates to a storm center.
    file_mrms_maxima: Path to file containing local maxima in MRMS azimuthal wind shear.
    buffer: Size of buffer (# of gridpoints) from each edge of the WoFS domain when storm centering.
    
    """
    
    df_center = pd.read_csv(file_coords,header=0)
    ds_maxima_mrms = xr.open_dataset(file_mrms_maxima)
    valid_times = df_center.columns.values.tolist()[1:]
    
    df_center = df_center.rename(columns={'Unnamed: 0': 'Member'}).set_index('Member')
    
    for i in range(len(valid_times)):
        col = df_center.iloc[:,i].str.replace('(', '').str.replace(')', '')#, regex=True)
        col = col.apply(lambda x: pd.Series(str(x).split(',')))
        col = col.rename(columns={0:f'{valid_times[i]} nx',
                                  1:f'{valid_times[i]} ny'})
        
        df_center = pd.merge(df_center, col, on='Member')
        
    df_center = df_center.iloc[:,len(valid_times):].astype(int)
    
    df_nx = df_center.iloc[:,1::2] + ds_maxima_mrms.attrs['EW_displacement']
    df_ny = df_center.iloc[:,0::2] + ds_maxima_mrms.attrs['NS_displacement']
    
    df_center = pd.concat([df_ny, df_nx], axis=1)
    df_center = df_center[list(sum(zip(df_ny.columns, df_nx.columns), ()))][-1:] - buffer
    
    return df_center


def diff_center_df(df, file_mrms_maxima):
    """
    Given a storm-centered Pandas DataFrame (in format output by create_coords_df() or create_mrmscenter_df() functions),
    returns a Pandas DataFrame with time difference of gridpoint location between the column time and the previous column 
    time.
    
    Variables:
    ----------
    df: Pandas dataframe containing nx and ny storm positions.
    file_mrms_maxima: Path to file containing local maxima in MRMS azimuthal wind shear.
    
    """
    
    ds_maxima_mrms = xr.open_dataset(file_mrms_maxima)
    
    df_nx = df.iloc[:,1::2] + ds_maxima_mrms.attrs['EW_displacement']
    df_ny = df.iloc[:,0::2] + ds_maxima_mrms.attrs['NS_displacement']
    
    df_diff = pd.concat([df_ny.diff(axis=1), df_nx.diff(axis=1)], axis=1)
    df_diff = df_diff[list(sum(zip(df_ny.columns, df_nx.columns), ()))]
    df_diff.iloc[:,0::2]
    
    return df_diff

def xy_mesh(case, state_time, nx=240, dx=3):
    """
    Generate 2D storm-relative x- and y- distance meshgrids with an origin at the
    mesocyclone center. 
    
    Variables:
    ----------
    case: string specifying WoFS run initialization time (format: 'YYYYMMDDHHmm')
    state_time: datetime object with format datetime.datetime(year,month,day,hour,minute)
    
    Returns:
    --------
    xmesh,ymesh: 2D meshgrids of storm-relative distances with an origin at the mesocyclone
                 center.
    
    """
    # Select case specs
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    # Define storm centers on storm-relative grid
    df_center = create_mrmscenter_df(file_coords, file_mrms_maxima)
    
    # Select nx and ny storm center position at chosen state time
    nx_center, ny_center = df_center[f'{str(state_time.hour).zfill(2)}{str(state_time.minute).zfill(2)} UTC ny'].values[0],\
                            df_center[f'{str(state_time.hour).zfill(2)}{str(state_time.minute).zfill(2)} UTC nx'].values[0]
    
    # Uncomment this if want to use a domain lag   
        # gives true distance in x and y to lag center
    #domain_lag_x = domain_lag*np.cos(np.deg2rad(storm_ang))
    #domain_lag_y = domain_lag*np.sin(np.deg2rad(storm_ang))

        # gives number of gridpoints in x and y to lag center
    #domain_lag_gridx = int(round(domain_lag_x/3))
    #domain_lag_gridy = int(round(domain_lag_y/3))
    
        # pull gridpoint storm center components from dataframe
    #nx_center, ny_center = df_center[f'{str(state_time.hour).zfill(2)}{str(state_time.minute).zfill(2)} UTC ny'].values[0] - \
    #                         domain_lag_gridx, \
    #                        df_center[f'{str(state_time.hour).zfill(2)}{str(state_time.minute).zfill(2)} UTC nx'].values[0] - \
    #                         domain_lag_gridy
    
    # Calculate storm-relative x and y distnace arrays
    x = np.arange(-nx_center, nx-nx_center, 1)*dx # initialize x-dimension of nx-km gridpoint distances
    y = np.arange(-ny_center, nx-ny_center, 1)*dx # initialize y-dimension of nx-km gridpoint distances
    
    # Create storm-relative distance meshgrids
    xmesh, ymesh = np.meshgrid(x,y)
    
    return xmesh, ymesh


def calc_storm_ang(sm_x, sm_y):
    """
    Given x- and y-components of storm motion, calculate the angle of storm motion
    (where 0° is the positive x-axis).
    
    Variables:
    ---------
    sm_x: x-component of storm motion (float)
    sm_y: y-component of storm motion (float)
    
    Returns:
    --------
    storm_ang: storm motion angle in degrees, relative to the +x axis (float) 
    
    """
    
    if sm_x > 0 and sm_y >= 0:
        storm_ang = np.arctan(abs(sm_y)/abs(sm_x))*180/np.pi
    
    if sm_x <= 0 and sm_y > 0:
        storm_ang = 90 + np.arctan(abs(sm_x)/abs(sm_y))*180/np.pi
    
    if sm_x < 0 and sm_y <= 0:
        storm_ang = 180 + np.arctan(abs(sm_y)/abs(sm_x))*180/np.pi
    
    if sm_x >= 0 and sm_y < 0:
        storm_ang = 270 + np.arctan(abs(sm_x)/abs(sm_y))*180/np.pi
    
    if sm_x == 0 and sm_y == 0:
        storm_ang = np.inf
    
    return storm_ang


def calc_ang_tot(xmesh, ymesh, storm_ang):
    """
    Given storm-centered 2D x- and y-meshgrids, return a 2D meshgrid of angles of respective storm-relative 
    gridpoint locations from the positive x-axis (zonal axis)
    
    Variables:
    ----------
    xmesh, ymesh: 2D meshgrids of distances from storm centers at a given time
    storm_ang: Angle of storm motion from the positive x-axis (zonal axis)
    
    """
    xmesh_q1 = np.where((xmesh<=0) | (ymesh<0), np.nan, xmesh)
    ymesh_q1 = np.where((xmesh<=0) | (ymesh<0), np.nan, ymesh)
    
    ang_q1 = np.arctan(ymesh_q1/xmesh_q1)*180/np.pi
    
    xmesh_q2 = np.where((xmesh>0) | (ymesh<0), np.nan, xmesh)
    ymesh_q2 = np.where((xmesh>0) | (ymesh<0), np.nan, ymesh)
    
    ang_q2 = -np.arctan(xmesh_q2/ymesh_q2)*180/np.pi + 90
    
    xmesh_q3 = np.where((xmesh>=0) | (ymesh>=0), np.nan, xmesh)
    ymesh_q3 = np.where((xmesh>=0) | (ymesh>=0), np.nan, ymesh)
    
    ang_q3 = np.arctan(ymesh_q3/xmesh_q3)*180/np.pi + 180
    
    xmesh_q4 = np.where((xmesh<0) | (ymesh>=0), np.nan, xmesh)
    ymesh_q4 = np.where((xmesh<0) | (ymesh>=0), np.nan, ymesh)
    
    ang_q4 = -np.arctan(xmesh_q4/ymesh_q4)*180/np.pi + 270
    
    ang_tot = (np.nanmean([ang_q1,ang_q2,ang_q3,ang_q4],axis=0) - storm_ang) % 360
    
    return ang_tot

def srw_ang_diff(case,state_time,zero_deg_ang=90):
    """
    Given a WoFS case and state time, calculates the difference between the angle of a gridpoint
    from the mesocyclone center and angle of the SR wind direction. Both angles are in the framework
    of hodograph wind angles (ie, 0° pointing south from origin, angle increases rotating clockwise).
    
    May not work well for cases where variability in either angle crosses the 0° line (ie, due south).
    Shifting the 0° line should work for cases in which SR wind variability occurs over a finite angular 
    range, but this may require additional tinkering.
    
    
    Variables:
    ----------
    case: 12-digit string corresponding to case initialization time in format: YYYYMMddHHmm
    state_time: datetime object with format datetime.datetime(year,month,day,hour,minute)
    
    Returns:
    """
    
    file = f'{drive_dir}/WOFS_output/wofs_{case[:8]}_{case[8:]}/wofs_center_i{case}_v{state_time.year}'+\
            f'{str(state_time.month).zfill(2)}{str(state_time.day).zfill(2)}{str(state_time.hour).zfill(2)}'+\
            f'{str(state_time.minute).zfill(2)}.nc'
    
    ds_sr = xr.open_dataset(file)
    
    # Calculate SR angles of gridpoints from mesocyclone center
    xmesh,ymesh = xy_mesh(case, state_time)
    sr_angle = (180 - calc_ang_tot(xmesh,ymesh,zero_deg_ang))%360 # transformation to get calc_ang_tot() in terms of storm motion
    
    # Calculate ensemble mean SR wind angle
    mean_srw = np.nanmean(calc_ang_tot(ds_sr.U10_SR,ds_sr.V10_SR,zero_deg_ang),axis=(0,1))
    
    # Calculate difference between the two angles
    srw_diff = np.abs((180+(180-mean_srw) - sr_angle))
    
    return srw_diff


def smooth_sm_alg(df_dx, df_dy, thres=5.0):
    """
    Function to identify where "spurious" storm positions may degrade storm motion estimates. 
    A rolling window of 3 storm motion vectors, centered at the current time, is selected. If 
    the difference between the current and previous storm motion vector exceeds the specified
    threshold value, the algorithm checks to see which the future-time motion vector is closer 
    to — the current-time vector or previous-time vector. If the future vector is closer to
    previous-time vector, then x- and y-motion components are averaged between the previous and
    future times. If the future vector is closer to the current-time vector, the current-time 
    x- and y-motion components remain unaltered.
    
    Variables:
    ----------
    df_dx: DataFrame containing x-component difference in storm positions (can also be x-speed).
    df_dy: DataFrame containing y-component difference in storm positions (can also be y-speed).
    thres: Threshold in vector difference between a current-time and previous-time vectors.
    
    Returns:
    --------
    df_dx_corr: Algorithmically corrected DataFrame of x-component differences.
    df_dy_corr: Algorithmically corrected DataFrame of y-component differences.
    
    """
    
    colvalsx,colvalsy = [],[]
    for i in range(len(df_dx.columns)): # iterate over columns (time)
        
        rowvalsx,rowvalsy = [],[]
        for j in range(len(df_dx[df_dx.columns[i]])): # iterate over columns (member)
            
            # Check if any indices in 3-value window (w1,w2,w3 elements) fall below (w1) or above (w3) 
            # the "sides" of the time domain. Window begins where w2 represents the first value in the
            # time domain to the final value within it, thus w1 and w3 can fall above or below it.
            
            if i == 0: # set w1 to nan if w1 falls below the time domain
                w_x1, w_x2, w_x3 = np.nan, df_dx[df_dx.columns[i]][j], df_dx[df_dx.columns[i+1]][j]
                w_y1, w_y2, w_y3 = np.nan, df_dy[df_dy.columns[i]][j], df_dy[df_dy.columns[i+1]][j]
                
            if i == len(df_dx.columns)-1: # set w3 to nan if w3 falls above the time domain
                w_x1, w_x2, w_x3 = df_dx[df_dx.columns[i-1]][j], df_dx[df_dx.columns[i]][j], np.nan
                w_y1, w_y2, w_y3 = df_dy[df_dy.columns[i-1]][j], df_dy[df_dy.columns[i]][j], np.nan
                
            else: # set w1, w2, w3 to the 1st, 2nd, 3rd values in the window if all fall within time domain
                w_x1, w_x2, w_x3 = df_dx[df_dx.columns[i-1]][j], df_dx[df_dx.columns[i]][j], df_dx[df_dx.columns[i+1]][j]
                w_y1, w_y2, w_y3 = df_dy[df_dy.columns[i-1]][j], df_dy[df_dy.columns[i]][j], df_dy[df_dy.columns[i+1]][j]
        
            # Apply algorithm
        
                # leave values at domain edges untouched
            if np.isnan([w_x1, w_x2]).all(): # Where w2 is first value in time domain
                finvalx = df_dx[df_dx.columns[i]][j]
                finvaly = df_dy[df_dy.columns[i]][j]

            elif np.isnan(w_x2) == False and np.isnan(w_x3) == True: # Where w2 is the 2nd to last value in the time domain
                finvalx = df_dx[df_dx.columns[i]][j]
                finvaly = df_dy[df_dy.columns[i]][j]
                
            elif np.isnan([w_x2, w_x3]).all(): # Where w2 is the last value in time domain
                finvalx = df_dx[df_dx.columns[i]][j]
                finvaly = df_dy[df_dy.columns[i]][j]
                
                # apply algorthim to values between domain edges
            else:
                # calculate vector difference between 1st and 2nd indices
                x_diff12 = df_dx[df_dx.columns[i]][j] - df_dx[df_dx.columns[i-1]][j]
                y_diff12 = df_dy[df_dy.columns[i]][j] - df_dy[df_dy.columns[i-1]][j]
                vector_diff12 = (x_diff12**2 + y_diff12**2)**0.5
                
                # calculate vector difference between 1st and 3rd indices
                x_diff13 = df_dx[df_dx.columns[i+1]][j] - df_dx[df_dx.columns[i-1]][j]
                y_diff13 = df_dy[df_dy.columns[i+1]][j] - df_dy[df_dy.columns[i-1]][j]
                vector_diff13 = (x_diff13**2 + y_diff13**2)**0.5
                
                # calculate vector difference between 2nd and 3rd indices
                x_diff23 = df_dx[df_dx.columns[i+1]][j] - df_dx[df_dx.columns[i]][j]
                y_diff23 = df_dy[df_dy.columns[i+1]][j] - df_dy[df_dy.columns[i]][j]
                vector_diff23 = (x_diff23**2 + y_diff23**2)**0.5                
                
                    # check if threshold exceeded and if 3rd vector is closer to 1st or 2nd vector
                if vector_diff12 > thres and vector_diff23 >= vector_diff13:
                    finvalx = (df_dx[df_dx.columns[i-1]][j] + df_dx[df_dx.columns[i+1]][j]) / 2
                    finvaly = (df_dy[df_dy.columns[i-1]][j] + df_dy[df_dy.columns[i+1]][j]) / 2
                    
                else: # if not exceeded, leave x/y values untouched
                    finvalx = df_dx[df_dx.columns[i]][j]
                    finvaly = df_dy[df_dy.columns[i]][j]
                    
            rowvalsx.append(finvalx)
            rowvalsy.append(finvaly)
        
        colvalsx.append(rowvalsx)
        colvalsy.append(rowvalsy)
        
    df_dx_corr = pd.DataFrame(np.array(colvalsx), columns=df_dx.index, index=df_dx.columns).T
    df_dy_corr = pd.DataFrame(np.array(colvalsy), columns=df_dy.index, index=df_dy.columns).T
    
    return df_dx_corr, df_dy_corr
    
    
def sens_summary(case, statetime, respvar, center, statvar='stdsens'):
    """
    Returns a 3D array of surface sensitivity statistics with dimensions case x x-coords x y-coords
    given a case and state time.
    
    Variables:
    ----------
    case: 12-character string with format YYYYMMDDHHmm
    statetime: 6-character string with format DDHHmm
    respvar: String giving sensitivity response variable used by sensitivity file string.
    center: If True, storm-relative stats used; if False, ground-relative stats used.
    
    Returns:
    ----------
    stats: 3D array of sensitivity statistics with dimensions case x x-coords x y-coords
    
    """
    
    sfcvars = ['T2','TD2','U10','V10','WND-MAG10']
    
    if center == False:
        sensfiles = glob.glob(f'{drive_dir}/sens_out/sens_gr_center_{case}/stats_{respvar}*{statetime[2:4]}_{statetime[4:6]}_00.nc')
    elif center == True:
        sensfiles = glob.glob(f'{drive_dir}/sens_out/sens_center_{case}/stats_{respvar}*{statetime[2:4]}_{statetime[4:6]}_00.nc')
    
    sensfiles = [ file for file in sensfiles if any(sfcvar in file[-28:] for sfcvar in sfcvars) ]
    
    stats = []
    for file in sensfiles:
        stat = abs(xr.open_dataset(file)[statvar]).values
        stats.append(stat)
    
    stats = np.array(stats)
    
    return stats
    
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def cref_cmap():
    """
    Returns properly truncated NWSReflectivity colormap.
    Recommended levels are np.arange(20,71,5).
    
    """
    
    norm, cmap_ref = ctables.registry.get_with_steps('NWSReflectivity', 16, 16)
    
    cmap_ref = truncate_colormap(cmap_ref, 0.2, 0.95)

    return cmap_ref

    
#################### PLOTTING FUNCTIONS #######################################


def stats_plot_center(case, moment, statevars, cmap='Reds', barbs=True, show_domain=False, gridlines=False):
    """
    Generates 4-panel plots and saves figures displaying state mean or standard deviation at 
    15-minute intervals for a chosen case until the response time.
    
    Variables:
    ----------
    case: 12-digit string corresponding to case initialization time in format: YYYYMMddHHmm
    moment: Moment to calculate ('mean' or 'std')
    statevars: List of 4 state variable strings (with underscores converted to hyphens)
    cmap: Matplotlib-supported colormap string
    barbs: Boolean for whether or not to plot wind barbs (in knots)
    show_domain: Boolean for whether or not to plot inflow domains
    gridlines: Boolean for whether or not to plot x=0 and y=0 lines
    
    """
    
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    
    # Define specs of the inflow domain
    
    ang_bound1, ang_bound2 = 120, 240 # angular bounds relative to the storm motion
    
    nf_dist = 40 # distance of near-inflow domain
    ff_dist = 120 # distance of far-inflow domain
    
    domain_lag = 0 # distance lag of domain origin to storm center in km
    
    
    # Define observed MRMS storm centers to center storm on plot
    df_center = create_mrmscenter_df(file_coords, file_mrms_maxima)
    
    for state_time in state_times:
        # Process date/time strings and open necessary files
        
            # process date/time strings
        respmin, resphour, respday, respmonth, respyear = resptime[14:16],resptime[11:13],resptime[8:10],resptime[5:7],resptime[0:4]
        
        str_statemin, str_statehour, str_statemonth = '{:02d}'.format(state_time.minute),'{:02d}'.format(state_time.hour), \
                                                    '{:02d}'.format(state_time.month)
        
        inityear, initmin, inithour, initday, initmonth = case[0:4], case[10:12], case[8:10], case[6:8], case[4:6]
        
        resp_time = datetime.datetime(int(respyear), int(respmonth), int(respday), int(resphour), int(respmin))
        
        file_ref = f'{drive_dir}/WOFS_output/wofs_{inityear}{initmonth}{initday}_{inithour}' + \
                f'{initmin}/wofs_center_i{inityear}{initmonth}{initday}{inithour}{initmin}_v{state_time.year}{str_statemonth}' + \
                f'{state_time.day}{str_statehour}{str_statemin}.nc'
        
            # open netCDF files
        ds_ref = xr.open_dataset(file_ref)
        
            # define plotting variables
        cref = np.where(np.nanmean(ds_ref.REFD_MAX, axis=0) < 10, np.nan, np.nanmean(ds_ref.REFD_MAX, axis=0))
        
        # Create storm-centered x and y meshgrids
        
            # pull angle of storm motion
        storm_ang = 30 # make this calculate it eventually
        
        xmesh, ymesh = xy_mesh(case, state_time)
        
        # Define angle bounds for inflow domain
        angles1 = np.deg2rad(np.linspace(0,ang_bound1+storm_ang,10000)) # above 0
        angles2 = np.deg2rad(np.linspace(ang_bound2+storm_ang,360,10000)) # below 360
        

        # Define plotting specs
        
        levels = np.arange(-60,61,5)
        ticks = levels[0::4]
        
        # Run for loop and plot
        
        nrows,ncols = 1,2
        
        fig, ax = plt.subplots(nrows,ncols,figsize=(11,5))
        
        for nrow in range(nrows):
            for ncol in range(ncols):
                
                if nrow==0 and ncol==0:
                    statevar=statevars[0]
                if nrow==0 and ncol==1:
                    statevar=statevars[1]
                if nrow==1 and ncol==0:
                    statevar=statevars[2]
                if nrow==1 and ncol==1:
                    statevar=statevars[3]
                
                # Calculate mean for plotting state variable
                if moment == 'mean':
                    moment_state = np.nanmean(ds_ref[statevar.replace('-','_')], axis=0)[0]                    
                if moment == 'std':
                    moment_state = np.nanstd(ds_ref[statevar.replace('-','_')], axis=0)[0]
                
                # Plot wind barbs at level of variable
                    # if variable at upper-air mandatory level, plot wind at that level
                    # if near-surface or convective variable, plot surface wind

                try:
                    u = (np.nanmean(ds_ref[f'U{statevar[-3:]}_SR'], axis=(0,1))*units.meter_per_second).to(units.knot)
                    v = (np.nanmean(ds_ref[f'V{statevar[-3:]}_SR'], axis=(0,1))*units.meter_per_second).to(units.knot)
                except:
                    u = (np.nanmean(ds_ref.U10_SR, axis=(0,1))*units.meter_per_second).to(units.knot)
                    v = (np.nanmean(ds_ref.V10_SR, axis=(0,1))*units.meter_per_second).to(units.knot)
                
                # Automate colorbar levels
                #cblimit_min = round(np.nanpercentile(mean_state, 10), 0)
                #cblimit_max = round(np.nanpercentile(mean_state, 90), 0)
                #cbstep = round((cblimit_max - cblimit_min)/21, 1)
    
                #levels_auto = np.arange(cblimit_min, cblimit_max + cbstep, cbstep)
                
                # Plot state mean and mean storm
                cb = ax[ncol].contourf(xmesh, ymesh, moment_state, cmap=cmap, extend='both', levels=21, zorder=0)
                ax[ncol].contour(xmesh, ymesh, cref[0], levels=[20,35,50], colors='k')
                
                # Plot mean wind barbs
                if barbs == True:
                    ax[ncol].barbs(xmesh[5::10,5::10], ymesh[5::10,5::10], u[5::10,5::10], v[5::10,5::10], length=5, 
                                   sizes={'spacing':0.2}, color='k', alpha=0.7, zorder=2)
                
                # Plot inflow domain if show_domain = True
                
                if show_domain == True:
                        # far-field
                    ax[ncol].plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x,
                                  (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # above 0
                        # near-field
                    ax[ncol].plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x,
                                  (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # above 0
                    ax[ncol].plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x,
                                  (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # below 360
                    ax[ncol].plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, 
                                  (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # below 360
                    
                        # plot angle bound lines
                    ax[ncol].plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                                  [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                                  color = 'dimgray',lw=2, ls='--') # above 0
                    ax[ncol].plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                                  [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                                  color = 'dimgray',lw=2, ls='--') # below 360
                
                
                ax[ncol].set_xlim(-125, 170)
                ax[ncol].set_ylim(-150,150)
    
                if nrow == 1:
                    ax[ncol].set_xlabel('East-West Distance (km)')
                if ncol == 0:
                    ax[ncol].set_ylabel('North-South Distance (km)')
                    
                ax[ncol].set_title(statevar)
                
                #if gridlines == True:
                #    ax[ncol].axhline(y=0, linestyle='--', color='dimgray', zorder=1) # y-axis
                #    ax[ncol].axvline(x=0, linestyle='--', color='dimgray', zorder=1) # x-axis
                
                cbar = fig.colorbar(cb, ax=ax[ncol])#, ticks=ticks)
                
                if moment == 'mean':
                    titlestr = 'Mean'
                    cbar.set_label(label=f'Mean ({wofunits.labels[statevar]})', size=8)
                if moment == 'std':
                    titlestr = 'Standard Deviation'
                    cbar.set_label(label=f'Std. Dev. ({wofunits.labels[statevar]})', size=8)
        
        plt.suptitle(f'{titlestr} at {int((resp_time - state_time).seconds/60)}-min lead time', weight='bold')
        plt.tight_layout()
        
        outdir = '/Users/williamfaletti/Documents/python/thesis/wofs_code/sens_outplots_center'
        
        plt.savefig(f'{outdir}/{case}/{moment}_center_{state_time.year}{str_statemonth}{state_time.day}' + \
                f'{str_statehour}_{str_statemin}_{statevars[0]}_{statevars[1]}.jpg', bbox_inches='tight', facecolor='w', dpi = 200)


def sens_plot_center(case, senstype, respvar, statevars, cmap='RdBu_r', xlims=(-125,170), ylims=(-150,150), barbs=True, show_domain=False, gridlines=False, save=True):
    """
    Generates 4-panel plots and saves figures displaying ensemble sensitivity at
    15-minute intervals for a chosen case until the response time.
    
    Variables:
    ----------
    case: 12-digit string corresponding to case initialization time in format: YYYYMMddHHmm
    senstype: Sensitivity type to calculate ('sens' or 'stdsens')
    respvar: Response variable string (with underscores converted to hyphens)
    statevars: List of 2 state variable strings (with underscores converted to hyphens)
    cmap: Matplotlib-supported colormap string
    barbs: Boolean for whether or not to plot wind barbs (in knots)
    show_domain: Boolean for whether or not to plot inflow domains
    gridlines: Boolean for whether or not to plot x=0 and y=0 lines
    
    """
    
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    
    # Define specs of the inflow domain
    show_domain = False
    gridlines = True 
    
    ang_bound1, ang_bound2 = 120, 240 # angular bounds relative to the storm motion
    
    nf_dist = 40 # distance of near-inflow domain
    ff_dist = 120 # distance of far-inflow domain
    
    domain_lag = 0 # distance lag of domain origin to storm center in km
    
    
    # Define observed MRMS storm centers to center storm on plot
    df_center = create_mrmscenter_df(file_coords, file_mrms_maxima)
    
    for state_time in state_times:
        # Process date/time strings and open necessary files
        
            # process date/time strings
        respmin, resphour, respday, respmonth, respyear = resptime[14:16],resptime[11:13],resptime[8:10],resptime[5:7],resptime[0:4]
        
        str_statemin, str_statehour, str_statemonth = '{:02d}'.format(state_time.minute), \
                                                    '{:02d}'.format(state_time.hour),'{:02d}'.format(state_time.month)
        
        inityear, initmin, inithour, initday, initmonth = case[0:4], case[10:12], case[8:10], case[6:8], case[4:6]
        
        resp_time = datetime.datetime(int(respyear), int(respmonth), int(respday), int(resphour), int(respmin))
        
        file_ref = f'{drive_dir}/WOFS_output/wofs_{inityear}{initmonth}{initday}_{inithour}' + \
                    f'{initmin}/wofs_center_i{inityear}{initmonth}{initday}{inithour}{initmin}_v{state_time.year}{str_statemonth}' + \
                    f'{state_time.day}{str_statehour}{str_statemin}.nc'
        
            # open netCDF files
        ds_ref = xr.open_dataset(file_ref)
        
            # define plotting variables
        cref = np.where(np.nanmean(ds_ref.REFD_MAX, axis=0) < 10, np.nan, np.nanmean(ds_ref.REFD_MAX, axis=0))[0]
        
        # Create storm-centered x and y meshgrids

        xmesh, ymesh = xy_mesh(case, state_time)
        
        # Define angle bounds for inflow domain
       
            # pull angle of storm motion
        storm_ang = 30 # make this calculate it eventually
        
        angles1 = np.deg2rad(np.linspace(0,ang_bound1+storm_ang,10000)) # above 0
        angles2 = np.deg2rad(np.linspace(ang_bound2+storm_ang,360,10000)) # below 360
        
        
        # Define plotting specs
        
        levels = np.arange(-60,61,5)
        ticks = levels[0::4]
        
        # Run for loop and plot
        
        nrows,ncols = 2,2
        
        fig, ax = plt.subplots(nrows,ncols,figsize=(10,8))
        
        for nrow in range(nrows):
            for ncol in range(ncols):
                
                if nrow==0 and ncol==0:
                    statevar=statevars[0]
                if nrow==0 and ncol==1:
                    statevar=statevars[1]
                if nrow==1 and ncol==0:
                    statevar=statevars[2]
                if nrow==1 and ncol==1:
                    statevar=statevars[3]
                
                file_sens = f'{drive_dir}/sens_out/sens_center_2019{initmonth}{initday}{inithour}{initmin}' + \
                            f'/stats_{respvar}_max_{resphour}_{respmin}_00_{statevar}_{str_statehour}_{str_statemin}_00.nc'
                
                ds_sens = xr.open_dataset(file_sens)
                
                if senstype == 'sens':
                    sens = ds_sens.sens
                if senstype == 'stdsens':
                    sens = ds_sens.stdsens
                
                
                # Plot sensitivity, significance hatching, and mean storm
                ax[nrow,ncol].contour(xmesh, ymesh, cref, levels=[20,35,50], colors='k')
                cb = ax[nrow,ncol].contourf(xmesh, ymesh, sens, cmap=cmap, levels=levels, extend='both')
                ax[nrow,ncol].contourf(xmesh, ymesh, abs(ds_sens['pstat'].values), 
                                       [0, 0.05], colors='none', hatches=['////'], extend='lower')
                
                # Plot wind barbs
                if barbs == True:
                    
                    # Plot wind barbs at level of variable
                        # if variable at upper-air mandatory level, plot wind at that level
                        # if near-surface or convective variable, plot surface wind
                    try:
                        u = (np.nanmean(ds_ref[f'U{statevar[-3:]}_SR'], axis=(0,1))*units.meter_per_second).to(units.knot)
                        v = (np.nanmean(ds_ref[f'V{statevar[-3:]}_SR'], axis=(0,1))*units.meter_per_second).to(units.knot)
                    except:
                        u = (np.nanmean(ds_ref.U10_SR, axis=(0,1))*units.meter_per_second).to(units.knot)
                        v = (np.nanmean(ds_ref.V10_SR, axis=(0,1))*units.meter_per_second).to(units.knot)
                    
                    ax[nrow,ncol].barbs(xmesh[5::15,5::15], ymesh[5::15,5::15], u[5::15,5::15], v[5::15,5::15], length=5, 
                                   sizes={'spacing':0.2}, color='k', alpha=0.7, zorder=2)
                
                
                # Plot inflow domain if show_domain = True
                if show_domain == True:
                        # far-field
                    ax[nrow,ncol].plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, 
                                    (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # above 0
                        # near-field
                    ax[nrow,ncol].plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, 
                                    (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # above 0
                    ax[nrow,ncol].plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, 
                                    (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # below 360
                    ax[nrow,ncol].plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, 
                                    (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = 'dimgray',lw=2, ls='--') # below 360
                    
                        # plot angle bound lines
                    ax[nrow,ncol].plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                                       [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                                       color = 'dimgray',lw=2, ls='--') # above 0
                    
                    ax[nrow,ncol].plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                                       [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                                       color = 'dimgray',lw=2, ls='--') # below 360
                
              
                ax[nrow,ncol].set_xlim(xlims[0],xlims[1])
                ax[nrow,ncol].set_ylim(ylims[0],ylims[1])
    
                if nrow == 1:
                    ax[nrow,ncol].set_xlabel('East-West Distance (km)')
                if ncol == 0:
                    ax[nrow,ncol].set_ylabel('North-South Distance (km)')
                    
                ax[nrow,ncol].set_title(statevar)
                
                #if gridlines == True:
                #    ax[nrow,ncol].axhline(y=0, linestyle='--', color='dimgray', zorder=1) # y-axis
                #    ax[nrow,ncol].axvline(x=0, linestyle='--', color='dimgray', zorder=1) # x-axis
                
                cbar = fig.colorbar(cb, ax=ax[nrow,ncol], ticks=ticks)
                
                if senstype == 'sens':
                    sensstr = 'Sensitivity'
                    cbar.set_label(label=f'{sensstr} [({wofunits.labels[respvar]}) ({wofunits.labels[statevar]})$^{{-1}}$]', size=8)
                if senstype == 'stdsens':
                    sensstr = 'Std. Sens.'
                    cbar.set_label(label=f'{sensstr} ({wofunits.labels[respvar]})', size=8)
                
        #plt.suptitle(f'{sensstr} of {respvar} at {int((resp_time - state_time).seconds/60)}-min lead time', weight='bold')
        plt.suptitle(f'{int((resp_time - state_time).seconds/60)}-min lead time', weight='bold')
        plt.tight_layout()
        
        outdir = '/Users/williamfaletti/Documents/python/thesis/wofs_code/sens_outplots_center'
        
        if save == True:
            
            outdir = f'{outdir}/{case}/{statevars[0]}_{statevars[1]}_{statevars[2]}_{statevars[3]}_{respvar}'
            
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
                
            plt.savefig(f'{outdir}/{senstype}_center_{respvar}_{state_time.year}{str_statemonth}{state_time.day}'+\
                        f'{str_statehour}_{str_statemin}_{statevars[0]}_{statevars[1]}_{statevars[2]}_{statevars[3]}.jpg', 
                        bbox_inches='tight', facecolor='w', dpi = 200)
        
    plt.show()
        
def sr_gr_compareplot(cases=['201905172100',
                             '201905202030',
                             '201905262000',
                             '201905282230'], respvar_sr='UH25-30MIN', nrows=2, ncols=2):
    """
    Returns a 4-panel plot to compare ground- vs. storm-relative mesocyclone 
    intensity response functions for the 4 WoFS cases analyzed in this research.
    
    Variables:
    ----------
    cases: List of 4 case strings in format YYYYMMddHHmm
    respvar_sr: Response variable string (with underscores converted to hyphens)
    nrows: Number of rows in panel plot
    ncols: Number of columns in panel plot
    
    """
    
    dir_resp_sr = '/Users/williamfaletti/Documents/python/thesis/wofs_code/respout_center'
    respvar_sr = respvar_sr.replace('-','_')
    
    # Convert storm-relative response variable names to ground-relative equivalent names
    if respvar_sr == 'UH25_05MIN':
        respvar_gr = 'UP_HELI_MAX'
    if respvar_sr == 'UH25_15MIN':
        respvar_gr = 'UH_15SWT25_WRF'
    if respvar_sr == 'UH25_30MIN':
        respvar_gr = 'UH_30SWT25_WRF'
        
    if respvar_sr == 'WZ25_05MIN':
        respvar_gr = 'WZ_05SWT25'
    if respvar_sr == 'WZ25_15MIN':
        respvar_gr = 'WZ_15SWT25'
    if respvar_sr == 'WZ25_30MIN':
        respvar_gr = 'WZ_30SWT25'
        
    # Define figure
    fig,ax=plt.subplots(2,2, figsize=(8.5,6.5))

    # Iterate through cases
    for i, case in enumerate(cases):
        print(case)
        # Assign axes to plot based on order of cases in list
        if case == cases[0]:
            nrow,ncol = 0,0
        if case == cases[1]:
            nrow,ncol = 0,1
        if case == cases[2]:
            nrow,ncol = 1,0
        if case == cases[3]:
            nrow,ncol = 1,1
        print('file I/O initiated')
        # Grab case specs
        wofs_casedir,file_latlons,file_coords,file_resp_coords,file_mrms_maxima,files_wofs,state_times,resptime = case_sel(case)
        print('file I/O succeeded')
        # Process date/time strings
        respmin, resphour, respday, respmonth, respyear = resptime[14:16],resptime[11:13],resptime[8:10],resptime[5:7],resptime[0:4]
        resp_time = datetime.datetime(int(respyear), int(respmonth), int(respday), int(resphour), int(respmin))
        
        # Define response files
            # storm-relative
        dir_resp_sr = dir_resp_sr
        file_resp_sr = f'{dir_resp_sr}/resp_{respvar_sr}_max_{resp_time.year}-{str(resp_time.month).zfill(2)}-' + \
                        f'{str(resp_time.day).zfill(2)}_{str(resp_time.hour).zfill(2)}_{str(resp_time.minute).zfill(2)}_00.nc'
            # ground-relative
        dir_resp_gr = f'{drive_dir}/sens_out'
        file_resp_gr = f'{dir_resp_gr}/sens_{case}/resp_{respvar_gr}_max_{resp_time.year}-{str(resp_time.month).zfill(2)}-' + \
                        f'{str(resp_time.day).zfill(2)}_{str(resp_time.hour).zfill(2)}_{str(resp_time.minute).zfill(2)}_00.nc'
        
        # Open response files
        print('Open GR response file')
        ds_resp_gr = xr.open_dataset(file_resp_gr)
        print('Success; open SR response file')
        ds_resp_sr = xr.open_dataset(file_resp_sr)
        print('Success')
        
        # Plot
            # SR vs GR response
        ax[nrow,ncol].scatter(ds_resp_gr[f'{respvar_gr}'], ds_resp_sr[f'{respvar_sr}'], c='tab:orange', lw=0.5, edgecolor='k')
            # 1:1 line
        ax[nrow,ncol].plot(np.arange(-100,800,1),np.arange(-100,800,1), c='k', zorder=0)
        
        ax[nrow,ncol].set_title(f'{resptime[:-9]} {resptime[-8:-6]}:{resptime[-5:-3]}')
        
        if 'UH' in respvar_sr:
            labelstr='UH'
            ax[nrow,ncol].set_xlim(-25,720)
            ax[nrow,ncol].set_ylim(-25,720)
        if 'WZ' in respvar_sr:
            labelstr='W$_z$'
            ax[nrow,ncol].set_xlim(-0.0005,0.0135)
            ax[nrow,ncol].set_ylim(-0.0005,0.0135)
            
        resplabel = wofunits.labels[f'{respvar_gr}']
        
        if nrow==1:
            ax[nrow,ncol].set_xlabel(f'Ground-Rel. Max. {labelstr} ({resplabel})')
        if ncol==0:
            ax[nrow,ncol].set_ylabel(f'Storm-Rel. Max. {labelstr} ({resplabel})')
    
    plt.suptitle('Storm-Relative vs. Ground-Relative Response Functions', weight='bold')
    
    plt.tight_layout()
    
    outdir = '/Users/williamfaletti/Documents/python/thesis/wofs_code/sens_outplots_center/'
    
    plt.savefig(f'{outdir}/resp_compare_sr_gr_{labelstr}.jpg', dpi=250)
    
    plt.show()


    
def avg_regression_center(case, statevars, respvar='UH25-05MIN', if_region='if', pthres=0.05, save=False):
    
    """
    Returns a panel plot of area-averaged linear regressions in the supercell 
    inflow region.
    
    Variables:
    ----------
    case: 12-character string of initialization date/time (in format YYYYMMDDhhmm)
    statevars: 4-string list of state variables.
    respvar: String of response variable
    if_region: String ('if' - total inflow; 'ff' - far field; 'nf' - near-field)
    pthres: Maximum p-value to consider for gridpoint averaging (between 0 and 1)
    
    """

    # Modify response string to input to some files
    resp_str = respvar.replace('-','_')
    
    # Grab case specs
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    # Define specs of the inflow domain
    ang_bound1, ang_bound2 = 120, 240 # angular bounds relative to the storm motion
    
    nf_dist = 40 # distance of near-inflow domain
    ff_dist = 120 # distance of far-inflow domain
    
    domain_lag = 0 # distance lag of domain origin to storm center in km
    
    
    # Define response file and pull out response variable
    dir_resp = '/Users/williamfaletti/Documents/python/thesis/wofs_code/respout_center'
    resp_time = datetime.datetime(int(resptime[0:4]), int(resptime[5:7]), 
                                  int(resptime[8:10]), int(resptime[11:13]), int(resptime[14:16]))
    file_resp = f'{dir_resp}/resp_{resp_str}_max_{resp_time.year}-{str(resp_time.month).zfill(2)}' + \
                f'-{str(resp_time.day).zfill(2)}_{str(resp_time.hour).zfill(2)}_{str(resp_time.minute).zfill(2)}_00.nc'
    ds_resp = xr.open_dataset(file_resp)
    resp = ds_resp[resp_str].values
    
    # Define observed MRMS storm centers to center storm on plot
    df_center = create_mrmscenter_df(file_coords, file_mrms_maxima)
    
    # Define some plotting variables
    figsize=(8,6)
    nrows,ncols=2,2
    axislabels = ['a)','b)','c)','d)']
    
    # Define PBL scheme indices
    ysu_idx = wofunits.schemeidx['ysu']
    myj_idx = wofunits.schemeidx['myj']
    mynn_idx = wofunits.schemeidx['mynn']
    
    for state_time in state_times:
        
        # Create storm-centered x and y meshgrids and distance arrays
        
            # create storm-centered meshgrid
        xmesh, ymesh = xy_mesh(case, state_time)
        
        dist = (xmesh**2 + ymesh**2)**0.5 # convert to 2D meshgrid
        
        # Define angle arrays
        
            # pull angle of storm motion
        storm_ang = 30 # make this calculate it eventually
        
            # define angle bounds for inflow domain
        angles1 = np.deg2rad(np.linspace(0,ang_bound1+storm_ang,10000)) # above 0
        angles2 = np.deg2rad(np.linspace(ang_bound2+storm_ang,360,10000)) # below 360
        
        ang_tot = calc_ang_tot(xmesh,ymesh,storm_ang)
        
        # Open WoFS file
        
        fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
        
        for i, statevar in enumerate(statevars):
            
            if statevar == statevars[0]:
                nrow,ncol = 0,0
            if statevar == statevars[1]:
                nrow,ncol = 0,1
            if statevar == statevars[2]:
                nrow,ncol = 1,0
            if statevar == statevars[3]:
                nrow,ncol = 1,1
            
            state_dir = f'{drive_dir}/WOFS_output/wofs_{case[:-4]}_{case[-4:]}'
            ds_state = xr.open_dataset(f'{state_dir}/wofs_center_i{case}_v{state_time.year}' + \
                                        f'{str(state_time.month).zfill(2)}' + \
                                        f'{str(state_time.day).zfill(2)}{str(state_time.hour).zfill(2)}' + \
                                       f'{str(state_time.minute).zfill(2)}.nc')
            
                
            # Grab p-values from sensitivity file for p-value thresholding
            sens_dir = f'{drive_dir}/sens_out'
            ds_sens = xr.open_dataset(f'{sens_dir}/sens_center_{case}/stats_{respvar}_max_{resptime[-8:]}_{statevar}_' + \
                              f'{str(state_time.hour).zfill(2)}_{str(state_time.minute).zfill(2)}_00.nc')
            state =  ds_state[f'{statevar}']
            pstat = np.abs(ds_sens.pstat)
            sens = ds_sens.sens
            
            if if_region == 'ff':
                var = np.where( (ang_tot > ang_bound1) & (ang_tot < ang_bound2) | (dist < nf_dist) | (dist > ff_dist) | \
                               (pstat > pthres), np.nan, state)
            if if_region == 'nf':
                var = np.where( (ang_tot > ang_bound1) & (ang_tot < ang_bound2) | (dist > nf_dist) | (pstat > pthres), np.nan, state)
            if if_region == 'if':
                var = np.where( (ang_tot > ang_bound1) & (ang_tot < ang_bound2) | (dist > ff_dist) | (pstat > pthres), np.nan, state)
            
                # calculate member means over respective inflow regions
            mean_var = np.nanmean(var, axis=(1,2,3))
                
                # calculate linear regressions
            m_full, b_full = np.polyfit(mean_var, resp, 1)
            m_ysu, b_ysu = np.polyfit(mean_var[ysu_idx], resp[ysu_idx], 1)
            m_myj, b_myj = np.polyfit(mean_var[myj_idx], resp[myj_idx], 1)
            m_mynn, b_mynn = np.polyfit(mean_var[mynn_idx], resp[mynn_idx], 1)
            
                # scatter plot
            ax[nrow,ncol].scatter(mean_var[ysu_idx], resp[ysu_idx], c='r')
            ax[nrow,ncol].scatter(mean_var[myj_idx], resp[myj_idx], c='b')
            ax[nrow,ncol].scatter(mean_var[mynn_idx], resp[mynn_idx], c='g')
            
                # plot regression lines
            ax[nrow,ncol].plot(mean_var, m_full*mean_var+b_full, c='k')
            ax[nrow,ncol].plot(mean_var[ysu_idx], m_ysu*mean_var[ysu_idx]+b_ysu, c='r')
            ax[nrow,ncol].plot(mean_var[myj_idx], m_myj*mean_var[myj_idx]+b_myj, c='b')
            ax[nrow,ncol].plot(mean_var[mynn_idx], m_mynn*mean_var[mynn_idx]+b_mynn, c='g')        
                
                # subplot formatting
            ax[nrow,ncol].text(0.04,0.98,axislabels[i]+' '+statevar, transform=ax[nrow,ncol].transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=1), size=12)
            
            ax[nrow,ncol].set_xlabel(f'{statevar} ({wofunits.labels[statevar]})', size=10) # x-axis label
            if ncol == 0:
                ax[nrow,ncol].set_ylabel(f'Updraft Helicity ({wofunits.labels[respvar]})', size=10) # y-axis label
            
            ax[nrow,ncol].tick_params(axis='x', labelsize=9)
            ax[nrow,ncol].tick_params(axis='y', labelsize=9)
            
            ax[nrow,ncol].set_ylim(-75, max(ds_resp[resp_str].values)+50)
            #ax[nrow,ncol].set_title(f'{statevar}')
            
        plt.suptitle(f'{int((resp_time - state_time).total_seconds()/60)}-min Lead Time', weight='bold')
        
        plt.tight_layout()
        
        if save == True:
            outdir = '/Users/williamfaletti/Documents/python/thesis/wofs_code/sens_outplots_center'
        
            plt.savefig(f'{outdir}/{case}/avg_regressions/avgregress_center_{state_time.year}{str(state_time.month).zfill(2)}' + \
                f'{str(state_time.day).zfill(2)}{str(state_time.hour).zfill(2)}_{str(state_time.minute).zfill(2)}_' + \
                f'{statevars[0]}_{statevars[1]}.jpg', bbox_inches='tight', facecolor='w', dpi = 200)
        

        
def sens_plot_gr(case, senstype, respvar, statevars, cmap='RdBu_r', xlims=(-125,170), ylims=(-150,150), barbs=True, show_domain=False, gridlines=False, save=False):
    """
    Generates 4-panel plots and saves figures displaying ensemble sensitivity at
    15-minute intervals for a chosen case until the response time.
    
    Variables:
    ----------
    case: 12-digit string corresponding to case initialization time in format: YYYYMMddHHmm
    senstype: Sensitivity type to calculate ('sens' or 'stdsens')
    respvar: Response variable string (with underscores converted to hyphens)
    statevars: List of 2 state variable strings (with underscores converted to hyphens)
    cmap: Matplotlib-supported colormap string
    barbs: Boolean for whether or not to plot wind barbs (in knots)
    show_domain: Boolean for whether or not to plot inflow domains
    gridlines: Boolean for whether or not to plot x=0 and y=0 lines
    
    """
    
    # Define shapefiles and their mapping parameters
    shapefiles = ['/Users/williamfaletti/Documents/python/thesis/mapping/us_counties_states/cb_2018_us_state_20m.shp']
    edgecolor = ['k']
    lw = [1.4]
    
    # Define case specs
    wofs_casedir, file_latlons, file_coords, file_resp_coords, file_mrms_maxima, files_wofs, state_times, resptime = case_sel(case)
    
    respvar=respvar.replace('_','-')
    
    # Define specs of the inflow domain
    show_domain = False
    gridlines = True 
    
    for state_time in state_times:
        # Process date/time strings and open necessary files
        
            # process date/time strings
        respmin, resphour, respday, respmonth, respyear = resptime[14:16],resptime[11:13],resptime[8:10],resptime[5:7],resptime[0:4]
        
        str_statemin, str_statehour, str_statemonth = '{:02d}'.format(state_time.minute), \
                                                    '{:02d}'.format(state_time.hour),'{:02d}'.format(state_time.month)
        
        inityear, initmin, inithour, initday, initmonth = case[0:4], case[10:12], case[8:10], case[6:8], case[4:6]
        
        resp_time = datetime.datetime(int(respyear), int(respmonth), int(respday), int(resphour), int(respmin))
        
        file_ref = f'{drive_dir}/WOFS_output/wofs_{inityear}{initmonth}{initday}_{inithour}' + \
                    f'{initmin}/wofs_i{inityear}{initmonth}{initday}{inithour}{initmin}_v{state_time.year}{str_statemonth}' + \
                    f'{state_time.day}{str_statehour}{str_statemin}.nc'
        
            # open netCDF files
        ds_ref = xr.open_dataset(file_ref)
        
            # define plotting variables
        cref = np.where(np.nanmean(ds_ref.REFD_MAX, axis=0) < 10, np.nan, np.nanmean(ds_ref.REFD_MAX, axis=0))[0]
        
        # Define plotting specs
        
        levels = np.arange(-60,61,5)
        ticks = levels[0::4]
        
        # Run for loop and plot
        
        nrows,ncols = 2,2
        
        fig, ax = plt.subplots(2, 2, figsize=(10,8),
                        subplot_kw={'projection': ccrs.Mercator()})
        
        for nrow in range(nrows):
            for ncol in range(ncols):
                
                if nrow==0 and ncol==0:
                    statevar=statevars[0]
                if nrow==0 and ncol==1:
                    statevar=statevars[1]
                if nrow==1 and ncol==0:
                    statevar=statevars[2]
                if nrow==1 and ncol==1:
                    statevar=statevars[3]
                
                file_sens = f'{drive_dir}/sens_out/sens_2019{initmonth}{initday}{inithour}{initmin}' + \
                            f'/stats_{respvar}_max_{resphour}_{respmin}_00_{statevar}_{str_statehour}_{str_statemin}_00.nc'
                
                ds_sens = xr.open_dataset(file_sens)
                
                lat = ds_sens.XLAT.values
                lon = ds_sens.XLONG.values
                
                if senstype == 'sens':
                    sens = ds_sens.sens
                if senstype == 'stdsens':
                    sens = ds_sens.stdsens
                
                # Plot wind barbs at level of variable
                    # if variable at upper-air mandatory level, plot wind at that level
                    # if near-surface or convective variable, plot surface wind
                try:
                    u = (np.nanmean(ds_ref[f'U{statevar[-3:]}'].values, axis=(0,1))*units.meter_per_second).to(units.knot)
                    v = (np.nanmean(ds_ref[f'V{statevar[-3:]}'].values, axis=(0,1))*units.meter_per_second).to(units.knot)
                except:
                    u = (np.nanmean(ds_ref.U10, axis=(0,1))*units.meter_per_second).to(units.knot)
                    v = (np.nanmean(ds_ref.V10, axis=(0,1))*units.meter_per_second).to(units.knot)

                # Plot sensitivity, significance hatching, and mean storm
                ax[nrow,ncol].contour(lon, lat, cref, levels=[20,35,50], colors='k', transform=ccrs.PlateCarree())
                cb = ax[nrow,ncol].contourf(lon, lat, sens, cmap=cmap, levels=levels, extend='both', transform=ccrs.PlateCarree())
                ax[nrow,ncol].contourf(lon, lat, abs(ds_sens['pstat'].values), 
                                       [0, 0.05], colors='none', hatches=['////'], extend='lower', transform=ccrs.PlateCarree())
                
                # Plot wind barbs
                if barbs == True:
                    ax[nrow,ncol].barbs(lon[5::15,5::15], lat[5::15,5::15], u.magnitude[5::15,5::15], 
                                        v.magnitude[5::15,5::15], length=5, 
                                   sizes={'spacing':0.2}, color='k', alpha=0.7, zorder=2, transform=ccrs.PlateCarree())
                
                for j in range(len(shapefiles)):
                    reader = shpreader.Reader(shapefiles[j])
                    vector_data = list(reader.geometries()) 
                    VECTOR_DATA = cfeature.ShapelyFeature(vector_data, ccrs.PlateCarree())
                    ax[nrow,ncol].add_feature(VECTOR_DATA, facecolor='none', edgecolor = edgecolor[j], lw = lw[j])
                
                # Plot
    
                if nrow == 1:
                    ax[nrow,ncol].set_xlabel('East-West Distance (km)')
                if ncol == 0:
                    ax[nrow,ncol].set_ylabel('North-South Distance (km)')
                    
                ax[nrow,ncol].set_title(statevar)
                
                ax[nrow,ncol].set_extent([-102.75,-98.25,38.6,41.9])
                
                #if gridlines == True:
                #    ax[nrow,ncol].axhline(y=0, linestyle='--', color='dimgray', zorder=1) # y-axis
                #    ax[nrow,ncol].axvline(x=0, linestyle='--', color='dimgray', zorder=1) # x-axis
                
                cbar = fig.colorbar(cb, ax=ax[nrow,ncol], ticks=ticks)
                
                if senstype == 'sens':
                    sensstr = 'Sensitivity'
                    cbar.set_label(label=f'{sensstr} [({wofunits.labels[respvar]}) ({wofunits.labels[statevar]})$^{{-1}}$]', size=8)
                if senstype == 'stdsens':
                    sensstr = 'Std. Sens.'
                    cbar.set_label(label=f'{sensstr} ({wofunits.labels[respvar]})', size=8)
                
                rect = plt.Rectangle((-100.25, 40), 1.5, 1.5, edgecolor='k', facecolor='None', 
                        alpha=0.8, linewidth=4, ls = '--', transform=ccrs.PlateCarree(), zorder=999)
                #ax[nrow,ncol].add_patch(rect)
                
        #plt.suptitle(f'{sensstr} of {respvar} at {int((resp_time - state_time).seconds/60)}-min lead time', weight='bold')
        plt.suptitle(f'{int((resp_time - state_time).seconds/60)}-min lead time', weight='bold')
        plt.tight_layout()
        
        outdir = '/Users/williamfaletti/Documents/python/thesis/wofs_code/sens_outplots_gr'
        
        if save == True:
            
            outdir = f'{outdir}/{case}/{statevars[0]}_{statevars[1]}_{statevars[2]}_{statevars[3]}_{respvar}'
            
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            
            plt.savefig(f'{outdir}/{senstype}_{respvar}_{state_time.year}{str_statemonth}{state_time.day}{str_statehour}' +\
                    f'{str_statemin}_{statevars[0]}_{statevars[1]}_{statevars[2]}_{statevars[3]}.jpg', 
                        bbox_inches='tight', facecolor='w', dpi = 200)
        
    plt.show()
    

def plot_domain(storm_ang, ang_bounds, if_region='if', lw=2, color='k', nf_dist=40, ff_dist=120, domain_lag=0, dashes=(5, 1.8), subplot=False, ax=None):
    """
    Given inflow domain specs for storm-relative x/y meshgrids (in km), plots domain bounds for
    near- and far-inflow.
    
    Variables:
    ----------
    storm_ang: Angle (in degrees) of storm motion counterclockwise from the positive x-axis (float)
    ang_bounds: Angles (in degrees) of angular domain bounds counterclockwise from the positive 
                x-axis (tuple)
    if_region: Inflow region to plot (str: 'nf', 'ff', or 'if')
    nf_dist: Radius (in km) from the mesocyclone defining the near-inflow region (float)
    ff_dist: Radius (in km) from the mesocyclone defining the far-inflow region (float)
    domain_lag: Distance (in km) to lag the domain center behind the mesocyclone, relative
                to its forward motion (float)
    ax: Determines whether function is being used for singular plot or subplot (tuple)
    ax_idx: If ax=True, this is the index value of the subplot being iterated through (int)
    
    """
    
    ### Compute domain bounds from given specs ###
    
        # define angle bounds
    angles1 = np.deg2rad(np.linspace(0,ang_bounds[0]+storm_ang,10000)) # above 0
    angles2 = np.deg2rad(np.linspace(ang_bounds[1]+storm_ang,360,10000)) # below 360
        # gives true distance in x and y to lag
    domain_lag_x = domain_lag*np.cos(np.deg2rad(storm_ang))
    domain_lag_y = domain_lag*np.sin(np.deg2rad(storm_ang))
    
        # gives number of gridpoints in x and y to lag
    domain_lag_gridx = int(round(domain_lag_x/3))
    domain_lag_gridy = int(round(domain_lag_y/3))
    
    ### Now plot ###
        # if single plot
    if subplot == False: 
        
        if if_region == 'nf':
                    # near-field
            plt.plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--',dashes=dashes) # above 0
            plt.plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                    # plot angle bound lines
            plt.plot([-domain_lag_x, (nf_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (nf_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot([-domain_lag_x, (nf_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (nf_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360
            
        elif if_region == 'if':
                # far-field
            plt.plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # plot angle bound lines
            plt.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360
        elif if_region == 'both':
                # far-field
            plt.plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # near-field
            plt.plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # plot angle bound lines
            plt.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            plt.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360
        # if subplot
    if subplot == True: 
        
        if if_region == 'nf':
                    # near-field
            ax.plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                    # plot angle bound lines
            ax.plot([-domain_lag_x, (nf_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (nf_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot([-domain_lag_x, (nf_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (nf_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360
            
        elif if_region == 'if':
                # far-field
            ax.plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # plot angle bound lines
            ax.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360
        elif if_region == 'both':
                # far-field
            ax.plot((ff_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot((ff_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (ff_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # near-field
            ax.plot((nf_dist+domain_lag_x)*np.cos(angles1)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles1)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot((nf_dist+domain_lag_x)*np.cos(angles2)-domain_lag_x, (nf_dist+domain_lag_y)*np.sin(angles2)-domain_lag_y, color = color,lw=lw, ls='--', dashes=dashes) # below 360
                # plot angle bound lines
            ax.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles1[-1])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles1[-1])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # above 0
            ax.plot([-domain_lag_x, (ff_dist+domain_lag_x)*np.cos(angles2[0])-domain_lag_x], 
                     [-domain_lag_y, (ff_dist+domain_lag_y)*np.sin(angles2[0])-domain_lag_y], 
                     color = color,lw=lw, ls='--', dashes=dashes) # below 360