import csv
import numpy as np

def data(file_path, complete_zeros):
    """
    Parses and processes breakthrough curve (BTC) data from a CSV file.
    
    The function extracts time series and concentration data for multiple BTCs 
    from a structured CSV file. It filters BTCs based on specific metadata conditions 
    and optionally extends each BTC with zero-padding before and after the observed data 
    to ensure consistent temporal coverage.
    
    Parameters:
        file_path (str): Path to the CSV file containing BTC data in a tabular format.
        complete_zeros (bool): If True, extends the BTCs with zero values before and after 
                               the observation period using interpolation. If False, returns 
                               the trimmed, background-corrected data.
    
    Returns:
        n_btcs (int): Number of valid BTCs processed.
        ts_m (list of np.ndarray): Time arrays for each BTC (possibly extended).
        btcs_m (list of np.ndarray): Concentration arrays for each BTC, background-corrected.
        xs_m (list of float): Streamwise positions (or equivalent spatial data) for each BTC.
        names_list (list of str): List of BTC identifiers or names from the data.
    """
    
    with open(file_path, 'r') as file:
        data_reader = csv.reader(file)
        data_mat = [[cell for cell in row] for row in data_reader]
        breakthrough_curves_data = [[[data_mat[i][i_btc*5+j] for i in range(len(data_mat))] for j in range(4)] for i_btc in range(len(data_mat[0])//5)]
    
    ts_m = [np.array([float(c) for c in btc[0][33:] if c!='']) for btc in breakthrough_curves_data if len([float(c) for c in btc[3][33:] if c!=''])>0 and btc[2][18] not in ['','0'] and btc[2][26] in ['','0','0.0']]
    btcs_m = [np.array([float(c) for c in btc[3][33:] if c!='']) for btc in breakthrough_curves_data if len([float(c) for c in btc[3][33:] if c!=''])>0 and btc[2][18] not in ['','0'] and btc[2][26] in ['','0','0.0']]
    xs_m = [btc[2][18] for btc in breakthrough_curves_data if len([float(c) for c in btc[3][33:] if c!=''])>0 and btc[2][18] not in ['','0'] and btc[2][26] in ['','0','0.0']]
    xs_m = [float(i) if i!='' else 0.0 for i in xs_m]
    background_cs_m = [btc[2][25] for btc in breakthrough_curves_data if len([float(c) for c in btc[3][33:] if c!=''])>0 and btc[2][18] not in ['','0'] and btc[2][26] in ['','0','0.0']]
    background_cs_m = [float(i) if i!='' else 0.0 for i in background_cs_m]
    names_list = [btc[0][31] for btc in breakthrough_curves_data if len([float(c) for c in btc[3][33:] if c!=''])>0 and btc[2][18] not in ['','0'] and btc[2][26] in ['','0','0.0']]
    n_btcs = len(ts_m)
    
    for i_btc in range(n_btcs):
        if len(ts_m[i_btc])<len(btcs_m[i_btc]):
            btcs_m[i_btc] = btcs_m[i_btc][:len(ts_m[i_btc])]
    T_ = int(max([t[-1]/t[np.argmax(btc)] for t,btc in zip(ts_m,btcs_m)]))+1
    
    if complete_zeros:
        ts_m_0 = []
        btcs_m_0 = []
        for i_btc in range(n_btcs):
            t,btc = ts_m[i_btc],btcs_m[i_btc]
            sorted_indices = np.argsort(t)
            t = t[sorted_indices]
            btc = btc[sorted_indices]
            t_unique,indices_unique = np.unique(t, return_index=True)
            if len(t_unique)!=len(t):
                btc_unique = np.zeros(len(indices_unique))
                for i in range(len(indices_unique)):
                    if i<len(indices_unique)-1:
                        btc_unique[i] = np.mean(btc[indices_unique[i]:indices_unique[i+1]])
                    else:
                        btc_unique[i] = np.mean(btc[indices_unique[i]:])
                t = t_unique
                btc = btc_unique
            deltat = np.max(t[1:]-t[:-1])
            t_ext = np.concatenate((np.arange(t[0],0,-deltat)[1:][::-1],t,np.arange(t[-1],T_*t[np.argmax(btc)],deltat)[1:]))
            btc_ext = np.interp(t_ext,t,btc-background_cs_m[i_btc],left = 0.0,right = 0.0)
            btc_ext = btc_ext
            ts_m_0.append(t_ext)
            btcs_m_0.append(btc_ext)
        return n_btcs, ts_m_0, btcs_m_0, xs_m, names_list
    
    else:
        return n_btcs, ts_m, [btcs_m[i_btc]-background_cs_m[i_btc] for i_btc in range(n_btcs)], xs_m, names_list