def extract_z_slice(cell_info):
    a = np.array(cell_info_data['slice'])
    z_slice = []
    for item in a:
        for inner_item in item:
            z_slice.append(inner_item[0])
    df = pd.DataFrame(z_slice)
    return (df)

'''
This function collects data from the 2 Photon Calcium analysis using Isaac Bianco's 
Inputs: 
1. metaFile = data structure in the form of a nested dictionary
2. traces = either z scores (roi_tsz) or calcium traces (roi_ts)

Outputs:
A list of lists containing the required data from all the z slices (4 slices in the case of our data)
'''
def get_calcium_data(metaFile, traces = 'roi_tsz'):
    zscore_access = metaFile['gmROIts']['z']
    files = []
    for file in zscore_access:
        files.extend(file[traces])
    return files

'''
This function collects data from the 2 Photon Calcium analysis using Isaac Bianco's pipeline.

1. metaFile = data structure in the form of a nested dictionary
2. traces = either z scores (roi_tsz) or calcium traces (roi_ts)

Outputs:
A list of lists containing the required data from all the z slices (4 slices in the case of our data)
'''
def get_behaviour_data(behavoir_data, data_of_interest = 'vis_traj_tt'):
    behave = behavoir_data['gmbt']['p']
    for data in behave:
        return data[data_of_interest]


def mean_activity(data, amp=3):
    """
    This function returns the average activity of all the neurones in the dataset. 
    Inputs: No arguments are needed. It uses the dictionary (Matlab structure) to extract all the zscores per neurone for the detected spike
    any spike with a zscore higher than 2.3 is considered a spike. The sum of spikes from all neurones are calculated and divided
    by the number of neurones to get the mean activity
    Output:
    Mean as one value and this can be taken to be compared across the different conditions
    """
    zslice_data = get_calcium_data(data, traces = 'roi_tsz')
    count = 0 
    total = len(zslice_data[0])
    for Slice in zslice_data:
        for neuron in Slice:
            for amplitude in neuron:
                if amplitude >= amp:
                    count += 1
    mean = count/total
    return mean
def extract_roi(data):
    '''
        This function extracts the activity by frame for each stimulus presented to each neuron. 
        Inputes: gmrxanat from the Bianco preprocessing pipeline
        output: a 2d matrix of nuerones by stimulus presented across all the frames
    '''
    activity = []
    for roi in range(len(data['gmrxanat']['roi'])):
        roit = data['gmrxanat']['roi'][roi]['Vprofiles']['meanprofile']
        temp_roi = np.array(roit)
        temp_stim = []
        for stim in temp_roi[0:17]:
            temp_stim.extend(stim)
        activity.append(temp_stim)
    return (activity)

def extract_signal_by_stimulus(data, stim=1):
    '''
        This function extracts the activity by frame for each stimulus presented to each neuron. 
        Inputes: gmrxanat from the Bianco preprocessing pipeline
        output: a 2d matrix of nuerones by stimulus presented across all the frames
    '''
    nuerones = []
    for roi in range(0,len(data['gmrxanat']['roi'])):
            nuerones.append(data['gmrxanat']['roi'][roi]['Vprofiles']['meanprofile'][stim])
    return (nuerones)

def extract_signal_by_stimulus_(data, stim=1, traces = 'roi_tsz'):
    '''
        This function extracts the activity by frame for each stimulus presented to each neuron. 
        Inputes: gmrxanat from the Bianco preprocessing pipeline
        output: a 2d matrix of nuerones by stimulus presented across all the frames
    '''
    nuerones = []
    for roi in range(0,len(data['gmROIts']['z'])):
            nuerones.append(data['gmrxanat']['z'][roi]['Vprofiles']['meanprofile'][stim])
    return (nuerones)

def estimate_zscore():
    pass
