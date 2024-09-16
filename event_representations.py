import numpy as np
from models import EffWNet
import torch


# assume polarites from {0, 1}
def events_to_voxel_grid(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240), temporal_bins=5):
    """
    Convert event-based data into a voxel grid representation.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_timestamps: Array of timestamps of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :param temporal_bins: Number of temporal bins for the voxel grid
    :return: A voxel grid of shape (temporal_bins, height, width)
    """
    
    # Initialize an empty voxel grid with given temporal bins and resolution
    voxel_grid = np.zeros((temporal_bins, resolution[1], resolution[0]), dtype=np.float32)

    # Extract the first and last timestamps
    timestamps = event_timestamps
    first_stamp = timestamps[0]
    last_stamp = timestamps[-1]
    deltaT = last_stamp - first_stamp
    if deltaT == 0:
        deltaT = 1.0  # Prevent division by zero if all timestamps are the same

    # Normalize timestamps to the range [0, temporal_bins - 1]
    normalized_timestamps = (temporal_bins - 1) * (timestamps - first_stamp) / deltaT

    # Flatten the voxel grid for easier indexing
    voxel_grid_flat = voxel_grid.ravel()

    # Convert normalized timestamps to integers (floor) and calculate the decimal part
    tis = normalized_timestamps.astype(int)
    dts = normalized_timestamps - tis

    # Process event polarities: convert {0, 1} to {-1, 1}
    polarities = event_polarities
    polarities = polarities.astype(int) * 2 - 1  # If polarities are from {-1, 1}, comment this line

    # Convert coordinates to integer indices
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)

    # Calculate left and right contributions for bilinear interpolation
    vals_left = polarities * (1.0 - dts)
    vals_right = polarities * dts

    # Apply contributions to the voxel grid
    valid_indices = tis < temporal_bins
    np.add.at(voxel_grid_flat, x_coords[valid_indices] + y_coords[valid_indices] * resolution[0]
              + tis[valid_indices] * resolution[0] * resolution[1], vals_left[valid_indices])

    valid_indices = (tis + 1) < temporal_bins
    np.add.at(voxel_grid_flat, x_coords[valid_indices] + y_coords[valid_indices] * resolution[0]
              + (tis[valid_indices] + 1) * resolution[0] * resolution[1], vals_right[valid_indices])

    # Reshape the flat voxel grid back to the original shape
    voxel_grid = np.reshape(voxel_grid_flat, (temporal_bins, resolution[1], resolution[0]))

    return voxel_grid


def events_to_two_channel_histogram(event_xs, event_ys, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into a two-channel histogram representation.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :return: A two-channel histogram of shape (2, height, width)
    """
    # Initialize an empty two-channel histogram
    histogram = np.zeros((2, resolution[1], resolution[0]), dtype=np.float32)

    # Convert coordinates to integer indices
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)

    # Separate positive and negative events
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0

    # Count positive events
    np.add.at(histogram[0], (y_coords[positive_events], x_coords[positive_events]), 1)

    # Count negative events
    np.add.at(histogram[1], (y_coords[negative_events], x_coords[negative_events]), 1)

    return histogram


def events_to_four_channel_representation(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into a four-channel representation.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_timestamps: Array of timestamps of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :return: A four-channel representation of shape (4, height, width)
    """
    # Initialize the four-channel representation
    representation = np.zeros((4, resolution[1], resolution[0]), dtype=np.float32)

    # Convert coordinates to integer indices
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)

    # Separate positive and negative events
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0

    # Channel 0: Count of positive events
    np.add.at(representation[0], (y_coords[positive_events], x_coords[positive_events]), 1)

    # Channel 1: Count of negative events
    np.add.at(representation[1], (y_coords[negative_events], x_coords[negative_events]), 1)

    # Normalize timestamps to [0, 1]
    normalized_timestamps = (event_timestamps - event_timestamps.min()) / (event_timestamps.max() - event_timestamps.min())

    # Channel 2: Most recent positive event timestamp
    np.maximum.at(representation[2], (y_coords[positive_events], x_coords[positive_events]), normalized_timestamps[positive_events])

    # Channel 3: Most recent negative event timestamp
    np.maximum.at(representation[3], (y_coords[negative_events], x_coords[negative_events]), normalized_timestamps[negative_events])

    return representation



def events_to_ev_surface(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240), time_window=1.0):
    """
    Convert event-based data into an EvSurface representation.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_timestamps: Array of timestamps of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :param time_window: Time window for event integration (in seconds)
    :return: An EvSurface representation of shape (4, height, width)
    """
    # Initialize the EvSurface representation
    ev_surface = np.zeros((4, resolution[1], resolution[0]), dtype=np.float32)

    # Convert coordinates to integer indices
    x_coords = event_xs.astype(int)
    y_coords = event_ys.astype(int)

    # Separate positive and negative events
    positive_events = event_polarities == 1
    negative_events = event_polarities == 0

    # Calculate the start time for the integration window
    start_time = event_timestamps[-1] - time_window

    # Channels 0 and 1: Integrated event count within the time window
    valid_events = event_timestamps >= start_time
    np.add.at(ev_surface[0], (y_coords[valid_events & positive_events], x_coords[valid_events & positive_events]), 1)
    np.add.at(ev_surface[1], (y_coords[valid_events & negative_events], x_coords[valid_events & negative_events]), 1)

    # Normalize timestamps to [0, 1]
    normalized_timestamps = (event_timestamps - start_time) / time_window
    normalized_timestamps = np.clip(normalized_timestamps, 0, 1)

    # Channel 2: Most recent positive event timestamp
    np.maximum.at(ev_surface[2], (y_coords[positive_events], x_coords[positive_events]), normalized_timestamps[positive_events])

    # Channel 3: Most recent negative event timestamp
    np.maximum.at(ev_surface[3], (y_coords[negative_events], x_coords[negative_events]), normalized_timestamps[negative_events])

    return ev_surface





def events_to_EvRep(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240)):
    """
    Convert event-based data into an EvRep representation using more efficient matrix operations.

    :param event_xs: Array of x-coordinates of events
    :param event_ys: Array of y-coordinates of events
    :param event_timestamps: Array of timestamps of events
    :param event_polarities: Array of polarities of events (assumed to be from {0, 1})
    :param resolution: Tuple (width, height) representing the resolution of the output grid
    :return: An EvRep representation of shape (3, height, width)
    """
    width, height = resolution

    # Initialize the three channels: spatial, polarity, and temporal
    E_C = np.zeros((height, width), dtype=np.int32)  # Event spatial channel
    E_I = np.zeros((height, width), dtype=np.int32)  # Event polarity channel
    E_T_sum = np.zeros((height, width), dtype=np.float32)  # For sum of timestamp deltas
    E_T_sq_sum = np.zeros((height, width), dtype=np.float32)  # For sum of squared deltas

    # Normalize event polarities to {-1, 1}
    event_polarities = np.where(event_polarities == 0, -1, event_polarities)

    # Bin events into the grid (spatial and polarity channels)
    np.add.at(E_C, (event_ys, event_xs), 1)  # Count of events at each pixel
    np.add.at(E_I, (event_ys, event_xs), event_polarities)  # Net polarity of events at each pixel

    # Sort events by pixel for temporal statistics (approximation using binning)
    sort_indices = np.lexsort((event_timestamps, event_ys, event_xs))
    sorted_xs = event_xs[sort_indices]
    sorted_ys = event_ys[sort_indices]
    sorted_timestamps = event_timestamps[sort_indices]

    # Calculate deltas for consecutive events at each pixel
    delta_timestamps = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])

    # Efficient temporal processing by binning consecutive deltas
    np.add.at(E_T_sum, (sorted_ys, sorted_xs), delta_timestamps)
    np.add.at(E_T_sq_sum, (sorted_ys, sorted_xs), delta_timestamps**2)

    # Calculate standard deviation for temporal channel
    E_T_counts = E_C.clip(min=1)  # Avoid division by zero
    delta_mean = E_T_sum / E_T_counts
    E_T = np.sqrt(np.maximum((E_T_sq_sum / E_T_counts) - delta_mean**2, 0))

    # Stack the channels to form the EvRep representation
    EvRep = np.stack([E_C, E_I, E_T], axis=0)

    return EvRep




def load_RepGen(device="cuda"):
    # RepGen assume batchfied data B x 3 x H x W

    model = EffWNet(n_channels=3, out_depth=1, inc_f0=1, bilinear=True, n_lyr=4, ch1=12, c_is_const=False, c_is_scalar=False, device=device)

    model_path = "RepGen.pth"


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    model.to(device=device)

    return model



def EvRep_to_EvRepSL(model, ev_rep, device="cuda"):

    ev_rep = torch.tensor(ev_rep, dtype=torch.float32).to(device=device)
    
    evrepsl = model(ev_rep)
    return evrepsl




if __name__ == "__main__":
    # Generate sample data
    num_events = 500000 
    resolution = (320, 240)
    event_xs = np.random.randint(0, 320, num_events)
    event_ys = np.random.randint(0, 240, num_events)
    event_timestamps = np.sort(np.random.rand(num_events))
    event_polarities = np.random.randint(0, 2, num_events)

    # EvRep representation
    ev_rep = events_to_EvRep(event_xs, event_ys, event_timestamps, event_polarities, resolution)

    print("EvRep Representation was generated!")


    # EvRepSL representation
    # RepGen assume batchfied data B x 3 x H x W

    ev_rep = np.expand_dims(ev_rep, axis=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_RepGen(device)
    
    ev_rep_sl = EvRep_to_EvRepSL(model, ev_rep, device)

    print("EvRepSL Representation was generated!")