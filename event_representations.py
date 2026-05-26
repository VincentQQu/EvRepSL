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


def events_to_tore(event_xs, event_ys, event_timestamps, event_polarities, resolution=(320, 240), K=5, tau=5e6, tau_min=150):
    """
    Compute the TORE volume from event data.

    Args:
        event_xs (array-like): x coordinates of events.
        event_ys (array-like): y coordinates of events.
        event_timestamps (array-like): Timestamps for each event.
        event_polarities (array-like): Polarities for each event (assumed >0 for positive, <=0 for negative).
        resolution (tuple): (width, height) of the sensor.
        K (int): Number of recent events to store per pixel per polarity.
        tau (float): Maximum time threshold (for clipping).
        tau_min (float): Minimum time sensitivity threshold.
        
    Returns:
        np.ndarray: TORE volume with shape (2, K, height, width)
    """
    # Unpack resolution. Here resolution is assumed as (width, height).
    width, height = resolution

    # Initialize FIFO for each pixel and polarity.
    # We'll use shape (2, height, width, K): 2 for polarities, last dimension for the FIFO.
    fifo = np.full((2, height, width, K), -np.inf)
    
    num_events = len(event_xs)
    
    # Process each event in order (assuming events are sorted by time)
    for i in range(num_events):
        x = int(event_xs[i])
        y = int(event_ys[i])
        t = event_timestamps[i]
        pol = event_polarities[i]
        # Use index 0 for positive events, 1 for negative events.
        p_idx = 0 if pol > 0 else 1
        
        # Update the FIFO for pixel (y, x) and polarity p_idx.
        # Shift existing values one position to the right (older events move right)
        fifo[p_idx, y, x, 1:] = fifo[p_idx, y, x, :-1]
        # Insert the new timestamp at the front.
        fifo[p_idx, y, x, 0] = t

    # Define current time as the maximum timestamp (or could be provided externally)
    current_time = max(event_timestamps)
    
    # Compute time differences: dt = current_time - FIFO + 1.
    # For any -inf values (i.e. pixels that never fired), the difference becomes infinity.
    dt = current_time - fifo + 1
    # Clamp dt from below by tau_min to avoid very small values (or -inf issues).
    dt = np.maximum(dt, tau_min)
    
    # Compute the logarithm of the time differences.
    log_dt = np.log(dt)
    
    # Clip the log values to lie between log(tau_min) and log(tau) as defined in the paper.
    log_dt = np.clip(log_dt, np.log(tau_min), np.log(tau))
    
    # Rearrange dimensions to obtain a TORE volume of shape: (2, K, height, width)
    volume = np.transpose(log_dt, (0, 3, 1, 2))
    return volume



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


# ---------------------------------------------------------------------------
# Learned generators: PIE-Net / PIE-Net-Lite (PIEM representation)
# Requires: pip install event-pienet
# ---------------------------------------------------------------------------

_piem_models = {}


def load_PIEM_generator(variant="pie-net", device="cuda", pretrained=True):
    """
    Load a cached PIE-Net or PIE-Net-Lite representation generator.

    Args:
        variant: "pie-net" (full) or "pie-net-lite" (lite)
        device: "cuda" or "cpu"
        pretrained: load shipped checkpoint weights
    """
    try:
        from pie_net import load_model, resolve_variant
    except ImportError as exc:
        raise ImportError(
            "PIE-Net generators require event-pienet. Install with: pip install event-pienet"
        ) from exc

    key = resolve_variant(variant)
    if key not in _piem_models:
        _piem_models[key] = load_model(pretrained=pretrained, device=device, variant=key)
        _piem_models[key].eval()
    return _piem_models[key]


def load_PIENet(device="cuda", pretrained=True):
    """Load PIE-Net (full model)."""
    return load_PIEM_generator(variant="pie-net", device=device, pretrained=pretrained)


def load_PIENetLite(device="cuda", pretrained=True):
    """Load PIE-Net-Lite (faster, smaller)."""
    return load_PIEM_generator(variant="pie-net-lite", device=device, pretrained=pretrained)


def reset_piem_states(variant=None):
    """Reset streaming state. Call between independent event sequences."""
    if variant is None:
        for model in _piem_models.values():
            model.reset_states()
        return

    from pie_net import resolve_variant

    model = _piem_models.get(resolve_variant(variant))
    if model is not None:
        model.reset_states()


def voxel_to_PIEM_representation(voxel, model=None, variant="pie-net", device="cuda"):
    """
    Map a 5-bin voxel grid to a PIEM representation.

    Returns a dict with individual PIEM maps and a stacked 5-channel tensor:
        mean_exp_z, var_exp_z, k, mean_f1, var_f1  — each [1, H, W]
        piem  — stacked [5, H, W] representation for downstream tasks

    Channel order in piem:
        0: mean_exp_z  (expected log-intensity change)
        1: var_exp_z   (uncertainty of Z)
        2: k           (PIEM scaling parameter)
        3: mean_f1     (reconstructed intensity)
        4: var_f1      (frame uncertainty)
    """
    from pie_net import stack_piem_representation

    if model is None:
        model = load_PIEM_generator(variant=variant, device=device)

    if not torch.is_tensor(voxel):
        voxel = torch.tensor(voxel, dtype=torch.float32, device=device)
    else:
        voxel = voxel.to(device=device, dtype=torch.float32)

    if voxel.dim() == 3:
        voxel = voxel.unsqueeze(0)

    with torch.inference_mode():
        out = model(voxel)

    piem = stack_piem_representation(out).squeeze(0).cpu().numpy()
    return {
        "mean_exp_z": out["mean_exp_z"].squeeze(0).cpu().numpy(),
        "var_exp_z": out["var_exp_z"].squeeze(0).cpu().numpy(),
        "k": out["k"].squeeze(0).cpu().numpy(),
        "mean_f1": out["mean_f1"].squeeze(0).cpu().numpy(),
        "var_f1": out["var_f1"].squeeze(0).cpu().numpy(),
        "piem": piem,
    }


def events_to_PIEM_representation(
    event_xs,
    event_ys,
    event_timestamps,
    event_polarities,
    resolution=(320, 240),
    temporal_bins=5,
    variant="pie-net",
    device="cuda",
    model=None,
):
    """End-to-end: raw events -> voxel grid -> PIEM representation [5, H, W]."""
    voxel = events_to_voxel_grid(
        event_xs,
        event_ys,
        event_timestamps,
        event_polarities,
        resolution=resolution,
        temporal_bins=temporal_bins,
    )
    return voxel_to_PIEM_representation(
        voxel,
        model=model,
        variant=variant,
        device=device,
    )


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

    # PIEM representation (PIE-Net / PIE-Net-Lite)
    try:
        piem_rep = events_to_PIEM_representation(
            event_xs,
            event_ys,
            event_timestamps,
            event_polarities,
            resolution=resolution,
            variant="pie-net",
            device=device,
        )
        print("PIEM representation was generated!", piem_rep["piem"].shape)
    except ImportError:
        print("Skipping PIEM demo (install with: pip install event-pienet)")