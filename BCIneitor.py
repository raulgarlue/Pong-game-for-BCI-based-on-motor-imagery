import mne
import mne_lsl.stream_viewer
import numpy as np
import mne_lsl
import time
import re
from threading import Thread
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from scipy import signal
import pywt


# Custom transformer for band power features
class BandPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, 
                 #bands=[(8, 12), (12, 16), (16,20), (20, 24), (24,30)]
                 bands=[(8,12), (12,20), (20,30)]
                 #bands=[(8,12), (12,30)]
                 #bands=[(8, 12), (12, 16), (16,20), (20, 24), (24,30), (30, 38), (38, 45)]
                 ):
        self.sfreq = sfreq
        self.bands = bands
        

    def fit(self, X, y=None):
        return self
        

    def transform(self, X):
        """
        X shape: (n_epochs, n_channels, n_samples)
        Returns: (n_epochs, n_channels * n_bands)
        """
        features = []
        
        for band in self.bands:            

            # Compute PSD using Welch's method
            freqs, psd = welch(X, self.sfreq, nperseg=self.sfreq//2, axis=2)
            
            # Select frequency indices corresponding to the band
            band_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
            
            # Compute mean power within the band
            band_power = np.mean(psd[:, :, band_idx], axis=2)
            
            # Apply log transformation
            log_band_power = np.log1p(band_power)
            
            features.append(log_band_power)

        return np.concatenate(features, axis=1)


class BandPowerTransformer_3(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, 
                 #bands=[(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)],
                 bands=[(4,12), (12,20), (20,30)],
                 log_transform=True,
                 normalize=True):
        self.sfreq = sfreq
        self.bands = bands
        self.log_transform = log_transform
        self.normalize = normalize
        self.baseline = None

    def fit(self, X, y=None):
        if self.normalize:
            # Calculate baseline power for normalization
            self.baseline = np.zeros((len(self.bands), X.shape[1]))
            for i, band in enumerate(self.bands):
                filtered_data = mne.filter.filter_data(
                    X, self.sfreq, band[0], band[1], method='fir', verbose=False)
                self.baseline[i] = np.mean(np.mean(filtered_data ** 2, axis=2), axis=0)
        return self

    def transform(self, X):
        """
        X shape: (n_epochs, n_channels, n_samples)
        Returns: (n_epochs, n_channels * n_bands)
        """
        features = []
        
        for i, band in enumerate(self.bands):
            # Filter data for each band (using FIR filter for better phase response)
            filtered_data = mne.filter.filter_data(
                X, self.sfreq, band[0], band[1], method='fir', verbose=False)
            
            # Calculate band power (mean of squared values)
            power = np.mean(filtered_data ** 2, axis=2)
            
            # Apply log transform if requested
            if self.log_transform:
                # Add small constant to avoid log(0)
                power = np.log(power + 1e-10)
            
            # Normalize by baseline if requested
            if self.normalize and self.baseline is not None:
                power = power / (self.baseline[i] + 1e-10)
            
            features.append(power)
        
        # Combine all band powers into a single feature vector
        return np.concatenate(features, axis=1)


class BandPowerTransformer_z(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, 
                 #bands=[(8, 12), (12, 16), (16,20), (20, 24), (24,30)]
                 bands=[(8,12), (12,20), (20,30)]
                 ):
        self.sfreq = sfreq
        self.bands = bands
        

    def fit(self, X, y=None):
        return self
        

    def transform(self, X):
        """
        X shape: (n_epochs, n_channels, n_samples)
        Returns: (n_epochs, n_channels * n_bands)
        """
        features = []
        
        for band in self.bands:
            # Filter data for each band
            filtered_data = mne.filter.filter_data(
                X, self.sfreq, band[0], band[1], method='iir', verbose=False)
            
            # Calculate band power (mean of squared values)
            power = np.mean(filtered_data ** 2, axis=2)
            features.append(power)
        
        # Combine all band powers into a single feature vector
        return np.concatenate(features, axis=1)




# Custom transformer for CSP features
class CSPTransformer_z(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, n_components=2, freq_band=(8, 30)):
        self.sfreq = sfreq
        self.n_components = n_components    # Must be smaller than number of channels
        self.freq_band = freq_band
        
        

    def fit(self, X, y=None):
        # X shape: (n_epochs, n_channels, n_samples)
        n_channels = X.shape[1]

        # Adjust n_components to be at most n_channels
        actual_components = min(self.n_components,n_channels)

        # Create CSP object with adjusted number of components
        self.csp = mne.decoding.CSP(n_components=actual_components, reg=0.8, log=True, norm_trace=False)
        
        # Filter data in the motor imagery frequency band (8-30 Hz)
        X_filtered = mne.filter.filter_data(
            X, self.sfreq, self.freq_band[0], self.freq_band[1], method='iir', verbose=False)
        
        # Fit CSP on filtered data
        self.csp.fit(X_filtered, y)
        return self
        

    def transform(self, X):
        # Filter and apply CSP transformation
        X_filtered = mne.filter.filter_data(
            X, self.sfreq, self.freq_band[0], self.freq_band[1], method='iir', verbose=False)
        return self.csp.transform(X_filtered)
    

    def get_patterns(self):
        """Return CSP patterns for plotting"""
        return self.csp.patterns_


# Custom transformer for CSP features
class CSPTransformer_old(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, n_components=4, freq_band=(8, 30)):
        self.sfreq = sfreq
        self.n_components = n_components    # Must be smaller than number of channels
        self.freq_band = freq_band
        
        

    def fit(self, X, y=None):
        # X shape: (n_epochs, n_channels, n_samples)
        n_channels = X.shape[1]

        # Adjust n_components to be at most n_channels
        actual_components = min(self.n_components,n_channels)

        # Create CSP object with adjusted number of components
        self.csp = mne.decoding.CSP(n_components=actual_components, reg=0.8, log=True, norm_trace=False)
        
        # Filter data in the motor imagery frequency band (8-30 Hz)
        X_filtered = mne.filter.filter_data(
            X, self.sfreq, self.freq_band[0], self.freq_band[1], method='fir', verbose=False)
        
        # Fit CSP on filtered data
        self.csp.fit(X_filtered, y)
        return self
        

    def transform(self, X):
        # Filter and apply CSP transformation
        X_filtered = mne.filter.filter_data(
            X, self.sfreq, self.freq_band[0], self.freq_band[1], method='fir', verbose=False)
        return self.csp.transform(X_filtered)
    

    def get_patterns(self):
        """Return CSP patterns for plotting"""
        return self.csp.patterns_

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, n_components=4, 
                 freq_bands=[(8, 12), (12,20), (20,30)],
                 reg='ledoit_wolf', 
                 log=True):
        self.sfreq = sfreq
        self.n_components = n_components
        self.freq_bands = freq_bands
        self.reg = reg
        self.log = log
        self.csp_list = []

    def fit(self, X, y=None):
        n_channels = X.shape[1]
        actual_components = min(self.n_components, n_channels)
        
        # Create a CSP for each frequency band
        self.csp_list = []
        for band in self.freq_bands:
            csp = mne.decoding.CSP(n_components=actual_components, 
                                   reg=self.reg, 
                                   log=self.log, 
                                   norm_trace=False)
            
            # Filter data in the current frequency band
            X_filtered = mne.filter.filter_data(
                X, self.sfreq, band[0], band[1], method='fir', verbose=False)
            
            # Fit CSP on filtered data
            csp.fit(X_filtered, y)
            self.csp_list.append((band, csp))
            
        return self

    def transform(self, X):
        # Apply CSP transformation for each frequency band
        all_features = []
        for band, csp in self.csp_list:
            X_filtered = mne.filter.filter_data(
                X, self.sfreq, band[0], band[1], method='fir', verbose=False)
            features = csp.transform(X_filtered)
            all_features.append(features)
            
        # Combine features from all bands
        return np.hstack(all_features)

    def get_patterns(self, band_idx=0):
        """Return CSP patterns for a specific band index"""
        if 0 <= band_idx < len(self.csp_list):
            return self.csp_list[band_idx][1].patterns_
        else:
            raise ValueError(f"Band index {band_idx} out of range")



class BCI:
    def __init__(self):
        self.bf_duration = 10       # in Seconds
        self.sf = 500               # Sampling frequency
        self.marker_data = []       # Stores markers in the form (timestamp, type)
        self.total_epochs = []      # Stores all epochs
        #self.channel_selection = ['C3','Cz','C4']   # 3 Channels
        self.channel_selection = ['C3','Cz','C4','CP1','CP2']  # 5 Channels     # ['F3','Fz','F4','FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6','P3','Pz','P4']
        #self.channel_selection = ['C3', 'Cz', 'C4', 'Cp1', 'Cp2', 'C1', 'C2', 'Cp3', 'Cp4']    # 9 Channels
        self.min_samples = 20       # Minimum samples needed for training
        self.accuracy_history = []  # Track classification accuracy over time
        self.prediction_accuracy = [] # Percentage of accuracy tracking
        self.mode = 'training'      # Default mode: 'training' or 'prediction'
        self.true_labels = []
        self.predicted_labels = []
        mne.set_log_level("WARNING")    # To remove verbosity

    def realeeg_stream(self):
        """
        - Finds available streams
        - Connects to EEG data stream
        - Selects channels of interest
        - Filters stream
        - Creates MNE info object
        """
        # Find available streams
        print("Resolving EEG streams...")
        available_streams = mne_lsl.lsl.resolve_streams()

        time.sleep(1)

        if not available_streams:
            raise RuntimeError("No Streams Found. Ensure hardware is connected and running")
        else:
            print(available_streams)
        
        # Connect to EEG data stream
        self.eeg_stream = mne_lsl.stream.StreamLSL(
            bufsize=self.bf_duration,
            name="BrainVision RDA",
            stype="EEG",
            source_id="RDA 127.0.0.1:51244"
        )

        self.eeg_stream.connect(
            acquisition_delay=0.001,
            processing_flags="all"
        )

        time.sleep(1)

        print("Available channels:", self.eeg_stream.ch_names)

        if self.eeg_stream.connected:
            print(f"Connected to EEG stream: {self.eeg_stream._name}\n")
        else:
            print("Error when connecting to stream")

        # Select channels and apply Rereferencing
        self.eeg_stream.pick(self.channel_selection)        # Select wanted channels only
        self.eeg_stream.set_eeg_reference('average')        # Apply Common Average Referencing
        print("Selected channels: ", self.eeg_stream.ch_names)

        # Filter the stream
        self.eeg_stream.filter(0.5, 40)
        self.eeg_stream.notch_filter(50)

        # Create MNE info for epoch creation
        self.info = mne.create_info(
            ch_names=self.eeg_stream.ch_names, 
            sfreq=self.sf,
            ch_types="eeg"
        )

        #mne.rename_channels(self.info, {'Cp1':'CP1', 'Cp2':'CP2'})

    def events_stream(self):
        """
        - Finds available streams
        - Connects to Marker stream
        """
        # Find available streams
        print("Resolving Events stream...")
        available_streams = mne_lsl.lsl.resolve_streams()

        time.sleep(5)

        if not available_streams:
            raise RuntimeError("No Streams Found. Ensure hardware is connected and running.")
        else:
            print(available_streams)

        # Connect to Marker stream
        self.info_markers = mne_lsl.lsl.StreamInfo(
            name="BrainVision RDA Markers",
            stype="Markers",
            n_channels=1,
            sfreq=0,
            dtype="string",
            source_id="RDA 127.0.0.1:51244 Marker"
        )

        self.markers_stream= mne_lsl.lsl.StreamInlet(
            sinfo=self.info_markers,
            processing_flags="all"
        )

        time.sleep(1)

        print(f"Connected to Markers stream: {self.info_markers.name}")


    def check_markers(self):
        """
        - Constantly checks for incoming markers
        - If marker is found store it and create epoch
        """
        while True:
            try:
                # Check for new markers
                marker, marker_timestamp = self.markers_stream.pull_sample(timeout=0.1)
                
                if marker:
                    marker_string = marker[0]
                    print(f"\nMarker detected at {marker_timestamp:.3f}s: {marker_string}")
                    
                    self.ts_trigger = time.time()
                    # Store marker data with timestamp
                    self.marker_data.append((marker_timestamp, marker_string))
                    
                    # Create an epoch for this marker
                    epoch = self.create_epoch_for_marker(marker_timestamp, marker_string)
                    
                    # Handle the epoch based on current mode
                    if self.mode == 'training':
                        print(f"Training mode: Collected {len(self.total_epochs)} epochs so far")
                        if len(self.total_epochs) >= self.min_samples:
                            print("Sufficient samples collected. Ready for training. Switch mode to 'prediction' to start classification.")
                    
                    elif self.mode == 'prediction' and hasattr(self, 'pipeline'):
                        # Use the pipeline to predict class
                        self.predict_motor_imagery(epoch)
                
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in check_markers: {e}")
                time.sleep(0.1)


    def create_epoch_for_marker(self, marker_timestamp, marker_string):
        """
        - Collects data and creates an epoch around the marker
        """
        try:
            data, timestamp = self.eeg_stream.get_data(2)  # The number indicates the seconds of data collected

            sample_index = np.argmin(np.abs(timestamp - marker_timestamp))  # Find the closest sample to marker
            event_id = self.parse_marker(marker_string)
            events = np.array([[sample_index, 0, event_id]])

            # Set event_id dict based on marker
            if event_id == 1:
                event_id_dict = {"right": 1}
            elif event_id == 2:
                event_id_dict = {"left": 2}
            elif event_id == 3:
                event_id_dict = {"rest": 3}
            else:
                event_id_dict = {f"marker_{event_id}": event_id}

            # Set montage for proper electrode positions
            montage = mne.channels.make_standard_montage('standard_1020')
            self.info.set_montage(montage)

            # Create epoch
            epoch = mne.EpochsArray(
                data=data.reshape(1, len(self.eeg_stream.ch_names), data.shape[1]),  # Reshape data
                info=self.info,
                events=events,
                tmin=-1,
                event_id=event_id_dict,
                #reject={"eeg": 500e-6}, 
                flat={"eeg": 1e-6},
                baseline=(-1, 0) 
            )

            print("Marker Timestamp: ", marker_timestamp)
            
            epoch_start_time = timestamp[0]
            epoch_end_time = timestamp[-1]

            print(f"Epoch goes from {epoch_start_time} to {epoch_end_time}")

            # Crop to include relevant time period
            #epoch.crop(0.25, 1.25)
            
            print(f"Created epoch for marker: {marker_string}")
            print(f"Epoch shape: {epoch.get_data().shape}")
                
            # Store the epoch
            self.total_epochs.append(epoch)
            
            # Return the epoch for use in prediction mode
            return epoch
            
        except Exception as e:
            print(f"Error creating epoch: {e}")
            return None


    def parse_marker(self, marker_string):
        """Parse marker string to get numeric marker ID"""
        match = re.search(r'(\d+)$', marker_string)
        if match:
            marker_id = int(match.group(1))

            if marker_id == 1:
                return 1        # right
            elif marker_id == 2:
                return 2        # left
            elif marker_id == 3:
                return 3        # rest
            else:
                return marker_id
        return 0  # Default return if no marker ID found


    def train_models(self):
        """Train both bandpower and CSP models and compare performance"""
        if len(self.total_epochs) < self.min_samples:
            print(f"Not enough samples. Have {len(self.total_epochs)}, need {self.min_samples}")
            return False
            
        # Prepare data
        epochs_data = []
        epochs_labels = []
        
        for epoch in self.total_epochs:
            # Skip rest class if only interested in left vs right
            class_label = list(epoch.event_id.keys())[0]
            if class_label == "rest":
                continue
            
            data = epoch.get_data()
            #data = epoch.get_data()[:,:,int(1*self.sf):int(3*self.sf)]
            label = 1 if class_label == "right" else 2  # Convert to numeric
            
            epochs_data.append(data)
            epochs_labels.append(label)
        
        if len(epochs_data) < self.min_samples:
            print(f"Not enough non-rest samples. Have {len(epochs_data)}, need {self.min_samples}")
            return False
            
        X = np.concatenate(epochs_data, axis=0)  # Combine all epochs: (n_epochs, n_channels, n_samples)
        y = np.array(epochs_labels)
        
        print(f"Training on {len(epochs_labels)} epochs ({X.shape})")
        
        # Create pipelines
        bandpower_pipeline = Pipeline([
            ('bandpower', BandPowerTransformer(sfreq=self.sf, bands=[(8,12), (12,20), (20,30)])),
            ('scaler', StandardScaler()),
            ('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        
        csp_pipeline = Pipeline([
            ('csp', CSPTransformer(sfreq=self.sf, n_components=4)),
            ('scaler', StandardScaler()),
            ('classifier', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Evaluate band power approach
        bp_scores = cross_val_score(bandpower_pipeline, X, y, cv=cv)

        print(f"Band Power + LDA: {bp_scores.mean():.2f} ± {bp_scores.std():.2f}")
        
        # Evaluate CSP approach
        csp_scores = cross_val_score(csp_pipeline, X, y, cv=cv)
        
        print(f"CSP + LDA: {csp_scores.mean():.2f} ± {csp_scores.std():.2f}")
        
        # Train the final model on all data
        if csp_scores.mean() >= bp_scores.mean():
            print("Using CSP + LDA as it performed better")
            self.pipeline = csp_pipeline
            self.pipeline.fit(X, y)
            
            # For plotting CSP patterns
            self.csp = self.pipeline.named_steps['csp']
            self.plot_csp_patterns()
        else:
            print("Using Band Power + LDA as it performed better")
            self.pipeline = bandpower_pipeline
            self.pipeline.fit(X, y)
        
        self.plot_classification_accuracy(bp_scores.mean(), csp_scores.mean(), bp_scores.std(), csp_scores.std())
        self.accuracy_history.append((time.time(), max(bp_scores.mean(), csp_scores.mean())))
        
        return True


    def plot_classification_accuracy(self, bp_accuracy, csp_accuracy, bp_std, csp_std):
        """
        Plot a comparison of classification accuracies with standard deviation error bars
        
        Parameters:
        -----------
        bp_accuracy : float
            Mean accuracy for Band Power + LDA method
        csp_accuracy : float
            Mean accuracy for CSP + LDA method
        bp_std : float, optional
            Standard deviation for Band Power + LDA method
        csp_std : float, optional
            Standard deviation for CSP + LDA method
        """
        plt.figure(figsize=(8, 6))
        accuracies = [bp_accuracy, csp_accuracy]
        methods = ['Band Power + LDA', 'CSP + LDA']
        colors = ['skyblue', 'salmon']
        
        # Add error bars if standard deviations are provided
        if bp_std is not None and csp_std is not None:
            std_devs = [bp_std, csp_std]
            plt.bar(methods, accuracies, color=colors, yerr=std_devs, capsize=10, 
                    error_kw={'ecolor': 'black', 'linewidth': 2})
        else:
            plt.bar(methods, accuracies, color=colors)
        
        plt.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
        plt.ylim([0, 1])
        plt.ylabel('Classification Accuracy')
        plt.title('Comparison of Feature Extraction Methods')
        plt.legend()  # Show the legend with the chance level
        plt.tight_layout()
        plt.savefig(f"classification_comparison_{time.strftime('%Y%m%d-%H%M%S')}.png")
        plt.close()

    def plot_csp_patterns_antiguo(self):
        """Plot CSP patterns if available"""
        if hasattr(self, "csp"):
            # Get the actual number of components (limited by number of channels)
            actual_components = min(self.csp.n_components, len(self.channel_selection))

            # Adjust figure size and number of subplots based on actual components
            n_rows = int(np.ceil(actual_components/2))
            fig, axes = plt.subplots(n_rows, min(2, actual_components), figsize=(10, 4 * n_rows))
            
            if actual_components==1:
                axes = np.array([axes])
            else:
                axes = np.array(axes).flatten()

            patterns = self.csp.get_patterns()
            print(f"Patterns shape: {patterns.shape}")

            for i, ax in enumerate(axes):
                if i < actual_components:
                    mne.viz.plot_topomap(patterns[:, i],
                                         pos=self.info, 
                                         axes=ax,
                                         show=False,
                                         cmap="RdBu_r",
                                         outlines="head")
                    ax.set_title(f"CSP pattern {i+1}")

            plt.tight_layout()
            plt.savefig(f"csp_patterns_{time.strftime('%Y%m%d-%H%M%S')}.png")
            plt.close()

    def plot_csp_patterns(self):
        """Plot CSP patterns if available"""
        if hasattr(self, "csp"):
            # Get the actual number of components (limited by number of channels)
            actual_components = min(self.csp.n_components, len(self.channel_selection))

            # Adjust figure size and number of subplots based on actual components
            n_rows = int(np.ceil(actual_components/2))
            fig, axes = plt.subplots(n_rows, min(2, actual_components), figsize=(10, 4 * n_rows))
            
            if actual_components==1:
                axes = np.array([axes])
            else:
                axes = np.array(axes).flatten()

            patterns = self.csp.get_patterns()
            print(f"Patterns shape: {patterns.shape}")
            params = self.csp.get_params()
            print(f"params: {params}")
            
            # Create a subset of the info object that matches our patterns array
            # This is the key fix - we need to ensure the info object matches the patterns data
            pattern_info = mne.pick_info(self.info, np.arange(patterns.shape[0]))
            
            for i, ax in enumerate(axes):
                if i < actual_components:
                    mne.viz.plot_topomap(patterns[:, i],
                                        pos=pattern_info, 
                                        axes=ax,
                                        show=False,
                                        cmap="RdBu_r",
                                        outlines="head")
                    ax.set_title(f"CSP pattern {i+1}")

            plt.tight_layout()
            plt.savefig(f"csp_patterns_{time.strftime('%Y%m%d-%H%M%S')}.png")
            plt.close()

    def predict_motor_imagery_old(self, epoch):
        """Predict motor imagery class for a single epoch"""
        if not hasattr(self, 'pipeline'):
            print("Model not trained yet")
            return None
        
        try:
            data = epoch.get_data()
            #data = epoch.get_data()[:,:,int(1*self.sf):int(3*self.sf)]  # Shape: (1, n_channels, n_samples)
            prediction = self.pipeline.predict(data)
            
            class_name = "right" if prediction[0] == 1 else "left"
            
            # Get probabilities
            if 'csp' in self.pipeline.named_steps:
                features = self.pipeline.named_steps['csp'].transform(data)
            else:
                features = self.pipeline.named_steps['bandpower'].transform(data)
                
            probabilities = self.pipeline.named_steps['classifier'].predict_proba(features)
            confidence = probabilities[0][prediction[0]-1]  # -1 because classes are 1 and 2
            
            ts_prediction = time.time()
            latency = ts_prediction - self.ts_trigger
            #latency = 0
            print(f"Predicted class: {class_name} (confidence: {confidence:.2f}), True: {list(epoch.event_id)[0]}, Latency: {latency}")

            # Track accuracy
            if class_name == list(epoch.event_id)[0]:
                self.prediction_accuracy.append(1)
            else:
                self.prediction_accuracy.append(0)
            print(f"Accuracy: {sum(self.prediction_accuracy)/len(self.prediction_accuracy)}")   
            return class_name, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    

    def predict_motor_imagery(self, epoch):
        """Predict motor imagery class for a single epoch"""
        if not hasattr(self, 'pipeline'):
            print("Model not trained yet")
            return None
        
        try:
            data = epoch.get_data()
            #data = epoch.get_data()[:,:,int(1*self.sf):int(3*self.sf)]  # Shape: (1, n_channels, n_samples)
            prediction = self.pipeline.predict(data)
            
            class_name = "right" if prediction[0] == 1 else "left"
            
            # Get probabilities
            if 'csp' in self.pipeline.named_steps:
                features = self.pipeline.named_steps['csp'].transform(data)
            else:
                features = self.pipeline.named_steps['bandpower'].transform(data)
                
            probabilities = self.pipeline.named_steps['classifier'].predict_proba(features)
            confidence = probabilities[0][prediction[0]-1]  # -1 because classes are 1 and 2
            
            ts_prediction = time.time()
            latency = ts_prediction - self.ts_trigger
            
            # Extract true label correctly
            true_label = list(epoch.event_id.values())[0]  # Extract the true numerical label
            true_class_name = "right" if true_label == 1 else "left"
            
            print(f"Predicted class: {class_name} (confidence: {confidence:.2f}), True: {true_class_name}, Latency: {latency}")

            # Store predictions for confusion matrix evaluation
            self.true_labels.append(true_label)
            self.predicted_labels.append(prediction[0])

            
            # Track accuracy - compare numerical labels instead of strings
            is_correct = (prediction[0] == true_label)
            self.prediction_accuracy.append(1 if is_correct else 0)
            
            current_accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)
            print(f"Accuracy: {current_accuracy:.3f} ({sum(self.prediction_accuracy)}/{len(self.prediction_accuracy)})")
            
            return class_name, confidence
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()  # This will help debug any remaining issues
            return None

    def confusion_matrix(self):
         
        conf_matrix = confusion_matrix(self.true_labels, self.predicted_labels, labels=[1, 2])
                
        # Map numerical labels to class names for display
        # Assuming 1=right, 2=left based on your class_name logic
        display_labels = ["Right", "Left"]                
        disp = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)
                
        plt.figure(figsize=(8, 6))
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix (n={len(self.true_labels)} samples)")
                
        # Save the figure
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()  # Close the figure to free memory


    def plot_accuracy(self):
        if not self.prediction_accuracy:
            print("No prediction data available for accuracy plot")
            return
            
        percentages_accuracy = []
        for i in range(1, len(self.prediction_accuracy) + 1):
            percentages_accuracy.append(sum(self.prediction_accuracy[:i]) / i * 100)

        # Create figure with proper size and DPI
        plt.figure(figsize=(10, 6), dpi=100)
        
        # Plot with improved styling
        plt.plot(percentages_accuracy, marker='o', color='#1f77b4', 
                linewidth=2, markersize=6, markerfacecolor='white', 
                markeredgecolor='#1f77b4', markeredgewidth=2)
        
        # Add rolling average for trend visualization
        window = min(5, len(percentages_accuracy))  # Ensure window isn't larger than data
        if window > 1:
            rolling_avg = np.convolve(percentages_accuracy, 
                                    np.ones(window)/window, 
                                    mode='valid')
            plt.plot(range(window-1, len(percentages_accuracy)), 
                    rolling_avg, 
                    linestyle='--', 
                    color='#ff7f0e', 
                    linewidth=2, 
                    label=f'{window}-trial Moving Average')
        
        # Add chance level
        plt.axhline(y=50, color='#d62728', linestyle='--', 
                    alpha=0.7, linewidth=1.5, label='Chance Level (50%)')
        
        # Improve titles and labels
        plt.title('Classification Accuracy Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Predictions', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        
        # Add grid but make it subtle
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(loc='best', frameon=True, framealpha=0.9)
        
        # Add annotation of final accuracy
        final_accuracy = percentages_accuracy[-1]
        plt.annotate(f'Final: {final_accuracy:.1f}%',
                    xy=(len(percentages_accuracy)-1, final_accuracy),
                    xytext=(len(percentages_accuracy)-1, final_accuracy+10 if final_accuracy < 90 else final_accuracy-10),
                    ha='right',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Improve tick marks
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Add timestamp and save with high quality
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        plt.tight_layout()
        plt.savefig(f"accuracy_progression_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def run_visualizer(self):
        """Run the EEG stream visualizer in a separate thread"""
        try:
            visualizer = mne_lsl.stream_viewer.StreamViewer(stream_name=self.eeg_stream.name)
            visualizer.start(bufsize=0.25)
        except Exception as e:
            print(f"Error starting visualizer: {e}")


    def save_epochs(self):
        """Save collected epochs to file"""
        if not self.total_epochs:
            print("No epochs to save")
            return
            
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            all_epochs = mne.concatenate_epochs(self.total_epochs)
            epochs_filename = f"eeg_epochs_{timestamp}-epo.fif"
            all_epochs.save(epochs_filename, overwrite=True)
            print(f"Saved {len(self.total_epochs)} epochs to {epochs_filename}")
        except Exception as e:
            print(f"Error saving epochs: {e}")


    def set_mode(self, mode):
        """Change the operation mode of the BCI"""
        if mode in ['training', 'prediction']:
            old_mode = self.mode
            self.mode = mode
            print(f"Mode changed from {old_mode} to {mode}")
            
            # If switching to prediction mode, make sure model is trained
            if mode == 'prediction' and not hasattr(self, 'pipeline'):
                if len(self.total_epochs) >= self.min_samples:
                    print("Training model before switching to prediction mode...")
                    self.train_models()
                else:
                    print(f"Need more training samples. Have {len(self.total_epochs)}, need {self.min_samples}")
                    self.mode = 'training'
        else:
            print(f"Invalid mode: {mode}. Use 'training' or 'prediction'")

    def plot_evoked_responses(self):
        """Plot evoked responses for each class"""
        if not self.total_epochs:
            print("No epochs available for plotting evoked responses")
            return
        
        try:
            # Combine all epochs
            all_epochs = mne.concatenate_epochs(self.total_epochs)
            
            # Plot for each class
            fig, axes = plt.subplots(2, 1, figsize=(10, 15))
            
            # Check which classes are available
            available_classes = list(all_epochs.event_id.keys())
            print(f"Available classes: {available_classes}")
            
            for i, class_name in enumerate(['right', 'left']):
                if class_name in available_classes:
                    # Get evoked response for this class
                    evoked = all_epochs[class_name].average()
                    
                    # Plot evoked response
                    evoked.plot(axes=axes[i], show=False, spatial_colors=True, gfp=True)
                    axes[i].set_title(f"Evoked Response: {class_name}")

            #plt.tight_layout()
            plt.savefig(f"evoked_responses_{time.strftime('%Y%m%d-%H%M%S')}.png")
            #plt.ion()
            
        except Exception as e:
            print(f"Error plotting evoked responses: {e}")


    def plot_topomaps(self):
        """Plot topographic maps for each class at different time points"""
        if not self.total_epochs:
            print("No epochs available for plotting topomaps")
            return
            
        try:
            # Combine all epochs
            all_epochs = mne.concatenate_epochs(self.total_epochs)
            
            # Define time points to plot (in seconds)
            #times = np.linspace(0.05, 0.8, 8)  # 8 time points from 50ms to 800ms
            times = [-0.8, -0.3, 0.05, 0.5, 1, 1.5, 2.5, 3.5]
            
            # Check which classes are available
            available_classes = list(all_epochs.event_id.keys())
            
            for class_name in available_classes:
                # Get evoked response for this class
                evoked = all_epochs[class_name].average()
                
                # Plot topomaps
                #fig = plt.figure(figsize=(12, 6))
                #title = f"Topographic Maps: {class_name}"
                fig = evoked.plot_topomap(times=times, time_unit='s', show_names=True, vlim=(-5,5), cmap='RdBu_r', outlines='head')
                fig.savefig(f"topomap_{class_name}_{time.strftime('%Y%m%d-%H%M%S')}.png")
                #plt.ion()
                
        except Exception as e:
            print(f"Error plotting topomaps: {e}")

    
    def plot_time_frequency(self):
        """Plot time-frequency decomposition for each class with proper topomaps and legends"""
        if not self.total_epochs:
            print("No epochs available for time-frequency analysis")
            return
            
        try:
            # Combine all epochs
            all_epochs = mne.concatenate_epochs(self.total_epochs)
            
            # Define frequencies of interest
            freqs = np.arange(8, 30, 2)  # 8-30 Hz, using fewer frequency bins
            n_cycles = freqs / 2  # Scale cycles with frequency
            
            # Check which classes are available
            available_classes = list(all_epochs.event_id.keys())
            
            for class_name in available_classes:
                # Check if we have enough epochs for this class
                class_epochs = all_epochs[class_name]
                print(f"Processing {len(class_epochs)} epochs for class '{class_name}'")
                
                if len(class_epochs) < 5:
                    print(f"Warning: Only {len(class_epochs)} epochs for class '{class_name}'. Results may be unreliable.")
                
                # Compute time-frequency representation
                power = mne.time_frequency.tfr_morlet(
                    class_epochs, 
                    freqs=freqs, 
                    n_cycles=n_cycles, 
                    return_itc=False,
                    decim=3, 
                    average=True, 
                    n_jobs=1
                )
                
                # Apply baseline correction
                power.apply_baseline(baseline=(-1, 0), mode='zscore')
                
                # Plot regular TFR for each channel (this part remains the same)
                for ch_idx, ch_name in enumerate(power.ch_names):
                    fig = plt.figure(figsize=(10, 6))
                    ax = fig.add_subplot(111)
                    power.plot([ch_idx], title=f'Time-Frequency: {class_name} - {ch_name}', axes=ax, show=False, vlim=(-8,8))
                    plt.tight_layout()
                    plt.savefig(f"tfr_{class_name}_{ch_name}_{time.strftime('%Y%m%d-%H%M%S')}.png")
                    plt.close(fig)
                
                # Plot combined channels
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                power.plot(
                    picks=self.channel_selection,  # Use selected channels
                    baseline=(-1, 0), 
                    mode='zscore',
                    title=f'Time-Frequency: {class_name} (Combined Channels)',
                    combine='mean',
                    axes=ax,
                    show=False,
                    vlim=(-8,8)
                )
                plt.tight_layout()
                plt.savefig(f"tfr_{class_name}_combined_{time.strftime('%Y%m%d-%H%M%S')}.png")
                plt.close(fig)
                
                # Create proper topographic maps for specific frequency bands and time points
                # Define time points and frequency bands of interest
                time_points = [-0.5, 0, 0.5, 1, 2, 3.5]  # Time points in seconds
                freq_bands = {
                    'mu': (8, 12),
                    'beta': (13, 30)
                }
                
                for band_name, (fmin, fmax) in freq_bands.items():
                    # Create figure for this frequency band
                    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 4))
                    fig.suptitle(f'{band_name.capitalize()} Band ({fmin}-{fmax} Hz) Topographic Maps: {class_name}')
                    
                    # Extract power data for this frequency band
                    freq_idx = np.where((power.freqs >= fmin) & (power.freqs <= fmax))[0]
                    
                    # Store all topomap plots to add a single colorbar later
                    topomap_plots = []
                    
                    for time_idx, time_point in enumerate(time_points):
                        # Find closest time point in the data
                        time_idx_in_data = np.argmin(np.abs(power.times - time_point))
                        
                        # Average power across the selected frequency range
                        topodata = np.mean(power.data[:, freq_idx, time_idx_in_data], axis=1)
                        
                        # Print debug info
                        print(f"Creating topomap for {band_name} band at {time_point}s")
                        print(f"Data shape: {topodata.shape}")
                        
                        # Plot topomap using MNE's plot_topomap function with colorbar
                        im, _ = mne.viz.plot_topomap(
                            topodata, 
                            pos=power.info,  # Get positions from info
                            names=self.channel_selection,
                            axes=axes[time_idx],
                            show=False,
                            cmap='RdBu_r',
                            outlines='head',
                            contours=6,
                            vlim=(-5,5),  # Set consistent scale range
                        )
                        
                        topomap_plots.append(im)
                        axes[time_idx].set_title(f'{time_point:.1f}s')
                    
                    # Add a single colorbar for all the topoplots
                    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
                    cbar = fig.colorbar(topomap_plots[0], cax=cbar_ax)
                    cbar.set_label('Z-score Power')
                    
                    #plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
                    plt.savefig(f"tfr_topo_{band_name}_{class_name}_{time.strftime('%Y%m%d-%H%M%S')}.png")
                    plt.close(fig)
                    
                # Create a combined figure showing mu vs beta topomap at a key time point (e.g., 0.4s)
                key_time = 0.8
                time_idx_in_data = np.argmin(np.abs(power.times - key_time))
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f'Mu vs Beta Band Comparison at {key_time}s: {class_name}')
                
                topomap_plots = []
                
                for idx, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
                    freq_idx = np.where((power.freqs >= fmin) & (power.freqs <= fmax))[0]
                    topodata = np.mean(power.data[:, freq_idx, time_idx_in_data], axis=1)
                    
                    im, _ = mne.viz.plot_topomap(
                        topodata, 
                        pos=power.info,
                        names=self.channel_selection, 
                        axes=axes[idx],
                        show=False,
                        cmap='RdBu_r',
                        outlines='head',
                        contours=6,
                        vlim=(-5,5)
                    )
                    
                    topomap_plots.append(im)
                    axes[idx].set_title(f'{band_name.capitalize()} Band ({fmin}-{fmax} Hz)')
                
                # Add colorbar to the comparison plot
                cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
                cbar = fig.colorbar(topomap_plots[0], cax=cbar_ax)
                cbar.set_label('Z-score Power')
                
                #plt.tight_layout(rect=[0, 0, 0.9, 1])
                plt.savefig(f"tfr_topo_comparison_{class_name}_{time.strftime('%Y%m%d-%H%M%S')}.png")
                plt.close(fig)
                
        except Exception as e:
            print(f"Error in time-frequency analysis: {e}")
        

                                                           
    def compare_class_differences(self):
        """Statistical comparison between classes"""
        if not self.total_epochs:
            print("No epochs available for statistical comparison")
            return
            
        try:
            # Combine all epochs
            all_epochs = mne.concatenate_epochs(self.total_epochs)
            
            # Check if both left and right classes are available
            if 'left' in all_epochs.event_id and 'right' in all_epochs.event_id:
                # Get evoked responses
                evoked_left = all_epochs['left'].average()
                evoked_right = all_epochs['right'].average()
                
                # Plot difference (left minus right)
                evoked_diff = mne.combine_evoked([evoked_left, evoked_right], weights=[1, -1])
                
                # Plot the difference topography (left minus right)
                #times = np.linspace(0.05, 0.8, 8)
                times = [-0.8, -0.3, 0.05, 0.5, 1.5, 2.5, 3.5, 4]
                fig = evoked_diff.plot_topomap(times=times, time_unit='s', show_names=True)
                #fig = evoked_diff.plot_topomap(times=times, time_unit='s', show_names=True, vlim=(-3e-06, 3e-06), cmap='RdBu_r')
                fig.savefig(f"diff_topomap_left_right_{time.strftime('%Y%m%d-%H%M%S')}.png")
                #plt.show()
                
                # Plot difference time course
                fig = evoked_diff.plot(spatial_colors=True, gfp=True)
                fig.savefig(f"diff_timecourse_left_right_{time.strftime('%Y%m%d-%H%M%S')}.png")
                #plt.show()
                
                print("Statistical comparison completed")
            else:
                print("Need both 'left' and 'right' classes for comparison")
                
        except Exception as e:
            print(f"Error in statistical comparison: {e}")

        
    def main(self):
        """Main execution function"""
        try:
            self.realeeg_stream()
            self.events_stream()

            # Start a thread for the visualizer
            #visualizer_thread = Thread(target=self.run_visualizer)
            #visualizer_thread.daemon = True
            #visualizer_thread.start()

            # Start a thread for marker checking
            marker_thread = Thread(target=self.check_markers)
            marker_thread.daemon = True
            marker_thread.start()

            print("Monitoring markers stream...")
            print("Starting in training mode. Collect at least", self.min_samples, "samples.")
            
            print("\nAvailable commands:")
            print("  train      - Switch to training mode")
            print("  predict    - Switch to prediction mode")
            print("  save       - Save epochs to file")
            print("  status     - Display current status")
            print("  train_now  - Train models with current data")
            print("  evoked     - Plot evoked responses")
            print("  topomap    - Plot topographic maps")
            print("  tfr        - Time-frequency analysis")
            print("  compare    - Statistical comparison between classes")
            print("  visualizer - Visualization of the EEG signal")
            print("  accuracy   - Plot accuracy over time")
            print("  exit       - Exit program")

            # Main loop for user interaction
            while True:
                command = input("\nEnter command: ").strip().lower()
            
                if command == 'train':
                    self.set_mode('training')
                
                elif command == 'predict':
                    self.set_mode('prediction')
                
                elif command == 'exit':
                    break
                
                elif command == 'status':
                    print(f"Current mode: {self.mode}")
                    print(f"Epochs collected: {len(self.total_epochs)}")
                    if hasattr(self, 'pipeline'):
                        print("Model is trained and ready for prediction")
                
                elif command == 'save':
                    self.save_epochs()
                
                elif command == 'train_now':
                    if len(self.total_epochs) >= self.min_samples:
                        print("Training models...")
                        self.train_models()
                    else:
                        print(f"Not enough epochs: {len(self.total_epochs)}/{self.min_samples}")
            
                elif command == 'evoked':
                    self.plot_evoked_responses()
            
                elif command == 'topomap':
                    self.plot_topomaps()
            
                elif command == 'tfr':
                    self.plot_time_frequency()
            
                elif command == 'compare':
                    self.compare_class_differences()
                
                elif command == 'visualizer':
                    # Start a thread for the visualizer
                    visualizer_thread = Thread(target=self.run_visualizer)
                    visualizer_thread.daemon = True
                    visualizer_thread.start()
                
                elif command == 'accuracy':
                    self.plot_accuracy()
                
                elif command == 'confusion':
                    self.confusion_matrix()
                else:
                    print("Unknown command. Available commands: train/predict/save/status/train_now/evoked/topomap/tfr/compare/visualizer/accuracy/exit")

        except KeyboardInterrupt:
            print("Stopping program")
        finally:
            if hasattr(self, 'markers_stream'):
                self.markers_stream.close_stream()
            if hasattr(self, 'eeg_stream'):
                self.eeg_stream.disconnect()
            self.save_epochs()
            print("Program stopped")


if __name__ == "__main__":
    bci = BCI()
    bci.main()