import pygame
import sys
import random
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

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)

# Game settings
BALL_SIZE = 20
#BALL_SPEED = 4  # Slow ball speed
PADDLE_WIDTH = SCREEN_WIDTH // 2  # Half screen width
PADDLE_HEIGHT = 20
PADDLE_Y = SCREEN_HEIGHT - 40  # Bottom position

# Custom transformer for band power features
class BandPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=500, 
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
    def __init__(self, game_instance=None):
        self.bf_duration = 10       # in Seconds
        self.sf = 500               # Sampling frequency
        self.marker_data = []       # Stores markers in the form (timestamp, type)
        self.total_epochs = []      # Stores all epochs
        self.channel_selection = ['C3','Cz','C4','CP1','CP2']  # 5 Channels
        self.min_samples = 30       # Minimum samples needed for training
        self.accuracy_history = []  # Track classification accuracy over time
        self.prediction_accuracy = [] # Percentage of accuracy tracking
        self.mode = 'training'      # Default mode: 'training' or 'prediction'
        self.true_labels = []
        self.predicted_labels = []
        self.game = game_instance   # Reference to game instance
        self.marker_thread = None
        self.is_running = False
        self.pipeline = None
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

    def send_marker(self, marker_value):
        """Send a marker to the EEG system"""
        try:
            # Create a marker string
            marker_string = f"Stimulus/S{marker_value}"
            
            # Send the marker - this would typically go to your EEG system
            # For now, we'll simulate it by calling the marker processing directly
            current_time = time.time()
            print(f"Sending marker: {marker_string} at {current_time}")
            
            # Add to marker data
            self.marker_data.append((current_time, marker_string))
            
            # Create epoch for this marker
            epoch = self.create_epoch_for_marker(current_time, marker_string)
            
            # Handle the epoch based on current mode
            if self.mode == 'training' and self.game:
                print(f"Training mode: Collected {len(self.total_epochs)} epochs so far")
                if len(self.total_epochs) >= self.min_samples:
                    self.game.training_ready_for_model = True
            
            elif self.mode == 'prediction' and hasattr(self, 'pipeline') and self.pipeline is not None:
                # Use the pipeline to predict class
                prediction = self.predict_motor_imagery(epoch)
                if prediction and self.game:
                    predicted_class, confidence = prediction
                    self.game.bci_prediction = predicted_class
                    self.game.prediction_confidence = confidence
                    
        except Exception as e:
            print(f"Error sending marker: {e}")

    def start_marker_monitoring(self):
        """Start monitoring for markers in a separate thread"""
        if self.marker_thread is None or not self.marker_thread.is_alive():
            self.is_running = True
            self.marker_thread = Thread(target=self.check_markers, daemon=True)
            self.marker_thread.start()

    def stop_marker_monitoring(self):
        """Stop marker monitoring"""
        self.is_running = False
        if self.marker_thread and self.marker_thread.is_alive():
            self.marker_thread.join(timeout=1)

    def check_markers(self):
        """
        - Constantly checks for incoming markers
        - If marker is found store it and create epoch
        """
        while self.is_running:
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
                    if self.mode == 'training' and self.game:
                        print(f"Training mode: Collected {len(self.total_epochs)} epochs so far")
                        if len(self.total_epochs) >= self.min_samples:
                            self.game.training_ready_for_model = True
                    
                    elif self.mode == 'prediction' and hasattr(self, 'pipeline') and self.pipeline is not None:
                        # Use the pipeline to predict class
                        prediction = self.predict_motor_imagery(epoch)
                        if prediction and self.game:
                            predicted_class, confidence = prediction
                            self.game.bci_prediction = predicted_class
                            self.game.prediction_confidence = confidence
                
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
                flat={"eeg": 1e-6},
                baseline=(-1, 0) 
            )

            print("Marker Timestamp: ", marker_timestamp)
            
            epoch_start_time = timestamp[0]
            epoch_end_time = timestamp[-1]

            print(f"Epoch goes from {epoch_start_time} to {epoch_end_time}")
                
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
        else:
            print("Using Band Power + LDA as it performed better")
            self.pipeline = bandpower_pipeline
            self.pipeline.fit(X, y)
        
        self.accuracy_history.append((time.time(), max(bp_scores.mean(), csp_scores.mean())))
        
        # Switch to prediction mode
        self.mode = 'prediction'
        
        return True
    
    def predict_motor_imagery(self, epoch):
        """Predict motor imagery class for a single epoch"""
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            print("Model not trained yet")
            return None
        
        try:
            data = epoch.get_data()
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
            #latency = ts_prediction - self.ts_trigger
            latency = 0
            
            # Extract true label correctly
            true_label = list(epoch.event_id.values())[0]  # Extract the true numerical label
            true_class_name = "right" if true_label == 1 else "left"
            
            print(f"Predicted class: {class_name} (confidence: {confidence:.2f})")

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
            traceback.print_exc()
            return None

class Ball:
    BALL_SPEED = 4

    def __init__(self, x=None, y=None, dx=None, dy=None):
        self.x = x if x is not None else SCREEN_WIDTH // 2
        self.y = y if y is not None else SCREEN_HEIGHT // 2
        self.dx = dx if dx is not None else random.choice([-self.BALL_SPEED, self.BALL_SPEED])
        self.dy = dy if dy is not None else -self.BALL_SPEED
        self.crossed_middle = False
        self.active = True
        self.marker_sent = False  # Track if marker was sent for this ball
        
    def move(self, bci_system=None):
        if not self.active:
            return
            
        self.x += self.dx
        self.y += self.dy
        
        # Bounce off left and right walls
        if self.x <= BALL_SIZE//2 or self.x >= SCREEN_WIDTH - BALL_SIZE//2:
            self.dx = -self.dx
            
        # Bounce off top wall
        if self.y <= BALL_SIZE//2:
            self.dy = -self.dy
            
        # Check if ball crosses middle line towards player and send marker
        middle_y = SCREEN_HEIGHT // 2
        if not self.marker_sent and self.y > middle_y and self.dy > 0:
            print("Ball crossed middle - sending marker")
            self.marker_sent = True
            self.crossed_middle = True
            
            # Send marker based on ball position
            if bci_system:
                # Determine which side the ball is on
                if self.x < SCREEN_WIDTH // 2:
                    bci_system.send_marker(2)  # Left side - marker 2
                else:
                    bci_system.send_marker(1)  # Right side - marker 1
                    
        elif self.y < middle_y:
            self.crossed_middle = False
            self.marker_sent = False
    
    def draw(self, screen):
        if self.active:
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_SIZE//2)
    
    def reset(self, x=None, y=None, dx=None, dy=None):
        self.x = x if x is not None else SCREEN_WIDTH // 2
        self.y = y if y is not None else SCREEN_HEIGHT // 2
        self.dx = dx if dx is not None else random.choice([-self.BALL_SPEED, self.BALL_SPEED])
        self.dy = dy if dy is not None else -self.BALL_SPEED
        self.crossed_middle = False
        self.active = True
        self.marker_sent = False

class Paddle:
    def __init__(self):
        self.position = "left"  # "left" or "right"
        self.x = 0  # Left half
        self.y = PADDLE_Y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
    
    def move_to(self, position):
        if position == "left":
            self.x = 0
            self.position = "left"
        elif position == "right":
            self.x = SCREEN_WIDTH // 2
            self.position = "right"
    
    def draw(self, screen):
        color = BLUE if self.position == "left" else RED
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
    
    def check_collision(self, ball):
        if not ball.active:
            return False
        # Check if ball hits the paddle
        if (ball.y + BALL_SIZE//2 >= self.y and 
            ball.y - BALL_SIZE//2 <= self.y + self.height and
            ball.x >= self.x and 
            ball.x <= self.x + self.width):
            return True
        return False

class Game:
    def __init__(self, bci_system=None):
        self.mode = "menu"  # "menu", "training", "game"
        self.ball = Ball()
        self.paddle = Paddle()
        self.score = 0
        self.last_score = 0
        self.max_score = 0
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.samples = 30
        
        # Training mode variables
        self.training_count = 0
        self.training_side = "right"  # Start with right side
        self.training_waiting = False
        self.training_wait_time = 0
        self.balls_hit_in_training = 0
        self.training_ready_for_model = False
        
        # BCI integration
        self.bci = bci_system
        self.use_bci_control = False  # Toggle between manual and BCI control
        self.bci_prediction = None
        self.prediction_confidence = 0.0
        
        # Control mode
        self.control_mode = "bci"  # "manual" or "bci"
        
    def start_training(self):
        self.mode = "training"
        self.training_count = 0
        self.training_side = "right"
        self.training_waiting = False
        self.balls_hit_in_training = 0
        self.training_ready_for_model = False
        
        # Set BCI to training mode
        if self.bci:
            self.bci.mode = 'training'
            self.bci.total_epochs = []  # Reset epochs
            
        self.spawn_training_ball()
    
    def spawn_training_ball(self):
        if self.training_side == "right":
            # Ball falls on right side
            x = SCREEN_WIDTH * 3 // 4  # 3/4 across screen (right side)
            self.ball.reset(x=x, y=50, dx=0, dy=Ball.BALL_SPEED)
        else:
            # Ball falls on left side
            x = SCREEN_WIDTH // 4  # 1/4 across screen (left side)
            self.ball.reset(x=x, y=50, dx=0, dy=Ball.BALL_SPEED)
    
    def update_training(self):
        self.ball.move(self.bci)

        # Automatic paddle movement when ball crosses middle line during training
        if self.ball.crossed_middle and self.control_mode == "bci":
            # Move paddle to correct position based on ball position
            if self.ball.x < SCREEN_WIDTH // 2:
                self.paddle.move_to("left")
                print("Training: Auto-moved paddle to LEFT")
                self.ball.crossed_middle = False
            else:
                self.paddle.move_to("right")
                print("Training: Auto-moved paddle to RIGHT")
                self.ball.crossed_middle = False

        # Check paddle collision
        if self.paddle.check_collision(self.ball):
            self.ball.dy = -abs(self.ball.dy)  # Bounce up
            self.balls_hit_in_training += 1
        
        # Check if ball hits the top - disappear and spawn next ball
        if self.ball.y <= BALL_SIZE//2:
            self.ball.active = False  # Make ball disappear
            self.training_count += 1
            
            if self.training_count >= self.samples:  # 10 right + 10 left = 20 total
                self.mode = "training_complete"
            else:
                # Switch sides and spawn next ball after a short delay
                self.training_side = "left" if self.training_side == "right" else "right"
                self.training_waiting = True
                self.training_wait_time = pygame.time.get_ticks() + 500  # 500ms delay
        
        # Check if ball goes off bottom (missed ball)
        elif self.ball.y > SCREEN_HEIGHT + BALL_SIZE:
            self.training_count += 1
            
            if self.training_count >= self.samples:  # 10 right + 10 left = 20 total
                self.mode = "training_complete"
            else:
                # Switch sides and spawn next ball after a short delay
                self.training_side = "left" if self.training_side == "right" else "right"
                self.training_waiting = True
                self.training_wait_time = pygame.time.get_ticks() + 500  # 500ms delay
    
    def update_game(self):
        self.ball.move(self.bci)
        
        # Handle BCI predictions if in BCI mode
        if self.control_mode == "bci" and self.bci_prediction:
            if self.bci_prediction == "left":
                self.paddle.move_to("left")
            elif self.bci_prediction == "right":
                self.paddle.move_to("right")
            # Reset prediction after using it
            self.bci_prediction = None
        
        # Check paddle collision
        if self.paddle.check_collision(self.ball):
            self.ball.dy = -abs(self.ball.dy)  # Bounce up
            self.score += 1
            
            if self.score > 5:
                Ball.BALL_SPEED = 6  # Slow ball speed
            elif self.score > 10:
                Ball.BALL_SPEED = 8
        
        # Check if ball goes off bottom (game over condition)
        if self.ball.y > SCREEN_HEIGHT + BALL_SIZE:
            print(f"Game Over! Final Score: {self.score}")
            self.last_score = self.score
            if self.last_score > self.max_score:
                self.max_score = self.last_score
            self.ball.reset()
            self.score = 0
    
    def toggle_control_mode(self):
        """Toggle between manual and BCI control"""
        if self.control_mode == "manual":
            if self.bci and hasattr(self.bci, 'pipeline') and self.bci.pipeline is not None:
                self.control_mode = "bci"
                print("Switched to BCI control mode")
            else:
                print("BCI model not trained yet!")
        else:
            self.control_mode = "manual"
            print("Switched to manual control mode")
    
    def train_bci_model(self):
        """Train the BCI model with collected data"""
        if self.bci:
            success = self.bci.train_models()
            if success:
                print("BCI model trained successfully!")
                return True
            else:
                print("Failed to train BCI model")
                return False
        return False
    
    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if self.mode == "menu":
                if event.key == pygame.K_t:
                    self.start_training()
                elif event.key == pygame.K_g:
                    if self.bci and hasattr(self.bci, 'pipeline') and self.bci.pipeline is not None:
                        self.mode = "game"
                        self.ball.reset()
                        self.score = 0
                    else:
                        print("Train the BCI model first!")
                elif event.key == pygame.K_q:
                    return False
            
            elif (self.mode == "training") and (self.control_mode == "manual"):
                # Manual control during training
                if event.key == pygame.K_LEFT:
                    self.paddle.move_to("left")
                elif event.key == pygame.K_RIGHT:
                    self.paddle.move_to("right")

            elif self.mode == "training_complete":
                if event.key == pygame.K_SPACE:
                    # Train the model
                    if self.train_bci_model():
                        self.mode = "menu"
                    else:
                        print("Failed to train model. Try collecting more data.")
                elif event.key == pygame.K_r:
                    # Restart training
                    self.start_training()
                elif event.key == pygame.K_m:
                    self.mode = "menu"
            
            elif self.mode == "game":
                if event.key == pygame.K_ESCAPE:
                    self.mode = "menu"
                elif event.key == pygame.K_c:
                    self.toggle_control_mode()
                elif self.control_mode == "manual":
                    # Manual control in game mode
                    if event.key == pygame.K_LEFT:
                        self.paddle.move_to("left")
                    elif event.key == pygame.K_RIGHT:
                        self.paddle.move_to("right")
    
        return True
    
    def update(self):
        current_time = pygame.time.get_ticks()
        
        if self.mode == "training":
            # Handle training wait time
            if self.training_waiting:
                if current_time >= self.training_wait_time:
                    self.training_waiting = False
                    self.spawn_training_ball()
            else:
                self.update_training()
        
        elif self.mode == "game":
            self.update_game()
    
    def draw(self, screen):
        screen.fill(BLACK)
        
        # Draw middle line
        pygame.draw.line(screen, WHITE, (0, SCREEN_HEIGHT//2), (SCREEN_WIDTH, SCREEN_HEIGHT//2), 2)
        
        if self.mode == "menu":
            self.draw_menu(screen)
        elif self.mode == "training":
            self.draw_training(screen)
        elif self.mode == "training_complete":
            self.draw_training_complete(screen)
        elif self.mode == "game":
            self.draw_game(screen)
    
    def draw_menu(self, screen):
        title = self.font.render("BCI Pong Game", True, WHITE)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 100))
        
        # Instructions
        instructions = [
            "T - Start Training Mode",
            "G - Play Game (requires trained model)",
            "Q - Quit"
        ]
        
        y_offset = 200
        for instruction in instructions:
            text = self.small_font.render(instruction, True, WHITE)
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y_offset))
            y_offset += 30
        
        # Show training status
        if self.bci and hasattr(self.bci, 'pipeline') and self.bci.pipeline is not None:
            status = self.small_font.render("BCI Model: TRAINED", True, GREEN)
        else:
            status = self.small_font.render("BCI Model: NOT TRAINED", True, RED)
        screen.blit(status, (SCREEN_WIDTH//2 - status.get_width()//2, y_offset + 50))
    
    def draw_training(self, screen):
        # Draw ball and paddle
        self.ball.draw(screen)
        self.paddle.draw(screen)
        
        # Training info
        title = self.font.render("Training Mode", True, WHITE)
        screen.blit(title, (20, 20))
        
        progress = self.small_font.render(f"Progress: {self.training_count}/{self.samples}", True, WHITE)
        screen.blit(progress, (20, 60))
        
        side_text = self.small_font.render(f"Current Side: {self.training_side.upper()}", True, WHITE)
        screen.blit(side_text, (20, 80))
        
        hits_text = self.small_font.render(f"Balls Hit: {self.balls_hit_in_training}", True, WHITE)
        screen.blit(hits_text, (20, 100))
        
        if self.bci:
            epochs_text = self.small_font.render(f"Epochs Collected: {len(self.bci.total_epochs)}", True, WHITE)
            screen.blit(epochs_text, (20, 120))
        '''
        # Controls
        controls = [
            "Use LEFT/RIGHT arrows to move paddle",
            "Hit balls when they appear on the corresponding side"
        ]
        
        y_offset = SCREEN_HEIGHT - 60
        for control in controls:
            text = self.small_font.render(control, True, YELLOW)
            screen.blit(text, (20, y_offset))
            y_offset += 20
        '''
        # Waiting indicator
        if self.training_waiting:
            waiting_text = self.font.render("Next ball incoming...", True, ORANGE)
            screen.blit(waiting_text, (SCREEN_WIDTH//2 - waiting_text.get_width()//2, SCREEN_HEIGHT//2 - 50))
    
    def draw_training_complete(self, screen):
        title = self.font.render("Training Complete!", True, GREEN)
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
        
        stats = [
            f"Total Balls: {self.training_count}",
            f"Balls Hit: {self.balls_hit_in_training}",
            f"Hit Rate: {(self.balls_hit_in_training/self.training_count)*100:.1f}%"
        ]
        
        y_offset = 200
        for stat in stats:
            text = self.small_font.render(stat, True, WHITE)
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y_offset))
            y_offset += 25
        
        if self.bci:
            epochs_text = self.small_font.render(f"Epochs Collected: {len(self.bci.total_epochs)}", True, WHITE)
            screen.blit(epochs_text, (SCREEN_WIDTH//2 - epochs_text.get_width()//2, y_offset + 20))
        
        # Instructions
        instructions = [
            "SPACE - Train BCI Model",
            "R - Restart Training",
            "M - Return to Menu"
        ]
        
        y_offset = 350
        for instruction in instructions:
            text = self.small_font.render(instruction, True, YELLOW)
            screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y_offset))
            y_offset += 25
        
        # Ready status
        if self.training_ready_for_model:
            ready_text = self.small_font.render("Ready to train model!", True, GREEN)
            screen.blit(ready_text, (SCREEN_WIDTH//2 - ready_text.get_width()//2, y_offset + 30))
    
    def draw_game(self, screen):
        # Draw ball and paddle
        self.ball.draw(screen)
        self.paddle.draw(screen)
        
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (20, 20))
        
        # Control mode indicator
        mode_color = GREEN if self.control_mode == "bci" else BLUE
        mode_text = self.small_font.render(f"Control: {self.control_mode.upper()}", True, mode_color)
        screen.blit(mode_text, (20, 60))
        
        max_score_text = self.small_font.render(f"MAX Score: {self.max_score}", True, WHITE)
        screen.blit(max_score_text, (20, 80))
            
        last_score_text = self.small_font.render(f"Last Score: {self.last_score}", True, WHITE)
        screen.blit(last_score_text, (20, 100))
        
        # Controls info
        controls = []
        if self.control_mode == "manual":
            controls = ["LEFT/RIGHT arrows to move", "C - Switch to BCI control"]
        else:
            controls = ["Think LEFT/RIGHT to move paddle", "C - Switch to manual control"]
        
        controls.append("ESC - Return to menu")
        
        y_offset = SCREEN_HEIGHT - 80
        for control in controls:
            text = self.small_font.render(control, True, YELLOW)
            screen.blit(text, (20, y_offset))
            y_offset += 20

def main():
    # Initialize display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BCI Pong Game")
    clock = pygame.time.Clock()
    
    # Initialize BCI system
    try:
        print("Initializing BCI system...")
        bci = BCI()
        
        # Connect to streams
        bci.realeeg_stream()
        bci.events_stream()
        
        # Start marker monitoring
        bci.start_marker_monitoring()
        
        print("BCI system initialized successfully!")
        
    except Exception as e:
        print(f"Warning: BCI initialization failed: {e}")
        print("Running in simulation mode without real EEG data")
        bci = BCI()  # Create BCI instance anyway for game compatibility
    
    # Initialize game with BCI reference
    game = Game(bci)
    bci.game = game  # Set bidirectional reference
    
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                running = game.handle_input(event)
                if not running:
                    break
        
        game.update()
        game.draw(screen)
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    # Cleanup
    if hasattr(bci, 'stop_marker_monitoring'):
        bci.stop_marker_monitoring()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()