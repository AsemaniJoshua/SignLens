"""SignLens core: shared preprocessing for training notebooks and inference.

Import this from the notebooks AND from model_endpoints so both sides
process landmarks identically.
"""
from .landmarks import FEATURE_DIM, hand_features, normalize_hand, hand_from_legacy_vector, scaling_stats
from .augment import augment_hand
from .sequences import (SEQ_LEN, SEQ_FEATURE_DIM, build_sequence, sequence_features, augment_seq_sample,
                        save_sequences, load_sequences)
from .data import (StaticSet, grouped_split, check_no_leak, sort_labels, save_labels,
                   load_repo_youtube_frames, load_repo_webcam_sequences, load_repo_video_sequences,
                   load_repo_video_frames)
from . import batches, evaluate  # noqa: F401  (extract/inference import cv2/mediapipe lazily)

__version__ = "2.1.0"
