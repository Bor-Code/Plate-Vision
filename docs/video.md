# Video and Stream Processing

The `plate_vision.video` package extends single-frame ALPR to video streams by combining multi-object bounding box tracking (`PlateTracker`) with confidence-weighted temporal voting (`TemporalVoter`).

## Architecture

1. **Multi-Object Tracking (`PlateTracker` / `MultiObjectTrackerProtocol`)**:
   - Associates frame-level `PlatePredictionResult` detections across sequential frames using Intersection-over-Union (`IoU`) greedy matching (`iou_threshold`, `max_missed_frames`).
   - Maintains persistent `track_id` lifecycles (`TrackState`) with age, hit count, missed frame counter, and smoothed bounding box state.
2. **Temporal Voting (`TemporalVoter`)**:
   - Accumulates frame-level OCR hypotheses (`plate_text`, `combined_confidence`, `is_valid_format`) per `track_id` within a configurable sliding window (`window_size`).
   - Applies a format-validity weight bonus (`valid_format_bonus`) so syntactically valid plates outrank transient OCR corruptions caused by motion blur or partial occlusion.
   - Emits `StabilizedTrackPrediction` with consensus `consensus_text`, `consensus_confidence`, `vote_count`, and `agreement_ratio`.
3. **Stream Processor (`VideoStreamProcessor`)**:
   - Processes sequences or iterators of `VideoFrameInput`, returning per-frame `VideoFrameResult` and aggregate `VideoStreamSummary`.
