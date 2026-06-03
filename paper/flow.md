```mermaid
graph LR
	main["main.py"]
	flask["server/flask_app.py\ncreate_app()"]
	routes["server/routes.py"]
	streaming["server/streaming.py\nSharedFrameStreamer"]
	frame["processors/frame_processor.py\nFrameProcessor"]
	keydet["detectors/keypoint_detector.py\nKeypointDetector"]
	behav["detectors/behavior_classifier.py\nBehaviorClassifier"]
	model["models/stgcn_model.py\nCatBehaviorSTGCN"]
	anomaly["processors/anomaly_detector.py\nAnomalyDetector"]
	viz["processors/visualizer.py\nVisualizer"]
	tracker["trackers/behavior_tracker.py\nImprovedBehaviorTracker"]
	csv["logutils/csv_logger.py\nCSVLogger / SegmentLogger"]
	nodered["communication/nodered_client.py\nNodeRedClient"]
	constants["utils/constants.py\nShared constants"]
	utils["utils/helpers.py\nget_ip()"]
	config["config.py\n(Flask/Model/Node-RED settings)"]

	main --> flask
	main --> nodered

	flask --> routes

	routes --> config
	routes --> streaming
	routes --> frame
	routes --> nodered
	routes --> tracker
	routes --> model
	routes --> utils

	streaming --> frame
	streaming --> config

	frame --> keydet
	frame --> behav
	frame --> anomaly
	frame --> viz
	frame --> tracker
	frame --> nodered
	frame --> csv
	frame --> model
	frame --> config
	frame --> constants

	behav --> model
	keydet --> "Ultralytics YOLO (external)"
	model --> config

	tracker --> config
	nodered --> config
	csv --> config
	viz --> config
	viz --> constants
	helpers --> constants

	style main fill:#f9f,stroke:#333,stroke-width:1px
	style config fill:#ffeb99,stroke:#333,stroke-width:1px
	style constants fill:#d9edf7,stroke:#333,stroke-width:1px
```