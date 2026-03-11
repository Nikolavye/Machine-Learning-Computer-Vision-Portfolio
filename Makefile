PYTHON ?= python3

.PHONY: test sensor-train sensor-predict sports-eval sports-track

test:
	$(PYTHON) -m unittest discover -s tests -v

sensor-train:
	$(PYTHON) sensor_anomaly_detection/sensor_clustering.py

sensor-predict:
	$(PYTHON) sensor_anomaly_detection/predict.py --csv sensor_anomaly_detection/data_sensors.csv

sports-eval:
	$(PYTHON) sports_player_tracking/eval_c_100frames.py

sports-track:
	$(PYTHON) sports_player_tracking/solution.py
