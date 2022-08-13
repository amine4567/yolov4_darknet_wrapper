import datetime
import os
import shutil
from collections import OrderedDict
from io import BytesIO
from typing import Dict, List, Tuple, Union

import cv2
import mlflow
import numpy as np
import onnxruntime
import pytz
from mlflow.tracking.client import MlflowClient
from PIL import ImageFile

from pyolov4.config import Config
from pyolov4.exceptions import ArtifactNotFoundError
from pyolov4.utils import add_bboxes_to_img, process_model_output


class Yolov4ONNXInference:
    def __init__(
        self,
        onnx_model_path_or_bytes: Union[str, bytes],
        classes_names: List[str],
        config_model: Config,
        run_metadata,
    ):
        self._onnx_model_path_or_bytes = onnx_model_path_or_bytes

        self.classes_names = classes_names

        self.session = onnxruntime.InferenceSession(self._onnx_model_path_or_bytes)

        try:
            assert self.session.get_outputs()[1].shape[2] == len(self.classes_names)
        except AssertionError:
            raise ValueError()

        input_shape = self.session.get_inputs()[0].shape
        self.input_h = input_shape[2]
        self.input_w = input_shape[3]

        self.confidence_threshold = config_model["obj_detection"][
            "confidence_threshold"
        ]
        self.nms_threshold = config_model["obj_detection"]["nms_threshold"]

        self.run_metadata = run_metadata

    @classmethod
    def load_from_mlflow(
        cls, model_version: int, config_model: Config, config_settings: Config
    ):
        tracking_uri = config_settings["paths"]["tracking_uri"]
        experiment_name = config_settings["paths"]["obj_detection"]["experiment_name"]
        model_name = config_settings["paths"]["obj_detection"]["model_name"]
        artifact_path = config_settings["paths"]["obj_detection"]["artifacts"]
        classes_names_filename = config_settings["paths"]["obj_detection"][
            "classes_names_filename"
        ]

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow_client = MlflowClient()
        model_infos = mlflow_client.get_model_version(model_name, model_version)
        run_metadata = mlflow_client.get_run(model_infos.run_id)

        artifacts_dwnldpath = mlflow_client.download_artifacts(
            model_infos.run_id, artifact_path
        )

        filenames = [
            filename
            for filename in os.listdir(artifacts_dwnldpath)
            if filename.startswith(classes_names_filename)
        ]
        if len(filenames) == 0:
            raise ArtifactNotFoundError(
                f"Aucun artifact avec un nom commençant par {classes_names_filename} "
                "n'existe."
            )

        with open(os.path.join(artifacts_dwnldpath, filenames[0]), "r") as fp:
            classes_names = fp.readlines()

            classes_names = [name.strip() for name in classes_names]

        shutil.rmtree(artifacts_dwnldpath)

        onnx_model = cls(
            mlflow.onnx.load_model(model_infos.source).SerializeToString(),
            classes_names,
            config_model,
            run_metadata,
        )
        return onnx_model

    def predict(
        self, img_path_or_buffer: Union[str, BytesIO]
    ) -> Tuple[List[Dict], ImageFile.ImageFile]:
        if isinstance(img_path_or_buffer, str):
            img_array = cv2.imread(img_path_or_buffer)
        elif isinstance(img_path_or_buffer, BytesIO):
            img_array = cv2.imdecode(
                np.frombuffer(img_path_or_buffer.read(), np.uint8), 1
            )

        # Prepare the network's input
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        img_input = cv2.resize(
            img_array, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        img_input = np.transpose(img_input, (2, 0, 1)).astype(np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        img_input /= 255.0

        # Inference
        inference_output = self.session.run(
            None, {self.session.get_inputs()[0].name: img_input}
        )
        pred_results = process_model_output(
            inference_output,
            conf_thresh=self.confidence_threshold,
            nms_thresh=self.nms_threshold,
            classes_names=self.classes_names,
        )

        # Generate output image with bounding boxes
        img_with_bboxes = add_bboxes_to_img(img_array, pred_results)

        return pred_results, img_with_bboxes

    def get_metrics(self):
        metrics = self.run_metadata.data.metrics
        metrics = {key: round(val, 2) for key, val in metrics.items()}
        return metrics

    def get_run_datetime(self):
        run_datetime_utc = pytz.timezone("utc").localize(
            datetime.datetime.fromtimestamp(
                self.run_metadata.info.start_time / 1000
            )  # since end_time is None
        )
        return run_datetime_utc

    def get_model_data(self):
        model_data = OrderedDict(
            {"run_datetime": self.get_run_datetime(), **self.get_metrics()}
        )

        return model_data


if __name__ == "__main__":
    config_settings = Config("settings")
    config_model = Config("model")
    model = Yolov4ONNXInference.load_from_mlflow(6, config_model, config_settings)
    model.get_model_data()
    print("bla")
