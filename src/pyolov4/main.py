import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import OrderedDict
from io import BytesIO
from typing import List, Tuple, Union
from urllib.request import urlretrieve

import mlflow
import onnx
import pandas as pd
import torch

from pyolov4.config import Config
from pyolov4.utils import parse_metrics_results


class DarknetYolov4Model:
    def __init__(
        self,
        config_settings: Config,
    ):
        # TODO: automate the creation of the backup folder
        self.config_settings = config_settings

        self.darknet_config_dir = os.path.join(
            Config.get_config_path("settings").parent, "darknet"
        )
        self.workdir = self.config_settings["paths"]["obj_detection"]["workdir"]
        self._prepare_darknet_binary()
        self._prepare_darknet_data_file()
        self._prepare_pretrained_weights()
        self.cfg_file_path = os.path.join(self.darknet_config_dir, "yolov4-model.cfg")

        self.trained_weights_path = os.path.join(
            self.data_conf["backup"],
            "yolov4-model_last.weights",
        )

    def _prepare_darknet_binary(self):
        self.darknet_repo_path = os.path.join(self.workdir, "darknet-master")
        if not os.path.exists(self.darknet_repo_path):
            print(
                "The darknet repo doesn't exist in the working directory. Cloning ..."
            )
            urlretrieve(
                self.config_settings["paths"]["obj_detection"]["darknet_repo_url"],
                self.darknet_repo_path + ".zip",
            )
            with zipfile.ZipFile(self.darknet_repo_path + ".zip", "r") as zip_ref:
                zip_ref.extractall(self.workdir)

            print("Cloning done.")

            print("Copying the custom Makefile to the local darknet repo ...")
            makefile_dst_path = os.path.join(self.darknet_repo_path, "Makefile")
            if os.path.exists(makefile_dst_path):
                os.remove(makefile_dst_path)
            shutil.copy(
                os.path.join(self.darknet_config_dir, "Makefile"),
                self.darknet_repo_path,
            )
            print("Copying done.")

        if os.path.exists(os.path.join(self.darknet_repo_path, "darknet")):
            print("The darknet binary already exists.")
        else:
            print("The darknet binary doesn't exist. Compiling ...")
            # Note for Databricks: Because of limitations, darknet can't be compiled
            # in DBFS. As a workaround, the whole darknet repo will be copied to the
            # VM's local storage, the executable will be compiled then copied to DBFS.
            darknet_local_dirpath = self.config_settings["paths"]["obj_detection"][
                "darknet_local_dirpath"
            ]
            shutil.copytree(self.darknet_repo_path, darknet_local_dirpath)
            os.system(
                "export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}; "
                f"cd {darknet_local_dirpath}; "
                "make"
            )
            shutil.copy(
                os.path.join(darknet_local_dirpath, "darknet"), self.darknet_repo_path
            )
            print("Compiling done.")

    def _prepare_darknet_data_file(self):
        classes_names_filepath = os.path.join(self.darknet_config_dir, "classes.names")
        with open(classes_names_filepath, "r") as fp:
            self.classes_names = fp.readlines()

        self.data_conf = OrderedDict(
            {
                "classes": len(self.classes_names),
                "train": os.path.join(self.workdir, "train.txt"),
                "test": os.path.join(self.workdir, "test.txt"),
                "valid": os.path.join(self.workdir, "test.txt"),
                "names": classes_names_filepath,
                "backup": os.path.join(self.workdir, "backup"),
            }
        )

        self.data_conf_path = os.path.join(self.workdir, "obj.data")
        with open(self.data_conf_path, "w") as fp:
            fp.writelines(
                [f"{key} = {value}\n" for key, value in self.data_conf.items()]
            )

    def _prepare_pretrained_weights(self):
        pretrained_weights_url = self.config_settings["paths"]["obj_detection"][
            "pretrained_weights_url"
        ]
        self.pretrained_weights_filepath = os.path.join(
            self.workdir, pretrained_weights_url.rsplit("/", maxsplit=1)[1]
        )
        if not os.path.exists(self.pretrained_weights_filepath):
            print("The pretrained weights file doesn't exist. Downloading ...")
            urlretrieve(
                pretrained_weights_url,
                os.path.join(self.workdir, self.pretrained_weights_filepath),
            )
            print("Download finished.")

    def _execute_darknet_with_args(
        self,
        args: List,
        stream_output: bool = False,
    ):
        cmd_list = ["./darknet", *args]
        cmd = " ".join(cmd_list)
        print(cmd)
        if stream_output:
            # TODO: kill the process when a cancel/interrupt signal is detected (SIGINT for example)
            process = subprocess.Popen(
                cmd_list, cwd=self.darknet_repo_path, stdout=subprocess.PIPE
            )
            print(f"Darknet PID: {process.pid}")
            for c in iter(lambda: process.stdout.read(1), b""):
                sys.stdout.write(c)
        else:
            os.system(f"cd {self.darknet_repo_path};" + cmd)

    def train(self, from_scratch: bool = False):
        if os.path.exists(self.trained_weights_path) and not from_scratch:
            init_weights_path = self.trained_weights_path
        else:
            init_weights_path = self.pretrained_weights_filepath

        self._execute_darknet_with_args(
            [
                "detector",
                "train",
                self.data_conf_path,
                self.cfg_file_path,
                init_weights_path,
                "-dont_show",
            ],
            stream_output=True,
        )

    def compute_metrics(self) -> Tuple[float, pd.DataFrame]:
        if not os.path.exists(self.trained_weights_path):
            raise FileNotFoundError(
                "No trained weights file found, can't compute metrics"
            )

        fp = tempfile.NamedTemporaryFile(mode="w", delete=True)

        self._execute_darknet_with_args(
            [
                "detector",
                "map",
                self.data_conf_path,
                self.cfg_file_path,
                self.trained_weights_path,
                "-dont_show",
                ">>",
                fp.name,
            ],
            stream_output=False,
        )

        print("Finished computing metrics. Parsing the results file ...")
        map_score, ap_per_class_df = parse_metrics_results(fp.name)
        print("Finished parsing the results file")

        fp.close()

        return map_score, ap_per_class_df

    def save_model_to_mlflow(self):
        model_path = self.config_settings["paths"]["obj_detection"]["trained_model"]
        experiment_name = self.config_settings["paths"]["obj_detection"][
            "experiment_name"
        ]
        model_name = self.config_settings["paths"]["obj_detection"]["model_name"]
        artifact_path = self.config_settings["paths"]["obj_detection"]["artifacts"]

        onnx_buffer = BytesIO()
        convert_darknet_to_onnx(
            self.cfg_file_path, self.trained_weights_path, onnx_buffer, -1
        )

        mlflow.set_tracking_uri(model_path)
        mlflow.set_experiment(experiment_name)

        print("Logging ONNX model to MLflow")
        onnx_buffer.seek(0)
        mlflow.onnx.log_model(
            onnx.load(onnx_buffer),
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

        print("Logging cfg and weights files as artefacts to MLflow")
        mlflow.log_artifact(self.cfg_file_path, artifact_path)
        mlflow.log_artifact(self.trained_weights_path, artifact_path)

        print("Computing metrics")
        map_score, ap_per_class_df = self.compute_metrics()

        print("Logging The Mean Average Precision score to MLflow")
        mlflow.log_metric("mAP", map_score)

        print("Logging The Average Precision per class to MLflow as a CSV artifact")
        with tempfile.NamedTemporaryFile(prefix="ap_per_class", suffix=".csv") as tempf:
            ap_per_class_df.to_csv(tempf, index=False)
            mlflow.log_artifact(tempf.name, artifact_path)

        print("Logging classes names file artifact to MLflow")
        mlflow.log_artifact(self.data_conf["names"], artifact_path)


def convert_darknet_to_onnx(
    cfgfile_path: str,
    weightfile_path: str,
    onnx_filepath_or_buffer: Union[str, BytesIO],
    batch_size: int = 1,
):
    from tool.darknet2pytorch import Darknet

    model = Darknet(cfgfile_path)

    model.print_network()
    model.load_weights(weightfile_path)
    print("Loading weights from %s... Done!" % weightfile_path)

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ["boxes", "confs"]

    if dynamic:
        x = torch.randn((1, 3, model.height, model.width), requires_grad=True)
        dynamic_axes = {
            "input": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "confs": {0: "batch_size"},
        }
        # Export the model
        print("Export the onnx model ...")
        torch.onnx.export(
            model,
            x,
            onnx_filepath_or_buffer,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print("Onnx model exporting done")

    else:
        x = torch.randn((batch_size, 3, model.height, model.width), requires_grad=True)
        torch.onnx.export(
            model,
            x,
            onnx_filepath_or_buffer,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
        )

        print("Onnx model exporting done")
