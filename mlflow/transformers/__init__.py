"""
The ``mlflow.transformers`` module provides an API for logging and loading
transformers pretrained models. This module exports transformers models with
the following flavors:

transformers (native) format
    This is the main flavor that can be loaded back into transformers.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch
    inference.
"""
import logging
import os
import yaml
import warnings

import numpy as np
import pandas as pd
import posixpath

import mlflow
import shutil
from mlflow import pyfunc
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import (
    TempDir,
    write_to,
)
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "transformers"

_SAVED_TRANSFORMERS_MODEL_DIR_NAME = "model.pth"
_SAVED_TRANSFORMERS_TOKENIZER_DIR_NAME = "tokenizer.pth"
_TRANSFORMERS_PIPELINE_TASK_FILE_NAME = "pipeline_task.txt"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """Get default pip requirements.

    :return: A list of default pip requirements for MLflow Models produced by
             this flavor.  Calls to :func:`save_model()` and
             :func:`log_model()` produce a pip environment that, at minimum,
             contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "transformers",
                "torch",
            ],
        )
    )


def get_default_conda_env():
    """
    :return: The default Conda environment as a dictionary for MLflow Models
             produced by calls to :func:`save_model()` and :func:`log_model()`.

    .. code-block:: python
        :caption: Example

        import mlflow.transformers

        # Log transformers model
        with mlflow.start_run() as run:
            mlflow.transformers.log_model(model, "model")

        # Fetch the associated conda environment
        env = mlflow.transformers.get_default_conda_env()
        print("conda env: {}".format(env))

    .. code-block:: text
        :caption: Output

        conda env {'name': 'mlflow-env',
                   'channels': ['conda-forge'],
                   'dependencies': ['python=3.10.8',
                                    {'pip': ['transformers==4.24.0',
                                             'mlflow',
                                             'cloudpickle==2.2.0']}]}
    """
    return _mlflow_conda_env(
        additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="transformers"))
def log_model(
    transformers_model,
    transformers_tokenizer,
    artifact_path,
    task=None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a transformers model as an MLflow artifact for the current run.

    :param transformers_model: The transformers model to be saved. Must be a
                               subclass of ``transformers.PreTrainedModel``.

                               Any code dependencies of the model's class,
                               including the class definition itself, should be
                               included in one of the following locations:

                               - The package(s) listed in the model's Conda
                                 environment, specified by the ``conda_env``
                                 parameter.
                               - One or more of the files specified by the
                               ``code_paths`` parameter.

    :param transformers_tokenizer: The transformers tokenizer to be saved.
                                   Must be a subclass of
                                   ``transformers.PreTrainedTokenizer`` or
                                   ``transformers.PreTrainedTokenizerFast``.

                                   Any code dependencies of the tokenizer's
                                   class, including the class definition
                                   itself, should be included in one of the
                                   following locations:

                                   - The package(s) listed in the model's Conda
                                     environment, specified by the
                                     ``conda_env`` parameter.
                                   - One or more of the files specified by the
                                   ``code_paths`` parameter.

    :param artifact_path: Run-relative artifact path.
    :param task: String name defining the ``transformers.pipeline`` task the
                 model and tokenizer are intended for.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file
                       dependencies (or directories containing file
                       dependencies). These files are *prepended* to the system
                       path when the model is loaded.

    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a
                                  registered model if one with the given name
                                  does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema
                      <mlflow.types.Schema>`.  The model signature can be
                      :py:func:`inferred <mlflow.models.infer_signature>` from
                      datasets with valid model input (e.g. the training
                      dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training
                      dataset), for example:

                      .. code-block:: python

                         from mlflow.models.signature import infer_signature
                         train = df.drop_column("target_label")
                         predictions = ... # compute model predictions
                         signature = infer_signature(train, predictions)

    :param input_example: Input example provides one or several instances of
                          valid model input. The example can be used as a hint
                          of what data to feed the model. The given example can
                          be a Pandas DataFrame where the given example will be
                          serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be
                          serialized to json by converting it to a list. Bytes
                          are base64-encoded.

    :param await_registration_for: Number of seconds to wait for the model
                                   version to finish being created and is in
                                   ``READY`` status. By default, the function
                                   waits for five minutes. Specify 0 or None to
                                   skip waiting.

    :param requirements_file:

        .. warning::

           ``requirements_file`` has been deprecated. Please use
           ``pip_requirements`` instead.

        A string containing the path to requirements file. Remote URIs are
        resolved to absolute filesystem paths. For example, consider the
        following ``requirements_file`` string:

        .. code-block:: python

           requirements_file = "s3://my-bucket/path/to/my_file"

        In this case, the ``"my_file"`` requirements file is downloaded from
        S3. If ``None``, no requirements file is added to the model.

    :param extra_files: A list containing the paths to corresponding extra
                        files. Remote URIs are resolved to absolute filesystem
                        paths.  For example, consider the following
                        ``extra_files`` list -

                        extra_files = ["s3://my-bucket/path/to/my_file1",
                                       "s3://my-bucket/path/to/my_file2"]

                        In this case, the ``"my_file1 & my_file2"`` extra file
                        is downloaded from S3.

                        If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``PreTrainedModel.save_pretrained``
                   method.

    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance
             that contains the metadata of the logged model.

    .. code-block:: python
       :caption: Example

       from transformers import (
           AutoModelForQuestionAnswering,
           AutoTokenizer,
       )

       model_name = "deepset/roberta-base-squad2"

       # Load model & tokenizer
       model = AutoModelForQuestionAnswering.from_pretrained(model_name)
       tokenizer = AutoTokenizer.from_pretrained(model_name)

       # Log the pretrained model (and tokenizer)
       with mlflow.start_run() as run:
           mlflow.transformers.log_model(
               model,
               tokenizer,
               "model",
               task="question-answering")

       # Fetch the logged model artifacts
       print("run_id: {}".format(run.info.run_id))
       artifact_path = "model/data"
       artifacts = [
           f.path for f in MlflowClient().list_artifacts(
               run.info.run_id,
               artifact_path)
       ]
       print("artifacts: {}".format(artifacts))

    .. code-block:: text
       :caption: Output

       run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
       artifacts: ['model/data/model.pth']
                   'model/data/tokenizer.pth']

    .. figure:: ../_static/images/transformers_logged_models.png

        transformers logged models
    """
    # TODO: update logged models image above

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.transformers,
        transformers_model=transformers_model,
        transformers_tokenizer=transformers_tokenizer,
        task=task,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        requirements_file=requirements_file,
        extra_files=extra_files,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="transformers"))
def save_model(
    transformers_model,
    transformers_tokenizer,
    path,
    task=None,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Save a transformers model and tokenizer to a path on the local file system.

    :param transformers_model: The transformers model to be saved. Must be a
                               subclass of ``transformers.PreTrainedModel``.

                               Any code dependencies of the model's class,
                               including the class definition itself, should be
                               included in one of the following locations:

                               - The package(s) listed in the model's Conda
                                 environment, specified by the ``conda_env``
                                 parameter.
                               - One or more of the files specified by the
                               ``code_paths`` parameter.

    :param transformers_tokenizer: The transformers tokenizer to be saved.
                                   Must be a subclass of
                                   ``transformers.PreTrainedTokenizer`` or
                                   ``transformers.PreTrainedTokenizerFast``.

                                   Any code dependencies of the tokenizer's
                                   class, including the class definition
                                   itself, should be included in one of the
                                   following locations:

                                   - The package(s) listed in the model's Conda
                                     environment, specified by the
                                     ``conda_env`` parameter.
                                   - One or more of the files specified by the
                                   ``code_paths`` parameter.

    :param path: Local path where the model is to be saved.
    :param task: String name defining the ``transformers.pipeline`` task the
                 model and tokenizer are intended for.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being
                         added to.
    :param code_paths: A list of local filesystem paths to Python file
                       dependencies (or directories containing file
                       dependencies). These files are *prepended* to the system
                       path when the model is loaded.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema
                      <mlflow.types.Schema>`.  The model signature can be
                      :py:func:`inferred <mlflow.models.infer_signature>` from
                      datasets with valid model input (e.g. the training
                      dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training
                      dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)

    :param input_example: Input example provides one or several instances of
                          valid model input. The example can be used as a hint
                          of what data to feed the model. The given example can
                          be a Pandas DataFrame where the given example will be
                          serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be
                          serialized to json by converting it to a list. Bytes
                          are base64-encoded.

    :param requirements_file:

        .. warning::

           ``requirements_file`` has been deprecated. Please use
           ``pip_requirements`` instead.

        A string containing the path to requirements file. Remote URIs are
        resolved to absolute filesystem paths. For example, consider the
        following ``requirements_file`` string:

        .. code-block:: python

           requirements_file = "s3://my-bucket/path/to/my_file"

        In this case, the ``"my_file"`` requirements file is downloaded from
        S3. If ``None``, no requirements file is added to the model.

    :param extra_files: A list containing the paths to corresponding extra
                        files. Remote URIs are resolved to absolute filesystem
                        paths.  For example, consider the following
                        ``extra_files`` list -

                        extra_files = ["s3://my-bucket/path/to/my_file1",
                                       "s3://my-bucket/path/to/my_file2"]

                        In this case, the ``"my_file1 & my_file2"`` extra file
                        is downloaded from S3.

                        If ``None``, no extra files are added to the model.

    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``PreTrainedModel.save_pretrained``
                   method.

    .. code-block:: python
       :caption: Example

       import os

       from transformers import (
           AutoModelForQuestionAnswering,
           AutoTokenizer,
           pipeline,
       )

       model_name = "deepset/roberta-base-squad2"

       # Load model & tokenizer
       model = AutoModelForQuestionAnswering.from_pretrained(model_name)
       tokenizer = AutoTokenizer.from_pretrained(model_name)

       # Save transformers model and tokenizer to current working directory
       with mlflow.start_run() as run:
           mlflow.transformers.save_model(
               model,
               tokenizer,
               "model",
               task="question-answering")

       # Load model and tokenizer for inference
       model_path = "model"
       model_uri = "{}/{}".format(os.getcwd(), model_path)
       qa_pipeline = mlflow.pytorch.load_model(model_uri)
       print(f"Loaded {model_path}:")
       question = "Who am I?"
       context = "My name is Brian and I live in New York."
       result = qa_pipeline(question=question, context=context)
       print(f"{question = }")
       print(f"{context = }")
       print(f"{result = }")

    .. code-block:: text
       :caption: Output

       Loaded model:
       question = 'Who am I?'
       context = 'My name is Brian and I live in New York.'
       result = {'score': 0.988544225692749, 'start': 11, 'end': 16,
                 'answer': 'Brian'}
    """

    import transformers

    _validate_env_arguments(
        conda_env, pip_requirements, extra_pip_requirements)

    if not isinstance(transformers_model, transformers.PreTrainedModel):
        raise TypeError(
            "Argument 'transformers_model' should be a "
            "transformers.PreTrainedModel"
        )
    if not isinstance(
            transformers_tokenizer, transformers.PreTrainedTokenizerBase):
        raise TypeError(
            "Argument 'transformers_tokenizer' should inherit from"
            "transformers.PreTrainedTokenizerBase"
        )
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    model_data_subpath = "data"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    # Persist task if specified
    # TODO: Stop persisting this information to the filesystem once we have a
    # mechanism for supplying the MLmodel configuration to
    # `mlflow.pytorch._load_pyfunc`
    if task is not None:
        task_path = os.path.join(
            model_data_path, _TRANSFORMERS_PIPELINE_TASK_FILE_NAME)
        with open(task_path, "w") as f:
            f.write(f'{task}\n')

    # Save transformers model
    model_path = os.path.join(
        model_data_path, _SAVED_TRANSFORMERS_MODEL_DIR_NAME)
    transformers_model.save_pretrained(model_path)

    # Save transformers tokenizer
    tokenizer_path = os.path.join(
        model_data_path, _SAVED_TRANSFORMERS_TOKENIZER_DIR_NAME)
    transformers_model.save_pretrained(tokenizer_path)

    transformers_artifacts_config = {}

    if extra_files:
        transformers_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file,
                    output_path=tmp_extra_files_dir.path(),
                )
                rel_path = posixpath.join(
                    _EXTRA_FILES_KEY, os.path.basename(extra_file))
                transformers_artifacts_config[_EXTRA_FILES_KEY].append(
                    {"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )

    if requirements_file:
        warnings.warn(
            "`requirements_file` has been deprecated. Please use "
            "`pip_requirements` instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not isinstance(requirements_file, str):
            raise TypeError("Path to requirements file should be a string")

        with TempDir() as tmp_requirements_dir:
            _download_artifact_from_uri(
                artifact_uri=requirements_file,
                output_path=tmp_requirements_dir.path(),
            )
            rel_path = os.path.basename(requirements_file)
            transformers_artifacts_config[_REQUIREMENTS_FILE_KEY] = \
                {"path": rel_path}
            shutil.move(tmp_requirements_dir.path(rel_path), path)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        transformers_version=str(transformers.__version__),
        code=code_dir_subpath,
        **transformers_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.transformers",
        data=model_data_subpath,
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during
            # the dependency inference, `mlflow_model.save` must be called
            # beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = \
            _process_pip_requirements(
                default_reqs,
                pip_requirements,
                extra_pip_requirements,
            )
    else:
        conda_env, pip_requirements, pip_constraints = \
            _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(
            os.path.join(path, _CONSTRAINTS_FILE_NAME),
            "\n".join(pip_constraints),
        )

    if not requirements_file:
        # Save `requirements.txt`
        write_to(
            os.path.join(path, _REQUIREMENTS_FILE_NAME),
            "\n".join(pip_requirements),
        )

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_model(path, **kwargs):
    """
    :param path: The path to a saved Transformers model and tokenizer.
    :param kwargs: Additional kwargs to pass to the
                   ``PreTrainedModel.from_pretrained`` method.
    """
    import transformers

    # `path` is a directory containing a saved transformers model and
    # tokenizer
    model_path = os.path.join(path, _SAVED_TRANSFORMERS_MODEL_DIR_NAME)
    tokenizer_path = os.path.join(
        path, _SAVED_TRANSFORMERS_TOKENIZER_DIR_NAME)
    task_path = os.path.join(
        path, _TRANSFORMERS_PIPELINE_TASK_FILE_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Specified path {path!r} does not contain model directory '
            f'{_SAVED_TRANSFORMERS_MODEL_DIR_NAME!r}'
        )
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f'Specified path {path!r} does not contain model directory '
            f'{_SAVED_TRANSFORMERS_TOKENIZER_DIR_NAME!r}'
        )

    model = transformers.PreTrainedModel.from_pretrained(
            model_path, kwargs=kwargs)
    # TODO: Consider support for tokenizer kwargs
    tokenizer = transformers.PreTrainedTokenizerBase.from_pretrained(
        tokenizer_path)

    path = None
    if os.path.exists(task_path):
        path = open(task_path).read().strip()

    return model, tokenizer, path


def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a transformers model and tokenizer from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for
      example:

      - ``/Users/me/path/to/local/model``
      - ``relative/path/to/local/model``
      - ``s3://my_bucket/path/to/model``
      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
      - ``models:/<model_name>/<model_version>``
      - ``models:/<model_name>/<stage>``

      For more information about supported URI schemes, see
      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model
                     artifact.  This directory must already exist. If
                     unspecified, a local output path will be created.

    :param kwargs: kwargs to pass to ``PreTrainedModel.from_pretrained``
                   method.
    :return: Tuple of transformers pretrained model and tokenizer. If no
             tokenizer was saved, the second tuple element is ``None``.

    .. code-block:: python
       :caption: Example

       from transformers import (
           AutoModelForQuestionAnswering,
           AutoTokenizer,
           pipeline,
       )

       model_name = "deepset/roberta-base-squad2"

       # Load model & tokenizer
       model = AutoModelForQuestionAnswering.from_pretrained(model_name)
       tokenizer = AutoTokenizer.from_pretrained(model_name)

       # Log the model
       with mlflow.start_run() as run:
           mlflow.transformers.log_model(
               model,
               tokenizer,
               "model",
               task="question-answering")

       # Inference after loading the logged model
       model_uri = "runs:/{}/model".format(run.info.run_id)
       qa_pipeline = mlflow.pytorch.load_model(model_uri)
       print(f"Loaded {model_path}:")
       question = "Who am I?"
       context = "My name is Brian and I live in New York."
       result = qa_pipeline(question=question, context=context)
       print(f"{question = }")
       print(f"{context = }")
       print(f"{result = }")

    .. code-block:: text
       :caption: Output

       Loaded model:
       question = 'Who am I?'
       context = 'My name is Brian and I live in New York.'
       result = {'score': 0.988544225692749, 'start': 11, 'end': 16,
                 'answer': 'Brian'}
    """

    import transformers

    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path)
    transformers_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, transformers_conf)

    if transformers.__version__ != transformers_conf["transformers_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed transformers "
            "version '%s'",
            transformers_conf["transformers_version"],
            transformers.__version__,
        )
    transformers_artifacts_path = os.path.join(
        local_model_path, transformers_conf["model_data"])
    model, tokenizer, task = _load_model(
        path=transformers_artifacts_path, **kwargs)

    if task is not None:
        return transformers.pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
        )
    else:
        return model, tokenizer


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the
                 ``transformers`` flavor.
    """
    if 'task' not in kwargs:
        raise KeyError(
            'Must include kwarg `task` when calling `mlflow.'
        )
    return _PipelineWrapper(_load_model(path, **kwargs))


class _PipelineWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as ``pandas.DataFrame``
    """

    def __init__(self, transformers_pipeline):
        self.transformers_pipeline = transformers_pipeline

    def predict(self, data):
        if isinstance(data, pd.DataFrame):
            inp_data = data.values
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, list):
            # Assume k x N inputs for N sets of k arguments to pipeline
            inp_data = np.vstack(data).transpose()
        elif isinstance(data, dict):
            inp_data = np.vstack(list(data.values())).transpose()
        else:
            raise TypeError(f"Unrecognized input type {type(data)}")

        results = []
        for inps in inp_data:
            results.append(self.pipeline(*inps))

        if isinstance(data, pd.DataFrame):
            results = pd.DataFrame.from_records(results)
            results.index = data.index

        return results
