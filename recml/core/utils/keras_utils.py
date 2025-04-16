# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for training Keras models on Jax backend."""

from collections.abc import Mapping
from typing import Any

from absl import logging
import jax
import keras
import orbax.checkpoint as ocp
import tensorflow as tf

ORBAX_CHECKPOINT_DEFAULT_KEY = "default"


def _assert_variables_built(model: keras.Model):
  if not model.built or not model.optimizer.built:
    raise ValueError(
        "To use methods on `KerasOrbaxCheckpointManager`, your model and"
        f" optimizer must be built. Model built: {model.built}, Optimizer"
        f" built: {model.optimizer.built}"
    )


class KerasOrbaxCheckpointManager(ocp.CheckpointManager):
  """An Orbax checkpoint manager for Keras 3."""

  def __init__(
      self,
      checkpoint_dir: str,
      max_to_keep: int = 5,
      save_interval_epochs: int = 1,
  ):
    """Initializes a KerasOrbaxCheckpointManager.

    Args:
      checkpoint_dir: The directory to save checkpoints to.
      max_to_keep: The maximum number of checkpoints to keep.
      save_interval_epochs: The interval (in epochs) to save checkpoints.
    """
    super().__init__(
        directory=checkpoint_dir,
        checkpointers=ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_epochs,
            max_to_keep=max_to_keep,
        ),
    )

  def save_model_variables(
      self,
      model: keras.Model,
      epoch: int,
      logs: Mapping[str, Any] | None = None,
  ):
    _assert_variables_built(model)
    state = model._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        # metrics_variables is default to False because we don't want to save
        # metrics variables in the checkpoint. The metrics varibles are reset
        # after each epoch. We need to recalculate them after restoring from
        # the checkpoint.
        metrics_variables=False,
    )
    logging.info("Writing checkpoint for epoch %s...", epoch)
    self.save(step=epoch, items=state, metrics=logs)

  def restore_model_variables(self, model: keras.Model, epoch: int):
    _assert_variables_built(model)
    state = model._get_jax_state(  # pylint: disable=protected-access
        trainable_variables=True,
        non_trainable_variables=True,
        optimizer_variables=True,
        purge_model_variables=True,
    )
    logging.info("Restoring checkpoint for epoch %s...", epoch)
    model._jax_state_synced = False  # pylint: disable=protected-access

    def _restore(value):
      if isinstance(value, jax.Array):
        return ocp.type_handlers.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=value.sharding,
            global_shape=value.shape,
            dtype=value.dtype,
        )
      return ocp.type_handlers.RestoreArgs(
          restore_type=type(value),
          dtype=value.dtype if hasattr(value, "dtype") else None,
      )

    restore_args = jax.tree.map(_restore, state)
    # TODO(zixiangzhou): 'transforms' is a walkaround to avoid the error of
    # loading a checkpoint that has a different number of variables than the
    # current state because we don't want to load metrics_variables. But this
    # might lead to future bugs when the checkpoint does not exactly match the
    # defined model state. Currently, 'transforms' won't work if the order of
    # the variables is different from the checkpoint or new variables are added.
    # A better solution is to add keys for variables when checkpointing to use
    # the 'transforms' API (mapping by variable keys).
    restored_state = self.restore(
        step=epoch,
        args=ocp.args.PyTreeRestore(
            state,
            transforms={},
            restore_args=restore_args,
        ),
        directory=str(self.directory),
    )
    logging.info("Restored checkpoint for epoch %s.", epoch)
    model._initial_epoch = epoch + 1  # pylint: disable=protected-access
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    ) = restored_state
    model._jax_state = {  # pylint: disable=protected-access
        "trainable_variables": trainable_variables,
        "non_trainable_variables": non_trainable_variables,
        "optimizer_variables": optimizer_variables,
    }
    model.jax_state_sync()


class EpochOrbaxCheckpointAndRestoreCallback(keras.callbacks.Callback):
  """A callback for checkpointing and restoring state using Orbax."""

  def __init__(
      self,
      checkpoint_manager: KerasOrbaxCheckpointManager,
      marker_path: str | None = None,
  ):
    if keras.backend.backend() != "jax":
      raise ValueError(
          "`EpochOrbaxCheckpointAndRestoreCallback` is only supported on a"
          " `jax` backend."
      )

    self._checkpoint_manager = checkpoint_manager
    self._marker_path = marker_path
    # Marks the callback as async safe so batch end callbacks can be dispatched
    # asynchronously.
    self.async_safe = True

  def on_train_begin(self, logs: Mapping[str, Any] | None = None):
    if not self.model.built or not self.model.optimizer.built:
      raise ValueError(
          "To use `EpochOrbaxCheckpointAndRestoreCallback`, "
          "your model and optimizer must be built before you call `fit()`."
      )

    latest_epoch = self._checkpoint_manager.latest_step()
    if latest_epoch is not None:
      self._checkpoint_manager.restore_model_variables(self.model, latest_epoch)

  def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None):
    self._checkpoint_manager.save_model_variables(self.model, epoch, logs)

  def on_train_end(self, logs: Mapping[str, Any] | None = None):
    self._checkpoint_manager.wait_until_finished()
    if self._marker_path is not None and jax.process_index() == 0:
      with tf.io.gfile.GFile(self._marker_path, "w") as f:
        f.write("COMPLETED")


def restore_keras_model(
    model: keras.Model,
    checkpoint_dir: str,
    step: int | None = None,
    restore_optimizer_vars: bool = True,
):
  """Restores a Keras 3 Jax backend model from an Orbax checkpoint.

  Args:
    model: The Keras model to restore.
    checkpoint_dir: The directory containing the Orbax checkpoints.
    step: The step to restore the model to. If `None` then the latest checkpoint
      will be restored.
    restore_optimizer_vars: Whether to restore the optimizer variables.

  Raises:
    FileNotFoundError: If no checkpoints are found in the checkpoint directory.
    ValueError: If the specified step is not found in the checkpoint directory
      or if the model or the optimizer is not built.
  """
  if keras.backend.backend() != "jax":
    raise ValueError(
        "This function only supports restoring a Keras 3 Jax backend model from"
        " a TF Saved Model."
    )

  _assert_variables_built(model)

  metadata = ocp.path.step.latest_step_metadata(
      checkpoint_dir, ocp.path.step.standard_name_format()
  )
  if metadata is None:
    raise FileNotFoundError(
        f"No checkpoints found in {checkpoint_dir}. Please ensure that the"
        " checkpoint directory contains Orbax checkpoints."
    )
  if step is None:
    step = metadata.step
  elif step not in ocp.path.step.checkpoint_steps(checkpoint_dir):
    raise ValueError(
        f"Step {step} not found in {checkpoint_dir}. Please ensure you specify "
        "a valid step. Available steps: "
        f"{ocp.path.step.checkpoint_steps(checkpoint_dir)}"
    )

  checkpointer = ocp.Checkpointer(
      ocp.CompositeCheckpointHandler(**{
          ORBAX_CHECKPOINT_DEFAULT_KEY: ocp.handlers.PyTreeCheckpointHandler()
      })
  )
  state = model._get_jax_state(  # pylint: disable=protected-access
      trainable_variables=True,
      non_trainable_variables=True,
      optimizer_variables=restore_optimizer_vars,
      purge_model_variables=True,
  )
  model._jax_state_synced = False  # pylint: disable=protected-access

  # Delete the state to save memory.
  abstract_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
  jax.tree.map(
      lambda x: x.delete() if isinstance(x, jax.Array) else None, state
  )
  checkpoint_path = ocp.path.step.build_step_path(
      checkpoint_dir, ocp.path.step.standard_name_format(), step
  )
  # See the comment above for why we need to use transforms={} here.
  restored_state = checkpointer.restore(
      checkpoint_path,
      args=ocp.args.Composite(**{
          ORBAX_CHECKPOINT_DEFAULT_KEY: ocp.args.PyTreeRestore(
              item=abstract_state,
              transforms={},
              restore_args=ocp.checkpoint_utils.construct_restore_args(
                  abstract_state
              ),
          ),
      }),
  )[ORBAX_CHECKPOINT_DEFAULT_KEY]
  (
      trainable_variables,
      non_trainable_variables,
  ) = restored_state[:2]
  model._jax_state = {  # pylint: disable=protected-access
      "trainable_variables": trainable_variables,
      "non_trainable_variables": non_trainable_variables,
  }
  if restore_optimizer_vars:
    model._initial_epoch = step + 1  # pylint: disable=protected-access
    optimizer_variables = restored_state[2]
    model._jax_state["optimizer_variables"] = optimizer_variables  # pylint: disable=protected-access
  model.jax_state_sync()


# TODO(b/343544467): Support logging metrics more frequently.
class EpochSummaryCallback(keras.callbacks.TensorBoard):
  """A custom summary callback that only reports epoch metrics."""

  def __init__(
      self,
      log_dir: str,
      steps_per_epoch: int,
      write_steps_per_second: bool = True,
  ):
    super().__init__(
        log_dir,
        write_steps_per_second=write_steps_per_second,
        update_freq="epoch",
        write_graph=False,
    )
    self._steps_per_epoch = steps_per_epoch
    self._num_params = None
    # Marks the callback as async safe so batch end callbacks can be dispatched
    # asynchronously.
    self.async_safe = True

  def _get_num_params(self, training: bool) -> dict[str, int]:
    if self._num_params is None:
      self._num_params = {
          "num_params/trainable": keras.src.utils.summary_utils.count_params(
              self.model.trainable_variables
          ),
          "num_params/non_trainable": (
              keras.src.utils.summary_utils.count_params(
                  self.model.non_trainable_variables
              )
          ),
          "num_params/optimizer": keras.src.utils.summary_utils.count_params(
              self.model.optimizer.variables
          ),
      }
      self._num_params["num_params/total"] = sum(self._num_params.values())
    if not training:
      return {"val_" + k: v for k, v in self._num_params.items()}
    return self._num_params

  def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
    if not logs:
      return

    step = epoch * self._steps_per_epoch
    train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
    val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
    train_logs = self._collect_learning_rate(train_logs)
    if self.write_steps_per_second:
      train_logs["steps_per_second"] = self._compute_steps_per_second()

    if train_logs:
      num_params = self._get_num_params(training=True)
      logs.update(num_params)
      train_logs.update(num_params)
      with self._train_writer.as_default():
        for name, value in train_logs.items():
          self.summary.scalar(name, value, step=step)

    if val_logs:
      num_params = self._get_num_params(training=False)
      logs.update(num_params)
      val_logs.update(num_params)
      with self._val_writer.as_default():
        for name, value in val_logs.items():
          self.summary.scalar(name.removeprefix("val_"), value, step=step)

  def _collect_learning_rate(self, logs: Any) -> Any:
    if not self.model:
      return logs
    optimizer = self.model.optimizer
    if isinstance(optimizer, keras.optimizers.Optimizer):
      if hasattr(optimizer, "learning_rates"):
        learning_rates = optimizer.learning_rates
        if isinstance(learning_rates, Mapping):
          for k, v in learning_rates.items():
            logs["learning_rate/" + k] = float(keras.ops.convert_to_numpy(v))
      else:
        logs["learning_rate"] = float(
            keras.ops.convert_to_numpy(optimizer.learning_rate)
        )
    return logs

  def on_test_end(self, logs=None):
    self._pop_writer()
