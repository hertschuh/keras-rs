import argparse
import importlib
import os
import libtpu

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = libtpu.get_library_path()
os.environ["TF_XLA_FLAGS"] = (
    "--tf_mlir_enable_mlir_bridge=true "
    "--tf_mlir_enable_convert_control_to_data_outputs_pass=true "
    "--tf_mlir_enable_merge_control_flow_pass=true"
)

import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

import keras_rs

from .dataloader import DataLoader
from .model import DLRMDCNV2

SEED = 1337


def main(
    file_pattern,
    val_file_pattern,
    dense_features,
    large_emb_features,
    small_emb_features,
    label,
    shuffle_buffer,
    embedding_dim,
    allow_id_dropping,
    max_ids_per_partition,
    max_unique_ids_per_partition,
    embedding_learning_rate,
    bottom_mlp_dims,
    top_mlp_dims,
    num_dcn_layers,
    dcn_projection_dim,
    learning_rate,
    global_batch_size,
    file_batch_size,
    num_epochs,
):
    print("### TF", tf.__version__)

    print("### Resolver")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    # print("### Connect")
    # tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    print("### Initialize")
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    print("### Devices", tf.config.list_logical_devices('TPU'))

    print("### TPU metadata")
    tpu_metadata = resolver.get_tpu_system_metadata()
    device_assignment = tf.tpu.experimental.DeviceAssignment.build(
        topology, num_replicas=tpu_metadata.num_cores
    )
    print("### Strategy...")
    strategy = tf.distribute.TPUStrategy(
        resolver, experimental_device_assignment=device_assignment
    )

    print("### strategy", type(strategy), strategy)
    print("### strategy.extended", strategy.extended)
    print("### strategy.extended.tpu_hardware_feature", strategy.extended.tpu_hardware_feature)
    print("### strategy.extended.tpu_hardware_feature.embedding_feature", strategy.extended.tpu_hardware_feature.embedding_feature)
    print("### strategy.extended.tpu_hardware_feature.num_embedding_devices_per_chip", strategy.extended.tpu_hardware_feature.num_embedding_devices_per_chip)

    num_processes = 1  # TODO distribution._num_process

    per_host_batch_size = global_batch_size // num_processes

    # === Distributed embeddings' configs for lookup features ===
    # Set XLA flags.
    # os.environ["XLA_FLAGS"] = (
    #     "--xla_sparse_core_max_ids_per_partition_per_sample="
    #     f"{max_ids_per_partition} "
    #     "--xla_sparse_core_max_unique_ids_per_partition_per_sample="
    #     f"{max_unique_ids_per_partition}"
    # )
    feature_configs = {}
    for large_emb_feature in large_emb_features:
        feature_name = large_emb_feature["new_name"]
        vocabulary_size = large_emb_feature["vocabulary_size"]
        feature_list_length = large_emb_feature["feature_list_length"]

        table_config = keras_rs.layers.TableConfig(
            name=f"{feature_name}_table",
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            # TODO(abheesht): Verify.
            initializer=keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_in",
                distribution="uniform",
                seed=SEED,
            ),
            optimizer=keras.optimizers.Adagrad(
                learning_rate=embedding_learning_rate
            ),
            combiner="sum",
            placement="sparsecore",
            # TODO: These two args are not getting passed down to
            # `jax-tpu-embedding` properly, seems like.
            max_ids_per_partition=max_ids_per_partition,
            max_unique_ids_per_partition=max_unique_ids_per_partition,
        )
        feature_configs[f"{feature_name}_id"] = keras_rs.layers.FeatureConfig(
            name=feature_name,
            table=table_config,
            # TODO: Verify whether it should be `(bsz, 1)` or
            # `(bsz, feature_list_length)`.
            input_shape=(per_host_batch_size, feature_list_length),
            output_shape=(per_host_batch_size, embedding_dim),
        )

    # === Instantiate model ===
    # We instantiate the model first, because we need to preprocess large
    # embedding feature inputs using the distributed embedding layer defined
    # inside the model class.
    print("===== Initialising model =====")
    with strategy.scope():
        model = DLRMDCNV2(
            large_emb_feature_configs=feature_configs,
            small_emb_features=small_emb_features,
            embedding_dim=embedding_dim,
            bottom_mlp_dims=bottom_mlp_dims,
            top_mlp_dims=top_mlp_dims,
            num_dcn_layers=num_dcn_layers,
            dcn_projection_dim=dcn_projection_dim,
            seed=SEED,
            dtype="float32",
            name="dlrm_dcn_v2",
        )
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adagrad(learning_rate=learning_rate),
            metrics=[keras.metrics.BinaryAccuracy()],
        )

    # === Load dataset ===
    print("===== Loading dataset =====")
    train_ds = DataLoader(
        file_pattern=file_pattern,
        batch_size=global_batch_size,
        file_batch_size=file_batch_size,
        dense_features=dense_features,
        large_emb_features=large_emb_features,
        small_emb_features=small_emb_features,
        label=label,
        training=True,
    ).create_dataset(
        # process_id=distribution._process_id, # TODO
        num_processes=num_processes,
        shuffle_buffer=shuffle_buffer,
    )
    # For the multi-host case, the dataset has to be distributed manually.
    # See note here:
    # https://github.com/keras-team/keras-rs/blob/main/keras_rs/src/layers/embedding/base_distributed_embedding.py#L352-L363.
    # if num_processes > 1:
    #     train_ds = distribution.distribute_dataset(train_ds)
    #     # eval_ds = distribution.distribute_dataset(eval_ds)
    #     distribution.auto_shard_dataset = False

    # Print one sample.
    for element in train_ds.take(1):
        print(">>> train sample", element[0])

    # def generator(dataset, training=False):
    #     """Converts tf.data Dataset to a Python generator and preprocesses
    #     large embedding features.
    #     """
    #     for features, labels in dataset:
    #         preprocessed_large_embeddings = model.embedding_layer.preprocess(
    #             features["large_emb_inputs"], training=training
    #         )

    #         x = {
    #             "dense_input": features["dense_input"],
    #             "large_emb_inputs": preprocessed_large_embeddings,
    #             "small_emb_inputs": features["small_emb_inputs"],
    #         }
    #         y = labels
    #         yield (x, y)

    # train_generator = generator(train_ds, training=True)
    # eval_generator = generator(eval_ds, training=False)
    with strategy.scope():
        for element in train_ds.take(1):
            model(element[0])

    # Train the model.
    with strategy.scope():
        model.fit(train_ds, epochs=num_epochs)


if __name__ == "__main__":
    keras.config.disable_traceback_filtering()

    print("====== Launching train script =======")
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the DLRM-DCNv2 model on the Criteo dataset (MLPerf)"
        )
    )
    parser.add_argument(
        "--config_name", type=str, help="Name of the `.py` config file."
    )
    args = parser.parse_args()

    print(f"===== Reading config from {args.config_name} ======")
    config = importlib.import_module(
        f".configs.{args.config_name}", package=__package__
    ).config

    # === Unpack args from config ===

    # == Dataset config ==
    ds_cfg = config["dataset"]
    # File path
    file_pattern = ds_cfg["file_pattern"]
    val_file_pattern = ds_cfg.get("val_file_pattern", None)
    # File batch size
    file_batch_size = ds_cfg.get("file_batch_size", None)
    # Shuffling
    shuffle_buffer = ds_cfg.get("shuffle_buffer", None)
    # Features
    label = ds_cfg["label"]
    dense_features = ds_cfg["dense"]
    emb_features = ds_cfg["lookup"]

    # == Model config ==
    model_cfg = config["model"]
    # Embedding
    embedding_dim = model_cfg["embedding_dim"]
    allow_id_dropping = model_cfg["allow_id_dropping"]
    embedding_threshold = model_cfg["embedding_threshold"]
    max_ids_per_partition = model_cfg["max_ids_per_partition"]
    max_unique_ids_per_partition = model_cfg["max_unique_ids_per_partition"]
    embedding_learning_rate = model_cfg["learning_rate"]
    # MLP
    bottom_mlp_dims = model_cfg["bottom_mlp_dims"]
    top_mlp_dims = model_cfg["top_mlp_dims"]
    # DCN
    num_dcn_layers = model_cfg["num_dcn_layers"]
    dcn_projection_dim = model_cfg["dcn_projection_dim"]

    # == Training config ==
    training_cfg = config["training"]
    learning_rate = training_cfg["learning_rate"]
    global_batch_size = training_cfg["global_batch_size"]
    num_epochs = training_cfg["num_epochs"]

    # For features which have vocabulary_size < embedding_threshold, we can
    # just do a normal dense lookup for those instead of having distributed
    # embeddings. We could ideally pass `placement = default_device` to
    # `keras_rs.layers.TableConfig` directly (and wouldn't have to do this
    # separation of features), but doing it that way will necessarily require
    # a separate optimiser for the embedding layer.
    small_emb_features = []
    large_emb_features = []
    for emb_feature in emb_features:
        if emb_feature["vocabulary_size"] < embedding_threshold:
            small_emb_features.append(emb_feature)
        else:
            large_emb_features.append(emb_feature)

    main(
        file_pattern,
        val_file_pattern,
        dense_features,
        large_emb_features,
        small_emb_features,
        label,
        shuffle_buffer,
        embedding_dim,
        allow_id_dropping,
        max_ids_per_partition,
        max_unique_ids_per_partition,
        embedding_learning_rate,
        bottom_mlp_dims,
        top_mlp_dims,
        num_dcn_layers,
        dcn_projection_dim,
        learning_rate,
        global_batch_size,
        file_batch_size,
        num_epochs,
    )
