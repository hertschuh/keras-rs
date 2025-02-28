import keras
from keras import ops
from keras.layers import deserialize
from keras.layers import serialize

from keras_rs.src import testing
from keras_rs.src.layers.retrieval import sampling_probability_correction


class SamplingProbabilityCorrectionTest(testing.TestCase):
    def test_call(self):
        shape = (10, 20)  # (num_queries, num_candidates)
        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, seed=rng)
        probs = keras.random.uniform(shape[1:], seed=rng)

        # Verifies logits are always less than corrected logits.
        layer = sampling_probability_correction.SamplingProbablityCorrection()
        corrected_logits = layer(logits, probs)
        self.assertAllClose(ops.less(logits, corrected_logits), ops.ones(shape))

        # Set some of the probabilities to 0.
        probs_with_zeros = ops.multiply(
            probs,
            ops.cast(
                ops.greater_equal(
                    keras.random.uniform(probs.shape, seed=rng), 0.5
                ),
                dtype="float32",
            ),
        )

        # Verifies logits are always less than corrected logits.
        corrected_logits_with_zeros = layer(logits, probs_with_zeros)
        self.assertAllClose(
            ops.less(logits, corrected_logits_with_zeros), ops.ones(shape)
        )

    def test_predict(self):
        # Note: for predict, we test with probabilities that have a batch dim.
        shape = (10, 20)  # (num_queries, num_candidates)

        layer = sampling_probability_correction.SamplingProbablityCorrection()
        in_logits = keras.layers.Input(shape=shape[1:])
        in_probs = keras.layers.Input(shape=shape[1:])
        out_logits = layer(in_logits, in_probs)
        model = keras.Model([in_logits, in_probs], out_logits)

        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, seed=rng)
        probs = keras.random.uniform(shape, seed=rng)

        model.predict([logits, probs], batch_size=10)

    def test_serialization(self):
        layer = sampling_probability_correction.SamplingProbablityCorrection()
        restored = deserialize(serialize(layer))
        self.assertDictEqual(layer.get_config(), restored.get_config())

    def test_model_saving(self):
        shape = (10, 20)  # (num_queries, num_candidates)

        layer = sampling_probability_correction.SamplingProbablityCorrection()
        in_logits = keras.layers.Input(shape=shape[1:])
        in_probs = keras.layers.Input(batch_shape=shape[1:])
        out_logits = layer(in_logits, in_probs)
        model = keras.Model([in_logits, in_probs], out_logits)

        rng = keras.random.SeedGenerator(42)
        logits = keras.random.uniform(shape, seed=rng)
        probs = keras.random.uniform(shape[1:], seed=rng)

        self.run_model_saving_test(model=model, input_data=[logits, probs])
