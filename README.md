# Keras Recommenders

Keras Recommenders is a library for building recommender systems on top of
Keras 3. Keras Recommenders works natively with TensorFlow, JAX, or PyTorch. It
provides a collection of building blocks which help with the full workflow of
creating a recommender system. As it's built on Keras 3, models can be trained
and serialized in any framework and re-used in another without costly
migrations.

This library is an extension of the core Keras API; all high-level modules
receive that same level of polish as core Keras. If you are familiar with Keras,
congratulations! You already understand most of Keras Recommenders.

## Configuring your backend

If you have Keras 3 installed in your environment (see installation above), you
can use Keras Recommenders with any of JAX, TensorFlow and PyTorch. To do so,
set the `KERAS_BACKEND` environment variable. For example:

```shell
export KERAS_BACKEND=jax
```

Or in Colab, with:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras_recommenders
```

> [!IMPORTANT]
> Make sure to set the `KERAS_BACKEND` **before** importing any Keras libraries;
> it will be used to set up Keras when it is first imported.

## Compatibility

We follow [Semantic Versioning](https://semver.org/), and plan to provide
backwards compatibility guarantees both for code and saved models built with our
components. While we continue with pre-release `0.y.z` development, we may break
compatibility at any time and APIs should not be considered stable.

## Citing Keras Recommenders

If Keras Recommenders helps your research, we appreciate your citations.
Here is the BibTeX entry:

```bibtex
@misc{kerasrecommenders2024,
  title={KerasRecommenders},
  author={Hertschuh, Fabien and  Chollet, Fran\c{c}ois and others},
  year={2024},
  howpublished={\url{https://github.com/keras-team/keras-recommenders}},
}
```

## Acknowledgements

Thank you to all of our wonderful contributors!

<a href="https://github.com/keras-team/keras-recommenders/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=keras-team/keras-recommenders" />
</a>