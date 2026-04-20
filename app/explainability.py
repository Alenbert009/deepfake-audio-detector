import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display


def find_last_conv_layer(model):
    """
    Find the last Conv2D layer in the trained CNN.

    Grad-CAM uses the last convolutional layer because
    it contains spatial information about the input.
    """

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise ValueError(
        "No Conv2D layer was found in the model."
    )


def make_gradcam_heatmap(
    model,
    spectrogram,
    predicted_class
):
    """
    Generate a Grad-CAM heatmap for a binary sigmoid CNN.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained CNN model.

    spectrogram : numpy.ndarray
        Model input with shape:
        (1, height, width, channels)

    predicted_class : int
        0 = Real
        1 = Fake

    Returns
    -------
    heatmap : numpy.ndarray
        Normalized Grad-CAM heatmap.
    """

    # --------------------------------------------------
    # 1. Find the last convolutional layer
    # --------------------------------------------------

    last_conv_layer = find_last_conv_layer(model)

    # --------------------------------------------------
    # 2. Create a model that gives us:
    #
    #    a) Last convolutional feature maps
    #    b) Final model prediction
    # --------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # 3. Convert input to TensorFlow tensor

    input_tensor = tf.cast(
        spectrogram,
        tf.float32
    )

    # 4. Record operations for gradient calculation
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False
        )

        # predictions shape:
        #
        # (1, 1)
        #
        # Example:
        # [[0.85]]
        #
        # 0.85 = Fake probability

        fake_probability = predictions[:, 0]

        #model is binary sigmoid:
        #
        # Fake score = P(Fake)
        # Real score = 1 - P(Fake)
        # ------------------------------------------------

        if predicted_class == 1:

            target_score = fake_probability

        else:

            target_score = 1.0 - fake_probability

    # --------------------------------------------------
    # 5. Calculate gradients
    #
    # How strongly does each convolutional feature
    # affect the selected class?
    # --------------------------------------------------

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )

    # --------------------------------------------------
    # 6. Global average pooling of gradients
    #
    # Gives an importance weight to every feature map.
    # --------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------
    # 7. Remove batch dimension
    # --------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    # --------------------------------------------------
    # 8. Weight each feature map by its importance
    # --------------------------------------------------

    weighted_features = (
        conv_outputs * pooled_gradients
    )

    # --------------------------------------------------
    # 9. Combine all feature maps
    # --------------------------------------------------

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # --------------------------------------------------
    # 10. Keep only positive contributions
    # --------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------
    # 11. Normalize heatmap to 0-1
    # --------------------------------------------------

    max_value = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def create_gradcam_figure(
    spectrogram,
    heatmap,
    alpha=0.45
):
    """
    Overlay Grad-CAM heatmap on the Mel Spectrogram.

    Parameters
    ----------
    spectrogram : numpy.ndarray
        Original Mel Spectrogram.

    heatmap : numpy.ndarray
        Grad-CAM heatmap.

    alpha : float
        Transparency of heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # --------------------------------------------------
    # 1. Resize heatmap to original spectrogram size
    # --------------------------------------------------

    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32
    )

    resized_heatmap = tf.image.resize(
        heatmap_tensor,
        (
            spectrogram.shape[0],
            spectrogram.shape[1]
        ),
        method="bilinear"
    )

    resized_heatmap = (
        resized_heatmap.numpy().squeeze()
    )

    # --------------------------------------------------
    # 2. Create figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    # --------------------------------------------------
    # 3. Display original Mel Spectrogram
    # --------------------------------------------------

    librosa.display.specshow(
        spectrogram,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    # --------------------------------------------------
    # 4. Overlay Grad-CAM
    # --------------------------------------------------

    image = ax.imshow(
        resized_heatmap,
        origin="lower",
        aspect="auto",
        cmap="jet",
        alpha=alpha
    )

    # --------------------------------------------------
    # 5. Add color bar
    # --------------------------------------------------

    fig.colorbar(
        image,
        ax=ax,
        label="Model Influence"
    )

    ax.set_title(
        "Grad-CAM — Model Decision Regions"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Mel Frequency"
    )

    fig.tight_layout()

    return fig