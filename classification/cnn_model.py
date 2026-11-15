# 1. libraries
import json
from pathlib import Path

import librosa 
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras import layers

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# 2. Paths - DCASE 2025 Task 4 Dataset
BASE_DIR = Path.cwd().parents[1]

train_path = (
    BASE_DIR
    / "Dataset"
    / "Development Dataset"
    / "final_0402_1"
    / "DCASE2025Task4Dataset"
    / "dev_set"
    / "sound_event"
    / "train"
)

valid_path = (
    BASE_DIR
    / "Dataset"
    / "Development Dataset"
    / "final_0402_1"
    / "DCASE2025Task4Dataset"
    / "dev_set"
    / "sound_event"
    / "valid"
)

# 3. parameters
sample_rate = 32_000
duration = 10.0

num_samples = int(
    sample_rate * duration
)

n_fft = 1024
hop_length = 512
n_mels = 128

f_min = 20
f_max = sample_rate // 2

batch_size = 8

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
}

# 4. use exactly the same class order everywhere
class_names = sorted(
    [
        p.name
        for p in train_path.iterdir()
        if p.is_dir()
    ]
)

class_to_index = {
    name: index
    for index, name
    in enumerate(class_names)
}

NUM_CLASSES = len(class_names)

print("Classes:")
for i, name in enumerate(class_names):
    print(i, name)

# 5. Read audio file paths
def collect_audio_files(
    data_path,
    class_to_index,
):

    file_paths = []
    labels = []

    for class_name, class_id in class_to_index.items():

        class_folder = (
            data_path
            / class_name
        )

        if not class_folder.exists():
            continue

        for audio_path in class_folder.rglob("*"):

            if (
                audio_path.suffix.lower()
                in AUDIO_EXTENSIONS
            ):

                file_paths.append(
                    str(audio_path)
                )

                labels.append(
                    class_id
                )

    return (
        np.asarray(file_paths),
        np.asarray(
            labels,
            dtype=np.int32,
        ),
    )


train_file_paths, train_labels = (
    collect_audio_files(
        train_path,
        class_to_index,
    )
)

valid_file_paths, valid_labels = (
    collect_audio_files(
        valid_path,
        class_to_index,
    )
)

# 6. train - test split
X_train, X_test, y_train, y_test = (
    train_test_split(
        train_file_paths,
        train_labels,
        test_size=0.20,
        random_state=SEED,
        shuffle=True,
        stratify=train_labels,
    )
)

X_val = valid_file_paths
y_val = valid_labels


print("Training samples:", len(X_train))

print("Validation samples:", len(X_val))

print("Test samples:", len(X_test))

# 7. Audio preprocessing
def decode_path(
    path_value
):

    if isinstance(
        path_value,
        np.ndarray,
    ):

        path_value = (
            path_value.item()
        )

    if isinstance(
        path_value,
        (bytes, np.bytes_),
    ):

        return path_value.decode(
            "utf-8"
        )

    return str(path_value)

# Fix audio duration to 10 seconds
def fix_length(y):

    if len(y) < num_samples:

        y = np.pad(
            y,
            (
                0,
                num_samples
                - len(y),
            ),
            mode="constant",
        )

    else:

        y = y[:num_samples]

    return y.astype(
        np.float32
    )

# 8. WAVEFORM DATA AUGMENTATION
def augment_waveform(y):

    # -----------------------------------------
    # Random gain
    # -----------------------------------------

    if np.random.rand() < 0.5:

        gain_db = np.random.uniform(
            -6.0,
            6.0,
        )

        y = (
            y
            * (
                10.0
                ** (
                    gain_db
                    / 20.0
                )
            )
        )


    # -----------------------------------------
    # Random temporal shift
    # maximum 0.30 s
    # -----------------------------------------

    if np.random.rand() < 0.5:

        max_shift = int(
            0.30
            * sample_rate
        )

        shift = np.random.randint(
            -max_shift,
            max_shift + 1,
        )

        if shift > 0:

            y = np.concatenate(
                [
                    np.zeros(
                        shift,
                        dtype=y.dtype,
                    ),
                    y[:-shift],
                ]
            )

        elif shift < 0:

            s = -shift

            y = np.concatenate(
                [
                    y[s:],
                    np.zeros(
                        s,
                        dtype=y.dtype,
                    ),
                ]
            )


    # -----------------------------------------
    # Additive noise
    # random SNR: 15 - 30 dB
    # -----------------------------------------

    if np.random.rand() < 0.6:

        signal_rms = np.sqrt(
            np.mean(
                y ** 2
            )
            + 1e-12
        )

        if signal_rms > 1e-6:

            snr_db = np.random.uniform(
                15.0,
                30.0,
            )

            noise = np.random.randn(
                len(y)
            ).astype(
                np.float32
            )

            noise_rms = np.sqrt(
                np.mean(
                    noise ** 2
                )
                + 1e-12
            )

            target_noise_rms = (
                signal_rms
                / (
                    10.0
                    ** (
                        snr_db
                        / 20.0
                    )
                )
            )

            noise = (
                noise
                * (
                    target_noise_rms
                    / noise_rms
                )
            )

            y = (
                y
                + noise
            )


    return y.astype(
        np.float32
    )

# 9. log-mel extraction
def extract_log_mel(
    audio_path,
    label,
    training=False,
):

    audio_path = decode_path(
        audio_path
    )

    y, _ = librosa.load(
        audio_path,

        sr=sample_rate,

        mono=True,
    )

    y = fix_length(y)


    # waveform augmentation
    # only during training
    if training:

        y = augment_waveform(y)


    mel = (
        librosa
        .feature
        .melspectrogram(

            y=y,

            sr=sample_rate,

            n_fft=n_fft,

            hop_length=hop_length,

            n_mels=n_mels,

            fmin=f_min,

            fmax=f_max,

            power=2.0,
        )
    )


    log_mel = (
        librosa
        .power_to_db(

            mel,

            ref=np.max,

            top_db=80,
        )
    )


    # -----------------------------------------
    # Normalization
    # -----------------------------------------

    log_mel = (
        log_mel
        - np.mean(log_mel)
    ) / (
        np.std(log_mel)
        + 1e-6
    )


    # add channel dimension
    log_mel = (
        log_mel[
            ...,
            np.newaxis
        ]
    )


    return (
        log_mel.astype(
            np.float32
        ),
        np.int32(label),
    )

# separate wrappers for train/evaluation

def extract_train_np(
    path,
    label,
):

    return extract_log_mel(
        path,
        label,
        training=True,
    )


def extract_eval_np(
    path,
    label,
):

    return extract_log_mel(
        path,
        label,
        training=False,
    )

# Determine input shape
# ------------------------------------------------------------

example_feature, _ = (
    extract_log_mel(
        X_train[0],
        y_train[0],
        training=False,
    )
)

INPUT_SHAPE = (
    example_feature.shape
)

print(
    "Input shape:",
    INPUT_SHAPE,
)

# expected:
# (128, 626, 1)

# 10. TensorFlow wrappers
def tf_extract_train(
    path,
    label,
):

    x, y = tf.numpy_function(

        extract_train_np,

        [
            path,
            label,
        ],

        [
            tf.float32,
            tf.int32,
        ],
    )

    x.set_shape(
        INPUT_SHAPE
    )

    y.set_shape(())

    return x, y


def tf_extract_eval(
    path,
    label,
):

    x, y = tf.numpy_function(

        extract_eval_np,

        [
            path,
            label,
        ],

        [
            tf.float32,
            tf.int32,
        ],
    )

    x.set_shape(
        INPUT_SHAPE
    )

    y.set_shape(())

    return x, y

def spec_augment(
    feature,
    label,
):

    # -----------------------------------------
    # Frequency masking
    # max 8 Mel bands
    # -----------------------------------------

    freq_width = (
        tf.random.uniform(

            [],

            minval=0,

            maxval=9,

            dtype=tf.int32,
        )
    )

    freq_start = (
        tf.random.uniform(

            [],

            minval=0,

            maxval=(
                n_mels
                - freq_width
                + 1
            ),

            dtype=tf.int32,
        )
    )


    freq_axis = (
        tf.range(
            n_mels
        )[
            :,
            None,
            None
        ]
    )


    freq_keep = tf.logical_or(

        freq_axis
        < freq_start,

        freq_axis
        >= (
            freq_start
            + freq_width
        ),
    )


    feature = (
        feature
        * tf.cast(
            freq_keep,
            feature.dtype,
        )
    )


    # -----------------------------------------
    # Time masking
    # max 40 frames
    # -----------------------------------------

    time_steps = (
        INPUT_SHAPE[1]
    )


    time_width = (
        tf.random.uniform(

            [],

            minval=0,

            maxval=41,

            dtype=tf.int32,
        )
    )


    time_start = (
        tf.random.uniform(

            [],

            minval=0,

            maxval=(
                time_steps
                - time_width
                + 1
            ),

            dtype=tf.int32,
        )
    )


    time_axis = (
        tf.range(
            time_steps
        )[
            None,
            :,
            None
        ]
    )


    time_keep = tf.logical_or(

        time_axis
        < time_start,

        time_axis
        >= (
            time_start
            + time_width
        ),
    )


    feature = (
        feature
        * tf.cast(
            time_keep,
            feature.dtype,
        )
    )


    return (
        feature,
        label,
    )

# 12. Dataset creation
def create_dataset(
    paths,
    labels,
    training=False,
):

    dataset = (
        tf.data
        .Dataset
        .from_tensor_slices(
            (
                paths,
                labels,
            )
        )
    )


    if training:

        dataset = (
            dataset.shuffle(

                buffer_size=len(paths),

                seed=SEED,

                reshuffle_each_iteration=True,
            )
        )


        dataset = (
            dataset.map(

                tf_extract_train,

                num_parallel_calls=2,
            )
        )


        # spectrogram augmentation
        dataset = (
            dataset.map(

                spec_augment,

                num_parallel_calls=2,
            )
        )


    else:

        dataset = (
            dataset.map(

                tf_extract_eval,

                num_parallel_calls=2,
            )
        )


    dataset = dataset.batch(
        batch_size
    )


    dataset = dataset.prefetch(
        2
    )


    return dataset


train_dataset = create_dataset(
    X_train,
    y_train,
    training=True,
)

validation_dataset = create_dataset(
    X_val,
    y_val,
    training=False,
)

test_dataset = create_dataset(
    X_test,
    y_test,
    training=False,
)

# 13. MODERATE CLASS WEIGHTS
raw_weights = (
    compute_class_weight(

        class_weight="balanced",

        classes=np.arange(
            NUM_CLASSES
        ),

        y=y_train,
    )
)


# less aggressive than the original weights
moderate_weights = np.sqrt(
    raw_weights
)

# prevent extremely high weights
moderate_weights = np.minimum(
    moderate_weights,
    4.0,
)

class_weights = {

    class_id: float(weight)

    for class_id, weight
    in enumerate(
        moderate_weights
    )
}


print("Class weights:")

for class_id, weight in class_weights.items():

    print(
        f"{class_names[class_id]:25s}: "
        f"{weight:.3f}"
    )

# 14. CNN ARCHITECTURE 
regularizer = (
    keras.regularizers.l2(
        1e-4
    )
)

def build_cnn():
    inputs = keras.Input(shape=INPUT_SHAPE)

    # BLOCK 1
    x = layers.Conv2D(32,kernel_size=3, padding='same', use_bias=False, 
                      kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    # BLOCK 2
    x = layers.Conv2D(64,kernel_size=3, padding='same', use_bias=False, 
                      kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    # BLOCK 3
    x = layers.Conv2D(128,kernel_size=3, padding='same', use_bias=False, 
                      kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=2)(x)

    # BLOCK 4
    x = layers.Conv2D(256,kernel_size=3, padding='same', use_bias=False, 
                      kernel_regularizer=regularizer)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="audio_cnn_v2")
    model.compile(optimizer=(keras.optimizers.Adam(learning_rate=1e-3)), 
                  loss = ("sparse_categorical_crossentropy"),
                  metrics=["accuracy"])

    return model

model = build_cnn()
model.summary()

# 15. Training Callbacks
training_callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=(
            "best_audio_cnn_v2.keras"
        ),

        monitor="val_loss",

        save_best_only=True,

        verbose=1,
    ),


    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1,
    ),


    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=10,

        restore_best_weights=True,

        verbose=1,
    ),
]

# 16. TRAIN the model
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=50,
                    callbacks=training_callbacks, class_weight=class_weights)

# 17. load best model
model = keras.models.load_model("best_audio_cnn_v2.keras")

# 18. test evaluation
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)


predicted_probabilities = (
    model.predict(
        test_dataset,
        verbose=1,
    )
)


predicted_labels = np.argmax(

    predicted_probabilities,

    axis=1,
)


print(
    classification_report(

        y_test,

        predicted_labels,

        target_names=(
            class_names
        ),

        digits=4,

        zero_division=0,
    )
)

# 19. Save final model
model.save(
    "final_audio_cnn_v2.keras"
)