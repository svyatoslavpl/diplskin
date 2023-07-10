import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from utils.data_processing import create_train_test_split
from utils.metrics import calculate_metrics


def evaluate_model():
    test_dir = 'data/HAM10000/test_dir'

    data_pd = pd.read_csv('data/HAM10000_metadata.csv')

    # Split train and test data
    _, test_df = create_train_test_split(data_pd, None, None)

    # Image generator
    batch_size = 16
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
    image_size = 224

    print("\nTest Batches:")
    test_batches = datagen.flow_from_directory(test_dir,
                                               target_size=(image_size, image_size),
                                               batch_size=batch_size,
                                               shuffle=False)

    # Load model
    model = load_model("models/ResNet50+SA.h5")

    # Predict
    predictions = model.predict(test_batches, steps=len(test_df) // batch_size, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_batches.classes
    y_prob = predictions

    # Calculate metrics
    targetnames = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    y_test = to_categorical(y_true)
    calculate_metrics(y_true, y_pred, y_prob, targetnames)
