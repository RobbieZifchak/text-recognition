# import keras

#model loading
from keras.models import load_model

#image processing
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from cv2 import imread, imshow, IMREAD_GRAYSCALE, THRESH_BINARY
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = load_model('CNN200epochs.h5')

labels_list = ['Anglesey',
 'German',
 'God',
 'Government',
 'I',
 'Labour',
 'Minister',
 'Sir',
 'a',
 'about',
 'after',
 'again',
 'against',
 'all',
 'also',
 'always',
 'an',
 'and',
 'any',
 'are',
 'as',
 'at',
 'back',
 'be',
 'been',
 'before',
 'being',
 'but',
 'by',
 'can',
 'come',
 'could',
 'did',
 'do',
 'down',
 'even',
 'ever',
 'first',
 'for',
 'found',
 'from',
 'get',
 'go',
 'good',
 'great',
 'had',
 'has',
 'have',
 'he',
 'her',
 'here',
 'him',
 'himself',
 'his',
 'how',
 'if',
 'in',
 'into',
 'is',
 'it',
 'its',
 'know',
 'last',
 'life',
 'like',
 'little',
 'long',
 'made',
 'make',
 'man',
 'many',
 'may',
 'me',
 'men',
 'might',
 'more',
 'most',
 'much',
 'must',
 'my',
 'never',
 'new',
 'no',
 'not',
 'now',
 'of',
 'off',
 'on',
 'one',
 'only',
 'or',
 'other',
 'our',
 'out',
 'over',
 'people',
 'per',
 'said',
 'same',
 'see',
 'she',
 'should',
 'so',
 'some',
 'still',
 'such',
 'than',
 'that',
 'the',
 'their',
 'them',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'thought',
 'through',
 'time',
 'to',
 'told',
 'too',
 'town',
 'two',
 'under',
 'up',
 'us',
 'used',
 'very',
 'was',
 'way',
 'we',
 'well',
 'went',
 'were',
 'what',
 'when',
 'which',
 'who',
 'will',
 'with',
 'would',
 'year',
 'years',
 'you',
 'your']



def shape_new_img(image_path):

    """ image_path = image file path
        function takes in image path, converts to grayscale,
        and reshapes to 4 dimensional array size to be compatible with ConvNet Model
    """
    desired_size = 64
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img, (desired_size, desired_size))
    finalarray = new_img_array.reshape(1, desired_size, desired_size, 1)

    return finalarray


def predict_word(word_path):
    """ Takes path to a word, reshapes, makes prediction, and returns predicted class label"""
    word = word_path
    reshaped_word = shape_new_img(word)
    pred = model.predict(reshaped_word)
    get_class = np.argmax(pred)
    prediction = labels_list[get_class]
    return prediction

path = '../text-recognition/out_of_data_samples/01-their.png'
predict_word(path)

def plot_grayscale(img_path):
    img = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    return plt.imshow(img, cmap=plt.cm.gray)


def predict_and_plot(word_path):
    word = word_path
    reshaped_word = shape_new_img(word)
    pred = model.predict(reshaped_word)
    get_class = np.argmax(pred)
    prediction = labels_list[get_class]
     #plot input image
    img = cv2.imread(word_path,  cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap=plt.cm.gray)

    return prediction
