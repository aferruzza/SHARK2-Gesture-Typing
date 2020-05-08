'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

import json
import math
import time
import numpy as np
from flask import Flask, request
from flask import render_template
from scipy.interpolate import interp1d

app = Flask(__name__, template_folder='templates', static_folder='static')

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240,
               170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50,
               120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []

    # TODO: Start sampling (12 points)
    # ediff1d gives difference between consecutive elements of the array
    # we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    # basically when words like mm or ii have no path / little path, use centroid
    if (distance[-1] == 0):
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        # get the proportion of line segments
        distance = distance / distance[-1]
        # scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        # generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        # use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()

    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

#Helper function to calculate euclidean distance
def euclid(X1, Y1, X2, Y2):
    return math.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 35
    # TODO: Do pruning (12 points)
    # From the paper, we compute start-start and end-end distances between gesture points and template sample points
    # If they are BOTH <= threshold, add a copy to the respective valid template sample points and add the respective
    # word to valid_words
    for i in range(len(template_sample_points_X)):
        #start-start
        start_start = euclid(gesture_points_X[0][0], gesture_points_Y[0][0],
                            template_sample_points_X[i][0], template_sample_points_Y[i][0])
        #end-end
        end_end = euclid(gesture_points_X[0][len(gesture_points_X)-1], gesture_points_Y[0][len(gesture_points_Y)-1],
                            template_sample_points_X[i][-1], template_sample_points_Y[i][-1])
        if (start_start <= threshold and end_end <= threshold):
            valid_words.append(words[i])
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])
            continue
    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                     valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 1
    # TODO: Calculate shape scores (12 points)
    #Normalize gesture inputs
    W = max(gesture_sample_points_X[0]) - min(gesture_sample_points_X[0])
    H = max(gesture_sample_points_Y[0]) - min(gesture_sample_points_Y[0])
    gx = L / max(W, H)
    gy = L / max(W, H)

    #Scale the templates and append the score / 100 - formula based on paper
    for i in range(len(valid_template_sample_points_X)):
        tempX = valid_template_sample_points_X[i]
        tempY = valid_template_sample_points_Y[i]
        z = 0
        tx = L / max(tempX)
        ty = L / max(tempY)
        for j in range(len(gesture_sample_points_X)):
            z += euclid(float(gesture_sample_points_X[0][j])*gx, float(gesture_sample_points_Y[0][j])*gy, float(tempX[j])*tx, float(tempY[j]) * ty)

        z = z / 100
        shape_scores.append(z)

    return shape_scores

#Helper function for get_location_scores, defined in the paper. This is to help calculate d(u,t) -- calculating
#distances between pairs
def pairs_d(px, py, qx, qy):
    to_return = 0
    for i in range(100):
        tx = px[i]
        ty = py[i]
        gx = qx[i]
        gy = qy[i]
        to_return += euclid(tx, ty, gx, gy)
    return to_return

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                        valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 10
    # TODO: Calculate location scores (12 points)
    for i in range(len(valid_template_sample_points_X)):
        flag = False
        d = 0
        tempX = valid_template_sample_points_X[i]
        tempY = valid_template_sample_points_Y[i]
        #Start with minimum inf as an initial flag
        for j in range(0, 100):
            minimum = float('inf')
            tx = tempX[j]
            ty = tempY[j]
            for k in range(0, 100):
                gx = gesture_sample_points_X[0]
                gy = gesture_sample_points_Y[0]
                z = euclid(tx, ty, gx[0], gy[0])
                #set the new minimum. the first one will be the first euclidean distance, then it is compared to that
                #each successive time
                minimum = min(minimum, z)
            dut = max(0, minimum - radius)
            #From the paper, if dut != 0, then we go down to the bottom. But if it is == 0, flag remains false
            #and we go up to top to calculate distances again
            if (dut > 0):
                flag = True
                break
        if (flag):
            d += pairs_d(tempX, tempY, gx, gy)
        location_scores.append(d)
    return location_scores

def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.9
    # TODO: Set your own location weight
    location_coef = 0.1
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    # TODO: Set your own range.
    n = 3
    # TODO: Get the best word (12 points)
    min = float('inf')
    if (len(valid_words) == 0):
        best_word = "gesture not recognized"
    else:
        keyScores = sorted(zip(integration_scores, valid_words))
        seperate_lists = [list(i) for i in zip(*keyScores)]
        sorted_scores = seperate_lists[0]
        sorted_words = seperate_lists[1]
        sorted_scores = sorted_scores[:n]
        sorted_words = sorted_words[:n]
        keyIndex = 0
        for i in range(len(sorted_words)):
            prob = probabilities[sorted_words[i]]
            final = prob * sorted_scores[i]
            if (final < min):
                final = min
                keyIndex = i
            else:
                continue

    best_word = sorted_words[keyIndex]

    return best_word

@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():
    start_time = time.time()
    data = json.loads(request.get_data())

    print(data)

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    gesture_points_X = [gesture_points_X]
    gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X,
                                                                                             gesture_points_Y,
                                                                                             template_sample_points_X,
                                                                                             template_sample_points_Y)


    print(valid_words)
    print(valid_template_sample_points_X)
    print(valid_template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X,
                                    valid_template_sample_points_Y)

    print(shape_scores)
    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y,
                                          valid_template_sample_points_X, valid_template_sample_points_Y)

    print(location_scores)
    integration_scores = get_integration_scores(shape_scores, location_scores)
    print(integration_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
