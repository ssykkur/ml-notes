def get_true_positives(Y_true, Y_pred):
    if len(Y_true) != len(Y_pred):
        return 'true/predict doesnt match'
    n = len(Y_true)
    true_positives = 0

    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]

        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives


def get_true_negatives(Y_true, Y_pred):
    if len(Y_true) != len(Y_pred):
        return 'true/predict doesnt match'

    n = len(Y_true)
    true_negatives = 0

    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]

        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives

def get_false_positives(Y_true, Y_pred):
    if len(Y_true) != len(Y_pred):
        return 'true/predict doesnt match'

    n = len(Y_true)
    false_positives = 0

    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]

        if true_label_i == 0 and predicted_label_i == 1:
            false_positives += 1
    return false_positives

def calc_accuracy(X):
    dataset_pred = []
    for email in X:
        prediction = naive_bayes(email, word_frequency, class_frequency)
        Y_pred.append(prediction)

    true_positives = get_true_positives(Y_test, Y_pred)
    true_negatives = get_true_negatives(Y_test, Y_pred)

    return (true_positives + true_negatives)/len(Y_test)


def calc_recall(Y_true, Y_pred):
    total_number_positive = Y_test.sum()
    true_positives = get_true_positives(Y_true, Y_pred)

    return true_positives/total_number_positive


def calc_precision(Y_true, Y_pred):
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)

    precision = true_positives/(true_positives + false_positives)
    return precision