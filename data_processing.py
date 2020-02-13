from os.path import isfile, join
from os import listdir
import nltk
import pickle
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from torch import tensor
def get_number_segments(folder):
    """
    
    Computes the number of segments in the files inside the folder
    
    Args:
        folder: string, path to the folder we want to examine
    Returns:
        number_segments: integer, number of segments inside the folder
    
    """
    
    number_segments = 0 
    
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    
    for f in files:

        with open(f, "rb") as f_opened:

            number_segments += len(pickle.load(f_opened))
            
    return number_segments
            
def sentence_serialization(sentence, word2idx, lower_case = True):
    """ 
    
    Transforms a sentence into a list of integers. No integer will be appended if the token is not present in word2idx.
    
    Args:
        sentence: string, sentence that we want to serialize.
        word2idx: dictionary, dictionary with words as keys and indexes as values.
        lower_case: boolean, turns all words in the sentence to lower case. Useful if word2idx 
        doesn't support upper case words.
    Returns: 
        s_sentence: list, list containing the indexes of the words present in the sentence. 
        s_sentence stands for serialized sentence.
        
    """
    
    s_sentence = []
    
    not_found = 0
    
    if lower_case: 
        
        tokens = map(str.lower,nltk.word_tokenize(sentence))
        
    else:
        
        tokens = nltk.word_tokenize(sentence)
    
    for token in tokens:       
        
        try:
            
            s_sentence.append(word2idx[token])
            
        except KeyError:
            
            not_found += 1
            
            print("Warning: At least one token is not present in the word2idx dict. For instance: " + token + 
                  
                  ". Not found: " + str(not_found))
        
    return s_sentence

def get_tokens(inputh_path, output_path, read = False):
    """
    
    Checks all the files in filespath and returns a set of all the words found in the files. The function will ignore all the folders inside filespath automatically.
    We set all the words to be lower case. The function will check if the a file with all the tokens is available. In that case this function
    will be much faster. 
    
    Args:
        filespath: string, path to the folder with all the files containing the words that we want to extract.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        dictionary: set, set containing all the different words found in the files. 
    
    """

    dictionary_path = join(output_path, "dictionary.pkl")
    
    if isfile(dictionary_path) and read:
        
        print("Loading from file dictionary.pkl")
        
        with open(dictionary_path,"rb") as dictionary_file:
        
            word2idx_dictionary = pickle.load(dictionary_file)
        
    else:
        
        print("Processing dataset ...")
    
        dictionary = set()

        files = [join(inputh_path, f) for f in listdir(inputh_path) if isfile(join(inputh_path, f))]
        files = [file for file in files if ".keep" not in file]
        #files.remove(join(inputh_path,".keep"))
              
        for f in files:

            opened_file = open(f,'r')

            for i, line in enumerate(opened_file):
                a = line.split('","')

                a[1] = map(str.lower,set(nltk.word_tokenize(a[1])))

                dictionary = dictionary.union(a[1])
                
        dictionary = sorted(dictionary)
        
        word2idx_dictionary = {None: 0}
        
        for i, word in enumerate(dictionary,1):
                
                word2idx_dictionary[word] = i

        with open(dictionary_path, "wb") as dictionary_file:

            pickle.dump(word2idx_dictionary, dictionary_file)
            
    return word2idx_dictionary


def get_policy_of_interest_tokens(paragraphs, output_path):

    all_tokens_path = join(output_path, "policy_of_interest_tokens.pkl")

    print("Processing all tokens of the policy of interest ...")

    all_tokens = set()

    for paragraph in paragraphs:
        segment = map(str.lower, set(nltk.word_tokenize(paragraph)))
        all_tokens = all_tokens.union(segment)

    all_tokens = sorted(all_tokens)

    word2idx_dictionary = {None: 0}

    for i, word in enumerate(all_tokens, 1):
        word2idx_dictionary[word] = i

    with open(all_tokens_path, "wb") as dictionary_file:

        pickle.dump(word2idx_dictionary, dictionary_file)

    return word2idx_dictionary

def label_to_vector(label, labels, count):
    """
    
    Returns a vector representing the label passed as an input.
    
    Args:
        label: string, label that we want to transform into a vector.
        labels: dictionary, dictionary with the labels as the keys and indexes as the values.
    Returns:
        vector: np.array, 1-D array of lenght 12.
        
    """
    
    vector = np.zeros((count))
    
    try:
    
        index = labels[label]
    
        vector[index] = 1
        
    except KeyError:
        
        vector = np.zeros((count))
    
    return vector

def get_glove_dicts(parent_folder, input_path, output_path, dims, read = False):
    """
    
    This functions returns two dictionaries that process the glove.6B folder and gets the pretrained 
    embedding vectors.
    
    Args:
        path: string, path to the folder containing the glove embeddings
        dims: integer, embeddings dimensionality to use.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        word2vector: dictionary, the keys are the words an the values are the embeddings associated with that word.
        word2idx: dictionary, the keys are the words and the values are the indexes associated with that word.
    
    """
    
    word2vector_path = "word2vector_globe_" + str(dims) + ".pkl"
    
    word2vector_path = join(output_path, word2vector_path)
    
    word2idx_path = "word2idx_globe_" + str(dims) + ".pkl"
    
    word2idx_path = join(output_path, word2idx_path)
    
    if isfile(word2vector_path) and read:
        
        print("Loading from file word2vector_globe_{}.pkl".format(dims))

        with open(word2vector_path,"rb") as word2vector_file:
        
            word2vector = pickle.load(word2vector_file)

    else:
        
        print("Processing dataset ...")

        words = [None]

        word2idx = {None: 0}

        idx = 1

        vectors = [np.zeros(dims)]

        with open(join(parent_folder, input_path, input_path + "." + str(dims) + "d.txt"), encoding="utf-8") as glove_file:

            for line in glove_file:

                split_line = line.split()

                word = split_line[0]

                words.append(word)

                word2idx[word] = idx

                vector = np.array(split_line[1:]).astype(np.float)

                vectors.append(vector)
                
                idx += 1
        
        word2vector = {w: vectors[word2idx[w]] for w in words}
        
        with open(word2vector_path,"wb") as word2vector_file:
        
            pickle.dump(word2vector, word2vector_file)

    return word2vector

def get_fast_text_dicts(input_path, output_path, dims, missing = True, read = False):
    """
    
    This functions returns two dictionaries that process the fasttext folder and gets the pretrained 
    embedding vectors.
    
    Args:
        path: string, path to the folder containing the glove embeddings
        dims: integer, embeddings dimensionality to use.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        word2vector: dictionary, the keys are the words an the values are the embeddings associated with that word.
        word2idx: dictionary, the keys are the words and the values are the indexes associated with that word.
    
    """
    
    def append_from_file(words, word2idx, vectors, idx, input_path, file):
        
        with open(join(input_path, file) , encoding="utf8") as fast_text_file:

            for line in fast_text_file:
                """if idx == 158090:"""


                split_line = line.split()

                word = split_line[0]

                words.append(word)

                word2idx[word] = idx

                vector = np.array(split_line[1:]).astype(np.float)

                vectors.append(vector)
                
                idx += 1
                
        return words, word2idx, vectors, idx
    
    if missing:     
    
        word2vector_path = "word2vector_fast_text_" + str(dims) + "_nomissing.pkl"

        word2vector_path = join(output_path, word2vector_path)

        word2idx_path = "word2idx_fast_text_" + str(dims) + "_nomissing.pkl"

        word2idx_path = join(output_path, word2idx_path)
        
    else:

        word2vector_path = "word2vector_fast_text_" + str(dims) + ".pkl"

        word2vector_path = join(output_path, word2vector_path)

        word2idx_path = "word2idx_fast_text_" + str(dims) + ".pkl"

        word2idx_path = join(output_path, word2idx_path)
    
    if isfile(word2vector_path) and read:
        
        print("Loading from file {}".format(word2vector_path))

        with open(word2vector_path,"rb") as word2vector_file:
        
            word2vector = pickle.load(word2vector_file)

    else:
        
        print("Processing dataset ...")

        words = [None]

        word2idx = {None: 0}

        idx = 1

        vectors = [np.zeros(dims)]
        
        words, word2idx, vectors, idx = append_from_file(words, word2idx, vectors, idx, input_path, '.vec')     
                
        if missing:
                        
            words, word2idx, vectors, idx = append_from_file(words, word2idx, vectors, idx, input_path, '.vec_missing')
                           
        word2vector = {w: vectors[word2idx[w]] for w in words}
        
        with open(word2vector_path,"wb") as word2vector_file:
        
            pickle.dump(word2vector, word2vector_file)

    return word2vector

def get_weight_matrix(dims, output_path, read = False, oov_random = True, **kwargs):
    """

    This function returns a matrix containing the weights that will be used as pretrained embeddings. It will read 
    weights_matrix.pkl file as long as it exists. This will make the code much faster. 

    Args:
        dictionary: dictionary, dictionary containing a word2idx of all the words present in the dataset.
        word2vector: dictionary, the keys are the words and the values are the embeddings.
        dims: integer, dimensionality of the embeddings.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        weights_matrix: np.array, matrix containing all the embeddings.

    """
    
    weights_path = "weights_matrix_" + str(dims) + ".pkl"
    
    weights_path = join(output_path, weights_path)

    if isfile(weights_path) and read:
        
        print("Loading from file weights_matrix_{}.pkl".format(dims))

        with open(weights_path,"rb") as weights_file:
        
            weights_matrix = pickle.load(weights_file , encoding="latin1")
        
    else:
        
        print("Processing dataset ...")
        
        # We add 1 to onclude the None value
        
        dictionary = kwargs['dictionary']
        
        word2vector = kwargs['word2vector']
        
        matrix_len = len(dictionary) + 1

        weights_matrix = np.zeros((matrix_len, dims))

        oov_words = 0

        for word, i in dictionary.items():

            try: 

                weights_matrix[i] = word2vector[word]

            except KeyError:
                
                if oov_random:

                    weights_matrix[i] = np.random.normal(scale=0.6, size=(dims, ))
                    
                oov_words += 1
                
        if oov_words != 0:
            
            print("Some words were missing in the word2vector. {} words were not found.".format(oov_words))

        with open(weights_path,"wb") as weights_file:

            pickle.dump(weights_matrix, weights_file)
            
    return weights_matrix

def process_dataset(labels, word2idx, read = False):
    """
    
    This function process all the privacy policy files and transforms all the segments into lists of integers. It also 
    transforms all the labels into a list of 0s except in the positions associated with the labels in which we will find 1s
    where we will find a 1. It will also place .pkl files into the processed_data folder so that we can load the data from 
    there instead of having to process the whole dataset.
    
    Args:
        path: string, path where all the files we want to process are located (all the privacy policies).
        word2idx: dictionary the keys are the words and the values the index where we can find the vector in 
        weights_matrix.
        labels: labels: dictionary, dictionary with the labels as the keys and indexes as the values.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not. 
    Returns:
        sentence_matrices: list, a list of lists of lists containing the segments of the files transformed into integers. 
        sentence_matrices[i][j][k] -> "i" is for the file, "j" for the line and "k" for the token. 
        labels_matrices: list, a list of lists of lists containing the labels of the dataset. labels_matrices[i][j][k] ->
        "i" is for the file, "j" for the line and "k" for the boolean variable specifying 
        the presence of the a label.
        
    """    
    
    """
    
    Helper functions
    
    """
    
    def pickle_matrix(matrix, path):
        
        with open(path,"wb") as output_file:

            pickle.dump(matrix, output_file)
    
    def unpickle_matrix(path):
        
        with open(path,"rb") as input_file:

            matrix = pickle.load(input_file)
        
        return matrix
    
    """
    
    main code of process_dataset
    
    """
    
    input_path = "agg_data"
    
    output_path = "processed_data"
    
    path_sentence_matrices = join(output_path, "all_sentence_matrices.pkl")

    path_labels_matrices = join(output_path, "all_label_matrices.pkl")
    
    if isfile(path_sentence_matrices) and isfile(path_labels_matrices) and read:
        
        print("Loading from " + path_sentence_matrices + " and " + path_labels_matrices)       
        
        sentence_matrices = unpickle_matrix(path_sentence_matrices)

        labels_matrices = unpickle_matrix(path_labels_matrices)

        return sentence_matrices, labels_matrices 
        
    else:
        
        print("Processing dataset ...")
        
        with open("agg_data/agg_data.pkl",'rb') as dataframe_file:

            opened_dataframe = pickle.load(dataframe_file)

        num_records = len(opened_dataframe)

        print('Num of unique segments segments: {}'.format(num_records))

        num_labels = len(opened_dataframe["label"].iloc[0])

        sentence_matrices = np.zeros(num_records, dtype = 'object')

        labels_matrices = np.zeros((num_records, num_labels))

        for index, row in opened_dataframe.iterrows():

            segment = row["segment"]

            label = row["label"]

            sentence_matrices[index] = sentence_serialization(segment, word2idx)

            labels_matrices[index] = label

        path_sentence_matrices = join(output_path, "all_sentence_matrices.pkl")

        path_labels_matrices = join(output_path, "all_label_matrices.pkl")

        pickle_matrix(sentence_matrices, path_sentence_matrices)

        pickle_matrix(labels_matrices, path_labels_matrices)

        return sentence_matrices, labels_matrices


def process_policy_of_interest(word2idx , segments):
    """

    This function process all the privacy policy files and transforms all the segments into lists of integers. It also
    transforms all the labels into a list of 0s except in the positions associated with the labels in which we will find 1s
    where we will find a 1. It will also place .pkl files into the processed_data folder so that we can load the data from
    there instead of having to process the whole dataset.

    Args:
        path: string, path where all the files we want to process are located (all the privacy policies).
        word2idx: dictionary the keys are the words and the values the index where we can find the vector in
        weights_matrix.
        labels: labels: dictionary, dictionary with the labels as the keys and indexes as the values.
        read: boolean, variable that allows us to decide wether to read from pre-processed files or not.
    Returns:
        sentence_matrices: list, a list of lists of lists containing the segments of the files transformed into integers.
        sentence_matrices[i][j][k] -> "i" is for the file, "j" for the line and "k" for the token.
        labels_matrices: list, a list of lists of lists containing the labels of the dataset. labels_matrices[i][j][k] ->
        "i" is for the file, "j" for the line and "k" for the boolean variable specifying
        the presence of the a label.

    """

    """

    Helper functions

    """

    def pickle_matrix(matrix, path):

        with open(path, "wb") as output_file:
            pickle.dump(matrix, output_file)

    def unpickle_matrix(path):

        with open(path, "rb") as input_file:
            matrix = pickle.load(input_file)

        return matrix


    def stack_segments(segments, clearance=2):

        segments_len = map(len, segments)

        max_len = max(segments_len)

        segments_list = []

        output_len = max_len + clearance * 2

        for i, segment in enumerate(segments):
            segment_array = np.array(segment)

            zeros_to_prepend = int((output_len - len(segment_array)) / 2)

            zeros_to_append = output_len - len(segment_array) - zeros_to_prepend

            resized_array = np.append(np.zeros(zeros_to_prepend), segment_array)

            resized_array = np.append(resized_array, np.zeros(zeros_to_append))

            segments_list.append(torch.tensor(resized_array, dtype=torch.int64))

            segments_tensor = torch.stack(segments_list).unsqueeze(1)

        return segments_tensor


    output_path = "processed_data"

    print("Processing policy of interest ...")

    num_records = len(segments)

    segments_matrices = np.zeros(num_records, dtype='object')

    for index in range(num_records):
        segment = segments[index]
        segments_matrices[index] = sentence_serialization(segment, word2idx)

    path_sentence_matrices = join(output_path, "policy_of_interest_paragraphs_matrices.pkl")

    pickle_matrix(segments_matrices, path_sentence_matrices)
    segments_tensor = stack_segments(segments_matrices)

    return segments_tensor

def collate_csv_data(word2idx ,attribute, mode, num_labels, read=False):

    def stack_segments(segments, clearance=2):

        segments_len = map(len, segments)

        max_len = max(segments_len)

        segments_list = []

        output_len = max_len + clearance * 2

        for i, segment in enumerate(segments):
            segment_array = np.array(segment)

            zeros_to_prepend = int((output_len - len(segment_array)) / 2)

            zeros_to_append = output_len - len(segment_array) - zeros_to_prepend

            resized_array = np.append(np.zeros(zeros_to_prepend), segment_array)

            resized_array = np.append(resized_array, np.zeros(zeros_to_append))

            segments_list.append(torch.tensor(resized_array, dtype=torch.int64))

            segments_tensor = torch.stack(segments_list).unsqueeze(1)


        return segments_tensor

    print("Processing dataset ...")
    attr_segment_matrices, attr_values_matrices, attr_value_tensor = process_attribute_level_testset(word2idx,attribute,mode,num_labels,read )

    segments_tensor = stack_segments(attr_segment_matrices)

    return segments_tensor,attr_value_tensor

def process_attribute_level_testset(word2idx ,attribute, mode, num_labels, read=False ):

    def pickle_matrix(matrix, path):
        with open(path, "wb") as output_file:
            pickle.dump(matrix, output_file)

    def unpickle_matrix(path):
        with open(path, "rb") as input_file:
            matrix = pickle.load(input_file)

        return matrix


    output_path = join("processed_data", attribute)

    path_attr_segment_matrices = join(output_path, "attr_segment_matrices.pkl")
    path_attr_values_matrices = join(output_path, "attr_values_matrices.pkl")

    label_file = "labels_" + attribute + ".pkl"
    with open(label_file, "rb") as labels_file:
        labels_dict = pickle.load(labels_file)

    if isfile(path_attr_segment_matrices) and isfile(path_attr_values_matrices) and read:
        print("Loading from " + path_attr_segment_matrices + " and " + path_attr_values_matrices)
        segment_matrices = unpickle_matrix(path_attr_segment_matrices)
        labels_matrices = unpickle_matrix(path_attr_values_matrices)
        return segment_matrices, labels_matrices
    else:
        print("Processing attribute separated data ...")
        file = join('attribute_dataset', attribute + '_' + mode + '.csv')
        with open(file, 'r') as opened_file:
            reader = pd.read_csv(opened_file, delimiter=',', names = ["index","segment","label"])
            reader['label'] = reader['label'].apply(lambda x: label_to_vector(x, labels_dict, num_labels))

        num_records = len(reader['label'])

        segment_matrices = np.zeros(num_records, dtype='object')
        labels_matrices = np.zeros((num_records, num_labels))
        for index, row in reader.iterrows():
            label = row["label"]
            segment = row["segment"]
            segment_matrices[index] = sentence_serialization(segment, word2idx)
            labels_matrices[index] = label


        pickle_matrix(segment_matrices, path_attr_segment_matrices)
        pickle_matrix(labels_matrices, path_attr_values_matrices)
        attr_value_tensor = torch.tensor(reader['label'])
        return segment_matrices, labels_matrices, attr_value_tensor



def process_attribute_level_dataset(word2idx ,attribute, mode, read=False):

    """

    Helper functions

    """

    def pickle_matrix(matrix, path):

        with open(path, "wb") as output_file:
            pickle.dump(matrix, output_file)

    def unpickle_matrix(path):

        with open(path, "rb") as input_file:
            matrix = pickle.load(input_file)

        return matrix


    output_path = join("processed_data", attribute)

    path_attr_segment_matrices = join(output_path, "attr_segment_matrices.pkl")
    path_attr_values_matrices = join(output_path, "attr_values_matrices.pkl")


    if isfile(path_attr_segment_matrices) and isfile(path_attr_values_matrices) and read:
        print("Loading from " + path_attr_segment_matrices + " and " + path_attr_values_matrices)
        segment_matrices = unpickle_matrix(path_attr_segment_matrices)
        labels_matrices = unpickle_matrix(path_attr_values_matrices)
        return segment_matrices, labels_matrices
    else:
        print("Processing attribute separated data ...")
        file = join('agg_data', attribute + '_' + mode + '.pkl')
        with open(file,'rb') as dataframe_file:
            opened_dataframe = pickle.load(dataframe_file)

        num_records = len(opened_dataframe)
        print('Num of unique  segments: {}'.format(num_records))
        num_labels = len(opened_dataframe["label"].iloc[0])
        segment_matrices = np.zeros(num_records, dtype = 'object')
        labels_matrices = np.zeros((num_records, num_labels))

        for index, row in opened_dataframe.iterrows():
            segment = row["segment"]
            label = row["label"]
            segment_matrices[index] = sentence_serialization(segment, word2idx)
            labels_matrices[index] = label

        path_sentence_matrices = join(output_path, "all_sentence_matrices.pkl")
        path_labels_matrices = join(output_path, "all_label_matrices.pkl")
        pickle_matrix(segment_matrices, path_sentence_matrices)
        pickle_matrix(labels_matrices, path_labels_matrices)
        return segment_matrices, labels_matrices


def aggregate_data(read=False):
    """

    This function processes raw_data and aggregates all the segments labels. Places all the files in the agg_data folder.

    Args:
        read: boolean, if set to true it will read the data from agg_data folder as long as all the files are found
        inside the
        folder.
    Returns:
        Nothing.

    """

    """

    Helper functions

    """

    def aggregate_files(input_path, output_path, labels_dict):

        files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        files = [file for file in files if ".keep" not in file]
        # files.remove(".keep")

        all_results = pd.DataFrame({'label': [], 'segment': []})

        for f in files:
            data = pd.read_csv(join(input_path, f), names=["idx", "segment", "label"])

            data['label'] = data['label'].apply(lambda x: label_to_vector(x, labels_dict, 12))

            labels_data = data[['idx', 'label']]

            labels = labels_data.groupby("idx").sum()

            segments = data[['idx', 'segment']].set_index('idx').drop_duplicates()

            result = pd.merge(labels, segments, left_index=True, right_index=True)

            all_results = pd.concat([all_results, result])

        all_results.reset_index(drop=True, inplace=True)

        folder_output_path = "agg_data"

        with open(join(output_path, "agg_data.pkl"), "wb") as output_file:
            pickle.dump(all_results, output_file)

    """

    main code of aggregate_data

    """

    input_path = "raw_data"

    output_path = "agg_data"

    with open("labels.pkl", "rb") as labels_file:

        labels_dict = pickle.load(labels_file)

    file_exists = isfile(join(output_path, "agg_data.pkl"))

    if file_exists and read:

        print("agg_data.pkl are already in agg_data/")

    else:

        print("Processing dataset in one file ...")

        aggregate_files(input_path, output_path, labels_dict)





def aggregate_data_attribute_level(attribute, mode, num_labels, read = False):


    def aggregate_lines(input_file, output_file, labels_dict):
        data = pd.read_csv(input_file, names=["idx", "segment", "label"])
        data['label'] = data['label'].apply(lambda x: label_to_vector(x, labels_dict, num_labels))
        labels_data = data[['idx', 'label']]
        labels = labels_data.groupby("idx").sum()
        segments = data[['idx', 'segment']].set_index('idx').drop_duplicates()
        result = pd.merge(labels, segments, left_index=True, right_index=True)
        # result.reset_index(drop=True, inplace=True)
        # all_results = pd.concat([all_results, result])
        # all_results.reset_index(drop=True, inplace=True)
        with open(output_file, "wb") as output_file:
            pickle.dump(result, output_file)




    label_file = "labels_" + attribute + ".pkl"
    with open(label_file, "rb") as labels_file:
        labels_dict = pickle.load(labels_file)

    output_file = join("agg_data", attribute + '_' + mode + '.pkl')
    input_file = join('attribute_dataset', attribute + '_' + mode + '.csv')

    file_exists = isfile(output_file)

    if file_exists and read:

        print("data already aggregated, reading ...")

    else:
        print("Processing csv file ...")
        aggregate_lines(input_file, output_file, labels_dict)



def get_absent_words(dictionary, word2vector):
    """
    
    This function check if the words inside dictionary are present in word2vector which is a dictionary coming from a word
    embedding.
    
    Args:
        dictionary: set, set containing strings of words
        word2vector: dictionary, the keys are the words and the values are the embeddings   
    Returns:
        absent_words: list, list containing all the words that weren't found in the word embeddings word2vector
    
    """

    absent_words = []

    for word in dictionary:

        try:

            word2vector[word]

        except KeyError:

            absent_words.append(word)
            
    return absent_words

def attr_value_labels(attribute):


    if attribute == 'Retention Period':
        labels = OrderedDict([('Stated Period', 0),
             ('Limited', 1),
             ('Indefinitely', 2),
             ('Unspecified', 3)])
    elif attribute == 'Retention Purpose':
        labels = OrderedDict([('Advertising', 0),
             ('Analytics/Research', 1),
             ('Legal requirement', 2),
             ('Marketing', 3),
             ('Perform service', 4),
             ('Service operation and security', 5),
             ('Unspecified', 6)])
    elif attribute == 'Notification Type':
        labels = OrderedDict([('General notice in privacy policy', 0),
             ('General notice on website', 1),
             ('No notification', 2),
             ('Personal notice', 3),
             ('Unspecified', 4)])
    elif attribute == 'Security Measure':
        labels = OrderedDict([('Generic', 0),
             ('Data access limitation', 1),
             ('Privacy review/audit', 2),
             ('Privacy training', 3),
             ('Privacy/Security program', 4),
             ('Secure data storage', 5),
             ('Secure data transfer', 6),
             ('Secure user authentication', 7),
             ('Unspecified', 8)])
    elif attribute == 'Audience Type':
        labels = OrderedDict([('Children', 0),
             ('Californians', 1),
             ('Citizens from other countries', 2),
             ('Europeans', 3)])
    elif attribute == 'User Type':
        labels = OrderedDict([('User with account', 0),
             ('User without account', 1),
             ('Unspecified', 2)])
    elif attribute == 'Access Scope':
        labels = OrderedDict([('Profile data', 0),
             ('Transactional data', 1),
             ('User account data', 2),
             ('Other data about user', 3),
             ('Unspecified', 4)])
    elif attribute == 'Does or Does Not':
        labels = OrderedDict([('Does', 0),
             ('Does Not', 1)])
    elif attribute == 'Access Type':
        labels = OrderedDict([('Deactivate account', 0),
             ('Delete account (full)', 1),
             ('Delete account (partial)', 2),
             ('Edit information', 3),
             ('View', 4),
             ('None', 5),
             ('Unspecified', 6)])
    elif attribute == 'Action First-Party':
        labels = OrderedDict([('Collect from user on other websites', 0),
             ('Collect in mobile app', 1),
             ('Collect on mobile website', 2),
             ('Collect on website', 3),
             ('Receive from other parts of company/affiliates', 4),
             ('Receive from other service/third-party (named)', 5),
             ('Receive from other service/third-party (unnamed)', 6),
             ('Track user on other websites', 7),
             ('Unspecified', 8)])
    elif attribute == 'Action Third-Party':
        labels = OrderedDict([('Collect on first party website/app', 0),
             ('Receive/Shared with', 1),
             ('See', 2),
             ('Track on first party website/app', 3),
             ('Unspecified', 4)])
    elif attribute == 'Third Party Entity':
        labels = OrderedDict([('Named third party', 0),
             ('Other part of company/affiliate', 1),
             ('Other users', 2),
             ('Public', 3),
             ('Unnamed third party', 4),
             ('Unspecified', 5)])
    elif attribute == 'Choice Scope':
        labels = OrderedDict([('Collection', 0),
             ('First party collection', 1),
             ('First party use', 2),
             ('Third party sharing/collection', 3),
             ('Third party use', 4),
             ('Both', 5),
             ('Use', 6),
             ('Unspecified', 7)])
    elif attribute == 'Choice Type':
        labels = OrderedDict([('Browser/device privacy controls', 0),
             ('Dont use service/feature', 1),
             ('First-party privacy controls', 2),
             ('Opt-in', 3),
             ('Opt-out link', 4),
             ('Opt-out via contacting company', 5),
             ('Third-party privacy controls', 6),
             ('Unspecified', 7)])
    elif attribute == 'User Choice':
        labels = OrderedDict([('None', 0),
             ('Opt-in', 1),
             ('Opt-out', 2),
             ('User participation', 3),
             ('Unspecified', 4)])
    elif attribute == 'Change Type':
        labels = OrderedDict([('In case of merger or acquisition', 0),
             ('Non-privacy relevant change', 1),
             ('Privacy relevant change', 2),
             ('Unspecified', 3)])
    elif attribute == 'Collection Mode':
        labels = OrderedDict([('Explicit', 0),
             ('Implicit', 1),
             ('Unspecified', 2)])
    elif attribute == 'Identifiability':
        labels = OrderedDict([('Aggregated or anonymized', 0),
             ('Identifiable', 1),
             ('Unspecified', 2)])
    elif attribute == 'Personal Information Type':
        labels = OrderedDict([('Computer information', 0),
             ('Contact', 1),
             ('Cookies and tracking elements', 2),
             ('Demographic', 3),
             ('Financial', 4),
             ('Generic personal information', 5),
             ('Health', 6),
             ('IP address and device IDs', 7),
             ('Location', 8),
             ('Personal identifier', 9),
             ('Social media data', 10),
             ('Survey data', 11),
             ('User online activities', 12),
             ('User profile', 13),
             ('Unspecified', 14)])
    elif attribute == 'Purpose':
        labels = OrderedDict([('Additional service/feature', 0),
             ('Advertising', 1),
             ('Analytics/Research', 2),
             ('Basic service/feature', 3),
             ('Legal requirement', 4),
             ('Marketing', 5),
             ('Merger/Acquisition', 6),
             ('Personalization/Customization', 7),
             ('Service operation and security', 8),
             ('Unspecified', 9)])
    elif attribute == 'Majority':
        labels = OrderedDict([('First Party Collection/Use', 0),
             ('Third Party Sharing/Collection', 1),
             ('User Access, Edit and Deletion', 2),
             ('Data Retention', 3),
             ('Data Security', 4),
             ('International and Specific Audiences', 5),
             ('Do Not Track', 6),
             ('Policy Change', 7),
             ('User Choice/Control', 8),
             ('Introductory/Generic', 9),
             ('Practice not covered', 10),
             ('Privacy contact information', 11)])

    file = 'labels_' + attribute + '.pkl'
    with open(file, "wb") as f:
        pickle.dump(labels, f)

    if isfile(file):
        labels_file = open(file, "rb")
        labels = pickle.load(labels_file)
        labels_file.close()
        # for label, index in labels.items():
        #     print(str(index) + '. ' + label)

    return labels