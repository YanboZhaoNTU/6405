import numpy as np
from skmultilearn.dataset import load_from_arff
from skmultilearn.model_selection import IterativeStratification


def read_arff(path, label_count, wantfeature=False):
    path_to_arff_file = path + ".arff"
    arff_file_is_sparse = False
    X, Y, feature_names, label_names = load_from_arff(
        path_to_arff_file,
        label_count=label_count,
        label_location="end",
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    if (~wantfeature):
        return X, Y, None
    else:
        featype = []
        for i in range(len(feature_names)):
            if (feature_names[i][1] == 'NUMERIC'):
                featype.append([0])
            else:
                if (not feature_names[i][1][0].isdigit()):
                    feature_nomimal = np.arange(0, len(feature_names[i][1]))
                    featype.append([int(number) for number in feature_nomimal])
                else:
                    featype.append([int(number) for number in feature_names[i][1]])
        return X, Y, featype


class ReadData:
    def __init__(self, datas=[], allIndices=[], genpath="arff/"):
        self.genpath = genpath
        '''ALL 80 datasets from KDIS (http://www.uco.es/kdis/mllresources/)'''
        datasnames_ = ["20NG", "3Sources_bbc1000", "3Sources_guardian1000", "3Sources_inter3000",
                       "3Sources_reuters1000", "Bibtex", "Birds", "Bookmarks", "CAL500", "CHD_49",
                       "Corel16k001", "Corel16k002", "Corel16k003", "Corel16k004", "Corel16k005", "Corel16k006",
                       "Corel16k007", "Corel16k008", "Corel16k009", "Corel16k010",
                       "Corel5k", "Delicious", "Emotions", "Enron", "EukaryoteGO", "EukaryotePseAAC", "Eurlex_dc_1",
                       "Eurlex_ev_1", "Eurlex_sm_1", "Flags",
                       "Foodtruck", "Genbase", "GnegativeGO", "GnegativePseAAC", "GpositiveGO", "GpositivePseAAC",
                       "HumanGO", "HumanPseAAC", "Image", "Imdb",
                       "Langlog", "Mediamill", "Medical", "Ohsumed", "Nuswide_BoW", "Nuswide_cVLAD", "PlantGO",
                       "PlantPseAAC", "Rcv1subset1", "Rcv1subset2",
                       "Rcv1subset3", "Rcv1subset4", "Rcv1subset5", "Reuters_K500", "Scene", "Slashdot", "Chemistry",
                       "Chess", "Coffee", "Cooking",
                       "CS", "Philosophy", "Tmc2007", "Tmc2007_500", "VirusGO", "VirusPseAAC", "Water_quality", "Arts",
                       "Business", "Computers",
                       "Education", "Entertainment", "Health", "Recreation", "Reference", "Science", "Social",
                       "Society", "Yeast", "Yelp"]
        if (len(allIndices) == 0):
            for i in range(len(datas)):
                allIndices.append(datasnames_.index(datas[i]))
        dimALL_ = [19300, 352, 302, 169, 294, 7395, 645, 87860, 502, 555, 13770, 13760, 13760, 13840, 13850, 13860,
                   13920, 13860, 13880, 13620,
                   5000, 16110, 593, 1702, 7766, 7766, 19350, 19350, 19350, 194, 407, 662, 1392, 1392, 519, 519, 3106,
                   3106, 2000, 120900,
                   1460, 43910, 978, 13930, 269599, 269600, 978, 978, 6000, 6000, 6000, 6000, 6000, 6000, 2407, 3782,
                   6961, 1675, 225, 10490,
                   9270, 3971, 28600, 28600, 207, 207, 1060, 7484, 11210, 12440, 12030, 12730, 9205, 12830, 8027, 6428,
                   12110, 14510, 2417, 10810]
        dimTrains_ = [12933, 240, 204, 112, 201, 4941, 432, 58862, 327, 372, 9241, 9165, 9130, 9223, 9259, 9287, 9317,
                      9250, 9284, 9084,
                      3332, 10743, 395, 1151, 5179, 5179, 17413, 17413, 17413, 132, 275, 444, 931, 931, 347, 347, 2053,
                      2053, 1501, 81363,
                      978, 29418, 659, 9301, 161789, 161789, 644, 644, 4045, 4045, 4021, 3997, 3964, 4014, 1618, 2546,
                      4686, 1107, 153, 6990,
                      6115, 2623, 19559, 19140, 131, 131, 710, 5056, 7523, 6270, 8092, 8569, 6158, 8641, 5346, 4283,
                      8135, 9685, 1629, 7240]
        dimTests_ = [6367, 112, 98, 57, 93, 2454, 213, 28994, 175, 183, 4525, 4596, 4630, 4614, 4588, 4572, 4598, 4614,
                     4600, 4534,
                     1668, 5362, 198, 551, 2587, 2587, 1935, 1935, 1935, 62, 132, 218, 461, 461, 172, 172, 1053, 1053,
                     499, 39556,
                     482, 14489, 319, 4628, 107859, 107859, 334, 334, 1955, 1955, 1979, 2003, 2036, 1986, 789, 1236,
                     2275, 568, 72, 3501,
                     3155, 1348, 9037, 9456, 76, 76, 350, 2428, 3691, 6174, 3938, 4161, 3047, 4187, 2681, 2145, 3976,
                     4827, 788, 3566]
        num_feat_ = [1006, 1000, 1000, 3000, 1000, 1836, 260, 2150, 68, 49, 500, 500, 500, 500, 500, 500, 500, 500, 500,
                     500,
                     499, 500, 72, 1001, 12690, 440, 5000, 5000, 5000, 19, 21, 1186, 1717, 440, 912, 440, 9844, 440,
                     294, 1001,
                     1004, 120, 1449, 1002, 501, 129, 3091, 440, 47240, 47240, 47240, 47230, 47240, 500, 294, 1079, 540,
                     585, 1763, 577,
                     635, 842, 49060, 500, 749, 440, 16, 23150, 21920, 34100, 27530, 32000, 30610, 30320, 39680, 37190,
                     52350, 31800, 103, 671]
        num_label_ = [20, 6, 6, 6, 6, 159, 19, 208, 174, 6, 153, 164, 154, 162, 160, 162, 174, 168, 173, 144, 374, 983,
                      6, 53, 22, 22, 412, 3993, 201, 7, 12, 27, 8, 8, 4, 4, 14, 14, 5, 28,
                      75, 101, 45, 23, 81, 81, 12, 12, 101, 101, 101, 101, 101, 103, 6, 22, 175, 227, 123, 400, 274,
                      233, 22, 22, 6, 6, 14, 26, 30, 33, 33, 21, 32, 22, 33, 40, 39, 27, 14, 5]
        self.datasnames, self.dimALL, self.dimTrains, self.dimTests, self.num_feats, self.num_labels = [], [], [], [], [], []
        for i in range(len(allIndices)):
            self.datasnames.append(datasnames_[allIndices[i]])
            self.dimALL.append(dimALL_[allIndices[i]])
            self.dimTrains.append(dimTrains_[allIndices[i]])
            self.dimTests.append(dimTests_[allIndices[i]])
            self.num_feats.append(num_feat_[allIndices[i]])
            self.num_labels.append(num_label_[allIndices[i]])

    def readData_org(self, index, wantfeature):
        # print(self.num_labels)
        label_count = self.num_labels[index]
        path = self.genpath + self.datasnames[index]
        X, Y, featype = read_arff(path, label_count, wantfeature)
        dimTrain = self.dimTrains[index]
        dimTest = self.dimTests[index]
        train_idx = np.arange(dimTrain)
        test_idx = np.arange(dimTrain, dimTrain + dimTest)
        return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx], featype

    def readDataall(self, index):
        label_count = self.num_labels[index]
        path = self.genpath + self.datasnames[index]
        X, Y, featype = read_arff(path, label_count, False)
        return np.array(X.todense()), np.array(Y.todense())

    def readData(self, index, wantfeature=False, wantarray=True):
        X, Y, Xt, Yt, f = self.readData_org(index, wantfeature)
        print(self.datasnames[index])
        if (wantarray):
            X, Y, Xt, Yt = np.array(X.todense()), np.array(Y.todense()), np.array(Xt.todense()), np.array(Yt.todense())
        if (wantfeature):
            return X, Y, Xt, Yt, f
        else:
            return X, Y, Xt, Yt

    def readData_CV(self, index, CV=10):
        label_count = self.num_labels[index]
        path = self.genpath + self.datasnames[index]
        X, Y, f = read_arff(path, label_count)
        k_fold = IterativeStratification(n_splits=CV, order=1)
        # for train, test in k_fold.split(X, Y):
        #     print(np.shape(train),np.shape(test))
        return k_fold, np.array(X.todense()), np.array(Y.todense())

    def getnum_label(self):
        return self.num_labels


if __name__ == "__main__":
    print('a')
    datasnames = ["Birds", "Enron", "Langlog", "Medical", "Scene", "VirusGO", "Yeast", "Yelp", "HumanGO", "Tmc2007_500"]
    rd = ReadData(datas=datasnames, genpath='arff/')
    print(rd.datasnames)
    print(rd.dimALL)
    print(rd.dimTrains)
    print(rd.dimTests)
    print(rd.num_labels)
    print(rd.num_feats)
