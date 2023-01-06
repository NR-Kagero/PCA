import numpy as np


class PCA:
    def __init__(self, com_num=100):
        self.__com_num = com_num
        self.__mean = None
        self.__cov = None
        self.__components = None

    def fit(self, data):
        self.__mean = np.mean(data, axis=0)
        data = data - self.__mean

        self.__cov = np.cov(data.T)

        eign_vec, eign_vla = np.linalg.eig(self.__cov)
        eign_vec = eign_vec.T

        indx = np.argsort(eign_vla)[::-1]

        eign_vla = eign_vla[indx]
        eign_vec = eign_vec[indx]

        self.__components = eign_vec[:self.__com_num]

    def transform(self, data):
        data = data - self.__mean
        projected_data = np.dot(data, self.__components.T)

        return projected_data

    def inverse_transform(self, p_data):
        inv_eign = np.linalg.pinv(self.__components)
        original = np.dot(p_data, inv_eign.T) + self.__mean
        return original

    def get_components_num(self):
        return self.__com_num

    def get_components(self):
        return self.__components

    def get_me(self):
        return self.__com_num

    def __str__(self):
        d=[str(x) for x in self.__components]
        x="components : \n"+str(d)+"\ncomponents counts = "+str(self.__com_num)
        return x