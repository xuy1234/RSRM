import numpy as np
from pandas import read_csv

from model.config import Config
from model.pipeline import Pipeline


def pipeline(csv_1, csv_2, const=False):
    """
    主函数 用于生成表达式
    :param csv_1: train csv文件 ， 分割
    :param csv_2: test csv文件 ， 分割
    :param const: 是否需要参数
    """
    csv1, csv2 = read_csv(csv_1, header=None), read_csv(csv_2, header=None)
    x, t = np.array(csv1).T[:-1], np.array(csv1).T[-1]
    x_test, t_test = np.array(csv2).T[:-1], np.array(csv2).T[-1]
    config = Config()
    config.init()
    config.config_basic(const_opt=const, consts=const, verbose=True)
    config.set_input(x=x, t=t, x_=x_test, t_=t_test)
    model = Pipeline(config=config)
    best_exp, rmse = model.fit()
    print(f'\nresult: {best_exp} {rmse}')


if __name__ == '__main__':
    pipeline("nyugen/4_train.csv", "nyugen/4_test.csv", False)
