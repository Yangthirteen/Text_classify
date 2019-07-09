from django.shortcuts import render
from django.shortcuts import HttpResponse
import multiprocessing
import pandas as pd
import numpy as np
import jieba
import yaml
# gensim:用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达。
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation, Bidirectional, Conv1D, GlobalMaxPooling1D
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

vocab_dim = 100  # 是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
n_exposures = 10  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
window_size = 7  # 表示当前词与预测词在一个句子中的最大距离是多少
cpu_count = multiprocessing.cpu_count()  # 参数控制训练的并行数
n_iterations = 1  # 迭代次数，默认为5
max_len = 10
input_length = 100
batch_size = 32
n_epoch = 11  # 训练次数

# Create your views here.
evaluate = []


'''
* ━━━━━━神兽出没━━━━━━
* 　　　┏┓　　　┏┓
* 　　┏┛┻━━━┛┻┓
* 　　┃　　　　　　　┃
* 　　┃　　　━　　　┃
* 　　┃　┳┛　┗┳　┃
* 　　┃　　　　　　　┃
* 　　┃　　　┻　　　┃
* 　　┃　　　　　　　┃
* 　　┗━┓　　　┏━┛Code is far away from bug with the animal protecting
* 　　　　┃　　　┃ 神兽保佑,代码无bug
* 　　　　┃　　　┃
* 　　　　┃　　　┗━━━┓
* 　　　　┃　　　　　　　┣┓
* 　　　　┃　　　　　　　┏┛
* 　　　　┗┓┓┏━┳┓┏┛
* 　　　　　┃┫┫　┃┫┫
* 　　　　　┗┻┛　┗┻┛
*
* ━━━━━━感觉萌萌哒━━━━━━
'''


def index(request):
    category = ""
    classes = ""
    response = ""

    if request.method == "POST":
        sentence = request.POST.get("sentence", None)
        temp = {"value": sentence}
        evaluate.append(temp)
        s = lstm_predict(sentence)
        print(s)
        if s == 0:
            category = "积极"
            classes = "此评论是积极评论"
            response = "谢谢小主的支持"
        else:
            category = "消极"
            classes = "第"+str(s)+"类评论"
            if s == 1:
                classes = classes + "  吐槽内容"
                response = "内容不好，非常抱歉"
            elif s == 2:
                classes = classes + "  吐槽服务"
                response = "服务不好，非常抱歉"
            elif s == 3:
                classes = classes + "  吐槽质量"
                response = "质量不好，非常抱歉"
            elif s == 4:
                classes = classes + "  吐槽物流"
                response = "物流不好，非常抱歉"
            elif s == 5:
                classes = classes + "  吐槽价格"
                response = "价格不好，非常抱歉"
            elif s == 6:
                classes = classes + "  吐槽酒店服务"
                response = "酒店服务不好，非常抱歉"
            elif s == 7:
                classes = classes + "  吐槽酒店设施"
                response = "酒店设施不好，非常抱歉"
            elif s == 8:
                classes = classes + "  吐槽周边环境"
                response = "周边环境不好，非常抱歉"
            elif s == 9:
                classes = classes + "  吐槽酒店价格"
                response = "酒店价格不好，非常抱歉"
    return render(request, "index.html", {"classes": classes,
                                          "category": category,
                                          "response": response,
                                          "evaluate": evaluate})


# 加载文件
def loadfile():
    # 读取文件
    neg = pd.read_excel('../static/data/neg1.xlsx', sheet_name=0, header=None)
    pos = pd.read_excel('../static/data/pos1.xls', sheet_name=0, header=None, index=None)

    negative = np.delete(np.array(neg[0]), 0)
    # pos = pos[:600]
    # print(len(pos))
    label = np.concatenate((np.zeros(len(pos), dtype=int), negative))
    text = np.concatenate((np.array(pos[0]), np.delete(np.array(neg[1]), 0)))
    return label, text


# 对句子进行分词，并取掉换行
def tokenizer(text):
    """Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    """
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(text):
    # 建立一个空的模型对象
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    # 遍历一次语料库建立词典
    model.build_vocab(text)
    # 第二次遍历语料库建立神经网络模型
    model.train(text, total_examples=model.corpus_count, epochs=model.epochs)
    # 查询结果训练可以通过model.save('fname')或model.save_word2vec_format(fname)来保存为文件
    model.save('../static/lstm_data/Word2vec_model.pkl')

    index_dict, word_vectors, text = create_dictionaries(model=model, text=text)
    return index_dict, word_vectors, text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        text=None):
    maxlen = 100  # 向量截断长度
    """Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
    """
    if (text is not None) and model is not None:
        # 可以理解为python中的字典对象, 其Key是字典中的词，其Val是词对应的唯一数值型ID
        gensim_dict = Dictionary()

        # 函数doc2bow()只是计算每个唯一的词的出现频率，将词转化整型词id并且将结果作为稀疏向量返回
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

        # 获取给定词的索引
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有词频数超过10的词语的索引  k->key v->value

        # 输出给定词的词向量
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有词频数超过10的词语的词向量

        # 文本变数字
        def parse_dataset(text):
            """
            Words become integers
            """
            data = []
            for sentence in text:
                # print(sentence) # 已经分词好的句子
                new_txt = []
                for word in sentence:
                    try:
                        # print(word)# 单个分词
                        new_txt.append(w2indx[word])
                        # print(w2indx[word]) # 索引
                    except:
                        new_txt.append(0)
                # print(new_txt) # 所有词向量
                data.append(new_txt)
            return data

        text = parse_dataset(text)

        # 将多个序列截断或补齐为相同长度
        text = sequence.pad_sequences(text, maxlen=maxlen)
        return w2indx, w2vec, text
    else:
        print('No data provide')


def get_data(index_dict, word_vectors, text, label):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，词频小于10的词语索引为0，所以加1
    # print(n_symbols) # 4683

    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    # print(embedding_weights)

    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    """
    随机划分训练集和测试集的函数
    参数：
    test_size：测试集大小。如果为浮点型，则在0.0-1.0之间，代表测试集的比例；如果为整数型，则为测试集样本的绝对数量；如果没有，则为训练集的补充。默认情况下，值为0.25 。此外，还与版本有关。
    train_size: 训练集大小。如果为浮点型，则在0.0-1.0之间，代表训练集的比例；如果为整数型，则为训练集样本的绝对数量；如果没有，则为测试集的补充。
    random_state：指定随机方式。一个整数或者RandomState实例，或者None 。如果为整数，则它指定了随机数生成器的种子；如果为RandomState实例，则指定了随机数生成器；如果为None，则使用默认的随机数生成器，随机选择一个种子
    shuffle：布尔值。是否在拆分前重组数据。如果shuffle=False，则stratify必须为None。
    stratify：array-like or None。如果不是None,则数据集以分层方式拆分，并使用此作为类标签。
    返回值：拆分得到的train和test数据集。
    """
    x_train, x_test, label_train, label_test = train_test_split(text, label, test_size=0.2, random_state=None)
    return n_symbols, embedding_weights, x_train, label_train, x_test, label_test


# 定义网络结构
def train_lstm(n_symbols, embedding_weights, text_train, label_train, text_test, label_test):
    print('Defining a simple Keras Model')
    kernel_size = 3
    filters = 250

    """
    顺序模型是多个网络层的线性堆叠。
    参数:
    input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
    output_dim：大于0的整数，代表全连接嵌入的维度
    embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考initializers
    embeddings_regularizer: 嵌入矩阵的正则项，为Regularizer对象
    embeddings_constraint: 嵌入矩阵的约束项，为Constraints对象
    mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。设置为True的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 1。
    input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，则必须指定该参数，否则Dense层的输出维度无法自动推断。
    """
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(n_symbols, embedding_weights))        # 使用Embedding层将每个词编码转换为词向量
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # 池化
    model.add(GlobalMaxPooling1D())
    model.add(Dense(label_train.shape[1], activation='softmax'))  # 第一个参数units: 全连接层输出的维度，即下一层神经元的个数。
    model.add(Dropout(0.2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # 本函数编译模型以供训练
    # binary_crossentropy二分类
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',  # 损失函数，为预定义损失函数名或一个目标函数
                  optimizer='rmsprop',              # 优化器，为预定义优化器名或优化器对象
                  metrics=['accuracy'])

    #  本函数用以训练模型
    #  参数：
    #  x_train输入数据
    #  y_train标签
    #  batch_size：整数，指定进行梯度下降时每个batch包含的样本数
    #  epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止
    #  verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
    #  validation_data：形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。此参数将覆盖validation_spilt
    print("Train...")
    history = model.fit(text_train, label_train,
                        batch_size=batch_size,
                        epochs=n_epoch,
                        verbose=2,
                        validation_data=(text_test, label_test))

    """
    本函数按batch计算在某些输入数据上模型的误差
    参数
    x：输入数据，与fit一样，是numpy array或numpy array的list
    y：标签，numpy array
    batch_size：整数，含义同fit的同名参数
    verbose：含义同fit的同名参数，但只能取0或1
    sample_weight：numpy array，含义同fit的同名参数
    """
    print("Evaluate...")
    score = model.evaluate(text_test, label_test,
                           batch_size=batch_size)

    #  model.to_yaml() 以 YAML 字符串的形式返回模型的表示
    yaml_string = model.to_yaml()

    # python open() 函数用于打开一个文件，创建一个 file 对象，相关的方法才可以调用它进行读写
    with open('D:/javafile/Text_classify_test/static/lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))

    #  model.save_weights(filepath) 将模型权重存储为 HDF5 文件
    model.save_weights('D:/javafile/Text_classify_test/static/lstm_data/lstm.h5')

    #  输出误差
    print(model.metrics_names)
    print('Test score:', score)

    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# 训练模型，并保存
def train():
    print('Loading Data...')
    label, text = loadfile()

    """
    将类向量（整数）转换为二进制类矩阵
    参数：
    y: 需要转换成矩阵的类矢量 (从 0 到 num_classes 的整数)。
    num_classes: 总类别数。
    dtype: 字符串，输入所期望的数据类型 (float32, float64, int32...)
    """

    print('Tokenising...')
    text = tokenizer(text)

    print('Training a Word2vec model...')
    index_dict, word_vectors, text = word2vec_train(text)

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, text_train, label_train, text_test, label_test \
        = get_data(index_dict, word_vectors, text, label)

    print(label_train)
    label_test = to_categorical(label_test)
    label_train = to_categorical(label_train)
    print(label_train.shape)
    print(label_train[0])

    train_lstm(n_symbols, embedding_weights, text_train, label_train, text_test, label_test)


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)

    # 使用model.load(fname)或model.load_word2vec_format(fname,encoding='utf-8')读取查询结果
    model = Word2Vec.load('static/lstm_data/Word2vec_model.pkl')
    _, _, text = create_dictionaries(model, words)
    return text


# 执行结果
def lstm_predict(string):
    print('loading model......')
    with open('static/lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)

    # 加载回来
    model = model_from_yaml(yaml_string)

    #  model.load_weights(filepath, by_name=False): 从 HDF5 文件（由 save_weights 创建）中加载权重
    print('loading weights......')
    model.load_weights('static/lstm_data/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)

    # 数组(2,2,-1)就是c的形状，一共有三阶，第三个数字是reshape后数组a中最小单元中元素个数，在这里是3，如果是-1则表示可以自动推测出
    data.reshape(1, -1)

    # print(data)
    # 本函数按batch产生输入数据的类别预测结果
    result = model.predict_classes(data)
    print(result)
    print(string, '：----》第', result[0], '类')
    return result[0]


if __name__ == '__main__':
    train()
    # loadfile()
