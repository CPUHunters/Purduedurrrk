# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""
import os
import numpy as np
import tensorflow as tf

import time

#imagePath = '/tmp/test.jpg'                                      # 추론을 진행할 이미지 경로
modelFullPath = './tmp/output_graph.pb'                                      # 읽어들일 graph 파일 경로
labelsFullPath = './tmp/output_labels.txt'                                   # 읽어들일 labels 파일 경로


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(count):
    abc=count
    path_dir = './test/%s'%abc
    file_list = os.listdir(path_dir)
    file_list.sort()

    #file=open("./result.txt","a")
    answer = 0

    # if not tf.gfile.Exists(imagePath):
    #     tf.logging.fatal('File does not exist %s', imagePath)
    #     return answer
    # print(file_list)


    for item in file_list:
        file_path = ("./test/%s/"%abc)+item
        #print(file_path)
        if not tf.gfile.Exists(file_path):
            tf.logging.fatal('File does not exist %s',item)
            return answer
        image_data = tf.gfile.FastGFile(file_path,'rb').read()

        create_graph()

        with tf.Session() as sess:

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            print('< In %s th frame, %s image is being classified. >' % (abc,item))

            top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w) for w in lines]
            order=1
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                # print('\'%s\' (score = %.5f) %d' % (human_string[2:-3], score,order))
                order = order+1
                if ("bird" in human_string) and (order == 2) :
                    #data=("\'%s-%s\'  1\n" % (abc,item))
                    print("BIRD DETECTED!!!")
                    answer = 1
                    return answer
                # elif (order == 2):
                #     print("not detect")
                    #data=("\'%s-%s\'  0\n" % (abc,item))
            #file.write(data)
            f.close()
    return answer

if __name__ == '__main__':
    run_inference_on_image()