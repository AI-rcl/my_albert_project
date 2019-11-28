import os,sys
import tensorflow as tf

BASE_DIR=os.path.dirname(__file__)
BASE_MODEL_PATH=os.path.join(BASE_DIR,'base_model')
sys.path.append(BASE_MODEL_PATH)
from base_model.predict import Cls
import time


class Cls_loader():
    def __init__(self):
        self.bots_dir='bots'
        self.model_dict={}
        self._get_model_path()
        self._create_model()
    def _get_model_path(self):
        self.bots_list=os.listdir(self.bots_dir)
        self.model_path_list=[os.path.join(self.bots_dir,cls,'model') for cls in self.bots_list]
    def _create_model(self):
        for bot,model_path in zip(self.bots_list,self.model_path_list):
            bot_model=Cls(model_path)
            self.model_dict.setdefault(bot,bot_model)


# cls=Cls_loader()
# while True:
#     text=input('>>')
#     print(cls.model_dict['base_sentence'].predict([text]))

# path='bots/ner_bots'
# print(os.listdir(path))
# text='华为在哪里'
# ner_loader=Ner_Loader()
# print(ner_loader.model_dict)
# print(ner_loader.model_dict['global_ner'].predict(text))
# cls_loader=Cls_loader()
# print(cls_loader.model_dict['base_sentence'].predict(text))