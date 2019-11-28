import os
import shutil

# os.system('python base_model/ner_base/train.py %s'%('ner002'))
# os.system('''echo hello;
#           echo world''')

# label_list='a,b,c,d'
# os.system('python system_test.py %s'%label_list)
def mk_dir(bot_id):

    bot_dir='bots/'+bot_id
    data_path = bot_dir + '/data'
    model_path = bot_dir + '/model'
    if os.path.exists(bot_dir):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    else:
        os.mkdir(bot_dir)
        os.mkdir(data_path)
        os.mkdir(model_path)
    return 0

def remove_dir(mode,bot_id):
    bot_dir = 'bots/cls/' + bot_id

    if os.path.exists(bot_dir):
        shutil.rmtree(bot_dir)
    return 0



def train_cls(args):
    train_file='base_model/run_classifier.py'
    ckpt2pd_file='base_model/ckpt2pd.py'
    cls_name=args['cls_name']
    if args['train_epoch']:
        train_epoch=args['train_epoch']
    else:
        train_epoch=5.0
    # if args['label_list']:
    #     if type(args['label_list']):
    #         label_list=','.join(args['label_list'])
    #     else:
    #         raise TypeError('label_list must be a list')
    # else:
    #     raise ValueError("label_list cant't be empty")
    mk_dir(cls_name)
    os.system('''python %s %d %s;
                python %s %s'''%(train_file,train_epoch,cls_name,ckpt2pd_file,cls_name))

cls_args={'cls_name':'query_sentence',
      'train_epoch':100,
      }
train_cls(cls_args)

