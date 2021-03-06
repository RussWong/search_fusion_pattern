# -*- coding: utf-8 -*-
# THIS FILE IS PART OF INTEL MLPC OOB PROJECT
# search_patterns_TF_bk.py - The core part of searching fusion pattern
# Copyright (c) 2021 zhiwei.huang@intel.com

import re, os
import tensorflow as tf
import pandas as pd
from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_util import GraphAnalyzer
from neural_compressor.model import get_model_type
#fix oob model_path
def fix_oob_model_path(df):
    return df
#build session 
def graph_session(model, **kwargs):
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=model, config=config)
    return sess

def graph_def_session(model, **kwargs):
    graph = tf.Graph()
    try:
        with graph.as_default():
            tf.import_graph_def(model, name='')
    except:
        print("run here.")

    return graph_session(graph, **kwargs)

def frozen_pb_session(model, **kwargs):
    graph_def = tf.compat.v1.GraphDef()
    model = model if model.endswith('.pb') else model + '.pb'
    with open(model, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def_session(graph_def, **kwargs)

def ckpt_session(model):    
    ckpt_prefix = [os.path.splitext(i)[0] for i in os.listdir(model) \
        if i.endswith(".meta")][0]
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(\
            os.path.join(model, ckpt_prefix + '.meta'), clear_devices=True)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, os.path.join(model, ckpt_prefix))
    return sess

#build session specially for df.model_path[35]
def thirty_five_ckpt_session(model):
    ckpt_prefix = [os.path.splitext(i)[0] for i in os.listdir(model) \
        if i.endswith(".meta")][4]
    print(ckpt_prefix)
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(\
            os.path.join(model, ckpt_prefix + '.meta'), clear_devices=True)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, os.path.join(model, ckpt_prefix))
    return sess

def Keras_session(model):
    assert tf.version.VERSION > '2.1.0', 'keras model need tensorflow version > 2.1.0....'
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    if not isinstance(model, tf.keras.Model):
        model = tf.keras.models.load_model(model)
        print(model)
    kwargs = dict(zip(model.input_names, model.inputs))
    if tf.version.VERSION > '2.2.0':
        from tensorflow.python.keras.engine import keras_tensor
        if keras_tensor.keras_tensors_enabled():
            for name, tensor in kwargs.items():
                kwargs[name] = tensor.type_spec
    full_model = tf.function(lambda **kwargs: model(kwargs.values()))
    concrete_function = full_model.get_concrete_function(**kwargs)
    frozen_model = convert_variables_to_constants_v2(concrete_function)
    graph_def = frozen_model.graph.as_graph_def()
    return graph_def_session(graph_def)

def Saved_model_session(model): 
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
    loader = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model)
    return sess

#The beginning op of the target patterns
def is_comp(op):
    comp = ['Conv2D','Conv2DBackpropFilter','Conv2DBackpropInput','Conv3D','Conv3DBackpropFilterV2','Conv3DBackpropInputV2','DepthwiseConv2dNative','DepthwiseConv2dNativeBackpropFilter','DepthwiseConv2dNativeBackpropInput','MatMul','BatchMatMul','BatchMatMulV2']
    if op in comp:
        return True
    else:
        return False

def dfs(nodedef_and_outputs_list):#(nodedef,outputs)
    mark_dict = {}
    for i in range(len(nodedef_and_outputs_list)):
        an = is_comp(nodedef_and_outputs_list[i].node.op)
        if an:
            begin = ' '
            sub_res = []
            pattern = []
            
            dfs_next(sess, nodedef_and_outputs_list[i].node, nodedef_and_outputs_list[i].outputs, pattern, sub_res, begin, mark_dict)    
        else:
            continue

def dfs_next(sess, node, outputs_nodename_op, pattern, sub_res, begin, mark_dict):
    mark_dict[node.name] = True
    if is_comp(node.op) and not pattern:
        pattern.append(node.op)
    elif is_comp(node.op) and pattern:
        mark_dict[node.name] = False
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return
    elif outputs_nodename_op and len(outputs_nodename_op) > 1:
        pattern.append(node.op)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return    
    elif not outputs_nodename_op:
        pattern.append(node.op)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return
    elif outputs_nodename_op and len(outputs_nodename_op) == 1 and outputs_nodename_op[0][0] in mark_dict and mark_dict[outputs_nodename_op[0][0]]:
        pattern.append(node.op)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return     
    elif not re.search('(.*(m|M)ul)|(Conv.*)', node.op) and not node.input:
        return 
    else:
        pattern.append(node.op)

    for i in range(len(outputs_nodename_op)):
        pattern1 = pattern
        if node.name == 'DetectionOutput/while/while/Identity':
            print(pattern1)#just for debug, ignore it
        (node, outputs) = graph_info[sess.graph.get_operation_by_name(outputs_nodename_op[i][0]).name]
        if node.name not in mark_dict or not mark_dict[node.name]: 
            dfs_next(sess, node, outputs, pattern1, sub_res, begin, mark_dict)

#Calculate the number of patterns
def get_ops(result):
    op_dict = {}   
    for res in result:
        op_dict[tuple(res)] = op_dict.get(tuple(res), 0) + 1
    return op_dict

def get_data(oob_model_statistic_dict):
    op_list = []
    op_dict = {}
    op_models = {}
    for _ ,value in oob_model_statistic_dict.items():
        for op,num in value[0].items():
            if op in op_list:
                op_dict[op] += num
                op_models[op] += 1
            else:
                op_list.append(op)
                op_dict[op] = num
                op_models[op] = 1
    return op_list,op_dict,op_models

if __name__ == "__main__":
    df = pd.read_excel("path/to/file", usecols=[0, 2])
    df = fix_oob_model_path(df)
    total = len(df.model_path)
    cur_graph = GraphAnalyzer()

    failed = []
    total_result = []
    total_success = 0
    total_failed = 0
    flag = False
    oob_model = {}
    for i in range(total):
        try:            
            if i == 35:
                sess = thirty_five_ckpt_session(df.model_path[i])
                cur_graph.graph = sess.graph.as_graph_def()
            else:
                m_t = get_model_type(df.model_path[i])
                if m_t == 'frozen_pb':
                    sess = frozen_pb_session(df.model_path[i])
                    cur_graph.graph = sess.graph_def  
                elif m_t == 'checkpoint':
                    sess = ckpt_session(df.model_path[i])
                    cur_graph.graph = sess.graph.as_graph_def()
                elif m_t == 'keras':
                    sess = Keras_session(df.model_path[i])
                    cur_graph.graph = sess.graph.as_graph_def()
                elif m_t == 'saved_model':
                    sess = Saved_model_session(df.model_path[i])
                    cur_graph.graph = sess.graph.as_graph_def()
            
            flag = True
            graph_info = cur_graph.parse_graph()
            res = []
            nodedef_and_outputs_list = list(graph_info.values())
        
            dfs(nodedef_and_outputs_list)
            
            result = [y for x in res for y in x]
        
            op_dict = get_ops(result)
            oob_model[str(i)] = []
            oob_model[str(i)].append(op_dict)
            total_result.append(result)
            print('the {} model loaded successfully!'.format(i))
            total_success += 1
        except Exception as e:
            print(e)
            print('the {} model failed to load!'.format(i))
            total_failed += 1
            if flag:
                failed.append((i, m_t))
            else:
                failed.append(i)
        flag = False

    op_list,op_dict,op_models = get_data(oob_model)
    output = pd.DataFrame([op_dict,op_models],index=["Count in All Models","Model Count having this OP"]).transpose()
    output.to_csv('./TF_op_pattern.csv',encoding='gbk')
