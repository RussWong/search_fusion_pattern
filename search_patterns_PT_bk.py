# -*- coding: utf-8 -*-
# THIS FILE IS PART OF INTEL MLPC OOB PROJECT
# search_patterns_TF_bk.py - The core part of searching fusion pattern 
# Copyright (c) 2021 zhiwei.huang@intel.com

import re, os
import pandas as pd
import joblib
import tkinter
import torch
import torch.fx
from torch.fx import symbolic_trace
from collections import namedtuple
def clean_op_type(op):
    if '(' in op:
        index = op.find('(')
        op_type = op[:index]
    else:
        op_type = op
    return op_type

def parse_graph(torch_graph, modules):
    node_name_details = {}
    node_details = namedtuple('node_details', ['node', 'op_type','outputs'])
    for node in torch_graph.nodes:
        if node.op == 'call_module':
            op_type = clean_op_type(str(modules[node.target]))
        elif node.op == 'call_function':
            op_type = clean_op_type(node.target.__name__)
        else:
            op_type = clean_op_type(node.target)
        each_node = node_details(node=node, op_type=op_type, outputs=[])
        if node.name not in node_name_details:
            node_name_details[node.name] = each_node
 
    for node_name, node_details in node_name_details.items():
        args = tuple(tkinter._flatten(node_details.node.args))
        for each_input in args:
            if isinstance(each_input, torch.fx.node.Node) and each_input.name in node_name_details:
                node_name_details[each_input.name].outputs.append((node_name, node_details.op_type))
    return node_name_details

def dfs(nodedef_and_outputs_list):
    mark_dict = {}
    for i in range(len(nodedef_and_outputs_list)):
        target =clean_op_type(nodedef_and_outputs_list[i].op_type)
        an = is_comp(target)
        if an:
            begin = ' '
            sub_res = []
            pattern = []
            
            dfs_next(nodedef_and_outputs_list[i].node, target, nodedef_and_outputs_list[i].outputs, pattern, sub_res, begin, mark_dict)    
        else:
            continue
        
def is_comp(op):
    comp = re.search('(.*(M|m)at(M|m)ul.*)|(.*(C|c)onv.*)|(.*mm.*)|(l|L)stm|(l|L)inear|LSTM',op)
    if comp:
        return True
    else:
        return False

def dfs_next(node, target, outputs_nodename_op, pattern, sub_res, begin, mark_dict):
    mark_dict[node.name] = True 
    if is_comp(target) and not pattern:
        pattern.append(target)
    elif is_comp(target) and pattern:
        mark_dict[node.name] = False
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return
    elif outputs_nodename_op and len(outputs_nodename_op) > 1:
        pattern.append(target)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return    
    elif not outputs_nodename_op:
        pattern.append(target)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return
    elif outputs_nodename_op and len(outputs_nodename_op) == 1 and outputs_nodename_op[0][0] in mark_dict and mark_dict[outputs_nodename_op[0][0]]:
        pattern.append(target)
        sub_res.append(pattern)
        pattern = []
        res.append(sub_res)
        sub_res = []
        return
    elif not is_comp(target) and not node.args:
        return 
    else:
        pattern.append(target)

    for i in range(len(outputs_nodename_op)):
        pattern1 = pattern
        (node, target, outputs) = graph_info[outputs_nodename_op[i][0]]
        if node.name not in mark_dict or not mark_dict[node.name]: 
            dfs_next(node, target, outputs, pattern1, sub_res, begin, mark_dict)

def get_ops(result):
    op_dict = {}   
    for res in result:
        op_dict[tuple(res)] = op_dict.get(tuple(res), 0) + 1
    return op_dict

def get_data(oob_model_statistic_dict):
    op_list = []
    op_dict = {}
    op_models = {}
    for key,value in oob_model_statistic_dict.items():
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
    model_path_list = []
    for root,dirs,files in os.walk('path/to/dir'):
        for file in files:
            model_path_list.append(os.path.join(root,file))
    total = len(model_path_list)
    failed = []
    total_result = []
    total_success = 0
    total_failed = 0
    flag = False
    oob_model = {}
    jit_or_trace = ''
    for i in range(total):
        try:
            print(model_path_list[i])
            try:
                loaded_gm = torch.load(model_path_list[i])
                jit_or_trace = 'trace'
            except:
                loaded_gm = torch.jit.load(model_path_list[i])
                jit_or_trace = 'jit'
            modules = dict(loaded_gm.named_modules())
            print('=======')
            print(type(loaded_gm.graph))
            graph_info = parse_graph(loaded_gm.graph, modules)
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
            print('the {} model failed to load! ERROR is {}'.format(i, e))
            total_failed += 1
            if flag:
                failed.append((i, jit_or_trace, e))
            else:
                failed.append(i)
        flag = False
    op_list,op_dict,op_models = get_data(oob_model)
    output = pd.DataFrame([op_dict,op_models],index=["Count in All Models","Model Count having this OP"]).transpose()
    output.to_csv('./PT_op_pattern.csv',encoding='gbk')

    pattern_nums = {}
    for result in total_result:
        for res in result:
            if len(res) > 1:
                pattern_nums[tuple(res)] = pattern_nums.get(tuple(res), 0) + 1
