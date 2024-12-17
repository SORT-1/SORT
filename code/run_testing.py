import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import json
import pickle
import queue
import threading
import traceback
import argparse

from utils.subgraph_util import *
from utils.torch_input_generator import *
from utils.invoke_str_generator import *

lock = threading.Lock()

def generate_topological_seq(subgraph):
    # Generate code for subgraph
    # Save in-degree of every vertex for topological ranking
    in_degree = {}
    # Save in-edges of every vertex for generating input
    in_edges = {}

    vertices = subgraph["vid2vlb"]
    edges = subgraph["vid2toids"]

    # Initialize in-degree && in-vertices
    for v in vertices.keys():
        in_degree[v] = 0
        in_edges[v] = []

    # Count in-degree  && save in-vertices
    for from_v in edges.keys():
        for to_v in edges[from_v]:
            to_v = str(to_v)
            in_degree[to_v] += 1
            in_edges[to_v].append(from_v)

    # Topological ranking && generate testing code
    cur_vertices = queue.Queue()
    for v in in_degree.keys():
        if in_degree[v] == 0:
            cur_vertices.put(v)

    results = []
    while not cur_vertices.empty():
        cur_v = cur_vertices.get(block=False)
        results.append(cur_v)
        # Update in-degree for each vertex
        for v in edges[cur_v]:
            v = str(v)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                cur_vertices.put(v)
    # print("Gid : %d" % subgraph["gid"])
    # print(results)

    return results, in_edges


def run_subgraph_testing(args, subgraph, index, config_data, api_def):
    print("Start Gid %d testing...." % subgraph["gid"])
    node_id_list, in_edges_dict = generate_topological_seq(subgraph)
    print("Node Topological list:")
    print(node_id_list)

    error_count = 0

    cpu_inputs = {}
    gpu_inputs = {}
    cpu_results = {}
    gpu_results = {}
    compare_results = []

    t = 0
    start = index * args.batch_size
    while t < args.batch_size:
        try:
    # All Tensors --> np.ndarray
            cpu_input_dict = {}
            gpu_input_dict = {}
            cpu_output_dict = {}
            gpu_output_dict = {}
            # Start running the API sequence
            compare_node_id = node_id_list[-1]

            for node_id in node_id_list:
                print("Current node: %s" % node_id)
                node_label = subgraph["vid2vlb"][node_id]
                if "DropPath" in node_label:
                    for v in subgraph["vid2toids"][node_id]:
                        print("Node : %d lose his father DropPath...." % v)
                        in_edges_dict[str(v)] = []
                    continue

                in_edges = in_edges_dict[node_id]

                api_name = node_label

                if len(in_edges) == 0:
                    # Source && Follow all the same for initial input
                    cpu_node_input = generate_true_input(api_name, config_data)
                    gpu_node_input = cpu_node_input.copy()
                    # cpu_node_input = generate_new_input(api_name)
                else:
                    cpu_node_input = generate_fix_input(api_name, config_data, cpu_output_dict[in_edges[0]])
                    # print(cpu_node_input)
                    cpu_node_input = change_pre_api_output(cpu_node_input, cpu_output_dict[in_edges[0]])
                    # print(cpu_node_input)
                    gpu_node_input = change_pre_api_output(cpu_node_input, gpu_output_dict[in_edges[0]])
                    # print(cpu_node_input)

                # print('Out' + str(cpu_node_input))
                cpu_input_dict[node_id] = cpu_node_input
                gpu_input_dict[node_id] = gpu_node_input

                # print(cpu_input_dict)
                # Invoke the API on CPU
                node_input = dict(cpu_node_input)
                # print(node_input)
                convert_to_cpu(node_input)
                # print(node_input)
                cpu_output = generate_and_run_invoke_str(api_name, node_input, api_def)
                cpu_output_dict[node_id] = cpu_output

                # Invoke the API on GPU
                node_input = dict(gpu_node_input)
                convert_to_gpu(node_input)
                gpu_output = generate_and_run_invoke_str(api_name, node_input, api_def)
                gpu_output_dict[node_id] = gpu_output

            compare_results.append([start, "No Error"])
            cpu_inputs[start] = cpu_input_dict
            gpu_inputs[start] = gpu_input_dict
            cpu_results[start] = cpu_output_dict
            gpu_results[start] = gpu_output_dict

            t += 1
            start += 1
        except ErrorStartException:
            print("Next")
            error_count += 1
            if error_count >= 500:
                compare_results.append([start, "Next"])
                t += 1
                start += 1
            continue


    # Save all the inputs && results
    if len(cpu_inputs.keys()) > 0:
        with open(f"{args.data_path}/cpu_inputs_%d_%d.pickle" %
                  (subgraph["gid"], index), "wb+") as f:
            pickle.dump(cpu_inputs, f)
    if len(gpu_inputs.keys()) > 0:
        with open(f"{args.data_path}/gpu_inputs_%d_%d.pickle" %
                  (subgraph["gid"], index), "wb+") as f:
            pickle.dump(gpu_inputs, f)
    if len(cpu_results.keys()) > 0:
        with open(f"{args.data_path}/cpu_results_%d_%d.pickle" %
                  (subgraph["gid"], index), "wb+") as f:
            pickle.dump(cpu_results, f)
    if len(gpu_results.keys()) > 0:
        with open(f"{args.data_path}/gpu_results_%d_%d.pickle" %
                  (subgraph["gid"], index), "wb+") as f:
            pickle.dump(gpu_results, f)


def batch_run_subgraph_testing(args):
    # pool = ThreadPoolExecutor(max_workers=5)
    # Read subgraph from .json file
    with open(args.subgraph_path, "r") as f:
        subgraph_list = json.load(f)

    with open(args.config_range, "r") as f_config:
        content = f_config.read()
        # 解析 JSON
        config_data = json.loads(content)

    with open(args.api_def, "r") as f_def:
        api_def = f_def.readlines()

    for subgraph in subgraph_list:
        i = 0
        while i < args.batch:
            run_subgraph_testing(args, subgraph, i, config_data, api_def)
            i += 1
    # run_subgraph_testing(subgraph_list[0], 0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Testing Argument')
    parser.add_argument('-b', '--batch', type=int, default=50, help='Number of batches for generating test cases')
    parser.add_argument('-bs', '--batch_size', type=int, default=10, help='Number of test cases generated per batch')
    parser.add_argument('-s', '--subgraph_path', type=str, help='File path for the subgraph being tested')
    parser.add_argument('-config', '--config_range', type=str, help='File path for the API input feature configuration of the subgraph being tested')
    parser.add_argument('-def', '--api_def', type=str, help='API definition file')
    parser.add_argument('-d', '--data_path', type=str, help='File path to save the input-output results of the test cases')
    args = parser.parse_args()

    if not os.path.exists('./results'):
        os.mkdir('./results')

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    try:
        batch_run_subgraph_testing(args)
    except OverflowError:
        print('OverflowError')
        traceback.print_exc()

