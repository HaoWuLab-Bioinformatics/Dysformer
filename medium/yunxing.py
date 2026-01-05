import subprocess
import os


def run_experiment():
    # 显式指定 Conda 解释器
    python_executable = "/home/mzy/anaconda3/envs/pytorch/bin/python3.10"

    # 设置工作目录到 medium
    os.chdir("/home/mzy/experments/Desktop/pythonProjectexperments/hyperbolic-transformer-master/medium")

    command = [
        python_executable, "main.py",
        "--dataset", "airport",
        "--method", "Dysformer",
        "--lr", "0.005",
        "--weight_decay", "1e-3",
        "--hidden_channels", "256",
        "--use_graph", "1",
        "--gnn_dropout", "0.4",
        "--gnn_use_bn", "1",
        "--gnn_num_layers", "3",
        "--gnn_use_init", "1",
        "--trans_num_layers", "1",
        "--trans_use_residual", "1",
        "--trans_use_bn", "0",
        "--graph_weight", "0.2",
        "--trans_dropout", "0.2",
        "--device", "0",
        "--runs", "1",
        "--power_k", "2.0",
        "--epochs", "5000",
        "--decoder", "hyp",
        "--k_in", "1.0",
        "--k_out", "2.0",
        "--data_dir", "../data",
        "--decoder_type", "hyp"
    ]

    # 运行 main.py，实时读取输出
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    # 逐行打印输出，避免缓存
    for line in process.stdout:
        print(line, end="")  # 让输出实时刷新
    for line in process.stderr:
        print(line, end="")  # 也打印错误信息

    process.stdout.close()
    process.stderr.close()
    process.wait()  # 等待进程结束


if __name__ == "__main__":
    run_experiment()