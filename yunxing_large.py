import subprocess
import os
CUDA_LAUNCH_BLOCKING=1

def run_experiment():
    python_executable = "/home/user012/anaconda3/envs/pytorch/bin/python3.9"
    workdir = "/home/user012/experments/Desktop/pythonProjectexperments/Dysformer/large"
    os.chdir(workdir)

    # 重要：强制无缓冲输出 + 设置环境变量
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"

    if not os.path.exists("main.py"):
        print(f"[ERR] main.py 不在 {workdir} 下！")
        return

    command = [
        python_executable, "-u", "main.py",          # -u: unbuffered
        "--dataset", "ogbn-arxiv",
        "--method", "Dysformer",
        "--lr", "0.001",
        "--weight_decay", "0.",
        "--gnn_use_weight", "1",
        "--gnn_use_residual", "1",
        "--hidden_channels", "256",
        "--epochs", "5000",
        "--use_graph", "1",
        "--gnn_dropout", "0.5",
        "--gnn_use_bn", "1",
        "--gnn_num_layers", "3",
        "--gnn_use_init", "1",
        "--trans_num_layers", "1",
        "--trans_num_heads", "2",
        "--trans_use_residual", "1",
        "--trans_use_bn", "0",
        "--graph_weight", "0.2",
        "--trans_dropout", "0.",
        "--device", "1",
        "--runs", "30",
        "--power_k", "2.0",
        "--decoder", "hyp",
        "--k_in", "2",
        "--k_out", "0.5",
        "--data_dir", "/mnt/mnt1/mzy/data",
        "--decoder_type", "hyp",
        "--sub_dataset", "gcn_data",
        "--protocol", "semi",
        "--rand_split", "1",
        "--display_step", "1",
        "--optimizer_type", "adam",
        "--hyp_optimizer_type", "radam",
        "--patience", "100",
    ]

    print(f"[RUN] cwd={os.getcwd()}")
    print(f"[RUN] python {' '.join(command[1:3])} ...")

    # 合并 stderr 到 stdout，避免双流读取的阻塞问题
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # 实时逐行读取
    for line in process.stdout:
        print(line, end="")

    ret = process.wait()
    print(f"[DONE] child exit code = {ret}")


if __name__ == "__main__":
    run_experiment()