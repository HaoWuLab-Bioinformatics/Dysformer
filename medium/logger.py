import torch
import os

class Logger:
    """
    仅记录每个 run 的最佳 (val_acc, test_acc) 各一行
    """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]   # 每行: [val_acc, test_acc]
        self.test = None                           # 供外部 output 时使用

    # ---------- 写入 ----------
    def add_result(self, run, result):
        """
        result: (best_val_acc, best_test_acc) —— 只写一次
        """
        assert len(result) == 2, "result 必须是 (val_acc, test_acc)"
        assert 0 <= run < len(self.results)
        self.results[run] = list(result)           # 覆盖式写入(一行)

    # ---------- 打印 ----------
    def print_statistics(self, run=None):
        if run is not None:                        # 打印单个 run
            if not self.results[run]:
                print(f'Run {run+1:02d}: (no data)')
                return
            val, test = [x*100 for x in self.results[run]]
            print(f'Run {run+1:02d}: Best Val Acc = {val:.2f}%  |  Test Acc = {test:.2f}%')
            self.test = test
        else:                                      # 汇总多个 runs
            res = torch.tensor(self.results, dtype=torch.float32) * 100  # shape:(runs,2)
            val_mean, val_std   = res[:,0].mean().item(), res[:,0].std().item()
            test_mean, test_std = res[:,1].mean().item(), res[:,1].std().item()
            print(f'{len(self.results)} runs summary:')
            print(f'Val  Acc: {val_mean:.2f} ± {val_std:.2f}')
            print(f'Test Acc: {test_mean:.2f} ± {test_std:.2f}')
            self.test = test_mean
        return

    # ---------- 文件输出 ----------
    def output(self, out_path, info):
        """
        将 info 与最终 test acc 追加写入文件
        """
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(info + '\n')
            if self.test is not None:
                f.write(f'test acc: {self.test:.2f}\n')

    def save(self, params, results, filename):
        """
        params: dict  (训练参数)
        results: 任意结果字符串 / 数值
        """
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f"{results}\n{params}\n{'=='*50}\n\n")

# --------- 单次实验结果保存（保留原接口） ---------
def save_result(args, results):
    """
    results: torch.Tensor 或 list，长度 = runs，元素 = test_acc
    """
    folder = f'results/{args.dataset}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{folder}/{args.method}.csv'
    print(f'Saving results to {filename}')
    with open(filename, 'a+', encoding='utf-8') as w:
        for k, v in vars(args).items():
            w.write(f'{k}:{v} ')
        results = torch.tensor(results, dtype=torch.float32)
        w.write(f'{results.mean():.2f} ± {results.std():.2f}\n')
