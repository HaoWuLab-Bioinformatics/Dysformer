import torch
import wandb
import csv
import os
from datetime import datetime
'''
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, args=None):
        self.args = args
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 7
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    @staticmethod
    def get_results_string(best_result):
        result_string = ''
        r = best_result[:, 0]
        result_string += f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 1]
        result_string += f'Highest Test: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 2]
        result_string += f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 3]
        result_string += f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\t'
        r = best_result[:, 4]
        result_string += f'   Final Test: {r.mean():.2f} ± {r.std():.2f}'

        return result_string

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            # Ensure all elements are tensors and convert them properly
            result = [torch.tensor(r) * 100 if isinstance(r, (float, int)) else torch.tensor(r) * 100 for r in self.results[run]]
            result = torch.stack(result)  # Stack the list of tensors into a single tensor

            if self.args.save_whole_test_result:
                now = datetime.now()
                _month_day = now.strftime("%m%d")
                timestamp = now.strftime("%m%d-%H%M%S")
                results_path = mkdirs(f'results/runs/{self.args.dataset}/{_month_day}/{self.args.wandb_name}')
                with open(f'{results_path}/{run}-{self.args.run_id}-results.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Write the header (optional)
                    writer.writerow(["Epoch", "Train Acc", "Val Acc", "Test Acc", "Val Loss"])

                    # Write the data
                    for epoch in range(len(self.results[run])):
                        # Add 1 to run and epoch indices to match with human counting
                        formatted_row = ['{:.4f}'.format(float(x)) for x in self.results[run][epoch]]
                        writer.writerow([epoch * self.args.eval_step] + formatted_row)
                    # Write the args
                    writer.writerow([])
                    writer.writerow(["Args"])
                    for key, value in vars(self.args).items():
                        writer.writerow([key, value])

                    print(f"Saved results to {self.args.dataset}-{self.args.wandb_name}-{run}-results.csv")

            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print('==========================')
            print_str1 = f'>> Run {run + 1:02d}:\n' + \
                        f'\t Highest Train: {result[:, 0].max():.2f} ' + \
                        f'\t Highest Valid: {result[:, 1].max():.2f} ' + \
                        f'\t Highest Test: {result[:, 2].max():.2f}\n' + \
                        f'\t Chosen epoch based on Valid loss: {argmin * self.args.eval_step} ' + \
                        f'\t Final Train: {result[argmin, 0]:.2f} ' + \
                        f'\t Final Valid: {result[argmin, 1]:.2f} ' + \
                        f'\t Final Test: {result[argmin, 2]:.2f}'
            print(print_str1)

            print_str=f'>> Run {run + 1:02d}:' + \
                f'\t Highest Train: {result[:, 0].max():.2f} ' + \
                f'\t Highest Valid: {result[:, 1].max():.2f} ' + \
                f'\t Highest Test: {result[:, 2].max():.2f}\n' + \
                f'\t Chosen epoch based on Valid acc: {ind * self.args.eval_step} ' + \
                f'\t Final Train: {result[ind, 0]:.2f} ' + \
                f'\t Final Valid: {result[ind, 1]:.2f} ' + \
                f'\t Final Test: {result[ind, 2]:.2f}'
            print(print_str)
            self.test = result[ind, 2]
        else:
            best_results = []
            max_val_epoch = 0

            for r in self.results:
                r = [torch.tensor(res) * 100 if isinstance(res, (float, int)) else torch.tensor(res) * 100 for res in r]
                r = torch.stack(r)  # Stack the list of tensors into a single tensor
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                    max_val_epoch = r[:, 1].argmax()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()
            # if self.args.use_wandb:
            #     wandb.log({
            #         'Average Highest Train': r.mean().item(),
            #         'Std Highest Train': r.std().item(),
            #         'Average Highest Test': best_result[:, 1].mean().item(),
            #         'Std Highest Test': best_result[:, 1].std().item(),
            #         'Average Highest Valid': best_result[:, 2].mean().item(),
            #         'Std Highest Valid': best_result[:, 2].std().item(),
            #         'Average Final Train': best_result[:, 3].mean().item(),
            #         'Std Final Train': best_result[:, 3].std().item(),
            #         'Average Final Test': best_result[:, 4].mean().item(),
            #         'Std Final Test': best_result[:, 4].std().item()
            #     })
            return self.get_results_string(best_result)

    def save(self, params, results, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(f"{results}\n")
            file.write(f"{params}\n")
            file.write('=='*50)
            file.write('\n')
            file.write('\n')

import os
def save_result(args, results):
    if args.save_result:
        if not os.path.exists(f'results/{args.dataset}'):
            os.makedirs(f'results/{args.dataset}')
        filename = f'results/{args.dataset}/{args.method}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            write_obj.write(
                f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
                f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
                f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")
'''
# logger.py
import os
import csv
from datetime import datetime
import torch
import pandas as pd


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Logger(object):
    """
    记录 & 打印 & 保存 10 个验证/测试指标
        0  val_loss       5  test_loss
        1  val_acc        6  test_acc
        2  val_f1         7  test_f1
        3  val_prec       8  test_prec
        4  val_rec        9  test_rec
    """
    def append_run_to_excel(self, run: int, file_path: str,
                            select_by: str = 'val_acc') -> None:
        """
        将指定 run 的最佳指标（10 列）追加写入同一个 Excel。
        若目标文件已存在则读取→去重→拼接→写回；否则直接创建。
        """
        # ---------- 准备行数据 ----------
        # self.results[run] 形状: [epochs, 10]
        metrics = torch.tensor(self.results[run])

        if select_by == 'val_acc':
            best_ep = metrics[:, 1].argmax()      # val_acc 最大的 epoch
        elif select_by == 'test_acc':
            best_ep = metrics[:, 6].argmax()      # test_acc 最大的 epoch
        else:
            raise ValueError("select_by 仅支持 'val_acc' 或 'test_acc'")

        # 构建 DataFrame（第一列记录 run 编号）
        row = [run] + metrics[best_ep].tolist()
        cols = [
            'run',
            'val/loss', 'val/acc', 'val/f1', 'val/precision', 'val/recall',
            'test/loss', 'test/acc', 'test/f1', 'test/precision', 'test/recall'
        ]
        df_new = pd.DataFrame([row], columns=cols)

        # ---------- 追加写入 ----------
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if os.path.exists(file_path):
            # 读取旧文件
            df_old = pd.read_excel(file_path)
            # 防止重复行：去除同 run 已有记录
            df_old = df_old[df_old['run'] != run]
            # 拼接并重置索引
            df_new = pd.concat([df_old, df_new], ignore_index=True)

        # 覆盖写回（相当于“追加”效果）
        df_new.to_excel(file_path, index=False)
        print(f'📥 已追加保存 run {run} → {file_path}')
    def __init__(self, runs, args=None):
        self.args = args
        # results[run] = [ [metric10], [metric10], ... ]  按 epoch 存
        self.results = [[] for _ in range(runs)]

    # ------------------------------------------------------------------
    # 每到一次 eval_step 调用，把 10 个指标 append
    # ------------------------------------------------------------------
    def add_result(self, run: int, result: list):
        """
        Parameters
        ----------
        run : 第几个 run
        result : 长度 10 的 list / tuple，顺序见类注释
        """
        assert len(result) == 10, "Logger.add_result 需要 10 个指标"
        assert 0 <= run < len(self.results)
        self.results[run].append([float(x) for x in result])

    # ------------------------------------------------------------------
    # 打印单个 / 所有 run 统计
    # ------------------------------------------------------------------
    @staticmethod
    def _tensorize(lst):
        return torch.tensor(lst) * 100.0      # 百分比显示

    def print_statistics(self, run=None, mode='max_acc'):
        """
        run=None  => 打印所有 run 汇总
        run=int   => 打印该 run 最高 / 最佳 epoch
        mode      => 'max_acc'(按 val_acc 最大选 epoch) 或 'min_loss'
        """
        # === 单个 run === ------------------------------------------------
        if run is not None:
            r = self._tensorize(self.results[run])        # shape [epochs, 10]
            # 依据 val_acc 最大或 val_loss 最小挑 epoch
            if mode == 'max_acc':
                best_epoch = r[:, 1].argmax().item()
            else:
                best_epoch = r[:, 0].argmin().item()

            print('==========================')
            print(f'>> Run {run + 1:02d}:')
            print(f'   ├─ Highest Val Acc : {r[:, 1].max():.2f}%  '
                  f'@ epoch {r[:, 1].argmax().item()}')
            print(f'   └─ Selected Epoch  : {best_epoch}  '
                  f' Val Acc {r[best_epoch,1]:.2f}% | '
                  f' Test Acc {r[best_epoch,6]:.2f}%')
            # 方便主程序访问 test_acc
            self.test = r[best_epoch, 6]

            # 若开启保存每轮结果
            if getattr(self.args, 'save_whole_test_result', False):
                self._dump_whole_run_csv(run)

        # === 汇总全部 run === --------------------------------------------
        else:
            summary = []
            for r in self.results:
                r = self._tensorize(r)
                # 最高 val_acc
                best_ep = r[:, 1].argmax()
                summary.append([
                    r[best_ep, 1].item(),    # val_acc
                    r[best_ep, 6].item()     # test_acc
                ])
            summary = torch.tensor(summary)   # [runs, 2]

            print('==========================')
            print('>> All Runs Summary (based on best Val Acc)')
            print(f'   Val Acc : {summary[:,0].mean():.2f} ± {summary[:,0].std():.2f}%')
            print(f'   Test Acc: {summary[:,1].mean():.2f} ± {summary[:,1].std():.2f}%')

            # 可选：把总结果写进 WANDB、CSV 等
            return summary

    # ------------------------------------------------------------------
    # 整个 run 所有 epoch 的 10 指标 → CSV
    # ------------------------------------------------------------------
    def _dump_whole_run_csv(self, run: int):
        now = datetime.now()
        stamp_dir = now.strftime("%m%d")
        timestamp = now.strftime("%m%d-%H%M%S")

        save_dir = mkdirs(
            f'results/runs/{self.args.dataset}/{stamp_dir}/{self.args.wandb_name}'
        )
        csv_path = f'{save_dir}/{run}-{self.args.run_id}-results.csv'

        header = [
            "Epoch",
            'val/loss', 'val/acc', 'val/f1', 'val/precision', 'val/recall',
            'test/loss', 'test/acc', 'test/f1', 'test/precision', 'test/recall'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, metrics in enumerate(self.results[run]):
                writer.writerow([idx * self.args.eval_step] + metrics)
            writer.writerow([]); writer.writerow(['Args'])
            for k, v in vars(self.args).items():
                writer.writerow([k, v])
        print(f'📄 Saved per-epoch metrics → {csv_path}')

    # ------------------------------------------------------------------
    # 训练全部完成后：一次性导出 Excel（每 run 一行，10 列）
    # 外部调用：logger.save_to_excel("results/exp_metrics.xlsx")
    # ------------------------------------------------------------------
    def save_to_excel(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        columns = [
            'val/loss', 'val/acc', 'val/f1', 'val/precision', 'val/recall',
            'test/loss', 'test/acc', 'test/f1', 'test/precision', 'test/recall'
        ]
        # 取各 run “最佳 val_acc epoch” 的 10 指标
        rows = []
        for r in self.results:
            r = torch.tensor(r)
            best_ep = r[:, 1].argmax()
            rows.append(r[best_ep].tolist())
        df = pd.DataFrame(rows, columns=columns)
        df.to_excel(file_path, index=False)
        print(f'✅ Excel saved : {file_path}')


# ----------------------------------------------------------------------
# 保留你原来的 save_result()，无需修改
# ----------------------------------------------------------------------
def save_result(args, results):
    if args.save_result:
        os.makedirs(f'results/{args.dataset}', exist_ok=True)
        filename = f'results/{args.dataset}/{args.method}.csv'
        print(f"Saving results to {filename}")
        with open(filename, 'a+') as f:
            f.write(
                f"{args.method} {args.kernel}: {args.weight_decay} "
                f"{args.dropout} {args.num_layers} {args.alpha}: "
                f"{args.hidden_channels}: "
                f"{results.mean():.2f} ± {results.std():.2f}\n"
            )
def save_to_excel(self, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    columns = [
        'run',
        'val/loss', 'val/acc', 'val/f1', 'val/precision', 'val/recall',
        'test/loss', 'test/acc', 'test/f1', 'test/precision', 'test/recall'
    ]
    rows = []

    for run_id, r in enumerate(self.results):
        r = torch.tensor(r)
        best_ep = r[:, 6].argmax()  # 基于 test_acc 最大
        row = [run_id] + r[best_ep].tolist()
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)

    # 若已存在则追加
    if os.path.exists(file_path):
        old_df = pd.read_excel(file_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(file_path, index=False)
    print(f'✅ Excel 已保存: {file_path}')
