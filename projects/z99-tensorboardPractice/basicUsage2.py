import numpy as np
from torch.utils.tensorboard import SummaryWriter


# ログをとる対象を増やしてみる
np.random.seed(111)
x1 = np.random.randn(100)
y1 = x1.cumsum()

x2 = np.random.randn(100)
y2 = x2.cumsum()

writer = SummaryWriter(log_dir="./projects/z99-tensorboardPractice/logs/basic2")

# tag = "group_name/value_name" の形式に
for i in range(100):
    writer.add_scalar("X/x1", x1[i], i)
    writer.add_scalar("Y/y1", y1[i], i)
    writer.add_scalar("X/x2", x2[i], i)
    writer.add_scalar("Y/y2", y2[i], i)

writer.close()