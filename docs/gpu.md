# 有显卡之后怎么跑

把本项目和 SAM3 源码拷到有 NVIDIA 显卡的电脑上再跑预标注。显卡让 SAM3 能够出框，并且每张图几秒内完成。框的位置由 `sam3.pt` 和提示词决定，换一块更快的卡不会让同一张图的框变得更准。

当前程序没有训练功能。文末是拿到人工改完的数据之后，用来提高准确度的做法。

## 机器要求

- NVIDIA 显卡，驱动正常。`nvidia-smi` 能看到卡的名字和显存。
- 显卡支持 bfloat16。30 系及更新的卡可以。更老的卡会在预标注时报错。
- 显存建议 16GB 或更多。12GB 可以试。8GB 经常装不下这个模型。
- 安装 Python 3.12 和 Node.js。SAM3 交付环境用的是 Python 3.12。

## 要拷贝什么

拷贝两个目录：

- 本项目 `tabletop-preannotation`
- SAM3 源码目录，里面要有 `sam3/` 这个包。现在的配置指向 `/home/ck/Downloads/sam3_preannotation_delivery_final`

不要拷贝 `.venv` 和 `frontend/node_modules`。到新电脑后重新安装。

权重单独放，不要和代码混在压缩包里传来传去也可以。把 `sam3.pt` 放到 SAM3 目录的 `models/sam3.pt`。`config.json` 里的 `checkpoint` 留空时，程序会自己找这个文件。

## 改配置

编辑项目根目录的 `config.json`，把 `sam3_root` 改成新电脑上 SAM3 源码的实际路径：

```json
{
  "sam3_root": "D:/tools/sam3_preannotation_delivery_final",
  "checkpoint": "",
  "threshold": 0.3,
  "nms_iou": 0.5
}
```

权重不在默认位置时，把 `checkpoint` 写成 `sam3.pt` 的绝对路径。

## 安装

在项目根目录执行。PyTorch 要用带 CUDA 的版本，不要用 CPU 版。下面的 CUDA 12.8 轮子与 SAM3 交付环境一致；新电脑驱动较旧时，改用与驱动匹配的 cu121 或 cu124 轮子。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r backend/requirements.txt
```

Windows 上激活环境用 `.venv\Scripts\activate`。

检查显卡和 bfloat16：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print('bf16', torch.cuda.is_bf16_supported())"
```

两行都正常，并且 `bf16` 是 `True`，再装界面依赖：

```bash
cd frontend
npm install
```

## 启动

开两个终端，都在项目根目录。

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8780
```

```bash
cd frontend
npm run dev
```

浏览器打开 http://127.0.0.1:5173 。

左侧应显示「检测可用」。如果仍是「未配置可用的 sam3.pt」「无法导入 torch」或「当前环境没有可用的 CUDA」，预标注不会出框。先看 `config.json` 的路径，以及当前终端是不是已经激活带 CUDA 的 `.venv`。

## 每天怎么标

1. 在「图片目录」填本机文件夹。目录里直接是图片，或里面有 `images/` 子目录，都可以。
2. 点「打开」。程序会在旁边建立 `labels/` 和 `classes.txt`。`classes.txt` 必须依次是 `item`、`robot`、`hand`。
3. 选中一张图，点「预标注」。结果只出现在画面上，还没写入文件。
4. 改框：选类别、添加、拖动、拉角、删除。机械臂是蓝框，物品是绿框，人手是红框。爪子里的物品和机械臂大框都要留。没有机械臂框时页面会提示。
5. 点「保存」。标签写到 `labels/图片名.txt`，格式是 YOLO 水平框：`类别 中心x 中心y 宽 高`。
6. 这三份东西一起留着：图片、`labels/`、`classes.txt`。LabelImg 和 X-AnyLabeling 也能打开。

置信度默认 0.30。漏检多就往下调，错框多就往上调。调完要重新点「预标注」，已经保存的文件不会自动改。

同一类叠在一起的框只留一个。不同类的框不会互相删掉。

## 后续：用改完的数据提高准确度

这一步现在还没有写进程序。数据准备好之后再做。

不要拿 SAM3 没改过的草稿去训练。只使用人确认并保存过的 `labels/*.txt`。

建议训练一个只认这三类的 YOLO 检测模型，而不是先去微调整份 SAM3。原因是：现场只有这一个俯视机位和这三类；标注已经是 YOLO 水平框；YOLO 用这份数据训练后，对这台机械臂、这张台面、这些小物件会比开放词汇提示词更稳，推理也更快。SAM3 继续负责第一批草稿。YOLO 在留出的数据上明显好过 SAM3 之后，再把预标注接口从 SAM3 换成这个 YOLO 权重。界面和保存格式不变。

数据这样交：

```text
dataset/
  images/
  labels/
  classes.txt
```

`classes.txt` 仍是这三行，顺序不能改。另外用文件夹或清单标出图片属于哪一天、哪一段录像。训练集和验证集按整天或整段录像拆开，不要把同一段里相邻的帧随机拆到两边。验证集不要参与训练。

每张图都应有机械臂框。没有手的图不要写 `hand`。爪子里的小物件和机械臂大框都保留。画面边缘的小物件如果要标，就标进训练集，这样模型才学得到。

训练时固定类别顺序 `item`、`robot`、`hand`。比较时看留出录像上的框是否更全、更少错，而不是只看训练损失。新一轮人工修改继续加进训练集，已经当过验证集的录像不要混回去。
