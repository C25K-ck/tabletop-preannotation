# 交给下一台电脑的说明

把本文件连同整个 `tabletop-preannotation` 项目、SAM3 源码和 `sam3.pt` 一起拷走。新对话里先读这份说明、`AGENTS.md` 和 `docs/gpu.md`，再改代码。

## 要做的事

给俯视快递台面的图片做三类预标注。画面里始终有一台机械臂小车，台上有 A4 纸、快递和小物件，过程中有时会出现人手。

模型先画草稿框，人再改，最后只保存 YOLO 水平框。LabelImg、X-AnyLabeling 以及原来的三类标注页都能打开这种 txt。

这不是快递传送带那套预标注。那套在 `/home/ck/Downloads/sam3_preannotation_delivery_final`，只认袋子、纸箱、异形包裹，丢掉画面左侧，并且要求视频是 640×480。台面图不能用那些规则。采集客户端 `XCoreDataTool` 也不要放这个项目。

## 已经定下的口径

`classes.txt` 只能是这三行，顺序就是类别编号：

```text
item
robot
hand
```

| 编号 | 名字 | 画面 | 颜色 |
| --- | --- | --- | --- |
| 0 | item | A4 纸、快递、小物件 | 绿 `#22c55e` |
| 1 | robot | 机械臂小车，每张图都应有一个 | 蓝 `#3b82f6` |
| 2 | hand | 人手；没有手就不写 | 红 `#ef4444` |

YOLO-HBB 每行是 `类别 中心x 中心y 宽 高`，相对原图宽高，6 位小数。不写旋转角，不交付 mask 或 COCO。

检测规则：

- SAM3 用英文提示词找东西。白纸、纸箱、小物件都归到 0，机械臂归到 1，人手归到 2。提示词在 `backend/app/taxonomy.py`。
- 只在同一类内部做 NMS，IoU 默认 0.5。不同类的框互不删除。
- 机械臂大框套住爪子里的物品时，两个框都留。
- 不使用左侧忽略线，不要求 640×480。
- 置信度默认 0.30。
- 模型结果是草稿。人改完再点保存，才写入 `labels/*.txt`。
- 关闭分割，不做跟踪，不做视频抽帧，不做切块。

## 程序怎么组成

界面是 Vite、React、TypeScript、Tailwind、shadcn/ui。按钮、列表、滑块、对话框用 shadcn。画框是 `frontend/src/components/box-canvas.tsx` 里的 canvas。

服务是 FastAPI，只监听 `127.0.0.1:8780`。页面开发地址是 `http://127.0.0.1:5173`，由 Vite 把 `/api` 转到 8780。

检测复用 SAM3 源码，不把那份源码复制进本仓库。`config.json` 的 `sam3_root` 指向 SAM3 目录。权重不提交。`checkpoint` 为空时，自动使用 `<sam3_root>/models/sam3.pt`。推理用 CUDA 和 bfloat16，分割头关闭。加载逻辑在 `backend/app/detector.py`，只读取 checkpoint 里 `detector.` 开头的权重。

图片目录有两种摆法：

- 文件夹里直接是图片，标签写到同级 `labels/`
- 文件夹里有 `images/`，标签写到同级 `labels/`。如果打开的就是名为 `images` 的目录，标签写到它的上一级

主要文件：

- `AGENTS.md`：以后改代码要遵守的规则
- `docs/gpu.md`：有显卡的电脑上安装、启动、每天标注的步骤
- `config.json`：SAM3 路径、权重、置信度、NMS
- `backend/app/main.py`：打开目录、读图、读写标签、预标注接口
- `backend/app/yolo.py`、`backend/app/nms.py`：YOLO 读写和同类 NMS
- `backend/tests/test_yolo_nms.py`：不需要显卡的测试
- `frontend/src/App.tsx`：图片列表、改类别、保存、缺机械臂提示、未保存时切换图片的对话框

## 在没有显卡的电脑上已经验证过什么

开发机没有可用 NVIDIA 驱动，交付包里也没有 `sam3.pt`，所以 SAM3 没有真正跑过一张图。

已经做过：

- `backend/tests/test_yolo_nms.py` 通过。覆盖 YOLO 往返、三类文件、同类重叠只留高分、机械臂框套住物品框时两个都留。
- 浏览器里打开图片目录，读出已有框，把机械臂改成人手，保存后的 txt 是合法 YOLO-HBB。
- 没有机械臂框时页面会提示，图片列表里已标注但缺少机械臂的图会标出「缺机械臂」。
- 未保存就切换图片时，对话框可以选择留下或放弃。
- 没有 CUDA 或没有权重时，页面说明原因，手动画框和保存仍然可用。

## 到有显卡的电脑上先做什么

按 `docs/gpu.md` 操作。摘要如下：

1. 拷贝本项目和 SAM3 源码。不要拷贝 `.venv` 和 `frontend/node_modules`。
2. 把 `sam3.pt` 放到 SAM3 目录的 `models/sam3.pt`。
3. 把 `config.json` 的 `sam3_root` 改成新电脑上的实际路径。现在写的是 `/home/ck/Downloads/sam3_preannotation_delivery_final`。
4. 用 Python 3.12 建虚拟环境，安装带 CUDA 的 PyTorch，再执行 `pip install -r backend/requirements.txt`。`requirements.txt` 故意没有写 torch。
5. 确认 `torch.cuda.is_bf16_supported()` 为真。30 系及更新的卡可以。显存建议 16GB，12GB 可以试。
6. `cd frontend && npm install`。
7. 一个终端启动 `python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8780`，另一个终端在 `frontend` 里执行 `npm run dev`。
8. 打开 http://127.0.0.1:5173 。左侧应显示「检测可用」。用一张真实台面图点「预标注」，看机械臂、纸和边上的小物件是否出框，再人工改完保存。

显卡只让模型跑起来并加快速度。同一份权重和同一张图，换更快的卡，框的位置基本一样。小物件和半只手仍可能漏，漏了先调置信度，再考虑以后做切块。

## 训练先不要写

等人工改完并保存的数据交来再写。数据要有 `images/`、`labels/`、`classes.txt`，并能分出哪一天或哪一段录像。训练集和验证集按整天或整段拆开，不要把同一段相邻帧随机拆开。不要用没改过的 SAM3 草稿训练。

到那时训练一个只认这三类的 YOLO，在留出的录像上和 SAM3 比较。YOLO 更全、错得更少之后，再把「预标注」从 SAM3 换成这个 YOLO。页面和 YOLO-HBB 格式不变。不要先微调整份 SAM3。

## 新对话可以这样开头

请先读项目里的 `docs/handoff.md`、`AGENTS.md` 和 `docs/gpu.md`。这是俯视台面的三类预标注，输出只能是 YOLO-HBB。请先把 SAM3 在这台显卡上跑通，用真实图片点一次预标注，再根据漏检和错框决定要不要改提示词或置信度。训练等我交来人工改完的数据再做。
