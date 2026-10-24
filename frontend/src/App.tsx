import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { api } from "@/api"
import { BoxCanvas } from "@/components/box-canvas"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Slider } from "@/components/ui/slider"
import type { BatchStatus, Box, ClassInfo, Frame, ImageRow, Meta } from "@/types"

function batchDoneMessage(status: BatchStatus) {
  if (status.cancel_requested) {
    const kept = status.kept > 0 ? `，${status.kept} 张已改过并保留` : ""
    return `已取消，撤回 ${status.reverted} 张预标注${kept}`
  }
  const skipped = status.skipped > 0 ? `，跳过 ${status.skipped} 张已有标签` : ""
  const errors = status.errors > 0 ? `，${status.errors} 张失败` : ""
  return `已预标注 ${status.written} 张${skipped}${errors}`
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [path, setPath] = useState("")
  const [images, setImages] = useState<ImageRow[]>([])
  const [current, setCurrent] = useState("")
  const [frame, setFrame] = useState<Frame | null>(null)
  const [boxes, setBoxes] = useState<Box[]>([])
  const [selected, setSelected] = useState(-1)
  const [adding, setAdding] = useState(false)
  const [drawClass, setDrawClass] = useState(0)
  const [dirty, setDirty] = useState(false)
  const [threshold, setThreshold] = useState(0.3)
  const [busy, setBusy] = useState(false)
  const [batch, setBatch] = useState<BatchStatus | null>(null)
  const [pendingName, setPendingName] = useState<string | null>(null)
  const currentRef = useRef("")
  const dirtyRef = useRef(false)
  const seenDone = useRef(-1)
  const batchFinishing = useRef(false)
  currentRef.current = current
  dirtyRef.current = dirty

  const classes = meta?.classes ?? []

  const refreshImages = useCallback(async () => {
    const listed = await api.images()
    setImages(listed.images)
    return listed.images
  }, [])

  useEffect(() => {
    api.meta().then((data) => {
      setMeta(data)
      setThreshold(data.threshold)
      if (data.dataset.open && data.dataset.root) setPath(data.dataset.root)
    }).catch((error: Error) => {
      toast.error(error.message)
    })
    api.batch().then(setBatch).catch((error: Error) => {
      toast.error(error.message)
    })
  }, [])

  useEffect(() => {
    if (!batch?.running) return
    let stopped = false
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const status = await api.batch()
          if (stopped) return
          setBatch(status)
          if (status.done !== seenDone.current) {
            seenDone.current = status.done
            if (status.current && status.current === currentRef.current) await reloadCurrent()
          }
          if (!status.running) {
            if (batchFinishing.current) return
            batchFinishing.current = true
            await refreshImages()
            await reloadCurrent()
            if (!stopped) toast.success(batchDoneMessage(status))
          }
        } catch (error) {
          if (!stopped) toast.error(error instanceof Error ? error.message : "预标注状态读取失败")
        }
      })()
    }, 400)
    return () => {
      stopped = true
      window.clearInterval(timer)
    }
  }, [batch?.running, refreshImages])

  async function openFolder() {
    setBusy(true)
    try {
      await api.open(path)
      const listed = await refreshImages()
      setCurrent("")
      setFrame(null)
      setBoxes([])
      setDirty(false)
      if (listed[0]) await showImage(listed[0].name, false)
      toast.success(`已打开 ${listed.length} 张图片`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "打开失败")
    } finally {
      setBusy(false)
    }
  }

  async function showImage(name: string, guardDirty: boolean) {
    if (guardDirty && dirty) {
      setPendingName(name)
      return
    }
    const data = await api.labels(name)
    setCurrent(name)
    setFrame(data)
    setBoxes(data.boxes)
    setSelected(-1)
    setAdding(false)
    setDirty(false)
  }

  function updateBoxes(next: Box[]) {
    setBoxes(next)
    setDirty(true)
  }

  async function save() {
    if (!current) return
    setBusy(true)
    try {
      const saved = await api.save(current, boxes)
      setBoxes(saved.boxes)
      setFrame(saved)
      setDirty(false)
      await refreshImages()
      toast.success("已写入 YOLO 标签")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存失败")
    } finally {
      setBusy(false)
    }
  }

  async function detect() {
    if (!current || !meta) return
    if (!meta.detector.ready) {
      toast.error(meta.detector.reasons.join("；"))
      return
    }
    setBusy(true)
    try {
      const found = await api.detect(current, threshold)
      setBoxes(found.boxes)
      setFrame({ ...found })
      setSelected(-1)
      setDirty(true)
      toast.success(`预标注 ${found.boxes.length} 个框，确认后点保存`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "预标注失败")
    } finally {
      setBusy(false)
    }
  }

  async function reloadCurrent() {
    const name = currentRef.current
    if (!name || dirtyRef.current) return
    const data = await api.labels(name)
    if (currentRef.current !== name || dirtyRef.current) return
    setFrame(data)
    setBoxes(data.boxes)
    setSelected(-1)
    setDirty(false)
  }

  async function startBatch() {
    if (!meta?.detector.ready) {
      toast.error(meta?.detector.reasons.join("；") || "检测不可用")
      return
    }
    seenDone.current = -1
    batchFinishing.current = false
    try {
      const status = await api.batchStart(threshold, dirty ? current : null)
      setBatch(status)
      if (!status.running) toast.success(status.message || "没有需要预标注的图片")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "一键预标注失败")
    }
  }

  async function cancelBatch() {
    try {
      const status = await api.batchCancel()
      setBatch(status)
      if (!status.running) {
        await refreshImages()
        await reloadCurrent()
        toast.success(batchDoneMessage(status))
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "取消预标注失败")
    }
  }

  function changeClass(classId: number) {
    setDrawClass(classId)
    if (selected < 0) return
    updateBoxes(boxes.map((box, index) => (index === selected ? { ...box, class_id: classId, score: null } : box)))
  }

  function removeSelected() {
    if (selected < 0) return
    updateBoxes(boxes.filter((_, index) => index !== selected))
    setSelected(-1)
  }

  function addBox(box: Box) {
    const next = [...boxes, { ...box, class_id: drawClass }]
    setBoxes(next)
    setSelected(next.length - 1)
    setAdding(false)
    setDirty(true)
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Delete" || event.key === "Backspace") {
        const target = event.target
        if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return
        removeSelected()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  const missingRobot = frame !== null && !boxes.some((box) => box.class_id === 1)
  const classOf = (id: number): ClassInfo | undefined => classes.find((item) => item.id === id)
  const activeClass = selected >= 0 ? boxes[selected]?.class_id : drawClass

  return (
    <div className="flex h-full min-h-0 bg-background">
      <aside className="flex w-80 shrink-0 flex-col border-r">
        <div className="space-y-3 p-4">
          <div>
            <h1 className="text-lg font-semibold">台面三类预标注</h1>
            <p className="text-sm text-muted-foreground">单张预标注确认后才保存。一键预标注会写入还没有标签的图片，取消可撤回没改过的结果。</p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="folder">图片目录</Label>
            <Input
              id="folder"
              value={path}
              placeholder="/path/to/images"
              onChange={(event) => setPath(event.target.value)}
            />
            <Button className="w-full" disabled={busy || batch?.running || !path} onClick={() => void openFolder()}>
              打开
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            {meta?.detector.ready
              ? "检测可用"
              : meta?.detector.reasons.join("；") || "正在检查检测环境"}
          </p>
          {missingRobot ? (
            <p className="text-xs text-muted-foreground">这张图还没有机械臂框。</p>
          ) : null}
          {batch?.running ? (
            <p className="truncate text-xs text-muted-foreground">
              一键预标注 {batch.done}/{batch.total}
              {batch.current ? ` ${batch.current}` : ""}
            </p>
          ) : null}
        </div>
        <Separator />
        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-1 p-2">
            {images.map((image) => (
              <Button
                key={image.name}
                variant={image.name === current ? "secondary" : "ghost"}
                className="h-auto justify-start px-2 py-1.5 text-left"
                onClick={() => void showImage(image.name, true)}
              >
                <span className="truncate">{image.name}</span>
                {image.labeled && !image.has_robot ? (
                  <Badge variant="outline" className="ml-auto">缺机械臂</Badge>
                ) : null}
              </Button>
            ))}
          </div>
        </ScrollArea>
      </aside>
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex flex-wrap items-center gap-2 border-b p-3">
          {classes.map((item) => (
            <Button
              key={item.id}
              variant={activeClass === item.id ? "default" : "outline"}
              onClick={() => changeClass(item.id)}
            >
              <span className="size-2.5 rounded-full" style={{ background: item.color }} />
              {item.id} {item.label}
            </Button>
          ))}
          <Separator orientation="vertical" className="mx-1 h-6" />
          <Button variant={adding ? "default" : "outline"} disabled={!frame} onClick={() => setAdding((value) => !value)}>
            {adding ? "拖动画框" : "添加框"}
          </Button>
          <Button variant="outline" disabled={selected < 0} onClick={removeSelected}>删除</Button>
          <Button disabled={!frame || busy || batch?.running} onClick={() => void save()}>{dirty ? "保存" : "已保存"}</Button>
          <Button variant="secondary" disabled={!frame || busy || batch?.running} onClick={() => void detect()}>预标注</Button>
          <Button
            variant="secondary"
            disabled={busy || batch?.running || !meta?.detector.ready || images.length === 0}
            onClick={() => void startBatch()}
          >
            {batch?.running ? `预标注 ${batch.done}/${batch.total}` : "一键预标注"}
          </Button>
          <Button
            variant="outline"
            disabled={!batch?.running && !batch?.undo_available}
            onClick={() => void cancelBatch()}
          >
            取消预标注
          </Button>
          <div className="flex min-w-40 items-center gap-2">
            <Label className="shrink-0">置信度 {threshold.toFixed(2)}</Label>
            <Slider
              value={[threshold]}
              min={0.05}
              max={0.9}
              step={0.05}
              onValueChange={(value) => setThreshold(value[0] ?? threshold)}
            />
          </div>
        </div>
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto bg-muted p-4">
          {frame && current ? (
            <BoxCanvas
              src={api.mediaUrl(current)}
              width={frame.width}
              height={frame.height}
              boxes={boxes}
              classes={classes}
              selected={selected}
              adding={adding}
              onSelect={setSelected}
              onChange={updateBoxes}
              onAdd={addBox}
            />
          ) : (
            <p className="text-sm text-muted-foreground">打开一个图片目录。</p>
          )}
        </div>
        <ScrollArea className="h-28 border-t">
          <div className="flex flex-wrap gap-2 p-3">
            {boxes.map((box, index) => (
              <Button
                key={`${index}-${box.class_id}`}
                variant={index === selected ? "secondary" : "outline"}
                onClick={() => setSelected(index)}
              >
                <span className="size-2.5 rounded-full" style={{ background: classOf(box.class_id)?.color }} />
                {index} {classOf(box.class_id)?.label}
              </Button>
            ))}
          </div>
        </ScrollArea>
      </main>
      <Dialog open={pendingName !== null} onOpenChange={(open) => { if (!open) setPendingName(null) }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>还有未保存的修改</DialogTitle>
            <DialogDescription>切换图片会丢掉当前框的修改。可以先保存，或放弃修改再切换。</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPendingName(null)}>留在这张图</Button>
            <Button
              variant="destructive"
              onClick={() => {
                const name = pendingName
                setPendingName(null)
                if (name) void showImage(name, false)
              }}
            >
              放弃并切换
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
