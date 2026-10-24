import { useEffect, useRef, useState } from "react"
import type { PointerEvent as ReactPointerEvent } from "react"
import type { Box, ClassInfo } from "@/types"

type Drag =
  | { kind: "move"; index: number; origin: Box; startX: number; startY: number }
  | { kind: "resize"; index: number; corner: number; origin: Box }
  | { kind: "add"; startX: number; startY: number; x: number; y: number }

type Props = {
  src: string
  width: number
  height: number
  boxes: Box[]
  classes: ClassInfo[]
  selected: number
  adding: boolean
  onSelect: (index: number) => void
  onChange: (boxes: Box[]) => void
  onAdd: (box: Box) => void
}

function clampBox(box: Box, width: number, height: number): Box {
  const x1 = Math.max(0, Math.min(box.x1, box.x2, width))
  const y1 = Math.max(0, Math.min(box.y1, box.y2, height))
  const x2 = Math.min(width, Math.max(box.x1, box.x2, 0))
  const y2 = Math.min(height, Math.max(box.y1, box.y2, 0))
  return { ...box, x1, y1, x2, y2 }
}

export function BoxCanvas({
  src,
  width,
  height,
  boxes,
  classes,
  selected,
  adding,
  onSelect,
  onChange,
  onAdd,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [ready, setReady] = useState(false)
  const [drag, setDrag] = useState<Drag | null>(null)

  useEffect(() => {
    const image = new Image()
    image.onload = () => {
      imageRef.current = image
      setReady(true)
    }
    image.src = src
    return () => {
      setReady(false)
    }
  }, [src])

  useEffect(() => {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !ready) return
    const context = canvas.getContext("2d")
    if (!context) return
    canvas.width = width
    canvas.height = height
    context.drawImage(image, 0, 0, width, height)
    boxes.forEach((box, index) => {
      const info = classes.find((item) => item.id === box.class_id)
      const color = info?.color ?? "#111111"
      context.strokeStyle = color
      context.lineWidth = index === selected ? 4 : 2
      context.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1)
      context.fillStyle = color
      context.font = "16px sans-serif"
      const score = box.score == null ? "" : ` ${box.score.toFixed(2)}`
      context.fillText(
        `${index} ${info?.label ?? box.class_id}${score}`,
        box.x1,
        Math.max(16, box.y1 - 6),
      )
      if (index === selected) {
        const corners: Array<[number, number]> = [
          [box.x1, box.y1],
          [box.x2, box.y1],
          [box.x1, box.y2],
          [box.x2, box.y2],
        ]
        for (const [x, y] of corners) context.fillRect(x - 5, y - 5, 10, 10)
      }
    })
    if (drag?.kind === "add") {
      context.save()
      context.setLineDash([6, 4])
      context.strokeStyle = "#111111"
      context.lineWidth = 2
      context.strokeRect(
        Math.min(drag.startX, drag.x),
        Math.min(drag.startY, drag.y),
        Math.abs(drag.x - drag.startX),
        Math.abs(drag.y - drag.startY),
      )
      context.restore()
    }
  }, [boxes, classes, drag, height, ready, selected, width])

  function point(event: ReactPointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) * width) / rect.width,
      y: ((event.clientY - rect.top) * height) / rect.height,
    }
  }

  function onPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    const place = point(event)
    event.currentTarget.setPointerCapture(event.pointerId)
    if (adding) {
      setDrag({ kind: "add", startX: place.x, startY: place.y, x: place.x, y: place.y })
      return
    }
    const handle = 12 * (width / event.currentTarget.getBoundingClientRect().width)
    for (let index = boxes.length - 1; index >= 0; index -= 1) {
      const box = boxes[index]
      const corners = [
        [box.x1, box.y1],
        [box.x2, box.y1],
        [box.x1, box.y2],
        [box.x2, box.y2],
      ]
      const corner = corners.findIndex(([x, y]) => Math.hypot(x - place.x, y - place.y) <= handle)
      const inside = place.x >= box.x1 && place.x <= box.x2 && place.y >= box.y1 && place.y <= box.y2
      if (corner >= 0 || inside) {
        onSelect(index)
        setDrag(
          corner >= 0
            ? { kind: "resize", index, corner, origin: box }
            : { kind: "move", index, origin: box, startX: place.x, startY: place.y },
        )
        return
      }
    }
    onSelect(-1)
    setDrag(null)
  }

  function onPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!drag) return
    const place = point(event)
    if (drag.kind === "add") {
      setDrag({ ...drag, x: place.x, y: place.y })
      return
    }
    if (drag.kind === "move") {
      const dx = place.x - drag.startX
      const dy = place.y - drag.startY
      const next = boxes.map((box, index) =>
        index === drag.index
          ? clampBox(
              {
                ...drag.origin,
                x1: drag.origin.x1 + dx,
                y1: drag.origin.y1 + dy,
                x2: drag.origin.x2 + dx,
                y2: drag.origin.y2 + dy,
              },
              width,
              height,
            )
          : box,
      )
      onChange(next)
      return
    }
    const next = boxes.map((box, index) => {
      if (index !== drag.index) return box
      const moved = { ...drag.origin }
      if (drag.corner % 2 === 0) moved.x1 = place.x
      else moved.x2 = place.x
      if (drag.corner < 2) moved.y1 = place.y
      else moved.y2 = place.y
      return clampBox(moved, width, height)
    })
    onChange(next)
  }

  function onPointerUp() {
    if (drag?.kind === "add") {
      const box = clampBox(
        {
          class_id: classes[0]?.id ?? 0,
          x1: drag.startX,
          y1: drag.startY,
          x2: drag.x,
          y2: drag.y,
          score: null,
        },
        width,
        height,
      )
      if (box.x2 - box.x1 >= 2 && box.y2 - box.y1 >= 2) onAdd(box)
    }
    setDrag(null)
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="max-h-full max-w-full bg-neutral-900"
      style={{ cursor: adding ? "crosshair" : "default" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    />
  )
}
