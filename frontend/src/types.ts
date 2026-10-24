export type ClassInfo = {
  id: number
  name: string
  label: string
  color: string
}

export type Box = {
  class_id: number
  x1: number
  y1: number
  x2: number
  y2: number
  score?: number | null
}

export type ImageRow = {
  name: string
  labeled: boolean
  has_robot: boolean
}

export type Frame = {
  name: string
  width: number
  height: number
  boxes: Box[]
  has_robot: boolean
}

export type BatchStatus = {
  running: boolean
  cancel_requested: boolean
  total: number
  done: number
  skipped: number
  written: number
  errors: number
  current: string
  last_error: string | null
  undo_available: boolean
  reverted: number
  kept: number
  message: string
}

export type DetectorStatus = {
  ready: boolean
  reasons: string[]
  checkpoint: string
  threshold: number
  nms_iou: number
}

export type Edge = [[number, number], [number, number]]

export type Meta = {
  classes: ClassInfo[]
  threshold: number
  nms_iou: number
  table_top: Edge | null
  table_bottom: Edge | null
  detector: DetectorStatus
  dataset: {
    open: boolean
    root?: string
    image_dir?: string
    label_dir?: string
  }
}
