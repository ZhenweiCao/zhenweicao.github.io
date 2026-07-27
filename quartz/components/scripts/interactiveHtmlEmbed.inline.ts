type InteractiveHtmlViewer = {
  embed: HTMLElement
  iframe: HTMLIFrameElement
  viewport: HTMLElement
  stage: HTMLElement
  scaleLabel: HTMLElement
  configuredHeight: number
  baseHeight: number
  baseWidth: number
  zoom: number
}

const interactiveHtmlViewers = new WeakMap<HTMLElement, InteractiveHtmlViewer>()

function viewerButton(label: string, title: string, action: () => void) {
  const button = document.createElement("button")
  button.type = "button"
  button.className = "interactive-html-control"
  button.textContent = label
  button.title = title
  button.setAttribute("aria-label", title)
  button.addEventListener("click", action)
  return button
}

function applyInteractiveHtmlScale(viewer: InteractiveHtmlViewer) {
  const { iframe, stage, baseWidth, baseHeight, zoom, scaleLabel } = viewer
  iframe.style.width = `${baseWidth}px`
  iframe.style.height = `${baseHeight}px`
  iframe.style.transform = `scale(${zoom})`
  stage.style.width = `${Math.round(baseWidth * zoom)}px`
  stage.style.height = `${Math.round(baseHeight * zoom)}px`
  scaleLabel.textContent = `${Math.round(zoom * 100)}%`
}

function fitInteractiveHtmlViewer(viewer: InteractiveHtmlViewer) {
  viewer.zoom = 1
  viewer.baseWidth = Math.max(1, viewer.viewport.clientWidth)
  viewer.baseHeight = viewer.embed.matches(":fullscreen")
    ? Math.max(viewer.configuredHeight, viewer.viewport.clientHeight)
    : viewer.configuredHeight
  applyInteractiveHtmlScale(viewer)
  viewer.viewport.scrollTo({ left: 0, top: 0 })
}

function setupInteractiveHtmlEmbed(embed: HTMLElement) {
  if (interactiveHtmlViewers.has(embed)) return

  const iframe = embed.querySelector<HTMLIFrameElement>(":scope > iframe")
  if (!iframe) return

  const toolbar = document.createElement("div")
  toolbar.className = "interactive-html-toolbar"

  const toolbarTitle = document.createElement("span")
  toolbarTitle.className = "interactive-html-toolbar-title"
  toolbarTitle.textContent = "交互图"

  const controls = document.createElement("div")
  controls.className = "interactive-html-controls"

  const scaleLabel = document.createElement("span")
  scaleLabel.className = "interactive-html-scale"
  scaleLabel.setAttribute("aria-live", "polite")

  const viewport = document.createElement("div")
  viewport.className = "interactive-html-viewport"

  const stage = document.createElement("div")
  stage.className = "interactive-html-stage"

  const configuredHeight = Number.parseFloat(
    getComputedStyle(iframe).getPropertyValue("--interactive-html-height"),
  )
  const viewer: InteractiveHtmlViewer = {
    embed,
    iframe,
    viewport,
    stage,
    scaleLabel,
    configuredHeight: Number.isFinite(configuredHeight) ? configuredHeight : 640,
    baseHeight: Number.isFinite(configuredHeight) ? configuredHeight : 640,
    baseWidth: 0,
    zoom: 1,
  }
  interactiveHtmlViewers.set(embed, viewer)

  const zoom = (factor: number) => {
    viewer.zoom = Math.min(3, Math.max(0.5, viewer.zoom * factor))
    applyInteractiveHtmlScale(viewer)
  }

  const fullscreen = viewerButton("⛶", "全屏查看交互图", () => {
    if (document.fullscreenElement === embed) {
      void document.exitFullscreen().catch(() => {})
    } else {
      void embed.requestFullscreen().catch(() => {})
    }
  })
  if (!("requestFullscreen" in embed)) {
    fullscreen.disabled = true
    fullscreen.title = "当前浏览器不支持全屏"
  }

  controls.append(
    viewerButton("−", "缩小交互图", () => zoom(1 / 1.2)),
    viewerButton("适应", "恢复为适应页面宽度", () => fitInteractiveHtmlViewer(viewer)),
    fullscreen,
    viewerButton("+", "放大交互图", () => zoom(1.2)),
  )
  toolbar.append(toolbarTitle, scaleLabel, controls)
  viewport.append(stage)
  stage.append(iframe)
  embed.prepend(toolbar, viewport)
  embed.dataset.interactiveReady = "true"

  requestAnimationFrame(() => fitInteractiveHtmlViewer(viewer))

  if (typeof ResizeObserver !== "undefined") {
    const resizeObserver = new ResizeObserver(() => {
      if (Math.abs(viewport.clientWidth - viewer.baseWidth) > 1) {
        requestAnimationFrame(() => fitInteractiveHtmlViewer(viewer))
      }
    })
    resizeObserver.observe(viewport)
  }
}

function setupInteractiveHtmlEmbeds() {
  document
    .querySelectorAll<HTMLElement>(".interactive-html-embed")
    .forEach(setupInteractiveHtmlEmbed)
}

document.addEventListener("fullscreenchange", () => {
  document
    .querySelectorAll<HTMLElement>(".interactive-html-embed[data-interactive-ready]")
    .forEach((embed) => {
      const viewer = interactiveHtmlViewers.get(embed)
      if (viewer) requestAnimationFrame(() => fitInteractiveHtmlViewer(viewer))
    })
})

let interactiveHtmlResizeTimer = 0
window.addEventListener("resize", () => {
  window.clearTimeout(interactiveHtmlResizeTimer)
  interactiveHtmlResizeTimer = window.setTimeout(() => {
    document
      .querySelectorAll<HTMLElement>(".interactive-html-embed[data-interactive-ready]")
      .forEach((embed) => {
        const viewer = interactiveHtmlViewers.get(embed)
        if (viewer) fitInteractiveHtmlViewer(viewer)
      })
  }, 150)
})

document.addEventListener("nav", setupInteractiveHtmlEmbeds)
document.addEventListener("render", setupInteractiveHtmlEmbeds)
setupInteractiveHtmlEmbeds()
