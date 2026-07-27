/**
 * Coordinates two related layout controls:
 * - the persistent user preference for the left explorer
 * - Reader Mode's temporary sidebar override
 *
 * Reader Mode is intentionally not written to the explorer preference. When
 * it ends, CSS falls back to the user's previously saved explorer state.
 */
const sidebarStorageKey = "quartz-left-sidebar"
type SidebarState = "open" | "closed"

function getSidebarState(): SidebarState {
  try {
    return localStorage.getItem(sidebarStorageKey) === "closed" ? "closed" : "open"
  } catch {
    return "open"
  }
}

function saveSidebarState(state: SidebarState) {
  try {
    localStorage.setItem(sidebarStorageKey, state)
  } catch {
    // Storage can be unavailable in private browsing contexts. The toggle
    // should still work for the current page.
  }
}

function applySidebarState(state: SidebarState) {
  const isOpen = state === "open"
  document.documentElement.dataset.leftSidebar = state

  for (const button of document.querySelectorAll<HTMLButtonElement>(".left-sidebar-toggle")) {
    const label = isOpen ? "收起左侧目录层级" : "展开左侧目录层级"
    button.setAttribute("aria-expanded", String(isOpen))
    button.setAttribute("aria-label", label)
    button.title = label
  }
}

function applyReaderModeControls(isReaderMode: boolean) {
  const label = isReaderMode ? "退出阅读模式" : "进入阅读模式"
  for (const button of document.querySelectorAll<HTMLButtonElement>(".readermode")) {
    button.setAttribute("aria-pressed", String(isReaderMode))
    button.setAttribute("aria-label", label)
    button.title = label
  }
}

function setupSidebarToggle() {
  applySidebarState(getSidebarState())
  applyReaderModeControls(document.documentElement.getAttribute("reader-mode") === "on")

  for (const button of document.querySelectorAll<HTMLButtonElement>(".left-sidebar-toggle")) {
    button.onclick = () => {
      const nextState: SidebarState =
        document.documentElement.dataset.leftSidebar === "closed" ? "open" : "closed"
      saveSidebarState(nextState)
      applySidebarState(nextState)
    }
  }
}

applySidebarState(getSidebarState())
document.addEventListener("nav", setupSidebarToggle)
document.addEventListener("render", setupSidebarToggle)
document.addEventListener("readermodechange", (event) => {
  const readerModeEvent = event as CustomEvent<{ mode: "on" | "off" }>
  applyReaderModeControls(readerModeEvent.detail.mode === "on")
})
window.addEventListener("storage", (event) => {
  if (event.key === sidebarStorageKey) {
    applySidebarState(event.newValue === "closed" ? "closed" : "open")
  }
})
