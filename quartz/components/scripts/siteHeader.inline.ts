let cleanupSiteHeader = () => {}

const bindSiteHeader = () => {
  cleanupSiteHeader()

  const menuButton = document.querySelector<HTMLButtonElement>(".site-menu-toggle")
  const explorerButton = document.querySelector<HTMLButtonElement>(".mobile-explorer")
  const explorer = explorerButton?.closest<HTMLElement>(".explorer")
  if (!menuButton || !explorerButton || !explorer) return

  const mobileViewport = window.matchMedia("(max-width: 799.98px)")

  const syncState = () => {
    const isOpen = mobileViewport.matches && !explorer.classList.contains("collapsed")
    menuButton.classList.toggle("is-open", isOpen)
    menuButton.setAttribute("aria-expanded", String(isOpen))
    menuButton.setAttribute("aria-label", isOpen ? "关闭知识导航" : "打开知识导航")
  }

  const openOrCloseMenu = () => {
    explorerButton.click()
    requestAnimationFrame(syncState)
  }

  const resetMenuForViewport = () => {
    const shouldBeCollapsed = mobileViewport.matches
    if (explorer.classList.contains("collapsed") !== shouldBeCollapsed) {
      explorerButton.click()
    }
    requestAnimationFrame(syncState)
  }

  const handleKeydown = (event: KeyboardEvent) => {
    if (
      event.key !== "Escape" ||
      !mobileViewport.matches ||
      explorer.classList.contains("collapsed")
    )
      return
    openOrCloseMenu()
    menuButton.focus()
  }

  const observer = new MutationObserver(syncState)
  observer.observe(explorer, { attributes: true, attributeFilter: ["class"] })
  menuButton.onclick = openOrCloseMenu
  mobileViewport.addEventListener("change", resetMenuForViewport)
  document.addEventListener("keydown", handleKeydown)
  requestAnimationFrame(resetMenuForViewport)

  cleanupSiteHeader = () => {
    observer.disconnect()
    menuButton.onclick = null
    mobileViewport.removeEventListener("change", resetMenuForViewport)
    document.removeEventListener("keydown", handleKeydown)
  }
}

document.addEventListener("nav", bindSiteHeader)
document.addEventListener("render", bindSiteHeader)
bindSiteHeader()
