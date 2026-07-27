let cleanupSiteHeader = () => {}

const bindSiteHeader = () => {
  cleanupSiteHeader()
  const cleanupCallbacks: Array<() => void> = []

  const navDropdowns = Array.from(
    document.querySelectorAll<HTMLDetailsElement>(".site-nav-dropdown"),
  )
  if (navDropdowns.length > 0) {
    const closeDropdowns = (except?: HTMLDetailsElement) => {
      for (const dropdown of navDropdowns) {
        if (dropdown !== except) dropdown.open = false
      }
    }
    const handleDocumentClick = (event: MouseEvent) => {
      const target = event.target as Node | null
      const activeDropdown = navDropdowns.find((dropdown) => target && dropdown.contains(target))
      closeDropdowns(activeDropdown)
    }
    const handleDropdownKeydown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return
      const openDropdown = navDropdowns.find((dropdown) => dropdown.open)
      if (!openDropdown) return
      openDropdown.open = false
      openDropdown.querySelector<HTMLElement>("summary")?.focus()
    }

    document.addEventListener("click", handleDocumentClick)
    document.addEventListener("keydown", handleDropdownKeydown)
    cleanupCallbacks.push(() => {
      document.removeEventListener("click", handleDocumentClick)
      document.removeEventListener("keydown", handleDropdownKeydown)
    })
  }

  const menuButton = document.querySelector<HTMLButtonElement>(".site-menu-toggle")
  const explorerButton = document.querySelector<HTMLButtonElement>(".mobile-explorer")
  const explorer = explorerButton?.closest<HTMLElement>(".explorer")
  if (menuButton && explorerButton && explorer) {
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

    cleanupCallbacks.push(() => {
      observer.disconnect()
      menuButton.onclick = null
      mobileViewport.removeEventListener("change", resetMenuForViewport)
      document.removeEventListener("keydown", handleKeydown)
    })
  }

  cleanupSiteHeader = () => cleanupCallbacks.forEach((cleanup) => cleanup())
}

document.addEventListener("nav", bindSiteHeader)
document.addEventListener("render", bindSiteHeader)
bindSiteHeader()
