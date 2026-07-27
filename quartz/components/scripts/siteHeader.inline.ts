let cleanupSiteHeader = () => {}

type TutorialExplorerScope = {
  label: string
  prefix: string
}

const tutorialExplorerScopes: TutorialExplorerScope[] = [
  {
    label: "GPU Kernel Learning",
    prefix: "gpu/gpu-kernel-learning",
  },
  {
    label: "Modern GPU Programming for MLSys",
    prefix: "gpu/modern-gpu-programming-for-mlsys",
  },
]

const activeTutorialExplorerScope = (): TutorialExplorerScope | undefined => {
  const slug = document.body.dataset.slug ?? ""
  return tutorialExplorerScopes.find(
    ({ prefix }) => slug === prefix || slug.startsWith(`${prefix}/`),
  )
}

const bindTutorialExplorerScope = (explorer: HTMLElement): (() => void) => {
  const list = explorer.querySelector<HTMLUListElement>(".explorer-ul")
  const heading = explorer.querySelector<HTMLElement>(".title-button h2")
  if (!list) return () => {}

  const defaultHeading = heading?.textContent ?? "Explorer"

  const applyScope = () => {
    const scope = activeTutorialExplorerScope()
    if (!scope) {
      delete list.dataset.tutorialScope
      delete explorer.dataset.tutorialScope
      if (heading) heading.textContent = defaultHeading
      return
    }

    const targetPath = `${scope.prefix}/index`
    const targetFolder = Array.from(
      list.querySelectorAll<HTMLElement>(".folder-container[data-folderpath]"),
    ).find((folder) => folder.dataset.folderpath === targetPath)
    const directEntries = list.querySelectorAll(":scope > li:not(.overflow-end)")

    if (!targetFolder && list.dataset.tutorialScope === scope.prefix && directEntries.length > 0) {
      return
    }
    if (!targetFolder) return

    const folderItem = targetFolder.closest<HTMLLIElement>("li")
    const scopedContent = folderItem?.querySelector<HTMLUListElement>(
      ":scope > .folder-outer > ul.content",
    )
    if (!scopedContent) return

    const fragment = document.createDocumentFragment()
    for (const child of Array.from(scopedContent.children)) fragment.appendChild(child)

    const overflowEnd =
      list.querySelector<HTMLElement>(":scope > .overflow-end")?.cloneNode(true) ??
      Object.assign(document.createElement("li"), { className: "overflow-end" })
    list.replaceChildren(fragment, overflowEnd)
    list.dataset.tutorialScope = scope.prefix
    explorer.dataset.tutorialScope = scope.prefix
    if (heading) heading.textContent = scope.label
  }

  const observer = new MutationObserver(applyScope)
  observer.observe(list, { childList: true, subtree: true })
  queueMicrotask(applyScope)
  return () => observer.disconnect()
}

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
  if (explorer) cleanupCallbacks.push(bindTutorialExplorerScope(explorer))
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
