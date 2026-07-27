import { componentRegistry } from "./registry"
import type { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { resolveNavigationItems } from "./navigation"

function resolveRegisteredComponent(name: string): QuartzComponent | undefined {
  const registered = componentRegistry.get(name)
  if (!registered) return undefined

  return componentRegistry.instantiate(
    registered.component as QuartzComponentConstructor,
    undefined,
  )
}

const SiteHeader: QuartzComponent = (props: QuartzComponentProps) => {
  const Search = resolveRegisteredComponent("search")
  const Darkmode = resolveRegisteredComponent("darkmode")
  const ReaderMode = resolveRegisteredComponent("reader-mode")
  const slug = props.fileData.slug ?? "index"
  const navItems = resolveNavigationItems(props.cfg.navigation, slug)
  const siteHeader = props.cfg.siteHeader
  const homeLabel = siteHeader?.homeLabel ?? `${props.cfg.pageTitle} 首页`

  return (
    <>
      <a class="skip-link" href="#quartz-main">
        跳到主要内容
      </a>
      <header class="site-header">
        <div class="site-header-inner">
          <a class="site-brand" href="/" aria-label={homeLabel}>
            <span class="site-brand-mark" aria-hidden="true">
              {siteHeader?.mark ?? props.cfg.pageTitle.slice(0, 1)}
            </span>
            <span class="site-brand-text">{props.cfg.pageTitle}</span>
          </a>

          <nav class="site-nav" aria-label="主导航">
            {navItems.map((item) => (
              <a
                key={`${item.label}-${item.href}`}
                class={item.isActive ? "is-active" : undefined}
                href={item.href}
                aria-current={item.isActive ? "page" : undefined}
              >
                {item.label}
              </a>
            ))}
          </nav>

          <div class="site-actions">
            {Search && (
              <div class="site-search">
                <Search {...props} />
                <span class="site-search-shortcut" aria-hidden="true">
                  ⌘K
                </span>
              </div>
            )}
            {ReaderMode && (
              <div class="site-action site-reader">
                <ReaderMode {...props} />
              </div>
            )}
            {Darkmode && (
              <div class="site-action site-theme">
                <Darkmode {...props} />
              </div>
            )}
            {siteHeader?.githubUrl && (
              <a
                class="site-action site-github"
                href={siteHeader.githubUrl}
                target="_blank"
                rel="noreferrer"
                aria-label={siteHeader.githubLabel ?? "在 GitHub 查看此站点"}
              >
                <svg
                  viewBox="0 0 24 24"
                  width="20"
                  height="20"
                  aria-hidden="true"
                  focusable="false"
                >
                  <path d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.2 1.77 1.2 1.04 1.77 2.72 1.26 3.38.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.28-5.27-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.16 1.18A10.9 10.9 0 0 1 12 6.09c.98 0 1.95.13 2.87.39 2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.07.79 2.16v3.25c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" />
                </svg>
              </a>
            )}
            <button
              class="site-action site-menu-toggle"
              type="button"
              aria-label="打开知识导航"
              aria-controls="quartz-left-sidebar"
              aria-expanded="false"
            >
              <svg
                class="site-menu-open"
                viewBox="0 0 24 24"
                width="22"
                height="22"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                aria-hidden="true"
              >
                <path d="M4 7h16M4 12h16M4 17h16" />
              </svg>
              <svg
                class="site-menu-close"
                viewBox="0 0 24 24"
                width="22"
                height="22"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                aria-hidden="true"
              >
                <path d="m6 6 12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        </div>
      </header>
    </>
  )
}

export default (() => SiteHeader) satisfies QuartzComponentConstructor
