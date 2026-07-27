import type { NavigationConfiguration, PinnedNavigationItem } from "../cfg"
import { slugTag } from "../util/path"

export type ResolvedNavigationItem = {
  label: string
  href: string
  matchPrefix: string
  isActive: boolean
}

export const DEFAULT_NAVIGATION: NavigationConfiguration = {
  pinnedLimit: 4,
  pinnedItems: [
    { label: "首页", href: "/" },
    {
      label: "知识库",
      directory: "GPU",
      href: "/gpu/gpu-知识库索引",
    },
    {
      label: "系统课程",
      directory: "GPU/GPU-Kernel-Learning",
      href: "/gpu/gpu-kernel-learning/readme",
    },
    { label: "标签", href: "/tags/" },
  ],
}

function stripQueryAndHash(value: string): string {
  return value.split(/[?#]/, 1)[0]
}

export function normalizeNavigationDirectory(directory: string): string {
  const normalizedSeparators = directory.trim().replaceAll("\\", "/")
  const withoutRoot = normalizedSeparators.replace(/^\/+/, "").replace(/^content\//i, "")
  const withoutTrailingSlash = withoutRoot.replace(/\/+$/, "")
  return withoutTrailingSlash ? slugTag(withoutTrailingSlash) : ""
}

function normalizeHref(href: string): string {
  const trimmed = href.trim()
  if (trimmed === "/") return "/"
  return `/${trimmed.replace(/^\/+/, "")}`
}

function hrefToMatchPrefix(href: string): string {
  const pathname = stripQueryAndHash(href)
  const withoutSlashes = pathname.replace(/^\/+|\/+$/g, "")
  return withoutSlashes || "index"
}

function resolveItem(item: PinnedNavigationItem): Omit<ResolvedNavigationItem, "isActive"> | null {
  const label = item.label.trim()
  const directory = item.directory ? normalizeNavigationDirectory(item.directory) : ""
  const href = item.href?.trim()
    ? normalizeHref(item.href)
    : directory
      ? `/${directory}/`
      : undefined

  if (!label || !href) return null

  return {
    label,
    href,
    matchPrefix: directory || hrefToMatchPrefix(href),
  }
}

function matchesSlug(slug: string, prefix: string): boolean {
  if (prefix === "index") return slug === "index"
  return slug === prefix || slug.startsWith(`${prefix}/`)
}

export function resolveNavigationItems(
  navigation: NavigationConfiguration | undefined,
  currentSlug: string,
): ResolvedNavigationItem[] {
  const config = navigation ?? DEFAULT_NAVIGATION
  const resolvedItems = config.pinnedItems.flatMap((item) => {
    const resolved = resolveItem(item)
    return resolved ? [resolved] : []
  })
  const configuredLimit = config.pinnedLimit ?? resolvedItems.length
  const limit = Math.max(0, Math.floor(configuredLimit))
  const visibleItems = resolvedItems.slice(0, limit)

  let activeIndex = -1
  let activeSpecificity = -1

  visibleItems.forEach((item, index) => {
    if (!matchesSlug(currentSlug, item.matchPrefix)) return

    const specificity = item.matchPrefix.split("/").length
    if (specificity > activeSpecificity) {
      activeIndex = index
      activeSpecificity = specificity
    }
  })

  return visibleItems.map((item, index) => ({
    ...item,
    isActive: index === activeIndex,
  }))
}
