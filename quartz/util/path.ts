import {
  getFileExtension as sharedGetFileExtension,
  slugifyFilePath as sharedSlugifyFilePath,
} from "@quartz-community/utils"
import type { FilePath } from "@quartz-community/utils"

// Re-export shared path utilities from @quartz-community/utils
export {
  isFilePath,
  isFullSlug,
  isSimpleSlug,
  isRelativeURL,
  isAbsoluteURL,
  getFullSlug,
  slugifyFilePath,
  simplifySlug,
  joinSegments,
  endsWith,
  trimSuffix,
  stripSlashes,
  getFileExtension,
  isFolderPath,
  getAllSegmentPrefixes,
  pathToRoot,
  resolveRelative,
  splitAnchor,
  slugTag,
  transformInternalLink,
  transformLink,
  normalizeHastElement,
} from "@quartz-community/utils"

export type { FilePath }
export type { FullSlug, SimpleSlug, RelativeURL, TransformOptions } from "@quartz-community/utils"

/**
 * Quartz treats .html as a page extension when generating slugs, but standalone HTML files in
 * content are assets. Keep their extension so static hosts return text/html instead of an
 * extensionless download.
 */
export function slugifyAssetFilePath(fp: FilePath): FilePath {
  const slug = sharedSlugifyFilePath(fp)
  return (sharedGetFileExtension(fp)?.toLowerCase() === ".html" ? `${slug}.html` : slug) as FilePath
}

// --- v5-specific exports below ---

export const QUARTZ = "quartz"

// from micromorph/src/utils.ts
// https://github.com/natemoo-re/micromorph/blob/main/src/utils.ts#L5
const _rebaseHtmlElement = (el: Element, attr: string, newBase: string | URL) => {
  const rebased = new URL(el.getAttribute(attr)!, newBase)
  el.setAttribute(attr, rebased.pathname + rebased.hash)
}
export function normalizeRelativeURLs(el: Element | Document, destination: string | URL) {
  el.querySelectorAll('[href=""], [href^="./"], [href^="../"]').forEach((item) => {
    _rebaseHtmlElement(item, "href", destination)
  })
  el.querySelectorAll('[src=""], [src^="./"], [src^="../"]').forEach((item) => {
    _rebaseHtmlElement(item, "src", destination)
  })
}
